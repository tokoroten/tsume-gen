//! The compact transposition table (`--table compact`, owner 2026-10-15).
//!
//! The exact search ([`Solver::solve`]) normally keeps two things per node: the
//! table entry (a 128-bit key and an 8-byte value in a `HashMap`, 28-57 bytes a
//! node with its load factor, twice that while it grows) and, with a graph, the
//! node and its moves (12 bytes plus 10 per recorded move).  The compact table
//! keeps only a **22-byte slot** per node -- the same 128-bit key, the node's
//! search id and its verdict -- in an open-addressing table that grows one
//! segment at a time up to the size the node budget needs, and builds no graph
//! while it searches.  Everything the published record reads from the graph is
//! recovered afterwards:
//!
//! * the **whole-graph numbers** -- the difficulty features' per-node counts and
//!   the per-depth node counts behind `nodesNonFutile` -- are summed while the
//!   search runs, over exactly the moves graph mode records (every node records
//!   its moves into a stack that is popped when the node is done);
//! * the **solution region** -- what the main line, rules B, C and G, the
//!   runtime graph and the quality shape read -- is rebuilt by a second,
//!   value-guided pass: from the root it regenerates each node's moves and reads
//!   every child's verdict and id from the table, exactly as graph mode recorded
//!   them ([`Solver::materialize`]), and keeps those nodes as a sparse
//!   [`Graph`] under their search ids;
//! * what that rebuild cannot recompute -- the verdicts rule F's credit oracle
//!   gave at a credit-line node -- is kept during the search, for the few nodes
//!   whose resolution reached the credit search, as their recorded moves
//!   ([`CompactState::patches`]);
//! * the counts that need reachability (`falseMoveDifficulty`, and
//!   `nodesNonFutile` when a 無駄合 edge has a child) are counted by walking the
//!   table from the root, once per wrong root move, on a solved mate only.
//!
//! The search itself is the sequential one, node for node: the same nodes in
//! the same order, so `stats` (`nodes`, `cacheHits`, the budget) and every
//! verdict are identical, and so is the record.
//!
//! ## Keys and collisions
//!
//! The key is the table key of the exact search ([`super::table_key`]): the
//! position's 128-bit Zobrist key XOR a 128-bit salt per remaining bound.  The
//! exact search's `HashMap` already stores nothing else, so the compact table
//! makes no new assumption.  For `n` distinct `(remaining, position)` pairs the
//! chance that two share a key is about `n^2 / 2^129`: 6e-23 at 2e8 nodes (a
//! 64-bit key would give 1e-3, unacceptable for an exact gate).
//! [`TableMode::CompactVerify`] also stores every position and asserts on every
//! hit that the key's position is the one asked for.
//!
//! ## Layout
//!
//! 64 segments chosen by the key's top six bits, each a Robin Hood
//! linear-probing array of packed slots (`k0, k1: u64`, `id: u32`, `value:
//! u16`).  A segment doubles when it is 7/8 full, up to the capacity the node
//! budget needs at 7/8 load, so the table never holds more than about 25 bytes
//! per budget node, and only one segment is copied at a time.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use super::graph::{CompactFacts, NO_MOVE, NO_NODE};
use super::{
    table_key, ChildInfo, FastHasher, Graph, GraphMove, GraphNode, Internal, Key, Outcome,
    SearchLimits, SolveResult, Solver, UselessInterposition,
};
use crate::features::NodeCounts;
use crate::position::{Move, Position};

/// Which transposition table [`Solver::solve`] uses ([`Solver::set_table`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TableMode {
    /// The `HashMap` table and, when asked, the full graph (the default).
    #[default]
    Exact,
    /// The compact table: 22 bytes a node, the solution region rebuilt
    /// afterwards.  The result is identical to `Exact`'s.
    Compact,
    /// `Compact`, also storing every position to assert that no two positions
    /// share a 128-bit key (verification only: it costs the full positions).
    CompactVerify,
}

impl TableMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            TableMode::Exact => "exact",
            TableMode::Compact => "compact",
            TableMode::CompactVerify => "compact-verify",
        }
    }

    /// Parses the CLI spelling (`exact` or `compact`), `None` for anything
    /// else.  [`TableMode::CompactVerify`] is for tests, through
    /// [`crate::solver::Solver::set_table`] (off the CLI since 2026-10-27).
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "exact" => Some(TableMode::Exact),
            "compact" => Some(TableMode::Compact),
            _ => None,
        }
    }

    /// `true` for both compact modes.
    pub const fn is_compact(self) -> bool {
        !matches!(self, TableMode::Exact)
    }
}

/// One table slot: the key, the node's search id and its packed verdict.
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Slot {
    k0: u64,
    k1: u64,
    id: u32,
    value: u16,
}

const _: () = assert!(std::mem::size_of::<Slot>() == 22);

/// `Slot::value` of an empty slot (outcome bits `3`, which no verdict has).
const EMPTY: u16 = u16::MAX;
/// The distance bits of "no distance".
const NO_DISTANCE: u16 = 0x3FFF;

const EMPTY_SLOT: Slot = Slot {
    k0: 0,
    k1: 0,
    id: 0,
    value: EMPTY,
};

/// Packs a verdict: the outcome in bits 0-1, the distance in bits 2-15.
#[inline]
pub(super) fn pack(outcome: Outcome, distance: Option<u32>) -> u16 {
    let outcome = match outcome {
        Outcome::Mate => 0,
        Outcome::Refuted => 1,
        Outcome::Unknown => 2,
    };
    let distance = match distance {
        Some(d) => {
            assert!(
                d < u32::from(NO_DISTANCE),
                "a mate distance fits the compact table"
            );
            d as u16
        }
        None => NO_DISTANCE,
    };
    outcome | (distance << 2)
}

#[inline]
pub(super) fn unpack(value: u16, id: u32) -> Internal {
    let outcome = match value & 3 {
        0 => Outcome::Mate,
        1 => Outcome::Refuted,
        _ => Outcome::Unknown,
    };
    let distance = value >> 2;
    Internal {
        outcome,
        distance: (distance != NO_DISTANCE).then_some(u32::from(distance)),
        id,
    }
}

/// Segments, chosen by the key's top bits.
const SEGMENT_BITS: u32 = 6;
const SEGMENTS: usize = 1 << SEGMENT_BITS;
/// The capacity a segment starts with (and returns to between searches).
const MIN_CAPACITY: usize = 64;
/// Slots a segment logs as filled (see [`Segment::filled`]) before it falls
/// back to clearing every slot: 64 x 32 Ki x 4 bytes = 8 MiB at most.
const FILLED_LOG_LIMIT: usize = 1 << 15;

#[derive(Default)]
struct Segment {
    slots: Vec<Slot>,
    len: usize,
    /// The slots inserts filled since the segment was last cleared: every
    /// insert turns exactly one empty slot into an occupied one (Robin Hood
    /// only moves keys between occupied slots), so emptying these empties the
    /// segment.  A small search on a large (preallocated) segment is then
    /// cleared in time proportional to its own size.
    filled: Vec<u32>,
    /// More than [`FILLED_LOG_LIMIT`] fills (or a capacity past `u32`): the
    /// next clear writes every slot.
    filled_overflow: bool,
}

/// The home slot of a key in a segment of `capacity` slots.
#[inline]
fn home(k1: u64, capacity: usize) -> usize {
    ((u128::from(k1) * capacity as u128) >> 64) as usize
}

impl Segment {
    fn new(capacity: usize) -> Self {
        Segment {
            slots: vec![EMPTY_SLOT; capacity],
            len: 0,
            filled: Vec::new(),
            filled_overflow: false,
        }
    }

    /// Empties every slot, keeping the allocation.
    fn clear(&mut self) {
        if self.filled_overflow {
            self.slots.fill(EMPTY_SLOT);
        } else {
            for &index in self.filled.iter() {
                self.slots[index as usize] = EMPTY_SLOT;
            }
        }
        debug_assert!(self.slots.iter().all(|slot| slot.value == EMPTY));
        self.filled.clear();
        self.filled_overflow = false;
        self.len = 0;
    }

    #[inline]
    fn get(&self, k0: u64, k1: u64) -> Option<(u32, u16)> {
        let capacity = self.slots.len();
        let mut index = home(k1, capacity);
        let mut probe = 0usize;
        loop {
            let slot = self.slots[index];
            if slot.value == EMPTY {
                return None;
            }
            if slot.k0 == k0 && slot.k1 == k1 {
                return Some((slot.id, slot.value));
            }
            // Robin Hood: a key never sits farther from home than a key it
            // passed, so a slot closer to its own home ends the search.
            let own = (index + capacity - home(slot.k1, capacity)) % capacity;
            if own < probe {
                return None;
            }
            index += 1;
            if index == capacity {
                index = 0;
            }
            probe += 1;
        }
    }

    /// Inserts a key that is not in the segment; the caller made room.
    #[inline]
    fn insert(&mut self, mut slot: Slot) {
        let capacity = self.slots.len();
        let mut index = home(slot.k1, capacity);
        let mut probe = 0usize;
        loop {
            let resident = &mut self.slots[index];
            if resident.value == EMPTY {
                *resident = slot;
                self.len += 1;
                if !self.filled_overflow {
                    match u32::try_from(index) {
                        Ok(index) if self.filled.len() < FILLED_LOG_LIMIT => {
                            self.filled.push(index)
                        }
                        _ => {
                            self.filled_overflow = true;
                            self.filled = Vec::new();
                        }
                    }
                }
                return;
            }
            let own = (index + capacity - home(resident.k1, capacity)) % capacity;
            if own < probe {
                std::mem::swap(resident, &mut slot);
                probe = own;
            }
            index += 1;
            if index == capacity {
                index = 0;
            }
            probe += 1;
        }
    }

    /// Rebuilds the segment at `capacity` slots.
    fn resize(&mut self, capacity: usize) {
        let old = std::mem::replace(&mut self.slots, vec![EMPTY_SLOT; capacity]);
        self.len = 0;
        self.filled.clear();
        self.filled_overflow = false;
        for slot in old {
            if slot.value != EMPTY {
                self.insert(slot);
            }
        }
    }
}

/// The compact table itself ([`TableMode::Compact`]).
pub(super) struct CompactTable {
    segments: Vec<Segment>,
    /// The segment capacity the node budget needs at 7/8 load: growth doubles
    /// up to it, and past it (a segment fuller than average) by a quarter.
    target: usize,
    /// [`TableMode::CompactVerify`]: every key's position.
    verify: Option<HashMap<Key, Position, BuildHasherDefault<FastHasher>>>,
}

impl CompactTable {
    fn new(verify: bool) -> Self {
        CompactTable {
            segments: (0..SEGMENTS).map(|_| Segment::new(MIN_CAPACITY)).collect(),
            target: MIN_CAPACITY,
            verify: verify.then(HashMap::default),
        }
    }

    /// Empties the table for a search of at most `node_budget + 1` nodes.
    ///
    /// A segment a previous search grew is released, so a small search after
    /// a large one neither clears nor keeps the large allocation (a mode that
    /// allocated every segment at the budget's capacity up front was removed
    /// on 2026-10-27).
    fn reset(&mut self, node_budget: u64) {
        // (budget + 1) nodes over the segments at 7/8 load, 2 % slack for the
        // spread between segments; computed in u128 so no budget overflows.
        let per_segment = (u128::from(node_budget) + 1).div_ceil(SEGMENTS as u128);
        let target = (per_segment * 8 / 7 * 102 / 100 + 16).min(1 << 40);
        self.target = (target as usize).max(MIN_CAPACITY);
        for segment in self.segments.iter_mut() {
            if segment.slots.len() > 4 * MIN_CAPACITY {
                *segment = Segment::new(MIN_CAPACITY);
            } else {
                segment.clear();
            }
        }
        if let Some(verify) = self.verify.as_mut() {
            *verify = HashMap::default();
        }
    }

    #[inline]
    fn segment(key: &Key) -> usize {
        (key[0] >> (64 - SEGMENT_BITS)) as usize
    }

    #[inline]
    fn get(&self, key: &Key, pos: &Position) -> Option<Internal> {
        let (id, value) = self.segments[Self::segment(key)].get(key[0], key[1])?;
        if let Some(verify) = self.verify.as_ref() {
            let stored = verify
                .get(key)
                .expect("compact-verify: every stored key has its position");
            assert!(
                stored.board == pos.board && stored.hands == pos.hands && stored.side == pos.side,
                "compact-verify: two positions share a 128-bit table key"
            );
        }
        Some(unpack(value, id))
    }

    #[inline]
    fn insert(&mut self, key: Key, value: Internal, pos: &Position) {
        let index = Self::segment(&key);
        let target = self.target;
        let segment = &mut self.segments[index];
        debug_assert!(
            segment.get(key[0], key[1]).is_none(),
            "a key is inserted once"
        );
        let capacity = segment.slots.len();
        if (segment.len + 1) * 8 > capacity * 7 {
            let grown = if capacity < target {
                (capacity * 2).min(target)
            } else {
                capacity + capacity / 4 + 1
            };
            segment.resize(grown);
        }
        segment.insert(Slot {
            k0: key[0],
            k1: key[1],
            id: value.id,
            value: pack(value.outcome, value.distance),
        });
        if let Some(verify) = self.verify.as_mut() {
            verify.insert(key, pos.clone());
        }
    }

    /// Bytes the slots take (measurement).
    pub(super) fn bytes(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.slots.len() * std::mem::size_of::<Slot>())
            .sum()
    }

    /// Keys stored.
    pub(super) fn len(&self) -> usize {
        self.segments.iter().map(|segment| segment.len).sum()
    }
}

/// A compact solver's state: the table and what the search records instead of
/// a graph.  Reset with every search.
pub(super) struct CompactState {
    pub(super) table: CompactTable,
    /// The recorded moves of every credit-line defence node whose resolution
    /// reached the credit search ([`Solver::resolve_with_credit`] past its
    /// first loop), by table key: the credit oracle's verdicts are not
    /// recomputed by the second pass.
    pub(super) patches: HashMap<Key, Box<[GraphMove]>, BuildHasherDefault<FastHasher>>,
    /// The difficulty-feature counts of every node created.
    pub(super) counts: NodeCounts,
    /// Nodes created per remaining bound (`nodesNonFutile` without 無駄合
    /// edges: every node at depth `max_ply - remaining`).
    pub(super) per_remaining: Vec<u64>,
    /// Whether some recorded move is a 無駄合 with a child (then
    /// `nodesNonFutile` needs a walk).
    pub(super) useless_with_child: bool,
    /// Whether some searched node or recorded move is `UNKNOWN`
    /// (`exact::graph_has_unknown` of the whole graph; a node created after
    /// the budget ran out counts too, see [`Solver::compact_graph`]).
    pub(super) any_unknown: bool,
    /// Whether this search records what the graph would hold (it was asked
    /// for a graph).  A search without one keeps the table only, exactly like
    /// the exact table's graph-free search.
    pub(super) recording: bool,
}

impl CompactState {
    pub(super) fn new(mode: TableMode) -> Self {
        CompactState {
            table: CompactTable::new(mode == TableMode::CompactVerify),
            patches: HashMap::default(),
            counts: NodeCounts::default(),
            per_remaining: Vec::new(),
            useless_with_child: false,
            any_unknown: false,
            recording: false,
        }
    }

    fn reset(&mut self, limits: &SearchLimits, recording: bool) {
        self.table.reset(limits.node_budget);
        self.patches = HashMap::default();
        self.counts = NodeCounts::default();
        self.per_remaining = vec![0; limits.max_ply as usize + 1];
        self.useless_with_child = false;
        self.any_unknown = false;
        self.recording = recording;
    }
}

/// `TSUME_COMPACT_STATS=1`: print the table's size after every compact search
/// to stderr (measurement only; the output is unchanged).
fn report_usage() -> bool {
    crate::diagnostics::compact_stats()
}

/// The solution region of the second pass: the nodes kept, by search id.
#[derive(Default)]
struct Region {
    index: HashMap<u32, usize>,
    /// `(id, node, moves, position, expanded)`.
    entries: Vec<(u32, GraphNode, Vec<GraphMove>, Position, bool)>,
}

impl Solver {
    /// Chooses the transposition table ([`TableMode`]).
    pub fn set_table(&mut self, mode: TableMode) {
        self.compact = mode.is_compact().then(|| Box::new(CompactState::new(mode)));
    }

    /// The table [`Solver::solve`] uses.
    pub fn table_mode(&self) -> TableMode {
        match self.compact.as_ref() {
            None => TableMode::Exact,
            Some(state) if state.table.verify.is_some() => TableMode::CompactVerify,
            Some(_) => TableMode::Compact,
        }
    }

    /// Bytes and keys of the compact table after the last search
    /// (measurement), `None` for the exact table.
    pub fn compact_table_usage(&self) -> Option<(usize, usize)> {
        self.compact
            .as_ref()
            .map(|state| (state.table.bytes(), state.table.len()))
    }

    /// The table's value at `key`, the key of `(remaining, pos)` (`pos` is
    /// read only by [`TableMode::CompactVerify`]).
    #[inline]
    pub(super) fn table_get(&self, key: &Key, pos: &Position) -> Option<Internal> {
        match self.compact.as_ref() {
            None => self.cache.get(key).map(|packed| packed.unpack()),
            Some(state) => state.table.get(key, pos),
        }
    }

    #[inline]
    pub(super) fn table_insert(&mut self, key: Key, value: Internal, pos: &Position) {
        match self.compact.as_mut() {
            None => {
                self.cache.insert(key, super::Packed::pack(value));
            }
            Some(state) => state.table.insert(key, value, pos),
        }
    }

    /// Whether the search records every node's moves (a compact search asked
    /// for a graph).
    #[inline]
    pub(super) fn compact_recording(&self) -> bool {
        self.compact.as_ref().is_some_and(|state| state.recording)
    }

    /// A node was created at `remaining` (compact only).
    #[inline]
    pub(super) fn compact_created(&mut self, remaining: u32) {
        if let Some(state) = self.compact.as_mut() {
            if let Some(count) = state.per_remaining.get_mut(remaining as usize) {
                *count += 1;
            }
        }
    }

    /// A searched node is done (compact only): its moves are
    /// `slab[slab_base..]`, recorded as graph mode records them.  Sums its
    /// feature counts and pops its moves (the root keeps them: they are the
    /// result's `moves`).
    pub(super) fn compact_node_done(
        &mut self,
        pos: &Position,
        attack: bool,
        slab_base: usize,
        outcome: Outcome,
        is_root: bool,
    ) {
        let Some(state) = self.compact.as_mut().filter(|state| state.recording) else {
            return;
        };
        let moves = &self.slab[slab_base..];
        state.counts = state
            .counts
            .merge(NodeCounts::of_parts(outcome, attack, moves, pos));
        if moves.iter().any(|m| m.useless() && m.next().is_some()) {
            state.useless_with_child = true;
        }
        if outcome == Outcome::Unknown || moves.iter().any(|m| m.outcome() == Outcome::Unknown) {
            state.any_unknown = true;
        }
        if !is_root {
            self.slab.truncate(slab_base);
        }
    }

    /// Keeps the recorded moves of a credit-line defence node whose
    /// resolution got past [`Solver::resolve_with_credit`]'s first loop (every
    /// reply that is not an interposition drop mates), compact only.
    /// `prune_base` / `moves_base` are the node's [`ChildInfo`] and slab starts.
    pub(super) fn compact_keep_credit_node(
        &mut self,
        pos: &Position,
        remaining: u32,
        range: std::ops::Range<usize>,
        line: super::CheckLine,
        prune_base: usize,
        moves_base: usize,
    ) {
        if !self.compact_recording() {
            return;
        }
        let count = range.len();
        let others_mate = (0..count).all(|index| {
            line.interposes(self.buf.buf[range.start + index])
                || self.prune_buf[prune_base + index].outcome == Outcome::Mate
        });
        if !others_mate {
            return;
        }
        let moves: Box<[GraphMove]> = self.slab[moves_base..moves_base + count].into();
        let key = table_key(remaining, pos.hash());
        if let Some(state) = self.compact.as_mut() {
            state.patches.insert(key, moves);
        }
    }

    /// [`Solver::solve`] under a compact table: the sequential search on the
    /// compact table, then (`build_graph`) the sparse graph of the solution
    /// region and the whole-graph counts.
    pub(super) fn solve_compact(
        &mut self,
        root_position: Position,
        attacker: u8,
        limits: &SearchLimits,
        build_graph: bool,
    ) -> SolveResult {
        self.build_graph = false;
        if let Some(state) = self.compact.as_mut() {
            state.reset(limits, build_graph);
        }
        let mut result = self.solve_sequential(root_position, attacker, limits);
        self.build_graph = build_graph;
        if report_usage() {
            let state = self.compact.as_ref().expect("a compact solver");
            let patch_moves: usize = state.patches.values().map(|moves| moves.len()).sum();
            eprintln!(
                "tsume-solver compact: nodes {} keys {} table_bytes {} patches {} patch_moves {} graph {}",
                result.stats.nodes,
                state.table.len(),
                state.table.bytes(),
                state.patches.len(),
                patch_moves,
                build_graph
            );
        }
        if build_graph {
            let graph = self.compact_graph(&result, limits);
            result.moves = graph.moves_of(0).to_vec();
            result.graph = Some(graph);
        }
        result
    }

    /// The second pass: the sparse graph of the solution region, with the
    /// whole-graph counts.
    fn compact_graph(&mut self, result: &SolveResult, limits: &SearchLimits) -> Graph {
        let root = Internal {
            outcome: result.outcome,
            distance: result.distance,
            id: 0,
        };
        let mut pos = result.root_position.clone();
        let mut region = Region::default();
        self.region_visit(&mut pos, limits.max_ply, root, true, &mut region);
        debug_assert_eq!(pos, result.root_position, "the walk restores the root");
        let state = self.compact.as_ref().expect("a compact solver");
        let mut facts = CompactFacts {
            counts: state.counts,
            reach: Vec::new(),
            non_futile: None,
            // A node created after the budget ran out is `UNKNOWN` and has no
            // moves; one exists exactly when the search made more nodes than
            // its budget.
            has_unknown: state.any_unknown || result.stats.nodes > self.node_budget,
        };
        // Only a mate pays for the walks (`falseMoveDifficulty`,
        // `nodesNonFutile`): no other root is published or counted.  A mate
        // beside an exhausted budget (the credit oracle's, or the main budget
        // in a refuted side branch) is walked too, as its graph is complete
        // where it is read; a node created after the budget has no moves.
        if let (Outcome::Mate, Some(distance)) = (result.outcome, result.distance) {
            let root_moves = region.entries[region.index[&0]].2.clone();
            let mut seen = vec![0u64; (result.stats.nodes as usize).div_ceil(64)];
            for gm in root_moves.iter() {
                if gm.outcome() == Outcome::Mate && gm.distance() == Some(distance) {
                    facts.reach.push(None);
                    continue;
                }
                let count = match gm.next() {
                    None => 0,
                    Some(_) => {
                        seen.fill(0);
                        let m = gm.mv();
                        let undo = pos.make(m);
                        let child = self.expect_lookup(limits.max_ply - 1, &pos);
                        let count =
                            self.reach_visit(&mut pos, limits.max_ply - 1, child.id, &mut seen);
                        pos.unmake(m, undo);
                        count
                    }
                };
                facts.reach.push(Some(count));
            }
            let state = self.compact.as_ref().expect("a compact solver");
            let non_futile = if state.useless_with_child {
                seen.fill(0);
                self.non_futile_visit(&mut pos, limits.max_ply, root, 0, distance, &mut seen)
            } else {
                // Every node is reached through moves that are not 無駄合, at
                // depth `max_ply - remaining`.
                let shallowest = limits.max_ply.saturating_sub(distance) as usize;
                state.per_remaining.iter().skip(shallowest).sum()
            };
            facts.non_futile = Some((distance, non_futile));
        }
        let mut entries: Vec<(u32, GraphNode, Vec<GraphMove>, Position)> = region
            .entries
            .into_iter()
            .map(|(id, node, moves, position, _)| (id, node, moves, position))
            .collect();
        entries.sort_unstable_by_key(|entry| entry.0);
        let mut graph = Graph::from_sparse(result.root_position.clone(), entries, facts);
        graph.omitted = std::mem::take(&mut self.omitted);
        graph
    }

    /// The table's value at `(remaining, pos)`, which the first pass stored.
    #[inline]
    fn expect_lookup(&self, remaining: u32, pos: &Position) -> Internal {
        self.table_get(&table_key(remaining, pos.hash()), pos)
            .expect("the first pass stored every child of a searched node")
    }

    /// Adds node `node` at `pos` to the region; `expand` opens its region
    /// edges, the ones `prune_solver_graph` walks: at an attack node the
    /// correct moves (`MATE` at the node's distance; none when the node is not
    /// a mate), whose other children are kept with their moves only (rule G
    /// and the runtime graph read their recorded reply), at a defence node
    /// every reply with a child.
    fn region_visit(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        node: Internal,
        expand: bool,
        region: &mut Region,
    ) {
        let moves = match region.index.get(&node.id) {
            Some(&index) => {
                let entry = &mut region.entries[index];
                if !expand || entry.4 {
                    return;
                }
                entry.4 = true;
                entry.2.clone()
            }
            None => {
                let base = self.materialize(pos, remaining, node);
                let moves = self.slab[base..].to_vec();
                self.slab.truncate(base);
                let mut raw = GraphNode::new(NO_NODE, NO_MOVE, 0, pos.side == self.attacker);
                raw.set_result(node.outcome, node.distance);
                region.index.insert(node.id, region.entries.len());
                region
                    .entries
                    .push((node.id, raw, moves.clone(), pos.clone(), expand));
                moves
            }
        };
        if !expand {
            return;
        }
        let attack = pos.side == self.attacker;
        for gm in moves.iter() {
            let Some(next) = gm.next() else {
                continue;
            };
            let open = !attack || (gm.outcome() == Outcome::Mate && gm.distance() == node.distance);
            let m = gm.mv();
            let undo = pos.make(m);
            let child = self.expect_lookup(remaining - 1, pos);
            debug_assert_eq!(child.id, next, "the table and the recorded move agree");
            self.region_visit(pos, remaining - 1, child, open, region);
            pos.unmake(m, undo);
        }
    }

    /// Pushes node `node`'s recorded moves at `pos` onto the slab, exactly as
    /// graph mode records them, and returns where they start.  Every child's
    /// verdict and id is read from the table; nothing is searched.
    pub(super) fn materialize(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        node: Internal,
    ) -> usize {
        let base = self.slab.len();
        if u64::from(node.id) >= self.node_budget {
            // Created after the budget ran out: never expanded, no moves.
            return base;
        }
        if pos.side == self.attacker {
            self.materialize_attack(pos, remaining);
        } else {
            self.materialize_defense(pos, remaining, node);
        }
        base
    }

    /// `search_attack`'s recorded moves.
    fn materialize_attack(&mut self, pos: &mut Position, remaining: u32) {
        if remaining == 0 {
            return;
        }
        let range = self.buf.gen_checks(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        for index in range.clone() {
            let m = self.buf.buf[index];
            let undo = pos.make(m);
            let child = self.expect_lookup(remaining - 1, pos);
            let distance = match child.outcome {
                Outcome::Mate => Some(child.distance.unwrap_or(0) + 1),
                _ => None,
            };
            let mut gm = GraphMove::new(m, child.outcome, distance, Some(child.id), None, false);
            if child.outcome == Outcome::Refuted {
                // `first_refuted_reply` of the child.
                let child_base = self.materialize(pos, remaining - 1, child);
                let code = self.slab[child_base..]
                    .iter()
                    .find(|reply| reply.outcome() == Outcome::Refuted)
                    .map_or(NO_MOVE, GraphMove::code);
                self.slab.truncate(child_base);
                gm.set_refutation_code(code);
            }
            pos.unmake(m, undo);
            self.slab.push(gm);
        }
        self.buf.buf.truncate(range.start);
    }

    /// `search_defense`'s recorded moves (with `record`).
    fn materialize_defense(&mut self, pos: &mut Position, remaining: u32, node: Internal) {
        let range = self.buf.gen_legal(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let count = range.len();
        if count == 0 {
            self.buf.buf.truncate(range.start);
            return;
        }
        let futile_base = self.futile_buf.len();
        let classical = self.useless == UselessInterposition::Classical
            && self.classify_futile_replies(pos, range.clone());
        let futile = |solver: &Solver, index: usize| -> Option<u32> {
            if classical {
                solver.futile_buf[futile_base + index]
            } else {
                None
            }
        };
        let all_futile = classical
            && self.futile_buf[futile_base..futile_base + count]
                .iter()
                .all(Option::is_some);
        if all_futile {
            for index in 0..count {
                let m = self.buf.buf[range.start + index];
                self.slab.push(GraphMove::new(
                    m,
                    Outcome::Mate,
                    futile(self, index),
                    None,
                    None,
                    true,
                ));
            }
        } else if remaining == 0 {
            if self.useless == UselessInterposition::Classical && node.id != 0 {
                // The horizon escape: the first reply that is not futile.
                let escape = (0..count)
                    .find(|&index| futile(self, index).is_none())
                    .expect("a node that is not all-futile has a reply that is not futile");
                let m = self.buf.buf[range.start + escape];
                self.slab
                    .push(GraphMove::new(m, Outcome::Refuted, None, None, None, false));
            } else {
                for index in 0..count {
                    let m = self.buf.buf[range.start + index];
                    let verdict = futile(self, index);
                    let outcome = if verdict.is_some() {
                        Outcome::Mate
                    } else {
                        Outcome::Refuted
                    };
                    self.slab.push(GraphMove::new(
                        m,
                        outcome,
                        verdict,
                        None,
                        None,
                        verdict.is_some(),
                    ));
                }
            }
        } else {
            self.materialize_replies(pos, remaining, node, range.clone(), classical, futile_base);
        }
        self.futile_buf.truncate(futile_base);
        self.buf.buf.truncate(range.start);
    }

    /// The searched replies of a defence node with plies left.
    fn materialize_replies(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        node: Internal,
        range: std::ops::Range<usize>,
        classical: bool,
        futile_base: usize,
    ) {
        let count = range.len();
        let base = self.slab.len();
        let prune = matches!(
            self.useless,
            UselessInterposition::Prune | UselessInterposition::Classical
        );
        let prune_base = self.prune_buf.len();
        let mut escape: Option<Move> = None;
        let mut any_unknown = false;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if classical {
                if let Some(futile) = self.futile_buf[futile_base + index] {
                    self.prune_buf.push(ChildInfo {
                        distance: futile,
                        useless: true,
                        outcome: Outcome::Mate,
                    });
                    self.slab.push(GraphMove::new(
                        m,
                        Outcome::Mate,
                        Some(futile),
                        None,
                        None,
                        true,
                    ));
                    continue;
                }
            }
            let undo = pos.make(m);
            let child = self.expect_lookup(remaining - 1, pos);
            pos.unmake(m, undo);
            let distance = match child.outcome {
                Outcome::Mate => Some(child.distance.unwrap_or(0) + 1),
                _ => None,
            };
            match child.outcome {
                Outcome::Refuted => {
                    if escape.is_none() {
                        escape = Some(m);
                    }
                }
                Outcome::Unknown => any_unknown = true,
                Outcome::Mate => {}
            }
            if prune {
                self.prune_buf.push(ChildInfo {
                    distance: distance.unwrap_or(0),
                    useless: false,
                    outcome: child.outcome,
                });
            }
            self.slab.push(GraphMove::new(
                m,
                child.outcome,
                distance,
                Some(child.id),
                None,
                false,
            ));
        }
        let credit_line = if self.useless == UselessInterposition::Classical && !classical {
            self.credit_line(pos, range.clone())
        } else {
            None
        };
        if let Some(line) = credit_line {
            let key = table_key(remaining, pos.hash());
            let patch = self
                .compact
                .as_ref()
                .and_then(|state| state.patches.get(&key));
            match patch {
                // The resolution reached the credit search: its moves as kept,
                // naming the children by the ids just read from the table
                // (the same ids as kept).
                Some(moves) => {
                    debug_assert_eq!(moves.len(), count);
                    for (gm, kept) in self.slab[base..base + count].iter_mut().zip(moves.iter()) {
                        let next = gm.next();
                        *gm = *kept;
                        gm.set_next(next);
                    }
                }
                // It stopped in its first loop: refuted by the first other
                // reply that is refuted, else UNKNOWN.
                None => {
                    let refuting = (0..count).find(|&index| {
                        let m = self.buf.buf[range.start + index];
                        !line.interposes(m) && self.slab[base + index].outcome() == Outcome::Refuted
                    });
                    debug_assert_eq!(
                        refuting.is_some(),
                        node.outcome == Outcome::Refuted,
                        "a credit-line node without a kept resolution stopped in its first loop"
                    );
                    if let Some(index) = refuting {
                        let escape = self.buf.buf[range.start + index];
                        for gm in self.slab[base..base + count].iter_mut() {
                            gm.set_refutation(Some(escape));
                        }
                    }
                }
            }
        } else if let Some(escape) = escape {
            for gm in self.slab[base..base + count].iter_mut() {
                gm.set_refutation(Some(escape));
            }
        } else if !any_unknown && prune {
            self.mark_useless_interpositions(pos, range.clone(), remaining, prune_base);
            for (index, gm) in self.slab[base..base + count].iter_mut().enumerate() {
                gm.set_useless(self.prune_buf[prune_base + index].useless);
            }
        }
        self.prune_buf.truncate(prune_base);
    }

    /// The moves of the node at `pos` that have a child (graph mode's `next`),
    /// pushed onto the move stack; the range is returned.  Which moves have a
    /// child depends on the position, the bound and rule F's futility test
    /// alone, never on a verdict.
    fn searched_moves(&mut self, pos: &mut Position, remaining: u32) -> std::ops::Range<usize> {
        let start = self.buf.buf.len();
        if remaining == 0 {
            // An attack node records no move; a defence node only witnesses.
            return start..start;
        }
        if pos.side == self.attacker {
            return self.buf.gen_checks(pos, true);
        }
        let range = self.buf.gen_legal(pos, true);
        if range.is_empty() || self.useless != UselessInterposition::Classical {
            return range;
        }
        let futile_base = self.futile_buf.len();
        if self.classify_futile_replies(pos, range.clone()) {
            // Keep the replies that are not futile, in place.
            let mut write = range.start;
            for index in 0..range.len() {
                if self.futile_buf[futile_base + index].is_none() {
                    self.buf.buf[write] = self.buf.buf[range.start + index];
                    write += 1;
                }
            }
            self.futile_buf.truncate(futile_base);
            self.buf.buf.truncate(write);
            return range.start..write;
        }
        range
    }

    /// `reachable_node_count` from node `id` at `pos`: the distinct nodes
    /// reachable through moves with a child.  `seen` holds one bit per id.
    fn reach_visit(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        id: u32,
        seen: &mut [u64],
    ) -> u64 {
        let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
        if seen[word] & bit != 0 {
            return 0;
        }
        seen[word] |= bit;
        if u64::from(id) >= self.node_budget {
            return 1;
        }
        let mut count = 1u64;
        let range = self.searched_moves(pos, remaining);
        for index in range.clone() {
            let m = self.buf.buf[index];
            let undo = pos.make(m);
            let child = self.expect_lookup(remaining - 1, pos);
            count += self.reach_visit(pos, remaining - 1, child.id, seen);
            pos.unmake(m, undo);
        }
        self.buf.buf.truncate(range.start);
        count
    }

    /// `non_futile_node_count` from node `node` at `pos`, `depth` plies below
    /// the root: the distinct nodes reachable without playing a reply flagged
    /// 無駄合, at most `mate_ply` plies deep.  Every node lies at depth
    /// `max_ply - remaining`, so a depth-first walk reaches the set the
    /// breadth-first count does.
    fn non_futile_visit(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        node: Internal,
        depth: u32,
        mate_ply: u32,
        seen: &mut [u64],
    ) -> u64 {
        let (word, bit) = (node.id as usize / 64, 1u64 << (node.id % 64));
        if seen[word] & bit != 0 {
            return 0;
        }
        seen[word] |= bit;
        if depth >= mate_ply {
            return 1;
        }
        let mut count = 1u64;
        let base = self.materialize(pos, remaining, node);
        let end = self.slab.len();
        for index in base..end {
            let gm = self.slab[index];
            if gm.useless() || gm.next().is_none() {
                continue;
            }
            let m = gm.mv();
            let undo = pos.make(m);
            let child = self.expect_lookup(remaining - 1, pos);
            count += self.non_futile_visit(pos, remaining - 1, child, depth + 1, mate_ply, seen);
            pos.unmake(m, undo);
        }
        self.slab.truncate(base);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    fn key(rng: &mut SplitMix64) -> Key {
        [rng.next_u64(), rng.next_u64()]
    }

    fn value(index: u64) -> Internal {
        let outcome = match index % 3 {
            0 => Outcome::Mate,
            1 => Outcome::Refuted,
            _ => Outcome::Unknown,
        };
        Internal {
            outcome,
            distance: (outcome == Outcome::Mate).then_some((index % 8000) as u32),
            id: index as u32,
        }
    }

    fn same(a: Option<Internal>, b: Option<Internal>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.outcome == b.outcome && a.distance == b.distance && a.id == b.id
            }
            _ => false,
        }
    }

    #[test]
    fn the_table_agrees_with_a_hash_map_while_it_grows_and_after_it_clears() {
        let pos = Position::empty();
        {
            let mut table = CompactTable::new(false);
            let mut rng = SplitMix64::new(7);
            // A large search (segments grow), small ones
            // cleared through the fill log, then a large one again.
            let rounds = [
                (200_000u64, 150_000u64),
                (200_000, 300),
                (50, 3000),
                (200_000, 120_000),
            ];
            for (round, (budget, count)) in rounds.into_iter().enumerate() {
                table.reset(budget);
                assert_eq!(table.len(), 0, "round {round}");
                let mut reference = std::collections::HashMap::new();
                for index in 0..count {
                    let k = key(&mut rng);
                    assert!(table.get(&k, &pos).is_none());
                    table.insert(k, value(index), &pos);
                    reference.insert(k, value(index));
                }
                assert_eq!(table.len(), count as usize);
                for (k, v) in reference.iter() {
                    assert!(same(table.get(k, &pos), Some(*v)), "round {round}");
                }
                for _ in 0..1000 {
                    assert!(table.get(&key(&mut rng), &pos).is_none());
                }
                // Within the budget's capacity: no growth past it.
                if count <= budget {
                    let budget_bytes = table.target * SEGMENTS * std::mem::size_of::<Slot>();
                    assert!(table.bytes() <= budget_bytes, "round {round}");
                }
            }
        }
    }

    #[test]
    fn a_large_budget_fits_the_size_types() {
        let mut table = CompactTable::new(false);
        table.reset(400_000_000);
        // 25.6 bytes per budget node at 7/8 load and 2 % slack.
        let bytes = table.target * SEGMENTS * std::mem::size_of::<Slot>();
        assert!((10_200_000_000..10_400_000_000).contains(&bytes), "{bytes}");
        table.reset(u64::MAX);
        assert_eq!(table.target, 1 << 40);
    }

    #[test]
    fn verdicts_round_trip_the_packing() {
        for outcome in [Outcome::Mate, Outcome::Refuted, Outcome::Unknown] {
            for distance in [None, Some(0), Some(15), Some(8014), Some(16_382)] {
                let packed = pack(outcome, distance);
                assert_ne!(packed, EMPTY);
                let back = unpack(packed, 42);
                assert_eq!(
                    (back.outcome, back.distance, back.id),
                    (outcome, distance, 42)
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "two positions share a 128-bit table key")]
    fn verify_mode_catches_a_key_shared_by_two_positions() {
        let mut table = CompactTable::new(true);
        table.reset(1000);
        let a = crate::sfen::parse_sfen("8k/9/9/9/7R1/9/9/9/9 b G 1").unwrap();
        let b = crate::sfen::parse_sfen("8k/9/9/9/7R1/9/9/9/9 b S 1").unwrap();
        let k = [1u64, 2u64];
        table.insert(k, value(0), &a);
        assert!(table.get(&k, &a).is_some());
        // A planted collision: `b` asks for the key `a` is stored under.
        let _ = table.get(&k, &b);
    }
}
