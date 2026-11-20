//! Bounded exact AND/OR mate search.
//!
//! Semantics are identical to `solveTsume` in `packages/solver/src/index.ts`:
//!
//! * Attack nodes consider only legal checking moves, in USI byte order, and
//!   take the shortest forced mate.  `remaining <= 0` is `REFUTED` with no moves.
//! * Defense nodes expand every legal reply.  No reply while in check is mate at
//!   distance 0; no reply while not in check is stalemate, which is `REFUTED`.
//!   `remaining <= 0` with replies is `REFUTED` and every reply is recorded as a
//!   witness.  Otherwise one refuted reply refutes the node (the first in USI
//!   order escapes), an unresolved reply makes the node `UNKNOWN`, and otherwise
//!   the node mates at the longest child distance.
//! * `UNKNOWN` comes only from node-budget exhaustion and is never reported as
//!   mate.
//!
//! ## Transposition table
//!
//! The TypeScript cache is keyed by `role | remaining | sfen-without-move-number`.
//! This solver keys by `(remaining, 128-bit Zobrist key)`.  That is the *same*
//! key: the Zobrist key covers board, hands and side to move, and the role is a
//! function of the side to move and the fixed attacker, so no extra sharing and
//! no extra splitting happens.  Results therefore match node for node, including
//! `stats.nodes` and `stats.cacheHits`, which the pipeline stores as difficulty
//! features `totalSearchNodes` and `transpositionCount`.
//!
//! A depth-independent table would be sound for pure existence queries -- a
//! proven `MATE d` holds for every `remaining >= d`, and a `REFUTED` at bound `r`
//! holds for every `remaining <= r` -- but it changes both the node counts and
//! the exported graph (a node proven at one bound would be reused with the move
//! list it recorded at that bound, and horizon defense nodes list every reply).
//! Because those numbers are published puzzle metadata, the bound stays in the
//! key.  `UNKNOWN` is never reused across bounds in either scheme.
//!
//! ## Options
//!
//! [`SolveOptions`] groups what a search is asked by meaning: the rules it
//! searches under ([`SearchRules`]: attacker, strict material, rule F), the
//! resources it may spend ([`SearchLimits`]: ply bound, node budget, the credit
//! oracle's budget) and the shape of its output (`build_graph`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

mod compact;
mod graph;

pub use compact::TableMode;

use crate::movegen::{checks_from_to, MoveBuf, Safety};
use crate::piece::{opponent, BLACK};
use crate::position::{Move, Position, NO_SQUARE};
use crate::strict_material;

/// Search verdict.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Mate,
    Refuted,
    Unknown,
}

impl Outcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Outcome::Mate => "MATE",
            Outcome::Refuted => "REFUTED",
            Outcome::Unknown => "UNKNOWN",
        }
    }
}

/// 無駄合 policy of `docs/rules.md` rule F, mirroring `UselessInterpositionMode`
/// in `packages/solver/src/index.ts`.
///
/// `Keep` is the bare solver contract: every legal interposition is resistance.
/// `Prune` is the composition reading: an interposition the checker simply
/// captures, buying nothing beyond its own two plies, is neither a defence nor
/// part of the mate length.
///
/// `Classical` is the composition reading since `PIPELINE_VERSION` 0.10.0 (owner
/// decision 2026-10-05, `docs/rules.md` rule F; `"classical"` in
/// `packages/solver`): `Prune` plus the classical tsume reading of 無駄合.  At a defense node whose legal replies are
/// *all* interposition drops on an interposable check line, each drop is tested
/// by a narrow futility check outside the main search ([`Solver::futile_drop`]);
/// futile drops are flagged `useless` and not searched, and when every reply is
/// futile the node is mate at distance 0, as if the check itself were mate.
/// Since `PIPELINE_VERSION` 0.11.0 (owner decision 2026-10-07) its ply bound also
/// counts rule-F plies: at a node with both interposition drops and other replies,
/// a drop F1 would flag but whose capture line the bound cannot prove is judged by
/// the credit search ([`Solver::credit_interposition`]), so neither the drop nor
/// its capture consumes the bound.  Elsewhere it behaves exactly as `Prune`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum UselessInterposition {
    #[default]
    Keep,
    Prune,
    Classical,
}

impl UselessInterposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            UselessInterposition::Keep => "keep",
            UselessInterposition::Prune => "prune",
            UselessInterposition::Classical => "classical",
        }
    }
    /// Parses the CLI / JSON spelling, `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "keep" => Some(UselessInterposition::Keep),
            "prune" => Some(UselessInterposition::Prune),
            "classical" => Some(UselessInterposition::Classical),
            _ => None,
        }
    }
}

/// `SolveStats` in the TypeScript solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolveStats {
    pub nodes: u64,
    pub cache_hits: u64,
    pub budget_exhausted: bool,
    pub max_ply: u32,
    pub exhaustive_within_bound: bool,
}

pub(crate) use graph::MAX_GRAPH_DISTANCE;
use graph::{encode_move, moves_range, NO_MOVE, NO_NODE};
pub use graph::{CompactFacts, Graph, GraphMove, GraphNode, Node, NodeSpec, OmittedReplies};

/// The largest ply bound [`Solver::solve`] accepts.  A recorded distance is at
/// most the bound plus a 無駄合 capture chain (at most 14 plies,
/// `docs/rules.md` rule F), and the graph stores distances exactly up to
/// 8,190 ([`graph`]'s 13-bit distances), so this leaves room.  (It was 65,534
/// before stage2c; no caller comes near either.)
pub const MAX_PLY: u32 = 8_000;

/// The rules a search runs under (`attacker`, `strictMaterial` and
/// `uselessInterposition` of `SolveOptions` in `packages/solver`).  They decide
/// the verdicts; the [`SearchLimits`] only decide where a search may stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchRules {
    /// The attacking side; `None` is the side to move.
    pub attacker: Option<u8>,
    /// Hand the defender every piece that is neither on the board nor in the
    /// attacker's hand (the strict-tsume convention).
    pub strict_material: bool,
    /// 無駄合 handling (rule F).  Default [`UselessInterposition::Keep`], which
    /// leaves the solver contract and the golden conformance set untouched.  The
    /// value is fixed for the whole call, so the `(remaining, zobrist)` cache key
    /// needs no extra discriminator.
    pub useless: UselessInterposition,
}

impl Default for SearchRules {
    fn default() -> Self {
        Self {
            attacker: None,
            strict_material: true,
            useless: UselessInterposition::Keep,
        }
    }
}

/// The resources a search may spend (`maxPly`, `nodeBudget`, `creditBudget`).
/// Running out of a budget is `UNKNOWN`, never a verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchLimits {
    /// The ply bound (rule-F plies under [`UselessInterposition::Classical`]).
    pub max_ply: u32,
    /// Nodes the search may create.
    pub node_budget: u64,
    /// Node budget of rule F's credit oracle ([`Solver::oracle`]), counted
    /// separately from `node_budget`; `None` means four times `node_budget`.
    pub credit_budget: Option<u64>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_ply: 11,
            node_budget: 100_000,
            credit_budget: None,
        }
    }
}

impl SearchLimits {
    /// The credit oracle's budget: `credit_budget`, else four times the node
    /// budget.
    pub fn oracle_budget(&self) -> u64 {
        self.credit_budget
            .unwrap_or(self.node_budget.saturating_mul(4))
    }
}

/// Options mirroring `SolveOptions` of `packages/solver`, grouped by meaning
/// (module docs, "Options").
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    pub rules: SearchRules,
    pub limits: SearchLimits,
    /// Build the exported graph (the default).  Turning it off keeps outcomes,
    /// distances and stats identical while skipping per-node bookkeeping; only
    /// the root move list survives, and refutations are not recorded.
    pub build_graph: bool,
}

/// The defaults of `solveTsume`: bound 11, budget 100,000, the side to move
/// attacks, strict material, rule F `keep`, a graph.
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            rules: SearchRules::default(),
            limits: SearchLimits::default(),
            build_graph: true,
        }
    }
}

impl SolveOptions {
    pub fn max_ply(mut self, max_ply: u32) -> Self {
        self.limits.max_ply = max_ply;
        self
    }
    pub fn node_budget(mut self, node_budget: u64) -> Self {
        self.limits.node_budget = node_budget;
        self
    }
    pub fn credit_budget(mut self, budget: u64) -> Self {
        self.limits.credit_budget = Some(budget);
        self
    }
    pub fn attacker(mut self, attacker: u8) -> Self {
        self.rules.attacker = Some(attacker);
        self
    }
    pub fn strict_material(mut self, strict: bool) -> Self {
        self.rules.strict_material = strict;
        self
    }
    pub fn useless_interposition(mut self, mode: UselessInterposition) -> Self {
        self.rules.useless = mode;
        self
    }
    pub fn build_graph(mut self, build: bool) -> Self {
        self.build_graph = build;
        self
    }
}

/// Result of a search.
#[derive(Clone, Debug)]
pub struct SolveResult {
    pub outcome: Outcome,
    pub distance: Option<u32>,
    pub stats: SolveStats,
    /// Root node moves: every legal checking move at an attack root, every legal
    /// reply at a defense root.
    pub moves: Vec<GraphMove>,
    pub graph: Option<Graph>,
    /// The position actually searched (after strict-material normalisation).
    pub root_position: Position,
    pub attacker: u8,
}

/// Multiplicative hasher for the `(remaining, zobrist)` key; SipHash would cost
/// more than the node it protects.
#[derive(Default)]
pub struct FastHasher {
    state: u64,
}

impl FastHasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        self.state = (self.state ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.state ^= self.state >> 29;
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.mix(byte as u64);
        }
    }
    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.mix(value as u64);
    }
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }
    #[inline]
    fn write_u128(&mut self, value: u128) {
        self.mix(value as u64);
        self.mix((value >> 64) as u64);
    }
}

/// Hasher for `[u64; 2]` keys that are already uniformly random (Zobrist
/// folds: [`table_key`], the oracle and board keys): the two words are folded
/// with a rotation instead of being mixed byte by byte, which is what
/// [`FastHasher`] does to a `[u64; 2]` (its `Hash` writes a length prefix and
/// sixteen bytes, 24 dependent multiplies per lookup).  Only for maps that are
/// never iterated in an order-dependent way: it changes where an entry sits,
/// never which entries a map holds.
#[derive(Default)]
pub struct KeyHasher {
    state: u64,
}

impl Hasher for KeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut words = bytes.chunks_exact(8);
        for word in &mut words {
            let value = u64::from_ne_bytes(word.try_into().expect("8 bytes"));
            self.state = self.state.rotate_left(23) ^ value;
        }
        for &byte in words.remainder() {
            self.state = (self.state ^ byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    /// The slice length prefix of a `[u64; 2]` key: the same for every key.
    #[inline]
    fn write_usize(&mut self, _len: usize) {}
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.state = self.state.rotate_left(23) ^ value;
    }
}

/// The table key `(remaining, position)` folded into 128 bits: the 128-bit
/// Zobrist key of the position XOR a fixed 128-bit salt per remaining bound.
/// Two keys collide only if two Zobrist keys collide in 128 bits, the same
/// premise the `(remaining, u128)` pair had.  `[u64; 2]` instead of `u128`
/// keeps the slot 8-byte aligned (24 bytes with [`Packed`] instead of 48).
type Key = [u64; 2];
type Table = HashMap<Key, Packed, BuildHasherDefault<FastHasher>>;

/// SplitMix64's finaliser.
#[inline]
const fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub(crate) fn table_key(remaining: u32, hash: u128) -> Key {
    let salt = mix64(0x7473_756d_6572_656d ^ u64::from(remaining));
    let salt2 = mix64(salt ^ 0x9E37_79B9_7F4A_7C15);
    [(hash as u64) ^ salt, ((hash >> 64) as u64) ^ salt2]
}

#[derive(Clone, Copy, Debug)]
struct Internal {
    outcome: Outcome,
    distance: Option<u32>,
    id: u32,
}

/// [`Internal`] as stored in the table: 8 bytes instead of 16 (the `u16`
/// distance fills the padding).  Distances are bounded by the ply bound, which
/// [`Solver::solve`] keeps below `u16::MAX`.
#[derive(Clone, Copy, Debug)]
struct Packed {
    id: u32,
    outcome: u8,
    distance: u16,
}

impl Packed {
    #[inline]
    fn pack(value: Internal) -> Self {
        Packed {
            id: value.id,
            outcome: match value.outcome {
                Outcome::Mate => 0,
                Outcome::Refuted => 1,
                Outcome::Unknown => 2,
            },
            distance: match value.distance {
                Some(d) => {
                    debug_assert!(d < u32::from(u16::MAX), "mate distance fits 16 bits");
                    d as u16
                }
                None => u16::MAX,
            },
        }
    }
    #[inline]
    fn unpack(self) -> Internal {
        Internal {
            outcome: match self.outcome {
                0 => Outcome::Mate,
                1 => Outcome::Refuted,
                _ => Outcome::Unknown,
            },
            distance: (self.distance != u16::MAX).then_some(u32::from(self.distance)),
            id: self.id,
        }
    }
}

/// A reusable solver.  Keeping one instance across many positions reuses the
/// move stack, node arena and table allocations, which matters when the reducer
/// re-solves a candidate dozens of times.
///
/// The table is *cleared* (not reused) at the start of every [`Solver::solve`]
/// call, because the TypeScript solver starts each `solveTsume` with an empty
/// cache and `stats.cacheHits` is published metadata.
pub struct Solver {
    buf: MoveBuf,
    cache: Table,
    /// The graph's nodes, in id order (only with `build_graph`).
    nodes: Vec<GraphNode>,
    /// The graph's moves, one slab in node-id order: a node reserves its slots
    /// as soon as it has generated its moves, before it creates any child.
    /// Without `build_graph` it holds the root's moves only.
    slab: Vec<GraphMove>,
    stats: SolveStats,
    /// The attacking side of the current solve (resolved from
    /// [`SearchRules::attacker`]).
    attacker: u8,
    node_budget: u64,
    build_graph: bool,
    useless: UselessInterposition,
    next_id: u32,
    /// Rule F scratch stack: one entry per reply of every defense node currently
    /// on the search stack, pushed only under [`UselessInterposition::Prune`].
    /// Pooled like [`MoveBuf`] so marking never allocates per node.
    prune_buf: Vec<ChildInfo>,
    /// [`UselessInterposition::Classical`] scratch: the futility verdict of each
    /// reply of every all-interposition defense node on the search stack.
    futile_buf: Vec<Option<u32>>,
    /// [`UselessInterposition::Classical`] memos, cleared with every solve.
    futility: Futility,
    /// Credit-search counters of the last solve ([`Solver::credit_stats`]).
    credit: CreditStats,
    /// Rule F's credit oracle (stage 2): decides credit searches outside the
    /// main search, see [`Solver::oracle`].
    oracle: Oracle,
    /// The replies the horizon escapes of this solve left out, for the
    /// difficulty features ([`Graph::omitted`]).
    omitted: OmittedReplies,
    /// The compact table and what it records instead of a graph
    /// ([`Solver::set_table`]); `None` for the exact table.
    compact: Option<Box<compact::CompactState>>,
}

/// One resolved reply of a defense node, for rule F.
#[derive(Clone, Copy, Debug)]
struct ChildInfo {
    /// Mate distance through this reply, `0` when the reply does not mate.
    distance: u32,
    /// Set by [`Solver::mark_useless_interpositions`].
    useless: bool,
    /// The child's verdict at `remaining - 1` (`MATE` for a futile drop).
    outcome: Outcome,
}

/// The verdict of one credit search ([`Solver::credit_interposition`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Credit {
    /// The drop is a 無駄合: the capture form that mates first, with `dist(E)`.
    Useless(u32),
    /// No capture form mates within `dist_other`: the drop is an effective one.
    Effective,
    /// No form mates and one ran out of budget.
    Undecided,
}

/// Internal counters of the credit search (rule F, `PIPELINE_VERSION` 0.11.0).
/// Not part of [`SolveStats`] or any published output; the CLI prints them with
/// `--report-credit` for measurement and regression checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CreditStats {
    /// Credit searches run: one per capture form searched, table hits included.
    pub calls: u64,
    /// Search nodes created inside outermost credit searches (nested ones are
    /// counted once, by the outermost).
    pub nodes: u64,
    /// Drops judged useless by a credit search.
    pub useless: u64,
    /// Credit searches whose mating capture form came after an UNKNOWN one.
    pub unknown_then_mate: u64,
    /// First-pass drops the credit search left undecided.
    pub first_pass_undecided: u64,
    /// Defense nodes refuted by an effective drop after an undecided one.
    pub undecided_then_effective: u64,
    /// Second passes stopped (D UNKNOWN) by an undecided drop.
    pub second_pass_stops: u64,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    pub fn new() -> Self {
        Self {
            buf: MoveBuf::new(),
            cache: Table::default(),
            nodes: Vec::new(),
            slab: Vec::new(),
            stats: SolveStats {
                nodes: 0,
                cache_hits: 0,
                budget_exhausted: false,
                max_ply: 0,
                exhaustive_within_bound: true,
            },
            attacker: BLACK,
            node_budget: 0,
            build_graph: true,
            useless: UselessInterposition::Keep,
            next_id: 0,
            prune_buf: Vec::new(),
            futile_buf: Vec::new(),
            futility: Futility::default(),
            credit: CreditStats::default(),
            oracle: Oracle::new(),
            omitted: OmittedReplies::default(),
            compact: None,
        }
    }

    /// Credit-search counters of the last [`Solver::solve`] (rule F,
    /// `PIPELINE_VERSION` 0.11.0).  All zero for `Keep` and `Prune`, and for a
    /// classical search that never needed one.
    pub fn credit_stats(&self) -> CreditStats {
        self.credit
    }

    /// Lowers the item-3 table caps ([`ORACLE_TABLE_CAP`], [`FUTILE_MEMO_CAP`])
    /// so a test can make the garbage collections run on small searches.  The
    /// caps are part of the oracle's deterministic node count, so nothing but a
    /// test may change them.
    #[doc(hidden)]
    pub fn set_table_caps_for_tests(&mut self, oracle: usize, futility: usize) {
        self.oracle.cap = oracle.max(1);
        self.futility.cap = futility.max(1);
    }

    /// Garbage collections of the credit oracle's table since the solver was
    /// built (measurement and tests only).
    pub fn oracle_collections(&self) -> u64 {
        self.oracle.collections
    }

    /// Runs a bounded search from `position`.
    pub fn solve(&mut self, position: &Position, options: &SolveOptions) -> SolveResult {
        let _prof = crate::prof::engine(if options.build_graph {
            crate::prof::Engine::ExactGraph
        } else {
            crate::prof::Engine::ExactNoGraph
        });
        let SolveOptions {
            rules,
            limits,
            build_graph,
        } = *options;
        assert!(
            limits.max_ply <= MAX_PLY,
            "the ply bound must be at most {MAX_PLY}"
        );
        let attacker = rules.attacker.unwrap_or(position.side);
        let root_position = if rules.strict_material {
            strict_material::normalize(position, attacker)
        } else {
            position.clone()
        };
        self.attacker = attacker;
        self.node_budget = limits.node_budget;
        self.build_graph = build_graph;
        self.useless = rules.useless;
        if self.compact.is_some() {
            // The compact table (`compact.rs`).
            return self.solve_compact(root_position, attacker, &limits, build_graph);
        }
        self.solve_sequential(root_position, attacker, &limits)
    }

    /// The search of [`Solver::solve`] (on either table);
    /// the rules and the output shape are already set on `self`.
    fn solve_sequential(
        &mut self,
        root_position: Position,
        attacker: u8,
        limits: &SearchLimits,
    ) -> SolveResult {
        self.next_id = 0;
        self.nodes.clear();
        self.slab.clear();
        self.omitted = OmittedReplies::default();
        self.prune_buf.clear();
        self.futile_buf.clear();
        self.futility.clear();
        self.credit = CreditStats::default();
        self.oracle.reset(limits.oracle_budget());
        self.cache.clear();
        self.buf.buf.clear();
        self.stats = SolveStats {
            nodes: 0,
            cache_hits: 0,
            budget_exhausted: false,
            max_ply: limits.max_ply,
            exhaustive_within_bound: true,
        };
        let mut work = root_position.clone();
        let root = self.search(&mut work, limits.max_ply, NO_NODE, NO_MOVE);
        debug_assert_eq!(work, root_position, "search must restore the root position");
        self.credit.nodes = self.oracle.nodes;
        if self.oracle.exhausted {
            // The credit oracle ran out of its own node budget: some verdict
            // may be UNKNOWN, exactly as when the main search runs out.
            self.stats.budget_exhausted = true;
            self.stats.exhaustive_within_bound = false;
        }
        let graph = if self.build_graph {
            let mut graph = Graph::from_parts(
                root.id,
                root_position.clone(),
                std::mem::take(&mut self.nodes),
                std::mem::take(&mut self.slab),
            );
            graph.omitted = std::mem::take(&mut self.omitted);
            Some(graph)
        } else {
            None
        };
        let moves = match &graph {
            Some(graph) => graph
                .node(root.id)
                .map(|node| node.moves().to_vec())
                .unwrap_or_default(),
            // Only the root recorded its moves.
            None => self.slab.clone(),
        };
        SolveResult {
            outcome: root.outcome,
            distance: root.distance,
            stats: self.stats,
            moves,
            graph,
            root_position,
            attacker,
        }
    }

    /// Searches `pos`, reached from node `parent` by the move code `mv`
    /// ([`NO_NODE`] / [`NO_MOVE`] at the root).
    fn search(&mut self, pos: &mut Position, remaining: u32, parent: u32, mv: u16) -> Internal {
        let attack = pos.side == self.attacker;
        let key: Key = table_key(remaining, pos.hash());
        if let Some(hit) = self.table_get(&key, pos) {
            self.stats.cache_hits += 1;
            return hit;
        }
        let id = self.next_id;
        self.next_id += 1;
        let is_root = id == 0;
        if self.build_graph {
            // The node's moves start where the slab ends now: it reserves them
            // before it creates any child (`reserve_moves`).
            self.nodes
                .push(GraphNode::new(parent, mv, self.slab.len(), attack));
        }
        self.compact_created(remaining);
        let visited = self.stats.nodes;
        self.stats.nodes += 1;
        if visited >= self.node_budget {
            self.stats.budget_exhausted = true;
            self.stats.exhaustive_within_bound = false;
            let result = Internal {
                outcome: Outcome::Unknown,
                distance: None,
                id,
            };
            self.table_insert(key, result, pos);
            return result;
        }
        // A compact search asked for a graph records every node's moves as
        // graph mode does, on a stack it pops when the node is done
        // (`compact.rs`).
        let compact = self.compact_recording();
        let record = self.build_graph || is_root || compact;
        let slab_base = self.slab.len();
        let result = if attack {
            self.search_attack(pos, remaining, id, record)
        } else {
            self.search_defense(pos, remaining, id, record)
        };
        if self.build_graph {
            self.nodes[id as usize].set_result(result.outcome, result.distance);
        }
        if compact {
            self.compact_node_done(pos, attack, slab_base, result.outcome, is_root);
        }
        self.table_insert(key, result, pos);
        result
    }

    /// The table's value at `(remaining, pos)`.
    fn lookup(&self, remaining: u32, pos: &Position) -> Option<Internal> {
        self.table_get(&table_key(remaining, pos.hash()), pos)
    }

    /// Reserves `count` slab slots for the moves of the node being expanded,
    /// when it records them, and returns the first.  Called before the node
    /// creates any child, which keeps the slab in node-id order.
    #[inline]
    fn reserve_moves(&mut self, record: bool, count: usize) -> usize {
        let base = self.slab.len();
        if record {
            self.slab.resize(base + count, GraphMove::EMPTY);
        }
        base
    }

    /// The code of the first `REFUTED` reply recorded at the finished node
    /// `id`, [`NO_MOVE`] when there is none.
    fn first_refuted_reply(&self, id: u32) -> u16 {
        self.slab[moves_range(&self.nodes, self.slab.len(), id)]
            .iter()
            .find(|gm| gm.outcome() == Outcome::Refuted)
            .map_or(NO_MOVE, GraphMove::code)
    }

    fn search_attack(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        id: u32,
        record: bool,
    ) -> Internal {
        if remaining == 0 {
            return Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            };
        }
        // Only checking moves, ordered by USI bytes.
        let range = self.buf.gen_checks(pos, true);
        let write = range.end;
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let count = write - range.start;
        let base = self.reserve_moves(record, count);
        let mut min_mate: Option<u32> = None;
        let mut any_unknown = false;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            let undo = pos.make(m);
            let child = self.search(pos, remaining - 1, id, encode_move(m));
            pos.unmake(m, undo);
            let distance = match child.outcome {
                Outcome::Mate => Some(child.distance.unwrap_or(0) + 1),
                _ => None,
            };
            match child.outcome {
                Outcome::Mate => {
                    let d = distance.unwrap();
                    min_mate = Some(min_mate.map_or(d, |best| best.min(d)));
                }
                Outcome::Unknown => any_unknown = true,
                Outcome::Refuted => {}
            }
            if record {
                let mut gm =
                    GraphMove::new(m, child.outcome, distance, Some(child.id), None, false);
                if self.build_graph && child.outcome == Outcome::Refuted {
                    gm.set_refutation_code(self.first_refuted_reply(child.id));
                }
                self.slab[base + index] = gm;
            }
        }
        self.buf.buf.truncate(range.start);
        // An unresolved alternative might be a shorter mate, so no exact shortest
        // answer may be claimed until every checking move is resolved.
        match (min_mate, any_unknown) {
            (_, true) => Internal {
                outcome: Outcome::Unknown,
                distance: None,
                id,
            },
            (Some(distance), false) => Internal {
                outcome: Outcome::Mate,
                distance: Some(distance),
                id,
            },
            (None, false) => Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            },
        }
    }

    fn search_defense(
        &mut self,
        pos: &mut Position,
        remaining: u32,
        id: u32,
        record: bool,
    ) -> Internal {
        // A horizon node with a legal board move is REFUTED under every rule F
        // mode: it has a reply, and that reply is not an interposition drop, so
        // the all-futile reading below cannot apply.  Nothing is recorded here,
        // so the reply list is not needed.
        if remaining == 0 && !record && self.buf.has_legal_board_move(pos) {
            return Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            };
        }
        let range = self.buf.gen_legal(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let count = range.len();
        if count == 0 {
            self.buf.buf.truncate(range.start);
            // Reaching a no-legal-move defense node through a check is checkmate.
            return if pos.is_in_check(pos.side) {
                Internal {
                    outcome: Outcome::Mate,
                    distance: Some(0),
                    id,
                }
            } else {
                Internal {
                    outcome: Outcome::Refuted,
                    distance: None,
                    id,
                }
            };
        }
        // Classical 無駄合: decided before the ply bound, because a futile drop is
        // not resistance and does not consume plies.  `classical` is set only when
        // every reply is an interposition drop; `futile_buf` then holds one verdict
        // per reply from `futile_base`.
        let futile_base = self.futile_buf.len();
        let classical = self.useless == UselessInterposition::Classical
            && self.classify_futile_replies(pos, range.clone());
        if classical
            && self.futile_buf[futile_base..futile_base + count]
                .iter()
                .all(Option::is_some)
        {
            // Every reply is futile: the check is mate.  The replies stay in the
            // graph, flagged and unsearched (`next: None`), so defense coverage
            // still sees every legal answer.
            let base = self.reserve_moves(record, count);
            if record {
                for index in 0..count {
                    self.slab[base + index] = GraphMove::new(
                        self.buf.buf[range.start + index],
                        Outcome::Mate,
                        self.futile_buf[futile_base + index],
                        None,
                        None,
                        true,
                    );
                }
            }
            self.futile_buf.truncate(futile_base);
            self.buf.buf.truncate(range.start);
            return Internal {
                outcome: Outcome::Mate,
                distance: Some(0),
                id,
            };
        }
        if remaining == 0 {
            // The defense has a legal answer but no attacker ply remains.
            // Under the classical reading a drop already proven futile keeps
            // its MATE / useless record, so the refutation the parent picks
            // (the first REFUTED reply) is never a futile interposition.
            //
            // Horizon escape (stage 2, classical, not the root): only that
            // escape is recorded -- the first reply in USI order that is not
            // futile, which is all the parent's `refutation` and the runtime's
            // selected reply read (rule G asks its own questions since 0.16.0).  The other replies are neither searched nor
            // counted, so node counts and verdicts do not change; the root keeps
            // every witness so a horizon-refuted root stays explainable.
            if self.useless == UselessInterposition::Classical && id != 0 {
                let escape = (0..count)
                    .find(|&index| !classical || self.futile_buf[futile_base + index].is_none())
                    .expect("a node that is not all-futile has a reply that is not futile");
                let base = self.reserve_moves(record, 1);
                if record {
                    self.slab[base] = GraphMove::new(
                        self.buf.buf[range.start + escape],
                        Outcome::Refuted,
                        None,
                        None,
                        None,
                        false,
                    );
                    // The other replies still count in the difficulty features.
                    for index in (0..count).filter(|&index| index != escape) {
                        let useless = classical && self.futile_buf[futile_base + index].is_some();
                        let m = self.buf.buf[range.start + index];
                        self.omitted.add_reply(pos, m, useless);
                    }
                }
                self.futile_buf.truncate(futile_base);
                self.buf.buf.truncate(range.start);
                return Internal {
                    outcome: Outcome::Refuted,
                    distance: None,
                    id,
                };
            }
            let base = self.reserve_moves(record, count);
            if record {
                for index in 0..count {
                    let futile = if classical {
                        self.futile_buf[futile_base + index]
                    } else {
                        None
                    };
                    self.slab[base + index] = GraphMove::new(
                        self.buf.buf[range.start + index],
                        if futile.is_some() {
                            Outcome::Mate
                        } else {
                            Outcome::Refuted
                        },
                        futile,
                        None,
                        None,
                        futile.is_some(),
                    );
                }
            }
            self.futile_buf.truncate(futile_base);
            self.buf.buf.truncate(range.start);
            return Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            };
        }
        let base = self.reserve_moves(record, count);
        let prune = matches!(
            self.useless,
            UselessInterposition::Prune | UselessInterposition::Classical
        );
        let prune_base = self.prune_buf.len();
        let mut escape: Option<Move> = None;
        let mut any_unknown = false;
        let mut longest = 0u32;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if classical {
                if let Some(futile) = self.futile_buf[futile_base + index] {
                    // A futile drop is neither searched nor counted: flagged with
                    // its capture-chain length, outside the main tree.
                    self.prune_buf.push(ChildInfo {
                        distance: futile,
                        useless: true,
                        outcome: Outcome::Mate,
                    });
                    if record {
                        self.slab[base + index] =
                            GraphMove::new(m, Outcome::Mate, Some(futile), None, None, true);
                    }
                    continue;
                }
            }
            let undo = pos.make(m);
            let child = self.search(pos, remaining - 1, id, encode_move(m));
            pos.unmake(m, undo);
            let distance = match child.outcome {
                Outcome::Mate => Some(child.distance.unwrap_or(0) + 1),
                _ => None,
            };
            match child.outcome {
                Outcome::Mate => longest = longest.max(distance.unwrap()),
                Outcome::Refuted => {
                    if escape.is_none() {
                        escape = Some(m);
                    }
                }
                Outcome::Unknown => any_unknown = true,
            }
            if prune {
                self.prune_buf.push(ChildInfo {
                    distance: distance.unwrap_or(0),
                    useless: false,
                    outcome: child.outcome,
                });
            }
            if record {
                self.slab[base + index] =
                    GraphMove::new(m, child.outcome, distance, Some(child.id), None, false);
            }
        }
        // Rule F's credit search (`PIPELINE_VERSION` 0.11.0): a classical node
        // with both interposition drops and other replies is judged by
        // [`Solver::resolve_with_credit`] instead of the reading below.  Without
        // a credit search and without budget exhaustion the two agree.
        let credit_line = if self.useless == UselessInterposition::Classical && !classical {
            self.credit_line(pos, range.clone())
        } else {
            None
        };
        let result = if let Some(line) = credit_line {
            let result = self.resolve_with_credit(
                pos,
                range.clone(),
                remaining,
                prune_base,
                line,
                record,
                base,
                id,
            );
            self.compact_keep_credit_node(pos, remaining, range.clone(), line, prune_base, base);
            result
        } else if let Some(escape) = escape {
            if record {
                for gm in self.slab[base..base + count].iter_mut() {
                    gm.set_refutation(Some(escape));
                }
            }
            Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            }
        } else if any_unknown {
            Internal {
                outcome: Outcome::Unknown,
                distance: None,
                id,
            }
        } else {
            // Every legal reply had to mate for the node to be MATE; rule F only
            // decides which of them count as resistance.
            if prune {
                // Under Classical a node that reached here through `classical` has
                // no non-interposition reply, so this marks nothing further and
                // only the futile flags already pushed apply.
                self.mark_useless_interpositions(pos, range.clone(), remaining, prune_base);
                longest = 0;
                let flags = &self.prune_buf[prune_base..prune_base + count];
                for info in flags.iter() {
                    if !info.useless {
                        longest = longest.max(info.distance);
                    }
                }
                if record {
                    for (gm, info) in self.slab[base..base + count].iter_mut().zip(flags.iter()) {
                        gm.set_useless(info.useless);
                    }
                }
            }
            Internal {
                outcome: Outcome::Mate,
                distance: Some(longest),
                id,
            }
        };
        self.prune_buf.truncate(prune_base);
        self.futile_buf.truncate(futile_base);
        // Truncated only here: rule F reads the reply list after every child has
        // been resolved.
        self.buf.buf.truncate(range.start);
        result
    }

    /// The check line of a defense node rule F's credit search applies to: the
    /// check is interposable ([`sliding_check_line`]) and the replies in `range`
    /// hold both interposition drops and other replies.  `None` otherwise.
    fn credit_line(&self, pos: &Position, range: std::ops::Range<usize>) -> Option<CheckLine> {
        let replies = &self.buf.buf[range];
        // Cheap pre-test: in check, a drop is always an interposition drop.
        if !replies.iter().any(|m| m.is_drop()) || replies.iter().all(|m| m.is_drop()) {
            return None;
        }
        let line = sliding_check_line(pos, self.attacker)?;
        let drops = replies.iter().filter(|&&m| line.interposes(m)).count();
        (drops > 0 && drops < replies.len()).then_some(line)
    }

    /// Rule F under [`UselessInterposition::Classical`] since `PIPELINE_VERSION`
    /// 0.11.0 (`docs/rules.md`, "The ply bound counts rule-F plies"): the verdict
    /// of a defense node with both interposition drops and other replies, every
    /// reply already searched at `remaining - 1` (`prune_buf` from `base` and
    /// the recorded moves from `moves_base` are index-parallel to `range`).
    ///
    /// 1. A REFUTED other reply refutes the node (the first in USI order, which
    ///    comes before every drop); else an UNKNOWN one makes it UNKNOWN.
    /// 2. `dist_other` is the longest other reply.  First pass, over the drops
    ///    whose child is REFUTED, in USI order: with `dist_other <= k - 2` the
    ///    drop is effective; else the credit search
    ///    ([`Solver::credit_interposition`]) decides.  The first effective drop
    ///    refutes the node and ends the pass.  A useless one is recorded `MATE`
    ///    / `useless` with `2 + dist(E)`, an undecided one `UNKNOWN`; a drop whose
    ///    child is UNKNOWN is undecided.  With no effective drop and an undecided
    ///    one the node is UNKNOWN.
    /// 3. Second pass, over the drops whose child is MATE, in USI order: a
    ///    capture form proven within `k - 2` decides by table lookup as before
    ///    (`shortest dist(E) <= dist_other`); else with `dist_other >= k - 1` the
    ///    credit search decides, and the first undecided drop makes the node
    ///    UNKNOWN; else the drop is effective.
    /// 4. The node mates at the longest edge distance of its replies that are
    ///    not useless.
    // The slab holds the moves only when `record`, so the reply index drives
    // the loops.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn resolve_with_credit(
        &mut self,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        remaining: u32,
        base: usize,
        line: CheckLine,
        record: bool,
        moves_base: usize,
        id: u32,
    ) -> Internal {
        let count = range.len();
        let unknown = Internal {
            outcome: Outcome::Unknown,
            distance: None,
            id,
        };
        let refute = |escape: Move, slab: &mut [GraphMove]| {
            if record {
                for gm in slab[moves_base..moves_base + count].iter_mut() {
                    gm.set_refutation(Some(escape));
                }
            }
            Internal {
                outcome: Outcome::Refuted,
                distance: None,
                id,
            }
        };
        let mut dist_other = 0u32;
        let mut other_unknown = false;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if line.interposes(m) {
                continue;
            }
            let info = self.prune_buf[base + index];
            match info.outcome {
                Outcome::Refuted => return refute(m, &mut self.slab),
                Outcome::Unknown => other_unknown = true,
                Outcome::Mate => dist_other = dist_other.max(info.distance),
            }
        }
        if other_unknown {
            return unknown;
        }
        let mut probe = CaptureProbe::default();
        let mut undecided = false;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if !line.interposes(m) {
                continue;
            }
            match self.prune_buf[base + index].outcome {
                Outcome::Mate => continue,
                Outcome::Unknown => {
                    undecided = true;
                    continue;
                }
                Outcome::Refuted => {}
            }
            let verdict = if dist_other + 2 <= remaining {
                Credit::Effective
            } else {
                self.credit_interposition(pos, m, line.checker, dist_other, &mut probe)
            };
            match verdict {
                Credit::Effective => {
                    if undecided {
                        self.credit.undecided_then_effective += 1;
                    }
                    return refute(m, &mut self.slab);
                }
                Credit::Useless(after) => {
                    let info = &mut self.prune_buf[base + index];
                    info.useless = true;
                    info.distance = after + 2;
                    if record {
                        let gm = &mut self.slab[moves_base + index];
                        gm.set_outcome(Outcome::Mate);
                        gm.set_distance(Some(after + 2));
                        gm.set_useless(true);
                    }
                }
                Credit::Undecided => {
                    undecided = true;
                    self.credit.first_pass_undecided += 1;
                    if record {
                        self.slab[moves_base + index].set_outcome(Outcome::Unknown);
                    }
                }
            }
        }
        if undecided {
            return unknown;
        }
        // The second pass flags nothing until it has judged every drop: an
        // undecided drop makes D UNKNOWN, and F1 flags only replies of MATE nodes.
        let mut flagged: Vec<usize> = Vec::new();
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if !line.interposes(m) || self.prune_buf[base + index].outcome != Outcome::Mate {
                continue;
            }
            // The capture line is in the table at `remaining - 2`: the child
            // attack node mated at `remaining - 1`, so it resolved every checking
            // move, each one entered in the table (a lookup, not a new search).
            let undo = pos.make(m);
            let forms = match probe.get(m.to) {
                Some(forms) => forms,
                None => {
                    let forms = self.capture_forms(pos, line.checker, m.to);
                    probe.insert(m.to, forms);
                    forms
                }
            };
            let mut shortest: Option<u32> = None;
            for form in forms.iter().flatten() {
                let capture = pos.make(*form);
                let after = self.lookup(remaining - 2, pos);
                pos.unmake(*form, capture);
                let Some(after) = after else {
                    debug_assert!(false, "the capture line is already in the table");
                    continue;
                };
                debug_assert_ne!(
                    after.outcome,
                    Outcome::Unknown,
                    "a mating child resolved it"
                );
                if after.outcome == Outcome::Mate {
                    let distance = after.distance.unwrap_or(0);
                    shortest = Some(shortest.map_or(distance, |best: u32| best.min(distance)));
                }
            }
            pos.unmake(m, undo);
            let useless = match shortest {
                Some(distance) => distance <= dist_other,
                None if dist_other + 1 >= remaining => {
                    match self.credit_interposition(pos, m, line.checker, dist_other, &mut probe) {
                        Credit::Useless(_) => true,
                        Credit::Effective => false,
                        Credit::Undecided => {
                            self.credit.second_pass_stops += 1;
                            return unknown;
                        }
                    }
                }
                None => false,
            };
            if useless {
                flagged.push(index);
            }
        }
        for index in flagged {
            self.prune_buf[base + index].useless = true;
            if record {
                self.slab[moves_base + index].set_useless(true);
            }
        }
        let longest = self.prune_buf[base..base + count]
            .iter()
            .filter(|info| !info.useless)
            .map(|info| info.distance)
            .max()
            .unwrap_or(0);
        Internal {
            outcome: Outcome::Mate,
            distance: Some(longest),
            id,
        }
    }

    /// Rule F's **credit search** (`PIPELINE_VERSION` 0.11.0) for the
    /// interposition drop `m` at the defense node `pos`, whose only checker
    /// stands on `checker`: the checker's capture forms (non-promoting first,
    /// then promoting, legal checks only) and, for each, the defense node E after
    /// it, asked of the credit oracle ([`Solver::oracle`]) at `bound` =
    /// `dist_other`.  The first form that mates makes the drop useless and ends
    /// the search; an UNKNOWN form does not stop the next one.  Neither the drop
    /// nor the capture consumes any of the bound: E has one defender piece fewer
    /// in hand, so (bound, defender hand size) still strictly decreases.
    fn credit_interposition(
        &mut self,
        pos: &mut Position,
        m: Move,
        checker: u8,
        bound: u32,
        probe: &mut CaptureProbe,
    ) -> Credit {
        let undo = pos.make(m);
        let forms = match probe.get(m.to) {
            Some(forms) => forms,
            None => {
                let forms = self.capture_forms(pos, checker, m.to);
                probe.insert(m.to, forms);
                forms
            }
        };
        let mut verdict = Credit::Effective;
        let mut saw_unknown = false;
        for form in forms.iter().flatten() {
            self.credit.calls += 1;
            let capture = pos.make(*form);
            let after = self.oracle(pos, bound);
            pos.unmake(*form, capture);
            match after {
                // Parity: E is not mate within k - 2, and dist(E) and `bound`
                // are both even, so a credit mate is at the bound.
                Outcome::Mate => {
                    if saw_unknown {
                        self.credit.unknown_then_mate += 1;
                    }
                    verdict = Credit::Useless(bound);
                    break;
                }
                Outcome::Unknown => {
                    saw_unknown = true;
                    verdict = Credit::Undecided;
                }
                Outcome::Refuted => {}
            }
        }
        if matches!(verdict, Credit::Useless(_)) {
            self.credit.useless += 1;
        }
        pos.unmake(m, undo);
        verdict
    }

    /// Rule F's **credit oracle** (`PIPELINE_VERSION` 0.11.0, stage 2): whether
    /// `pos` is mate within `bound` rule-F plies under
    /// [`UselessInterposition::Classical`], as a verdict only (`MATE`,
    /// `REFUTED`, or `UNKNOWN` when the oracle's node budget runs out).
    ///
    /// The credit search asks a yes/no question of E, so it is answered here,
    /// outside the main search, like F2's futility test: the oracle creates no
    /// graph node, adds nothing to `stats.nodes` or `stats.cacheHits`, and keeps
    /// its own table ([`Oracle`]) and its own node budget
    /// ([`SolveOptions::credit_budget`]).  With a sufficient budget its verdict
    /// is the main search's outcome at the same bound, so the results of the
    /// main search do not depend on it being an oracle; only the cost does.
    ///
    /// Unlike the main search it stops early: an attack node at its first
    /// mating check, a defense node at its first reply that does not mate.  Moves
    /// are tried in USI order, like the main search.  The table keeps, per
    /// position, the smallest bound proven mate and one more than the largest
    /// bound proven not mate; that is sound because a mate within `b` is a mate
    /// within every larger bound and a refutation at `b` holds at every smaller
    /// one (`docs/rules.md` rule F, "The ply bound counts rule-F plies").  An
    /// `UNKNOWN` is never stored.  Every call that the table does not answer is
    /// one oracle node.
    ///
    /// A defense node D with interposition drops and other replies (the credit
    /// line) mates within `k` exactly when every other reply's child mates
    /// within `k - 1` and every drop `r` satisfies
    /// `G_r = (E_r mates within q AND W) OR C_r mates within k - 1`, where `q`
    /// is the largest even bound `<= k`, `E_r` is the defense node after one of
    /// the checker's capture forms (non-promoting first), and
    /// `W` = "dist_other(D) = q" = some other reply's child is not mate within
    /// `q - 2`.  The oracle evaluates `E_r` first, then `W` (once per node), then
    /// `C_r` only when still needed: a useless drop is then settled without
    /// reading the attack node after it.  Every other defense node mates within
    /// `k` when every reply not proven F2-futile has a child mating within
    /// `k - 1` (an all-futile node mates at any bound, as in the main search).
    pub(crate) fn oracle(&mut self, pos: &mut Position, bound: u32) -> Outcome {
        let key = oracle_key(pos.hash());
        if let Some(entry) = self.oracle.table.get(&key) {
            if let Some((outcome, pure)) = entry.answer(bound) {
                if outcome == Outcome::Mate {
                    self.oracle.pure = pure;
                }
                return outcome;
            }
        }
        // Pure-mate dominance (owner idea 4'): the same board and side to move,
        // proven mate within `bound` by a pure proof with no more attacker hand.
        let board = board_key(pos);
        let hand = pos.hands[self.attacker as usize];
        if let Some(entries) = self.oracle.dominance.get(&board) {
            if dominated(entries, hand, bound) {
                self.oracle.pure = true;
                return Outcome::Mate;
            }
        }
        if self.oracle.nodes >= self.oracle.budget {
            self.oracle.exhausted = true;
            return Outcome::Unknown;
        }
        let started = self.oracle.nodes;
        self.oracle.nodes += 1;
        let (result, pure) = if pos.side == self.attacker {
            self.oracle_attack(pos, bound)
        } else {
            self.oracle_defense(pos, bound)
        };
        // Only bounds a byte holds exactly are remembered (`TABLE_BOUND_MAX`);
        // `u8::MAX` is the "none" sentinel and is never stored as a bound.
        if result != Outcome::Unknown && bound <= TABLE_BOUND_MAX {
            let work = (self.oracle.nodes - started).min(u64::from(u32::MAX)) as u32;
            self.oracle.record(key, result, pure, bound, work);
            if result == Outcome::Mate && pure {
                self.oracle.record_dominance(board, hand, bound);
            }
        }
        self.oracle.pure = pure;
        result
    }

    /// An attack node of the oracle: the verdict, and whether a `MATE` is pure.
    fn oracle_attack(&mut self, pos: &mut Position, bound: u32) -> (Outcome, bool) {
        if bound == 0 {
            return (Outcome::Refuted, false);
        }
        let range = self.buf.gen_checks(pos, true);
        let write = range.end;
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let mut result = Outcome::Refuted;
        let mut pure = false;
        for index in range.start..write {
            let m = self.buf.buf[index];
            let undo = pos.make(m);
            let child = self.oracle(pos, bound - 1);
            pos.unmake(m, undo);
            result = or3(result, child);
            if result == Outcome::Mate {
                pure = self.oracle.pure;
                break;
            }
        }
        self.buf.buf.truncate(range.start);
        (result, pure)
    }

    /// A defense node of the oracle: the verdict, and whether a `MATE` is pure.
    fn oracle_defense(&mut self, pos: &mut Position, bound: u32) -> (Outcome, bool) {
        // At the bound, a legal board move is a reply that is not an
        // interposition drop, so F2 cannot make the node mate: REFUTED, as
        // below, without the reply list.
        if bound == 0 && self.buf.has_legal_board_move(pos) {
            return (Outcome::Refuted, false);
        }
        let range = self.buf.gen_legal(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let count = range.len();
        if count == 0 {
            self.buf.buf.truncate(range.start);
            return if pos.is_in_check(pos.side) {
                (Outcome::Mate, true)
            } else {
                (Outcome::Refuted, false)
            };
        }
        let futile_base = self.futile_buf.len();
        let classical = self.classify_futile_replies(pos, range.clone());
        let result = if classical
            && self.futile_buf[futile_base..futile_base + count]
                .iter()
                .all(Option::is_some)
        {
            // Mate by rule F2: not pure.
            (Outcome::Mate, false)
        } else if bound == 0 {
            (Outcome::Refuted, false)
        } else {
            let line = if classical {
                None
            } else {
                self.credit_line(pos, range.clone())
            };
            match line {
                None => self.oracle_replies(pos, range.clone(), bound, classical, futile_base),
                Some(line) => self.oracle_credit_line(pos, range.clone(), bound, line),
            }
        };
        self.futile_buf.truncate(futile_base);
        self.buf.buf.truncate(range.start);
        result
    }

    /// Every reply not proven futile must mate within `bound - 1`.
    fn oracle_replies(
        &mut self,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        bound: u32,
        classical: bool,
        futile_base: usize,
    ) -> (Outcome, bool) {
        let mut result = Outcome::Mate;
        let mut pure = true;
        for (offset, index) in range.enumerate() {
            if classical && self.futile_buf[futile_base + offset].is_some() {
                // A reply skipped by rule F2: the proof is not pure.
                pure = false;
                continue;
            }
            let m = self.buf.buf[index];
            let undo = pos.make(m);
            let child = self.oracle(pos, bound - 1);
            pos.unmake(m, undo);
            pure &= child == Outcome::Mate && self.oracle.pure;
            result = and3(result, child);
            if result == Outcome::Refuted {
                break;
            }
        }
        (result, pure)
    }

    /// A credit-line defense node (see [`Solver::oracle`]).
    fn oracle_credit_line(
        &mut self,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        bound: u32,
        line: CheckLine,
    ) -> (Outcome, bool) {
        // A credit-line node has a non-interposition reply, whose child (an
        // attack node) cannot mate within `bound - 1 = 0`: not mate within 1.
        // Decided here so that `q - 2` below never goes under zero (even when
        // the oracle's budget ran out and that child came back UNKNOWN).
        if bound < 2 {
            return (Outcome::Refuted, false);
        }
        let mut result = Outcome::Mate;
        let mut pure = true;
        // The other replies (board moves sort before drops in USI order).
        for index in range.clone() {
            let m = self.buf.buf[index];
            if line.interposes(m) {
                continue;
            }
            let undo = pos.make(m);
            let child = self.oracle(pos, bound - 1);
            pos.unmake(m, undo);
            pure &= child == Outcome::Mate && self.oracle.pure;
            result = and3(result, child);
            if result == Outcome::Refuted {
                return (result, false);
            }
        }
        // With every other reply mating within k - 1, dist_other <= k, and a
        // drop can be useless by credit only when dist_other = q.  `bound >= 2`
        // here, so q >= 2.
        let q = bound & !1;
        let mut witness: Option<Outcome> = None;
        let mut probe = CaptureProbe::default();
        for index in range {
            let m = self.buf.buf[index];
            if !line.interposes(m) {
                continue;
            }
            let credit = if witness == Some(Outcome::Refuted) {
                Outcome::Refuted
            } else {
                self.oracle_capture(pos, m, line.checker, q, &mut probe)
            };
            if credit != Outcome::Refuted {
                let w = match witness {
                    Some(w) => w,
                    None => {
                        let w = self.oracle_witness(pos, line, q);
                        witness = Some(w);
                        w
                    }
                };
                if and3(credit, w) == Outcome::Mate {
                    // Useless by credit (rule F1): not pure.
                    pure = false;
                    continue;
                }
            }
            let undo = pos.make(m);
            let child = self.oracle(pos, bound - 1);
            pos.unmake(m, undo);
            pure &= child == Outcome::Mate && self.oracle.pure;
            let saved = if credit == Outcome::Refuted {
                Outcome::Refuted
            } else {
                and3(credit, witness.unwrap_or(Outcome::Unknown))
            };
            result = and3(result, or3(child, saved));
            if result == Outcome::Refuted {
                return (result, false);
            }
        }
        (result, pure)
    }

    /// `W` of a credit-line node: `MATE` when some non-interposition reply's
    /// child is not mate within `q - 2` (so dist_other = q, given they all mate
    /// within `k - 1`), `REFUTED` when every one is, else `UNKNOWN`.
    fn oracle_witness(&mut self, pos: &mut Position, line: CheckLine, q: u32) -> Outcome {
        let range = self.buf.gen_legal(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let mut result = Outcome::Refuted;
        for index in range.clone() {
            let m = self.buf.buf[index];
            if line.interposes(m) {
                continue;
            }
            let undo = pos.make(m);
            let child = self.oracle(pos, q - 2);
            pos.unmake(m, undo);
            result = or3(result, not3(child));
            if result == Outcome::Mate {
                break;
            }
        }
        self.buf.buf.truncate(range.start);
        result
    }

    /// Whether E after one of the checker's capture forms of the drop `m`
    /// (non-promoting first) mates within `q`: `MATE` at the first form that
    /// does, else `UNKNOWN` if a form is undecided, else `REFUTED`.
    fn oracle_capture(
        &mut self,
        pos: &mut Position,
        m: Move,
        checker: u8,
        q: u32,
        probe: &mut CaptureProbe,
    ) -> Outcome {
        let undo = pos.make(m);
        let forms = match probe.get(m.to) {
            Some(forms) => forms,
            None => {
                let forms = self.capture_forms(pos, checker, m.to);
                probe.insert(m.to, forms);
                forms
            }
        };
        let mut result = Outcome::Refuted;
        for form in forms.iter().flatten() {
            let capture = pos.make(*form);
            let after = self.oracle(pos, q);
            pos.unmake(*form, capture);
            result = or3(result, after);
            if result == Outcome::Mate {
                break;
            }
        }
        pos.unmake(m, undo);
        result
    }

    /// Rule F (`docs/rules.md`), applied to one proven-mate defense node: flags
    /// every 無駄合 among its replies.
    ///
    /// `range` still holds the replies in [`MoveBuf`], index-parallel to the
    /// [`ChildInfo`] entries from `base`.  An interposition drop is useless
    /// exactly when the checker can capture the dropped piece with check, that
    /// capture is a proven mate, and the defense node after it resists no longer
    /// than the best non-interposition reply -- so the drop bought nothing beyond
    /// the two plies it spent.  Distances read here are already rule-F distances,
    /// because children were searched first.
    ///
    /// Where `packages/solver` looks the capture up in the child attack node's
    /// recorded move list, this reads the *transposition table*: an attack node
    /// only reports `MATE` after resolving every one of its legal checking moves,
    /// so the defense node behind each of them is already in the table at
    /// `remaining - 2`, carrying exactly what the recorded edge would
    /// (`edge.distance == node.distance + 1`).  That keeps the graph-free path --
    /// the one the reducer uses -- identical to the graph path.
    fn mark_useless_interpositions(
        &mut self,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        remaining: u32,
        base: usize,
    ) {
        debug_assert!(remaining >= 2, "a mating defense node has two plies left");
        if remaining < 2 {
            return;
        }
        let Some(line) = sliding_check_line(pos, self.attacker) else {
            return;
        };
        let count = range.len();
        // dist_other: with no non-interposition reply to compare against, nothing
        // at this node is useless.
        let mut dist_other: Option<u32> = None;
        for index in 0..count {
            if line.interposes(self.buf.buf[range.start + index]) {
                continue;
            }
            let distance = self.prune_buf[base + index].distance;
            dist_other = Some(dist_other.map_or(distance, |best: u32| best.max(distance)));
        }
        let Some(dist_other) = dist_other else {
            return;
        };
        // Which capture forms are legal checks depends only on the square the
        // interposition lands on, never on which piece was dropped there: the
        // occupant is an enemy piece either way, and hands change neither legality
        // nor check.  One probe therefore serves every drop onto the same square,
        // and `between` holds at most seven squares.
        let mut probed: [(u8, [Option<Move>; 2]); 8] = [(NO_SQUARE, [None; 2]); 8];
        let mut probed_len = 0usize;
        for index in 0..count {
            let m = self.buf.buf[range.start + index];
            if !line.interposes(m) {
                continue;
            }
            let undo = pos.make(m);
            let forms = match probed[..probed_len]
                .iter()
                .find(|(square, _)| *square == m.to)
            {
                Some((_, forms)) => *forms,
                None => {
                    let forms = self.capture_forms(pos, line.checker, m.to);
                    probed[probed_len] = (m.to, forms);
                    probed_len += 1;
                    forms
                }
            };
            // The checker recaptures on the interposition square.  When both forms
            // are legal checks the shorter mate wins, so only the shortest proven
            // one can flag; a form that does not mate never can.
            let mut shortest: Option<u32> = None;
            for form in forms.iter().flatten() {
                let capture = pos.make(*form);
                let after = self.lookup(remaining - 2, pos);
                pos.unmake(*form, capture);
                let Some(after) = after else {
                    debug_assert!(false, "the capture line is already in the table");
                    continue;
                };
                if after.outcome == Outcome::Mate {
                    let distance = after.distance.unwrap_or(0);
                    shortest = Some(shortest.map_or(distance, |best: u32| best.min(distance)));
                }
            }
            pos.unmake(m, undo);
            if shortest.is_some_and(|distance| distance <= dist_other) {
                self.prune_buf[base + index].useless = true;
            }
        }
    }

    /// [`UselessInterposition::Classical`]: when the defense node `pos` is
    /// interposable (rule F's [`sliding_check_line`]) and **every** reply in
    /// `range` is an interposition drop, pushes one futility verdict per reply
    /// onto `futile_buf` and returns `true`.  Otherwise pushes nothing and returns
    /// `false`, and the node is handled exactly as under `Prune`.
    ///
    /// The work is [`Futility::classify`], shared with the df-pn prober.
    fn classify_futile_replies(
        &mut self,
        pos: &mut Position,
        range: std::ops::Range<usize>,
    ) -> bool {
        self.futility.classify(
            &mut self.buf,
            self.attacker,
            pos,
            range,
            &mut self.futile_buf,
        )
    }

    /// [`capture_forms`] on this solver's move stack.
    fn capture_forms(&mut self, pos: &mut Position, checker: u8, target: u8) -> [Option<Move>; 2] {
        capture_forms(&mut self.buf, self.attacker, pos, checker, target)
    }
}

/// The credit oracle's state ([`Solver::oracle`]), reset with every solve.
struct Oracle {
    table: HashMap<[u64; 2], OracleEntry, BuildHasherDefault<KeyHasher>>,
    /// Pure-mate dominance (owner idea 4'): board-and-side key (hands left out)
    /// to at most [`DOMINANCE_ENTRIES`] (attacker hand, smallest bound proven
    /// mate by a pure proof) pairs, oldest first.
    dominance: DominanceTable,
    /// Whether the last `MATE` [`Solver::oracle`] returned is pure.
    pure: bool,
    nodes: u64,
    budget: u64,
    exhausted: bool,
    /// Garbage collections of [`Oracle::table`] since the solver was built
    /// (measurement only).
    collections: u64,
    /// [`ORACLE_TABLE_CAP`] (lowered only by tests).
    cap: usize,
}

/// One oracle table entry.  A mate proof is **pure** when no node of it relied
/// on rule F: no reply skipped as F2-futile, no all-futile node, no drop saved
/// by credit.  It is then an ordinary AND/OR proof, which stays a proof when the
/// attacker holds more and (strict material) the defender less: every attacker
/// move of it stays legal (drop legality -- 二歩, 行き所のない駒, 打ち歩詰め --
/// depends on the board alone: a pawn-drop check is adjacent, so whether it mates
/// does not depend on the defender's hand), and the defender's replies are a
/// subset (fewer drops, the same board moves).  `docs/experiments/rule-f-bound/
/// stage2/LOG.md` §4'.
#[derive(Clone, Copy, Debug)]
struct OracleEntry {
    /// Smallest bound proven mate, `u8::MAX` if none.
    mate: u8,
    /// One more than the largest bound proven not mate, `0` if none.
    refuted_below: u8,
    /// Smallest bound proven mate by a pure proof, `u8::MAX` if none.
    pure: u8,
    /// Oracle nodes spent on this position's verdicts (saturating), the
    /// garbage collection's measure of what an entry is worth
    /// ([`Oracle::collect`]).
    work: u32,
}

impl OracleEntry {
    const EMPTY: Self = OracleEntry {
        mate: u8::MAX,
        refuted_below: 0,
        pure: u8::MAX,
        work: 0,
    };

    /// The entry's answer at `bound`: `MATE` with its purity, `REFUTED`, or
    /// `None` when the entry does not decide that bound.
    fn answer(&self, bound: u32) -> Option<(Outcome, bool)> {
        if self.mate != u8::MAX && u32::from(self.mate) <= bound {
            return Some((
                Outcome::Mate,
                self.pure != u8::MAX && u32::from(self.pure) <= bound,
            ));
        }
        (bound < u32::from(self.refuted_below)).then_some((Outcome::Refuted, false))
    }

    /// Records a decided verdict at `bound <= TABLE_BOUND_MAX`.
    fn record(&mut self, outcome: Outcome, pure: bool, bound: u32) {
        debug_assert!(bound <= TABLE_BOUND_MAX);
        let bound = bound as u8;
        match outcome {
            Outcome::Mate => {
                self.mate = self.mate.min(bound);
                if pure {
                    self.pure = self.pure.min(bound);
                }
            }
            Outcome::Refuted => self.refuted_below = self.refuted_below.max(bound + 1),
            Outcome::Unknown => {}
        }
    }
}

/// Whether a pure mate within `bound` is recorded for this board with no more
/// attacker hand than `hand` (owner idea 4').
pub(crate) fn dominated(entries: &[([u8; 7], u8)], hand: [u8; 7], bound: u32) -> bool {
    entries
        .iter()
        .any(|&(held, pure)| u32::from(pure) <= bound && hand_le(held, hand))
}

/// Records a pure mate within `bound <= TABLE_BOUND_MAX` with attacker hand
/// `hand`: the same hand keeps the smaller bound, a new hand is appended and the
/// oldest is dropped past [`DOMINANCE_ENTRIES`].
pub(crate) fn record_dominance(entries: &mut Vec<([u8; 7], u8)>, hand: [u8; 7], bound: u32) {
    debug_assert!(bound <= TABLE_BOUND_MAX);
    let bound = bound as u8;
    match entries.iter_mut().find(|(held, _)| *held == hand) {
        Some(found) => found.1 = found.1.min(bound),
        None => {
            if entries.len() >= DOMINANCE_ENTRIES {
                entries.remove(0);
            }
            entries.push((hand, bound));
        }
    }
}

/// Board-and-side key to (attacker hand, smallest pure mate bound) pairs.
type DominanceTable = HashMap<[u64; 2], Vec<([u8; 7], u8)>, BuildHasherDefault<KeyHasher>>;

/// The largest bound the oracle table keeps (a byte, `u8::MAX` being "none" and
/// `TABLE_BOUND_MAX + 1` the largest "refuted below").  Larger ones are not
/// remembered, never rounded.
const TABLE_BOUND_MAX: u32 = 253;

/// Hands kept per board in the dominance table (oldest dropped first).
const DOMINANCE_ENTRIES: usize = 32;

/// The most positions the credit oracle's table keeps (solver v3, item 3):
/// 7/8 of 2^22, the largest count a 2^22-bucket table holds without growing,
/// so the table stays at about 100 MB.  Past it `Oracle::collect` drops the
/// cheaper half.  Part of the deterministic contract of the oracle's node count
/// (`ORACLE_TABLE_CAP` in `packages/solver`).
pub const ORACLE_TABLE_CAP: usize = (1 << 22) / 8 * 7;

/// The most boards the pure-mate dominance table keeps; a new board past it
/// empties the table first (`DOMINANCE_BOARD_CAP` in `packages/solver`).
pub const DOMINANCE_BOARD_CAP: usize = (1 << 17) / 8 * 7;

/// The most positions the F2 futility memo keeps; a new position past it
/// empties the memo first.  The memo only caches a pure function of the
/// position, so emptying it changes cost, never a verdict or a count.
pub const FUTILE_MEMO_CAP: usize = (1 << 22) / 8 * 7;

impl Oracle {
    fn new() -> Self {
        Oracle {
            table: HashMap::default(),
            dominance: DominanceTable::default(),
            pure: false,
            nodes: 0,
            budget: 0,
            exhausted: false,
            collections: 0,
            cap: ORACLE_TABLE_CAP,
        }
    }

    fn reset(&mut self, budget: u64) {
        self.table.clear();
        self.dominance.clear();
        self.pure = false;
        self.nodes = 0;
        self.budget = budget;
        self.exhausted = false;
    }

    /// Records a decided verdict at `bound <= TABLE_BOUND_MAX` that cost `work`
    /// oracle nodes.  A new position first makes room when the table is full
    /// ([`ORACLE_TABLE_CAP`], [`Oracle::collect`]).
    fn record(&mut self, key: [u64; 2], outcome: Outcome, pure: bool, bound: u32, work: u32) {
        if self.table.len() >= self.cap && !self.table.contains_key(&key) {
            self.collect();
        }
        let entry = self.table.entry(key).or_insert(OracleEntry::EMPTY);
        entry.record(outcome, pure, bound);
        entry.work = entry.work.saturating_add(work);
    }

    /// [`record_dominance`] for `board`, emptying the dominance table first when
    /// a new board would take it past [`DOMINANCE_BOARD_CAP`].
    fn record_dominance(&mut self, board: [u64; 2], hand: [u8; 7], bound: u32) {
        if self.dominance.len() >= DOMINANCE_BOARD_CAP && !self.dominance.contains_key(&board) {
            self.dominance.clear();
        }
        record_dominance(self.dominance.entry(board).or_default(), hand, bound);
    }

    /// The table's deterministic garbage collection (solver v3, item 3): keeps
    /// the table at most [`ORACLE_TABLE_CAP`] positions, so one thread's oracle
    /// has a fixed memory cap.  It drops every entry whose `work` falls in the
    /// lowest power-of-two classes, the fewest classes that hold at least half
    /// of the entries.  Which entries go is a function of the entries alone,
    /// never of where they sit in the table, so the table's contents depend
    /// only on the sequence of verdicts recorded (the TypeScript oracle drops
    /// the same entries).  A dropped verdict is recomputed when asked again:
    /// the cost changes, a verdict never does.
    fn collect(&mut self) {
        let mut classes = [0usize; 33];
        for entry in self.table.values() {
            classes[work_class(entry.work)] += 1;
        }
        let target = self.table.len().div_ceil(2);
        let mut dropped = 0usize;
        let mut limit = 0usize;
        for (class, &count) in classes.iter().enumerate() {
            dropped += count;
            limit = class;
            if dropped >= target {
                break;
            }
        }
        self.table.retain(|_, entry| work_class(entry.work) > limit);
        self.collections += 1;
    }
}

/// The power-of-two class of a work count: 0 for 0, else 1 + floor(log2).
#[inline]
fn work_class(work: u32) -> usize {
    (u32::BITS - work.leading_zeros()) as usize
}

/// Every hand count of `a` at most that of `b`.
#[inline]
fn hand_le(a: [u8; 7], b: [u8; 7]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

/// The Zobrist key of the board and the side to move, both hands left out.
pub(crate) fn board_key(pos: &Position) -> [u64; 2] {
    let keys = &crate::zobrist::KEYS;
    let mut hash = pos.hash();
    for color in 0..2 {
        for piece in 0..7 {
            hash ^= keys.hand[color][piece][pos.hands[color][piece] as usize];
        }
    }
    oracle_key(hash)
}

/// Three-valued AND / OR over [`Outcome`] (`MATE` true, `REFUTED` false).
fn and3(a: Outcome, b: Outcome) -> Outcome {
    match (a, b) {
        (Outcome::Refuted, _) | (_, Outcome::Refuted) => Outcome::Refuted,
        (Outcome::Mate, Outcome::Mate) => Outcome::Mate,
        _ => Outcome::Unknown,
    }
}

fn or3(a: Outcome, b: Outcome) -> Outcome {
    match (a, b) {
        (Outcome::Mate, _) | (_, Outcome::Mate) => Outcome::Mate,
        (Outcome::Refuted, Outcome::Refuted) => Outcome::Refuted,
        _ => Outcome::Unknown,
    }
}

fn not3(a: Outcome) -> Outcome {
    match a {
        Outcome::Mate => Outcome::Refuted,
        Outcome::Refuted => Outcome::Mate,
        Outcome::Unknown => Outcome::Unknown,
    }
}

/// The oracle table key: the position's 128-bit Zobrist key as `[u64; 2]`.
#[inline]
fn oracle_key(hash: u128) -> [u64; 2] {
    [hash as u64, (hash >> 64) as u64]
}

/// [`oracle_key`] for the other bound-independent tables of the crate.
#[inline]
pub(crate) fn oracle_key_of(hash: u128) -> [u64; 2] {
    oracle_key(hash)
}

/// [`capture_forms`] for the other engines of the crate.
pub(crate) fn capture_forms_of(
    buf: &mut MoveBuf,
    attacker: u8,
    pos: &mut Position,
    checker: u8,
    target: u8,
) -> [Option<Move>; 2] {
    capture_forms(buf, attacker, pos, checker, target)
}

/// Classical futility memo: a defence position's hash with the attacker's hand
/// masked out, to [`Futility::futile_defense`]; `None` means not futile.
/// Stored as a byte (`u8::MAX` for `None`; a chain is at most 2 x 7 plies) under
/// a `[u64; 2]` key, so a slot is 24 bytes.  (Stage 2 dropped the per-drop memo:
/// it held up to ten entries per node of the search and its verdict is the OR of
/// at most two lookups here.)
type FutileMemo = HashMap<[u64; 2], u8, BuildHasherDefault<KeyHasher>>;

/// The classical 無駄合 test of rule F2 (`docs/rules.md`, `PIPELINE_VERSION`
/// 0.10.0), shared by the exact [`Solver`] and the df-pn prober
/// ([`crate::dfpn`]) so the two engines cannot drift apart on it.
///
/// It is pure move generation outside any main search: it creates no search
/// node, is charged to no node budget and is bounded by no ply bound (each
/// capture brings the checker strictly closer to the king, so a chain is at most
/// seven captures deep).  The two memos only cache verdicts that are functions of
/// the position, so clearing them changes cost, never an answer.
pub struct Futility {
    memo: FutileMemo,
    /// [`FUTILE_MEMO_CAP`] (lowered only by tests).
    cap: usize,
}

impl Default for Futility {
    fn default() -> Self {
        Self::new()
    }
}

impl Futility {
    pub fn new() -> Self {
        Futility {
            memo: FutileMemo::default(),
            cap: FUTILE_MEMO_CAP,
        }
    }

    /// Drops the memo.
    pub fn clear(&mut self) {
        self.memo.clear();
    }

    /// Memoises one verdict, emptying the memo first when a new position would
    /// take it past [`FUTILE_MEMO_CAP`] (solver v3, item 3).
    fn remember(&mut self, key: [u64; 2], verdict: u8) {
        if self.memo.len() >= self.cap {
            self.memo.clear();
        }
        self.memo.insert(key, verdict);
    }

    /// The futility verdicts of the defense node `pos` (defender to move, in
    /// check by `attacker`), whose legal replies are `buf.buf[range]`.
    ///
    /// When the check is interposable ([`sliding_check_line`]) and **every** reply
    /// is an interposition drop, pushes one verdict per reply onto `out`, in
    /// `range` order -- `Some(plies of the capture chain)` for a futile drop,
    /// `None` otherwise -- and returns `true`.  Otherwise pushes nothing and returns
    /// `false`: the node has a reply that is not an interposition drop, and F2 does
    /// not apply to it.  `buf` is used as scratch above `range.end` and restored.
    pub fn classify(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        out: &mut Vec<Option<u32>>,
    ) -> bool {
        let line = sliding_check_line(pos, attacker);
        self.classify_on(buf, attacker, pos, range, out, line)
    }

    /// [`Futility::classify`] with `line`, which must be
    /// [`sliding_check_line`]`(pos, attacker)`.
    pub fn classify_on(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
        range: std::ops::Range<usize>,
        out: &mut Vec<Option<u32>>,
        line: Option<CheckLine>,
    ) -> bool {
        let Some(line) = line else {
            return false;
        };
        if !buf.buf[range.clone()].iter().all(|&m| line.interposes(m)) {
            return false;
        }
        let mut probe = CaptureProbe::default();
        for index in range {
            let m = buf.buf[index];
            let verdict = self.futile_drop(buf, attacker, pos, line.checker, m, &mut probe);
            out.push(verdict);
        }
        true
    }

    /// Classical futility of the interposition drop `m` at defense node `pos`
    /// whose only checker stands on `checker`.  The checker captures the dropped
    /// piece (non-promoting form first, then promoting; a form must be legal and
    /// give check) and the drop is futile when the defense node after either form
    /// is futile in the sense of [`Futility::futile_defense`].  Returns the plies
    /// of the capture chain through the first passing form (`2 +` that node's
    /// distance), or `None`.
    ///
    /// Memoised like [`Futility::futile_defense`], keyed by the defence node and
    /// the drop.  `probe` shares the capture forms between drops onto the same
    /// square of one node.
    fn futile_drop(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
        checker: u8,
        m: Move,
        probe: &mut CaptureProbe,
    ) -> Option<u32> {
        let undo = pos.make(m);
        let forms = match probe.get(m.to) {
            Some(forms) => forms,
            None => {
                let forms = capture_forms(buf, attacker, pos, checker, m.to);
                probe.insert(m.to, forms);
                forms
            }
        };
        let mut verdict = None;
        for form in forms.iter().flatten() {
            let capture = pos.make(*form);
            let after = self.futile_defense(buf, attacker, pos);
            pos.unmake(*form, capture);
            if let Some(distance) = after {
                verdict = Some(distance + 2);
                break;
            }
        }
        pos.unmake(m, undo);
        verdict
    }

    /// Whether the defense node `pos` (defender to move, in check) is mate in the
    /// classical sense: no legal reply (`Some(0)`), or an interposable check whose
    /// every legal reply is an interposition drop that [`Futility::futile_drop`]
    /// proves futile (`Some(longest capture chain)`).  `None` otherwise.
    ///
    /// Memoised per position with the attacker's hand masked out: neither the
    /// defender's legal replies nor the checker's capture depend on what the
    /// attacker holds, so the verdict does not either.
    fn futile_defense(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
    ) -> Option<u32> {
        let hash = hash_without_hand(pos, attacker);
        let key = [hash as u64, (hash >> 64) as u64];
        if let Some(&known) = self.memo.get(&key) {
            return (known != u8::MAX).then_some(u32::from(known));
        }
        // A legal board move is a reply that is no interposition drop: the
        // verdict is `None` without the reply list.
        debug_assert_eq!(pos.side, opponent(attacker));
        let safety = Safety::new(pos);
        if buf.has_legal_board_move_with(pos, &safety) {
            self.remember(key, u8::MAX);
            return None;
        }
        self.futile_defense_drops(buf, attacker, pos, &safety, key)
    }

    /// [`Futility::futile_defense`] of a defense node in check that the caller
    /// knows has no legal board move (`safety` is its [`Safety`]): `Some(0)`
    /// when it has no legal reply at all, `Some(d)` when every reply is a
    /// futile interposition drop, `None` otherwise.  Memoised the same way
    /// (2026-10-27: the prober's bound-0 defense nodes ask exactly this).
    pub(crate) fn futile_defense_without_board_moves(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
        safety: &Safety,
    ) -> Option<u32> {
        let hash = hash_without_hand(pos, attacker);
        let key = [hash as u64, (hash >> 64) as u64];
        if let Some(&known) = self.memo.get(&key) {
            return (known != u8::MAX).then_some(u32::from(known));
        }
        self.futile_defense_drops(buf, attacker, pos, safety, key)
    }

    /// The rest of [`Futility::futile_defense`] once no legal board move is
    /// known: the drops, memoised under `key`.
    fn futile_defense_drops(
        &mut self,
        buf: &mut MoveBuf,
        attacker: u8,
        pos: &mut Position,
        safety: &Safety,
        key: [u64; 2],
    ) -> Option<u32> {
        let range = buf.gen_legal_with(pos, true, safety);
        let verdict = if range.is_empty() {
            Some(0)
        } else {
            match safety.check_line() {
                Some(line) if buf.buf[range.clone()].iter().all(|&m| line.interposes(m)) => {
                    let mut longest = Some(0u32);
                    let mut probe = CaptureProbe::default();
                    for index in range.clone() {
                        let m = buf.buf[index];
                        match self.futile_drop(buf, attacker, pos, line.checker, m, &mut probe) {
                            Some(distance) => longest = longest.map(|d| d.max(distance)),
                            None => {
                                longest = None;
                                break;
                            }
                        }
                    }
                    longest
                }
                _ => None,
            }
        };
        buf.buf.truncate(range.start);
        debug_assert!(verdict.is_none_or(|d| d < 255));
        self.remember(key, verdict.map_or(u8::MAX, |d| d.min(254) as u8));
        verdict
    }
}

/// The checker's capture of an interposition on `target`, as the non-promoting
/// (`[0]`) and promoting (`[1]`) forms, keeping only the ones the generator
/// offers *and* that give check -- exactly the moves the child attack node
/// recorded.
///
/// `pos` is that attack node: the position after the interposition drop.  `buf`
/// is used as scratch above its current end and restored.
fn capture_forms(
    buf: &mut MoveBuf,
    attacker: u8,
    pos: &mut Position,
    checker: u8,
    target: u8,
) -> [Option<Move>; 2] {
    debug_assert_eq!(pos.side, attacker);
    checks_from_to(buf, pos, checker, target)
}

/// The checker's capture forms per interposition square of one defence node.
///
/// Which forms are legal checks depends only on the square, never on which piece
/// was dropped there (see [`Solver::mark_useless_interpositions`]), and a check
/// line has at most seven squares strictly between checker and king.
#[derive(Default)]
struct CaptureProbe {
    entries: [(u8, [Option<Move>; 2]); 8],
    len: usize,
}

impl CaptureProbe {
    fn get(&self, square: u8) -> Option<[Option<Move>; 2]> {
        self.entries[..self.len]
            .iter()
            .find(|(probed, _)| *probed == square)
            .map(|(_, forms)| *forms)
    }
    fn insert(&mut self, square: u8, forms: [Option<Move>; 2]) {
        self.entries[self.len] = (square, forms);
        self.len += 1;
    }
}

/// `pos.hash()` with `color`'s hand treated as empty.
fn hash_without_hand(pos: &Position, color: u8) -> u128 {
    let keys = &crate::zobrist::KEYS.hand[color as usize];
    let hand = &pos.hands[color as usize];
    (0..7).fold(pos.hash(), |key, piece| {
        key ^ keys[piece][hand[piece] as usize] ^ keys[piece][0]
    })
}

/// `stats.nodesNonFutile` (owner decision 2026-09-25, `docs/rules.md` rule F):
/// the number of nodes of `graph` that a path from the root of **at most
/// `mate_ply` plies** reaches **without playing a reply flagged `useless`**.
///
/// Mirrors `nonFutileNodeCount` in `packages/solver/src/index.ts`, including the
/// breadth-first visit, so the two agree whenever the graphs agree.  Every edge
/// such a path uses costs one ply of the bound, so all such paths to a node have
/// the same length, and a mate in M that needed a bound P > M to prove its 無駄合
/// (under `Prune`) no longer pays for the deeper search.  With no flag, no credit
/// search and `stats.maxPly == mate_ply` the result is `stats.nodes`.
pub fn non_futile_node_count(graph: &Graph, mate_ply: u32) -> u64 {
    if let Some(facts) = graph.compact_facts() {
        // A compact table's sparse graph: its search counted it (`compact.rs`).
        let (ply, count) = facts
            .non_futile
            .expect("the compact search counted nodesNonFutile of its mate");
        assert_eq!(
            ply, mate_ply,
            "nodesNonFutile is counted at the mate distance"
        );
        return count;
    }
    if graph.node(graph.root).is_none() {
        return 0;
    }
    let mut depth: Vec<Option<u32>> = vec![None; graph.len()];
    depth[graph.root as usize] = Some(0);
    let mut queue: Vec<u32> = vec![graph.root];
    let mut head = 0usize;
    while head < queue.len() {
        let id = queue[head];
        head += 1;
        let d = depth[id as usize].unwrap_or(0);
        if d >= mate_ply {
            continue;
        }
        for m in graph.moves_of(id).iter() {
            let Some(next) = m.next() else {
                continue;
            };
            if m.useless() || depth[next as usize].is_some() {
                continue;
            }
            depth[next as usize] = Some(d + 1);
            queue.push(next);
        }
    }
    queue.len() as u64
}

/// A single sliding check that an interposition could block (rule F).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckLine {
    /// Square of the only checker.
    pub checker: u8,
    /// Bitboard of the empty squares strictly between the checker and the king.
    pub between: u128,
}

impl CheckLine {
    /// `true` when `m` is an *interposition drop*: a drop onto a square strictly
    /// between the checker and the king.  Board-move interpositions (移動合) are
    /// never classified useless in v1.
    #[inline]
    pub fn interposes(&self, m: Move) -> bool {
        m.is_drop() && self.between & (1u128 << m.to) != 0
    }
}

/// Rule F's *interposable* test, mirroring `slidingCheckLine` in
/// `packages/solver/src/index.ts`: the defender king is attacked by exactly one
/// checker, and that checker stands on a line at distance >= 2 -- which can only
/// be a sliding attack -- with every square strictly between it and the king
/// empty.
///
/// Every possible checker is either the first piece along one of the eight rays
/// from the king or a knight on one of the four knight squares, so the scan is
/// bounded.  `None` when no interposition drop at this node could ever be useless.
pub fn sliding_check_line(pos: &Position, attacker: u8) -> Option<CheckLine> {
    // One checker, not next to the king and not a knight: exactly the checks
    // that leave empty squares between (`Position::king_scan`, 2026-10-26).
    let scan = pos.king_scan(opponent(attacker), false)?;
    if scan.checkers != 1 || scan.between == 0 {
        return None;
    }
    Some(CheckLine {
        checker: scan.checker_squares.trailing_zeros() as u8,
        between: scan.between,
    })
}

/// Convenience wrapper for one-off searches.
pub fn solve(position: &Position, options: &SolveOptions) -> SolveResult {
    Solver::new().solve(position, options)
}

/// Convenience wrapper that also returns the graph, matching `solveTsume`.
pub fn solve_graph(position: &Position, max_ply: u32, node_budget: u64) -> SolveResult {
    solve(
        position,
        &SolveOptions::default()
            .max_ply(max_ply)
            .node_budget(node_budget),
    )
}

/// `true` when `m` is a legal checking move for the side to move.
pub fn is_checking_move(position: &Position, m: Move) -> bool {
    crate::movegen::gives_check(position, m)
}

#[cfg(test)]
mod oracle_table_tests {
    //! The credit oracle's table entries and the pure-mate dominance table
    //! (stage 2; Codex review of stage 2, items 1 and 6).
    use super::*;

    /// The legal move of `pos` whose USI is `usi` (test lookup only).
    fn legal(solver: &mut Solver, pos: &mut Position, usi: &str) -> Move {
        let range = solver.buf.gen_legal(pos, true);
        let found = solver.buf.buf[range.clone()]
            .iter()
            .copied()
            .find(|m| m.to_usi().to_string() == usi);
        solver.buf.buf.truncate(range.start);
        found.unwrap_or_else(|| panic!("{usi} is not legal"))
    }

    #[test]
    fn an_unknown_capture_form_then_a_mating_one_makes_the_drop_useless() {
        // D: the rook on 5e checks the king on 5a; the defender can drop its gold
        // on 5c, where the rook's capture has two forms (5e5c, then 5e5c+), or move
        // the king.  With the oracle's budget at zero every question it has not
        // answered before is UNKNOWN; the table is prepared so that E after the
        // promoting form is proven mate within 2.  The non-promoting form comes
        // back UNKNOWN, the next form is still read, and its mate makes the drop
        // useless (`unknown_then_mate`), with the oracle exhausted.
        let mut solver = Solver::new();
        solver.attacker = BLACK;
        solver.useless = UselessInterposition::Classical;
        solver.oracle.reset(0);
        let mut d = crate::sfen::parse_sfen("4k4/9/9/9/4R4/9/9/9/9 w g 1").unwrap();
        let line = sliding_check_line(&d, BLACK).expect("an interposable check");
        let drop = legal(&mut solver, &mut d, "G*5c");
        let undo = d.make(drop);
        let promote = legal(&mut solver, &mut d, "5e5c+");
        let capture = d.make(promote);
        let e2 = oracle_key(d.hash());
        d.unmake(promote, capture);
        d.unmake(drop, undo);
        let mut entry = OracleEntry::EMPTY;
        entry.record(Outcome::Mate, false, 2);
        solver.oracle.table.insert(e2, entry);
        let mut probe = CaptureProbe::default();
        let verdict = solver.credit_interposition(&mut d, drop, line.checker, 2, &mut probe);
        assert_eq!(verdict, Credit::Useless(2));
        assert_eq!(solver.credit.unknown_then_mate, 1);
        assert_eq!(solver.credit.calls, 2, "both forms were asked");
        assert!(solver.oracle.exhausted);
        // Without the prepared entry the drop is undecided, not effective.
        solver.oracle.reset(0);
        let verdict = solver.credit_interposition(&mut d, drop, line.checker, 2, &mut probe);
        assert_eq!(verdict, Credit::Undecided);
    }

    #[test]
    fn an_entry_never_answers_from_its_sentinels() {
        let mut entry = OracleEntry::EMPTY;
        entry.record(Outcome::Refuted, false, 0);
        // "No mate proven" is u8::MAX: no bound, however large, reads it as mate.
        for bound in [0u32, 1, 254, 255, 256, 10_000] {
            assert_ne!(
                entry.answer(bound).map(|a| a.0),
                Some(Outcome::Mate),
                "{bound}"
            );
        }
        assert_eq!(entry.answer(0), Some((Outcome::Refuted, false)));
        assert_eq!(entry.answer(1), None);
        entry.record(Outcome::Mate, false, TABLE_BOUND_MAX);
        assert_eq!(entry.answer(TABLE_BOUND_MAX - 1), None);
        assert_eq!(entry.answer(TABLE_BOUND_MAX), Some((Outcome::Mate, false)));
        assert_eq!(entry.answer(100_000), Some((Outcome::Mate, false)));
    }

    #[test]
    fn purity_is_kept_per_bound() {
        // An impure mate within 3, then a pure one within 5: mate from 3 on,
        // pure only from 5 on.
        let mut entry = OracleEntry::EMPTY;
        entry.record(Outcome::Mate, false, 3);
        entry.record(Outcome::Mate, true, 5);
        assert_eq!(entry.answer(2), None);
        assert_eq!(entry.answer(3), Some((Outcome::Mate, false)));
        assert_eq!(entry.answer(4), Some((Outcome::Mate, false)));
        assert_eq!(entry.answer(5), Some((Outcome::Mate, true)));
        // A later impure proof at a smaller bound does not make 5 impure.
        entry.record(Outcome::Mate, false, 1);
        assert_eq!(entry.answer(1), Some((Outcome::Mate, false)));
        assert_eq!(entry.answer(5), Some((Outcome::Mate, true)));
    }

    #[test]
    fn dominance_is_used_only_towards_a_larger_attacker_hand() {
        let mut entries = Vec::new();
        let base = [1u8, 0, 0, 0, 1, 0, 0];
        record_dominance(&mut entries, base, 5);
        assert!(dominated(&entries, base, 5));
        assert!(
            !dominated(&entries, base, 4),
            "a pure mate within 5 says nothing about 4"
        );
        assert!(dominated(&entries, [2, 0, 0, 0, 1, 0, 0], 7));
        assert!(
            !dominated(&entries, [0, 0, 0, 0, 1, 0, 0], 5),
            "less attacker hand"
        );
        assert!(
            !dominated(&entries, [2, 0, 0, 0, 0, 0, 0], 5),
            "not comparable"
        );
        // The same hand keeps the smaller bound.
        record_dominance(&mut entries, base, 3);
        record_dominance(&mut entries, base, 7);
        assert_eq!(entries, vec![(base, 3)]);
    }

    #[test]
    fn the_dominance_list_drops_its_oldest_hand() {
        let mut entries = Vec::new();
        let hand = |i: usize| [i as u8, 0, 0, 0, 0, 0, 1];
        for i in 0..DOMINANCE_ENTRIES {
            record_dominance(&mut entries, hand(i), 9);
        }
        assert_eq!(entries.len(), DOMINANCE_ENTRIES);
        assert!(dominated(&entries, hand(0), 9));
        // The 33rd hand drops the first.  hand(0) is then dominated by nothing
        // left (every other hand has a pawn).
        record_dominance(&mut entries, [0, 0, 0, 0, 0, 1, 0], 9);
        assert_eq!(entries.len(), DOMINANCE_ENTRIES);
        assert_eq!(entries[0].0, hand(1));
        assert!(!dominated(&entries, hand(0), 9));
        assert!(dominated(&entries, hand(1), 9));
    }
}
