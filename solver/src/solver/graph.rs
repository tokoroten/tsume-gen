//! The exported search graph, in compact form (owner 2026-10-07).
//!
//! The graph is published (`--graph full|pruned`, `stats.nodesNonFutile`, the
//! pipeline's main-line rules), so its *contents* are fixed: every node, every
//! edge, every field, in creation order.  (Under rule F `classical` a horizon
//! node other than the root records only its escape, `docs/rules.md` rule F
//! "The horizon escape"; [`OmittedReplies`] keeps the feature counts of the
//! rest.)  Only the in-memory representation is compact:
//!
//! * a [`GraphNode`] (12 bytes) keeps no position.  It records the node that
//!   created it and the move that led there, so [`Graph::position`] rebuilds a
//!   position by replaying at most `max_ply` moves from the root, and
//!   [`Graph::for_each_position`] visits every position in id order with about
//!   one move per node.  Its id is its index, outcome and side are bits, and
//!   the distance is 13 bits (a byte plus the five high bits of its flags
//!   byte, exact up to 8,190);
//! * every node's moves live in **one** slab, in node-id order: a node stores
//!   where its moves start, and they end where the next node's start.  The
//!   sequential search reserves a node's moves the moment it has generated
//!   them, before it creates any child, so the starts are in id order;
//! * a [`GraphMove`] (10 bytes) stores its move and its refutation as 16-bit
//!   move codes ([`encode_move`]), its child as a 32-bit id and its distance as
//!   13 bits: one byte plus the five high bits of the flags byte, whose low
//!   three bits hold the outcome and the 無駄合 flag.
//!
//! Accessors return the published values (`Option<u32>`, [`Usi`], [`Outcome`]),
//! so readers never see the encoding.

use super::Outcome;
use crate::piece::{ptype, EMPTY, PT_K, PT_P};
use crate::position::{Move, Position, Usi};
use crate::sfen::to_sfen;

/// "No move" as a 16-bit move code (a real code never sets bit 15).
pub(crate) const NO_MOVE: u16 = u16::MAX;
/// "No node" as a 32-bit node id.
pub(crate) const NO_NODE: u32 = u32::MAX;
/// A distance is 13 bits: the low 8 in the `distance` byte, the high 5 in
/// bits 3-7 of the `flags` byte (bits 0-2 hold the outcome and the useless /
/// attack flag).  All ones is "no distance".  Every distance up to
/// [`MAX_GRAPH_DISTANCE`] is exact, and none is ever rounded: a larger one
/// panics, and [`super::MAX_PLY`] keeps every distance a search records below
/// it (Codex review of stage2c, 2026-10-08).
const NO_DISTANCE: u16 = 0x1FFF;
/// The largest distance a graph stores.
pub(crate) const MAX_GRAPH_DISTANCE: u32 = 0x1FFE;
/// Where the high distance bits sit in `flags`.
const DISTANCE_HIGH_SHIFT: u32 = 3;
/// The `flags` bits that are not distance bits.
const LOW_FLAGS: u8 = (1 << DISTANCE_HIGH_SHIFT) - 1;
/// The distance byte and the `flags` high bits of "no distance".
const NO_DISTANCE_LOW: u8 = NO_DISTANCE as u8;
const NO_DISTANCE_HIGH: u8 = ((NO_DISTANCE >> 8) as u8) << DISTANCE_HIGH_SHIFT;
/// [`GraphMove::flags`] and [`GraphNode::flags`]: bits 0-1 hold the outcome.
const OUTCOME_MASK: u8 = 0b11;
/// [`GraphMove::flags`]: the reply is a 無駄合.
const USELESS: u8 = 1 << 2;
/// [`GraphNode::flags`]: an attack node.
const ATTACK: u8 = 1 << 2;

/// The 16-bit code of a move: origin in bits 0-6 (a square `0..81`, or `81 +`
/// the piece type for a drop), destination in bits 7-13, promotion in bit 14.
#[inline]
pub(crate) fn encode_move(m: Move) -> u16 {
    let from = if m.is_drop() {
        81 + u16::from(m.piece)
    } else {
        u16::from(m.from)
    };
    debug_assert!(from < 128 && m.to < 81, "a move fits its code");
    from | (u16::from(m.to) << 7) | (u16::from(m.promote) << 14)
}

/// The move of a code.  A board move comes back with `piece = PT_P`, exactly
/// like [`Move::from_usi`]: [`Position::make`] reads the mover from the board.
#[inline]
pub(crate) fn decode_move(code: u16) -> Move {
    debug_assert_ne!(code, NO_MOVE);
    let from = (code & 0x7F) as u8;
    let to = ((code >> 7) & 0x7F) as u8;
    if from >= 81 {
        Move::drop(from - 81, to)
    } else {
        Move::board(from, to, PT_P, code & (1 << 14) != 0)
    }
}

#[inline]
const fn outcome_bits(outcome: Outcome) -> u8 {
    match outcome {
        Outcome::Mate => 0,
        Outcome::Refuted => 1,
        Outcome::Unknown => 2,
    }
}

#[inline]
const fn outcome_of(flags: u8) -> Outcome {
    match flags & OUTCOME_MASK {
        0 => Outcome::Mate,
        1 => Outcome::Refuted,
        _ => Outcome::Unknown,
    }
}

/// The distance byte and the high bits for `flags` (already shifted).
#[inline]
fn distance_parts(distance: Option<u32>) -> (u8, u8) {
    let value = match distance {
        Some(d) => {
            assert!(
                d <= MAX_GRAPH_DISTANCE,
                "a graph distance is at most {MAX_GRAPH_DISTANCE}"
            );
            d as u16
        }
        None => NO_DISTANCE,
    };
    (value as u8, ((value >> 8) as u8) << DISTANCE_HIGH_SHIFT)
}

#[inline]
fn distance_of(byte: u8, flags: u8) -> Option<u32> {
    let value = u16::from(byte) | (u16::from(flags >> DISTANCE_HIGH_SHIFT) << 8);
    (value != NO_DISTANCE).then_some(u32::from(value))
}

/// One edge of the solution graph: 10 bytes.
///
/// `refutation` is the defender's escape a refuted edge records; `useless` marks
/// a 無駄合 (`docs/rules.md` rule F).  Only ever set under
/// [`super::UselessInterposition::Prune`] and
/// [`super::UselessInterposition::Classical`], such a reply is excluded from
/// its node's distance and from the designated main line, yet it stays in the
/// graph, so defense-coverage checks still see every legal answer.  Under
/// `Prune` it keeps its `next`.  A drop `Classical` proved futile was never
/// searched by the main tree, so it has `next: None`, `outcome: MATE` and
/// `distance` = the plies of its capture chain (2 + the distance of the defense
/// node after the capture).  A drop `Classical` proved useless by the credit
/// search while its child was REFUTED at the bound has `outcome: MATE`,
/// `distance` = 2 + the distance of the defense node after the capture, and its
/// searched `next`, whose own outcome is REFUTED.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GraphMove {
    mv: u16,
    refutation: u16,
    /// The child id, little half first ([`NO_NODE`] for none); two halves keep
    /// the struct 2-byte aligned, so it packs into 10 bytes.
    next: [u16; 2],
    distance: u8,
    flags: u8,
}

impl GraphMove {
    /// A placeholder for a reserved slot, overwritten before anyone reads it.
    pub(crate) const EMPTY: GraphMove = GraphMove {
        mv: NO_MOVE,
        refutation: NO_MOVE,
        next: [u16::MAX; 2],
        distance: NO_DISTANCE_LOW,
        flags: outcome_bits(Outcome::Unknown) | NO_DISTANCE_HIGH,
    };

    pub fn new(
        m: Move,
        outcome: Outcome,
        distance: Option<u32>,
        next: Option<u32>,
        refutation: Option<Move>,
        useless: bool,
    ) -> Self {
        let mut gm = GraphMove {
            mv: encode_move(m),
            ..GraphMove::EMPTY
        };
        gm.set_outcome(outcome);
        gm.set_distance(distance);
        gm.set_next(next);
        gm.set_refutation(refutation);
        gm.set_useless(useless);
        gm
    }

    /// The move's USI text.
    #[inline]
    pub fn usi(&self) -> Usi {
        decode_move(self.mv).to_usi()
    }
    /// The move's 16-bit code ([`encode_move`]).
    #[inline]
    pub(crate) fn code(&self) -> u16 {
        self.mv
    }
    /// The move (`piece = PT_P` for a board move, like [`Move::from_usi`]).
    #[inline]
    pub fn mv(&self) -> Move {
        decode_move(self.mv)
    }
    #[inline]
    pub fn outcome(&self) -> Outcome {
        outcome_of(self.flags)
    }
    #[inline]
    pub fn distance(&self) -> Option<u32> {
        distance_of(self.distance, self.flags)
    }
    #[inline]
    pub fn next(&self) -> Option<u32> {
        let id = u32::from(self.next[0]) | (u32::from(self.next[1]) << 16);
        (id != NO_NODE).then_some(id)
    }
    #[inline]
    pub fn refutation(&self) -> Option<Usi> {
        self.refutation_move().map(|m| m.to_usi())
    }
    #[inline]
    pub fn refutation_move(&self) -> Option<Move> {
        (self.refutation != NO_MOVE).then(|| decode_move(self.refutation))
    }
    #[inline]
    pub fn useless(&self) -> bool {
        self.flags & USELESS != 0
    }

    #[inline]
    pub fn set_outcome(&mut self, outcome: Outcome) {
        self.flags = (self.flags & !OUTCOME_MASK) | outcome_bits(outcome);
    }
    #[inline]
    pub fn set_distance(&mut self, distance: Option<u32>) {
        let (low, high) = distance_parts(distance);
        self.distance = low;
        self.flags = (self.flags & LOW_FLAGS) | high;
    }
    #[inline]
    pub fn set_next(&mut self, next: Option<u32>) {
        let id = match next {
            Some(id) => {
                assert_ne!(id, NO_NODE, "node ids stay below u32::MAX");
                id
            }
            None => NO_NODE,
        };
        self.next = [id as u16, (id >> 16) as u16];
    }
    #[inline]
    pub fn set_refutation(&mut self, refutation: Option<Move>) {
        self.refutation = refutation.map_or(NO_MOVE, encode_move);
    }
    #[inline]
    pub(crate) fn set_refutation_code(&mut self, code: u16) {
        self.refutation = code;
    }
    #[inline]
    pub fn set_useless(&mut self, useless: bool) {
        if useless {
            self.flags |= USELESS;
        } else {
            self.flags &= !USELESS;
        }
    }
    /// `self` with `useless` set, for building graphs by hand.
    pub fn with_useless(mut self, useless: bool) -> Self {
        self.set_useless(useless);
        self
    }
    /// `self` with `next` set, for building graphs by hand.
    pub fn with_next(mut self, next: Option<u32>) -> Self {
        self.set_next(next);
        self
    }
}

impl std::fmt::Debug for GraphMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.mv == NO_MOVE {
            return f.write_str("GraphMove(EMPTY)");
        }
        f.debug_struct("GraphMove")
            .field("usi", &self.usi())
            .field("outcome", &self.outcome())
            .field("distance", &self.distance())
            .field("next", &self.next())
            .field("refutation", &self.refutation())
            .field("useless", &self.useless())
            .finish()
    }
}

/// One node of the solution graph: 12 bytes.  Its id is its index in the
/// graph; read it through [`Graph::node`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GraphNode {
    /// The node that created this one ([`NO_NODE`] for the root): the first
    /// node to reach it, i.e. the node on top of the search stack.
    parent: u32,
    /// The first of this node's moves in the slab.
    start: u32,
    /// The move from `parent` to here ([`NO_MOVE`] for the root).
    mv: u16,
    distance: u8,
    flags: u8,
}

impl GraphNode {
    /// A node created by `parent` through the move code `mv`, whose moves start
    /// at `start`; outcome `UNKNOWN` until [`GraphNode::set_result`].
    #[inline]
    pub(crate) fn new(parent: u32, mv: u16, start: usize, attack: bool) -> Self {
        GraphNode {
            parent,
            start: slab_index(start),
            mv,
            distance: NO_DISTANCE_LOW,
            flags: outcome_bits(Outcome::Unknown)
                | if attack { ATTACK } else { 0 }
                | NO_DISTANCE_HIGH,
        }
    }
    #[inline]
    pub(crate) fn set_result(&mut self, outcome: Outcome, distance: Option<u32>) {
        let (low, high) = distance_parts(distance);
        self.flags = (self.flags & LOW_FLAGS & !OUTCOME_MASK) | outcome_bits(outcome) | high;
        self.distance = low;
    }
    #[inline]
    fn distance(&self) -> Option<u32> {
        distance_of(self.distance, self.flags)
    }
    #[inline]
    pub(crate) fn start(&self) -> usize {
        self.start as usize
    }
    #[inline]
    pub(crate) fn set_start(&mut self, start: usize) {
        self.start = slab_index(start);
    }
    #[inline]
    fn attack(&self) -> bool {
        self.flags & ATTACK != 0
    }
}

#[inline]
fn slab_index(index: usize) -> u32 {
    u32::try_from(index).expect("the move slab stays below 2^32 moves")
}

/// The slab range of node `id`'s moves: from its start to the next node's.
#[inline]
pub(crate) fn moves_range(nodes: &[GraphNode], slab_len: usize, id: u32) -> std::ops::Range<usize> {
    let index = id as usize;
    let end = nodes.get(index + 1).map_or(slab_len, GraphNode::start);
    nodes[index].start()..end
}

/// A hand-built node for [`Graph::from_specs`].
#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub attack: bool,
    pub position: Position,
    pub outcome: Outcome,
    pub distance: Option<u32>,
    pub moves: Vec<GraphMove>,
}

/// The replies a **horizon escape** (`docs/rules.md` rule F, stage 2) did not
/// record, counted the way the graph-wide difficulty features count a recorded
/// reply, so those features keep their values.  Every reply it counts is a
/// defender reply at a REFUTED defense node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OmittedReplies {
    pub drops: u64,
    /// Drops that are not 無駄合.
    pub interpositions: u64,
    /// Replies flagged useless (F2-futile drops).
    pub useless_interpositions: u64,
    pub promotions: u64,
    pub king_moves: u64,
}

impl OmittedReplies {
    /// Counts the reply `m` at `position` (the defense node's position).
    pub(crate) fn add_reply(&mut self, position: &Position, m: Move, useless: bool) {
        if m.is_drop() {
            self.drops += 1;
            if !useless {
                self.interpositions += 1;
            }
        } else {
            let c = position.board[m.from as usize];
            if c != EMPTY && ptype(c) == PT_K {
                self.king_moves += 1;
            }
        }
        if useless {
            self.useless_interpositions += 1;
        }
        if m.promote {
            self.promotions += 1;
        }
    }
}

/// The full search graph.  Node `i` has id `i`.
///
/// A graph of the compact table (`--table compact`, [`super::compact`]) is
/// **sparse**: it holds only the nodes the solution-region readers visit (the
/// main line, rules B, C and G, the runtime graph), each under its search id,
/// with an explicit position, and carries the whole-graph numbers
/// ([`CompactFacts`]) its search counted instead of the other nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    pub root: u32,
    /// What the horizon escapes left out of the graph (feature counts only).
    pub omitted: OmittedReplies,
    root_position: Position,
    nodes: Vec<GraphNode>,
    moves: Vec<GraphMove>,
    /// Only for graphs built by hand ([`Graph::from_specs`]), whose positions
    /// need not be connected by their moves, and for sparse graphs.
    explicit: Option<Vec<Position>>,
    /// A compact table's sparse graph: the search id of each stored node.
    sparse: Option<Box<Sparse>>,
}

/// The node ids of a sparse graph and its whole-graph numbers.
#[derive(Clone, Debug, Default, PartialEq)]
struct Sparse {
    /// Search id of the node at each index, ascending.
    ids: Vec<u32>,
    facts: CompactFacts,
}

/// What a compact-table search ([`super::compact`]) counted over its whole
/// graph, which a sparse [`Graph`] does not hold: the readers of whole-graph
/// numbers (`features::extract_with`, [`super::non_futile_node_count`],
/// `exact::complete_defense_coverage`) read these instead.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactFacts {
    /// The difficulty-feature counts of every node.
    pub(crate) counts: crate::features::NodeCounts,
    /// `reachable_node_count` of the child of each root move, by root-move
    /// index; `None` where it was not counted (a correct root move).
    pub(crate) reach: Vec<Option<u64>>,
    /// `(mate_ply, non_futile_node_count(graph, mate_ply))`, when counted.
    pub(crate) non_futile: Option<(u32, u64)>,
    /// `exact::graph_has_unknown` of the whole graph.
    pub(crate) has_unknown: bool,
}

impl CompactFacts {
    /// `reachable_node_count` of root move `index`'s child.
    ///
    /// # Panics
    /// When the search did not count it (only a resolved `MATE` root counts
    /// its wrong moves, the only ones `features::extract_with` reads).
    pub(crate) fn reach(&self, index: usize) -> usize {
        self.reach
            .get(index)
            .copied()
            .flatten()
            .expect("the compact search counted this root move's reach") as usize
    }
}

impl Graph {
    /// A searched graph: `root_position` is node `root`'s position, every
    /// other node replays from it.
    pub(crate) fn from_parts(
        root: u32,
        root_position: Position,
        nodes: Vec<GraphNode>,
        moves: Vec<GraphMove>,
    ) -> Self {
        Graph {
            root,
            omitted: OmittedReplies::default(),
            root_position,
            nodes,
            moves,
            explicit: None,
            sparse: None,
        }
    }

    /// A compact table's sparse graph: `entries` are `(search id, node, its
    /// moves, its position)` in ascending id order, the root (id 0) first.
    pub(crate) fn from_sparse(
        root_position: Position,
        entries: Vec<(u32, GraphNode, Vec<GraphMove>, Position)>,
        facts: CompactFacts,
    ) -> Self {
        let mut ids = Vec::with_capacity(entries.len());
        let mut nodes = Vec::with_capacity(entries.len());
        let mut moves = Vec::new();
        let mut positions = Vec::with_capacity(entries.len());
        for (id, mut node, node_moves, position) in entries {
            debug_assert!(ids.last().is_none_or(|&last| last < id), "ascending ids");
            node.set_start(moves.len());
            ids.push(id);
            nodes.push(node);
            moves.extend_from_slice(&node_moves);
            positions.push(position);
        }
        debug_assert_eq!(ids.first(), Some(&0), "the root is stored");
        Graph {
            root: 0,
            omitted: OmittedReplies::default(),
            root_position,
            nodes,
            moves,
            explicit: Some(positions),
            sparse: Some(Box::new(Sparse { ids, facts })),
        }
    }

    /// The whole-graph numbers of a compact table's sparse graph, `None` for
    /// every other graph.
    pub fn compact_facts(&self) -> Option<&CompactFacts> {
        self.sparse.as_ref().map(|sparse| &sparse.facts)
    }

    /// `true` for a compact table's sparse graph, which holds only the
    /// solution region (see [`Graph`]).
    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    /// The storage index of node `id`, `None` when the graph does not hold it.
    #[inline]
    fn index(&self, id: u32) -> Option<usize> {
        match &self.sparse {
            None => ((id as usize) < self.nodes.len()).then_some(id as usize),
            Some(sparse) => sparse.ids.binary_search(&id).ok(),
        }
    }

    /// The search id of the node stored at `index`.
    #[inline]
    fn id_at(&self, index: usize) -> u32 {
        match &self.sparse {
            None => index as u32,
            Some(sparse) => sparse.ids[index],
        }
    }

    /// A graph built by hand (tests): node `i` is `specs[i]`, with its own
    /// position, which need not follow from its parent's by a move.
    pub fn from_specs(root: u32, specs: Vec<NodeSpec>) -> Self {
        let mut nodes = Vec::with_capacity(specs.len());
        let mut moves = Vec::new();
        let mut positions = Vec::with_capacity(specs.len());
        for spec in specs {
            let mut node = GraphNode::new(NO_NODE, NO_MOVE, moves.len(), spec.attack);
            node.set_result(spec.outcome, spec.distance);
            nodes.push(node);
            moves.extend_from_slice(&spec.moves);
            positions.push(spec.position);
        }
        Graph {
            root,
            omitted: OmittedReplies::default(),
            root_position: positions.get(root as usize).cloned().unwrap_or_default(),
            nodes,
            moves,
            explicit: Some(positions),
            sparse: None,
        }
    }

    /// Number of nodes (of a sparse graph: the nodes it holds).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// Number of edges, over all nodes.
    pub fn move_count(&self) -> usize {
        self.moves.len()
    }

    pub fn node(&self, id: u32) -> Option<Node<'_>> {
        self.index(id).map(|_| Node { graph: self, id })
    }

    /// Every node, in id order.
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = Node<'_>> + '_ {
        (0..self.nodes.len()).map(move |index| Node {
            graph: self,
            id: self.id_at(index),
        })
    }

    /// Node `id`'s moves.
    pub fn moves_of(&self, id: u32) -> &[GraphMove] {
        let index = self.index(id).expect("a node of this graph");
        &self.moves[moves_range(&self.nodes, self.moves.len(), index as u32)]
    }

    /// The root node's position.
    pub fn root_position(&self) -> &Position {
        &self.root_position
    }

    /// Node `id`'s position, replayed from the root along the creating moves.
    pub fn position(&self, id: u32) -> Position {
        if let Some(explicit) = &self.explicit {
            return explicit[self.index(id).expect("a node of this graph")].clone();
        }
        let mut path: Vec<u16> = Vec::new();
        let mut current = id;
        loop {
            let node = &self.nodes[current as usize];
            if node.parent == NO_NODE {
                break;
            }
            path.push(node.mv);
            current = node.parent;
        }
        debug_assert_eq!(current, self.root, "every node descends from the root");
        let mut position = self.root_position.clone();
        for &code in path.iter().rev() {
            position.make(decode_move(code));
        }
        position
    }

    /// Calls `f` on every node with its position, in id order.
    ///
    /// A searched graph creates nodes depth-first, so a node's parent is the
    /// previous node or one of its ancestors: one stack of positions along the
    /// current path serves every node with one move each.
    pub fn for_each_position(&self, mut f: impl FnMut(Node<'_>, &Position)) {
        if let Some(explicit) = &self.explicit {
            for (node, position) in self.nodes().zip(explicit.iter()) {
                f(node, position);
            }
            return;
        }
        let mut stack: Vec<(u32, Position)> = Vec::new();
        for id in 0..self.nodes.len() as u32 {
            let raw = self.nodes[id as usize];
            if raw.parent == NO_NODE {
                stack.clear();
                stack.push((id, self.root_position.clone()));
            } else {
                while stack.last().is_some_and(|(top, _)| *top != raw.parent) {
                    stack.pop();
                }
                let position = match stack.last() {
                    Some((_, parent)) => {
                        let mut position = parent.clone();
                        position.make(decode_move(raw.mv));
                        position
                    }
                    // Not on the current path (never for a searched graph).
                    None => self.position(id),
                };
                stack.push((id, position));
            }
            let (_, position) = stack.last().expect("pushed above");
            f(Node { graph: self, id }, position);
        }
    }
}

/// A node of a [`Graph`], read through its accessors.
#[derive(Clone, Copy)]
pub struct Node<'g> {
    graph: &'g Graph,
    id: u32,
}

impl<'g> Node<'g> {
    #[inline]
    fn raw(self) -> &'g GraphNode {
        let index = self.graph.index(self.id).expect("a node of this graph");
        &self.graph.nodes[index]
    }
    #[inline]
    pub fn id(self) -> u32 {
        self.id
    }
    /// `true` for an attack node (`side: "attack"`).
    #[inline]
    pub fn attack(self) -> bool {
        self.raw().attack()
    }
    #[inline]
    pub fn outcome(self) -> Outcome {
        outcome_of(self.raw().flags)
    }
    #[inline]
    pub fn distance(self) -> Option<u32> {
        self.raw().distance()
    }
    #[inline]
    pub fn moves(self) -> &'g [GraphMove] {
        self.graph.moves_of(self.id)
    }
    /// The node that created this one, `None` for the root.
    #[inline]
    pub fn parent(self) -> Option<u32> {
        let parent = self.raw().parent;
        (parent != NO_NODE).then_some(parent)
    }
    /// The node's position, replayed from the root ([`Graph::position`]).
    pub fn position(self) -> Position {
        self.graph.position(self.id)
    }
    pub fn sfen(self) -> String {
        to_sfen(&self.position())
    }
    pub fn side_label(self) -> &'static str {
        if self.attack() {
            "attack"
        } else {
            "defense"
        }
    }
}

impl std::fmt::Debug for Node<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("attack", &self.attack())
            .field("outcome", &self.outcome())
            .field("distance", &self.distance())
            .field("moves", &self.moves())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(std::mem::size_of::<GraphMove>(), 10);
        assert_eq!(std::mem::size_of::<GraphNode>(), 12);
    }

    #[test]
    fn move_codes_round_trip() {
        for from in 0..81u8 {
            for to in 0..81u8 {
                for promote in [false, true] {
                    let m = Move::board(from, to, PT_P, promote);
                    assert_eq!(decode_move(encode_move(m)), m);
                }
            }
        }
        for piece in 0..7u8 {
            for to in 0..81u8 {
                let m = Move::drop(piece, to);
                assert_eq!(decode_move(encode_move(m)), m);
                assert_ne!(encode_move(m), NO_MOVE);
            }
        }
    }

    #[test]
    fn edge_fields_round_trip() {
        let m = Move::board(10, 20, PT_P, true);
        let r = Move::drop(3, 40);
        let gm = GraphMove::new(
            m,
            Outcome::Refuted,
            Some(7),
            Some(0x1234_5678),
            Some(r),
            true,
        );
        assert_eq!(gm.mv(), m);
        assert_eq!(gm.outcome(), Outcome::Refuted);
        assert_eq!(gm.distance(), Some(7));
        assert_eq!(gm.next(), Some(0x1234_5678));
        assert_eq!(gm.refutation_move(), Some(r));
        assert!(gm.useless());
        let plain = GraphMove::new(m, Outcome::Unknown, None, None, None, false);
        assert_eq!(plain.distance(), None);
        assert_eq!(plain.next(), None);
        assert_eq!(plain.refutation(), None);
        assert!(!plain.useless());
    }
}

#[cfg(test)]
mod distance_tests {
    use super::*;

    #[test]
    fn distances_are_exact_up_to_the_largest_one_and_keep_the_flags() {
        let m = Move::board(10, 1, PT_P, true);
        for distance in [
            None,
            Some(0),
            Some(14),
            Some(254),
            Some(255),
            Some(256),
            Some(4097),
            Some(MAX_GRAPH_DISTANCE),
        ] {
            for useless in [false, true] {
                for outcome in [Outcome::Mate, Outcome::Refuted, Outcome::Unknown] {
                    let mut gm = GraphMove::new(m, outcome, distance, Some(7), None, useless);
                    assert_eq!(gm.distance(), distance);
                    assert_eq!((gm.outcome(), gm.useless()), (outcome, useless));
                    gm.set_outcome(Outcome::Refuted);
                    gm.set_useless(!useless);
                    assert_eq!(gm.distance(), distance);
                    gm.set_distance(Some(3));
                    assert_eq!((gm.distance(), gm.useless()), (Some(3), !useless));
                }
            }
            for attack in [false, true] {
                let mut node = GraphNode::new(NO_NODE, NO_MOVE, 0, attack);
                assert_eq!(node.distance(), None);
                node.set_result(Outcome::Mate, distance);
                assert_eq!((node.distance(), node.attack()), (distance, attack));
                assert_eq!(outcome_of(node.flags), Outcome::Mate);
                node.set_result(Outcome::Refuted, None);
                assert_eq!((node.distance(), node.attack()), (None, attack));
            }
        }
        assert_eq!(GraphMove::EMPTY.distance(), None);
    }

    #[test]
    #[should_panic(expected = "a graph distance is at most")]
    fn a_larger_distance_is_never_rounded() {
        GraphMove::new(
            Move::drop(PT_P, 4),
            Outcome::Mate,
            Some(MAX_GRAPH_DISTANCE + 1),
            None,
            None,
            false,
        );
    }
}
