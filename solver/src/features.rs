//! Difficulty feature extraction (spec 14.2).
//!
//! Port of `extractDifficultyFeatures`.  Every field is computed from the
//! complete search graph, before pruning, exactly like the TypeScript stage.

use crate::exact::reachable_node_count;
use crate::jsonl::DifficultyFeatures;
use crate::piece::*;
use crate::position::Position;
use crate::solver::{GraphMove, Node, OmittedReplies, Outcome, SolveResult};

/// Extracts every published difficulty feature.
///
/// `position` is the record position (the value behind `normalizedSfen`), not the
/// strict-material normalisation the search ran on, matching the TypeScript call.
///
/// # Panics
/// Panics when the result carries no graph; features need the full graph.
pub fn extract(position: &Position, result: &SolveResult) -> DifficultyFeatures {
    extract_with_inert(position, result, None)
}

/// [`extract`] given rule G's piece count when the gate already analysed
/// the same graph for the same attacker (`inert`, the length of
/// [`crate::exact::ExactValidation::inert_pieces`]): the walk is a pure
/// function of the graph and the attacker, so the count is the one this would
/// recompute (`None` recomputes it).
pub fn extract_with_inert(
    position: &Position,
    result: &SolveResult,
    inert: Option<u64>,
) -> DifficultyFeatures {
    let graph = result
        .graph
        .as_ref()
        .expect("difficulty features require a search graph");
    // A compact-table graph (`--table compact`) holds only the solution region;
    // its search counted every node while it ran (`solver/compact.rs`).
    let compact = graph.compact_facts();
    // The graph keeps no positions: the walk replays them in id order.
    let counts = if let Some(facts) = compact {
        facts.counts
    } else {
        let mut counts = NodeCounts::default();
        graph.for_each_position(|node, position| {
            counts = counts.merge(NodeCounts::of(node, position));
        });
        counts
    };
    // The replies the horizon escapes did not record still count.
    let counts = counts.merge(NodeCounts::omitted(&graph.omitted));
    let root = graph.node(graph.root);
    let root_moves = root.map(|node| node.moves()).unwrap_or(&[]);
    let wrong_moves: Vec<_> = root_moves
        .iter()
        .filter(|m| !(m.outcome() == Outcome::Mate && m.distance() == result.distance))
        .collect();
    let defender_king = position
        .board
        .iter()
        .position(|&c| c != EMPTY && color_of(c) != position.side && ptype(c) == PT_K);
    let (king_file, king_rank) = match defender_king {
        Some(index) => (file_of(index) as i32, rank_of(index) as i32),
        None => (4, 4),
    };
    let visual_density_near_king = position
        .board
        .iter()
        .enumerate()
        .filter(|&(index, &c)| {
            c != EMPTY
                && (file_of(index) as i32 - king_file)
                    .abs()
                    .max((rank_of(index) as i32 - king_rank).abs())
                    <= 2
        })
        .count();
    let branch_count = counts.defense_branches;
    // One graph walk per wrong root move, summed below in root-move order.
    let wrong_reach: Vec<usize> = if let Some(facts) = compact {
        root_moves
            .iter()
            .enumerate()
            .filter(|(_, m)| !(m.outcome() == Outcome::Mate && m.distance() == result.distance))
            .map(|(index, _)| facts.reach(index))
            .collect()
    } else {
        wrong_moves
            .iter()
            .map(|m| reachable_node_count(graph, m.next()))
            .collect()
    };
    DifficultyFeatures {
        mate_ply: result.distance.unwrap_or(0),
        total_search_nodes: result.stats.nodes,
        solution_tree_node_count: counts.mate_nodes as u64,
        solution_tree_depth: result.distance.unwrap_or(0),
        defender_branch_count: branch_count as u64,
        max_defender_branching: counts.max_defense_branching as u64,
        mean_defender_branching: if counts.mate_defense_nodes == 0 {
            0.0
        } else {
            branch_count as f64 / counts.mate_defense_nodes as f64
        },
        legal_checking_moves_at_root: root_moves.len() as u64,
        plausible_wrong_checks: wrong_moves
            .iter()
            .filter(|m| m.outcome() != Outcome::Unknown)
            .count() as u64,
        false_move_difficulty: wrong_reach
            .iter()
            .map(|&count| (count as f64).ln_1p())
            .sum(),
        number_of_interpositions: counts.interpositions as u64,
        number_of_useless_interpositions: counts.useless_interpositions as u64,
        // Rule G: recomputed from the same complete search graph the reasons were
        // taken from, so the published count and the verdict cannot disagree.
        inert_piece_count: if result.outcome == Outcome::Mate && result.distance.is_some() {
            inert.unwrap_or_else(|| {
                crate::inert::analyze_inert_pieces(
                    graph,
                    position.side,
                    crate::inert::DEFAULT_RULE_G_BUDGET,
                )
                .count() as u64
            })
        } else {
            0
        },
        number_of_drops: counts.drops as u64,
        number_of_sacrifices: 0,
        number_of_promotions: counts.promotions as u64,
        king_move_count: counts.king_moves as u64,
        attacker_piece_count: position
            .board
            .iter()
            .filter(|&&c| c != EMPTY && color_of(c) == position.side)
            .count() as u64,
        board_piece_count: position.board_piece_count() as u64,
        visual_density_near_king: visual_density_near_king as u64,
        edge_distance: king_file
            .min(8 - king_file)
            .min(king_rank)
            .min(8 - king_rank) as u64,
        transposition_count: result.stats.cache_hits,
    }
}

/// The per-node counts behind the graph-wide features.  The compact table
/// (`solver/compact.rs`) sums them while it searches, since it keeps no graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeCounts {
    /// Proven-mate defence nodes, and the replies they list.
    mate_defense_nodes: usize,
    defense_branches: usize,
    max_defense_branching: usize,
    mate_nodes: usize,
    drops: usize,
    interpositions: usize,
    useless_interpositions: usize,
    promotions: usize,
    king_moves: usize,
}

impl NodeCounts {
    /// The counts of `node`, whose position is `position`.
    fn of(node: Node<'_>, position: &Position) -> Self {
        Self::of_parts(node.outcome(), node.attack(), node.moves(), position)
    }

    /// [`NodeCounts::of`] from a node's outcome, side and recorded moves.
    pub(crate) fn of_parts(
        outcome: Outcome,
        attack: bool,
        moves: &[GraphMove],
        position: &Position,
    ) -> Self {
        let mut counts = NodeCounts::default();
        if outcome == Outcome::Mate {
            counts.mate_nodes = 1;
            if !attack {
                counts.mate_defense_nodes = 1;
                counts.defense_branches = moves.len();
                counts.max_defense_branching = moves.len();
            }
        }
        for m in moves.iter() {
            let mv = m.mv();
            let is_drop = mv.is_drop();
            if is_drop {
                counts.drops += 1;
                // Rule F: a 無駄合 is not a defence, so it is not an interposition
                // the player has to see through either; it is counted separately.
                if !attack && !m.useless() {
                    counts.interpositions += 1;
                }
            }
            if !attack && m.useless() {
                counts.useless_interpositions += 1;
            }
            if mv.promote {
                counts.promotions += 1;
            }
            if !is_drop {
                let c = position.board[mv.from as usize];
                if c != EMPTY && ptype(c) == PT_K {
                    counts.king_moves += 1;
                }
            }
        }
        counts
    }

    /// The counts of the replies the horizon escapes left out.
    fn omitted(omitted: &OmittedReplies) -> Self {
        NodeCounts {
            drops: omitted.drops as usize,
            interpositions: omitted.interpositions as usize,
            useless_interpositions: omitted.useless_interpositions as usize,
            promotions: omitted.promotions as usize,
            king_moves: omitted.king_moves as usize,
            ..NodeCounts::default()
        }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            mate_defense_nodes: self.mate_defense_nodes + other.mate_defense_nodes,
            defense_branches: self.defense_branches + other.defense_branches,
            max_defense_branching: self.max_defense_branching.max(other.max_defense_branching),
            mate_nodes: self.mate_nodes + other.mate_nodes,
            drops: self.drops + other.drops,
            interpositions: self.interpositions + other.interpositions,
            useless_interpositions: self.useless_interpositions + other.useless_interpositions,
            promotions: self.promotions + other.promotions,
            king_moves: self.king_moves + other.king_moves,
        }
    }
}
