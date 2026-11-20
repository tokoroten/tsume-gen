//! Heuristic puzzle quality score (spec 33, spec 10.6).
//!
//! Spec 33 lists the components but not a formula; this module fixes one, records
//! every component, and documents the weights in `docs/generation.md` so they can
//! be re-tuned against human labels later (spec 14.5).  Nothing downstream treats
//! the score as correctness: `exact_validate` decides whether a puzzle is valid,
//! the score only decides which valid variant is *interesting*.
//!
//! The deliberate departure from a naive reducer is spec 10.6: compactness is not
//! "fewer pieces is better".  A piece is only penalised when it is **provably
//! irrelevant**, that is when removing it alone still leaves a strict-valid puzzle
//! of the same length.  A locally irreducible 12-piece puzzle therefore scores
//! full compactness, while an 8-piece puzzle with two removable pieces does not.

use serde_json::{json, Value};

use crate::jsonl::DifficultyFeatures;
use crate::piece::*;
use crate::position::Position;
use crate::solver::{Graph, Outcome, SolveResult};

/// Relative weights of the quality components.  The positive weights sum to 1,
/// the penalties are subtracted on top, and the final score is clamped to `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct QualityWeights {
    pub uniqueness: f64,
    pub false_move: f64,
    pub compactness: f64,
    pub solution_tree: f64,
    pub readability: f64,
    pub triviality: f64,
    pub clutter: f64,
}

/// The weights used by `distill` unless overridden.
pub const DEFAULT_WEIGHTS: QualityWeights = QualityWeights {
    uniqueness: 0.24,
    false_move: 0.22,
    compactness: 0.15,
    solution_tree: 0.20,
    readability: 0.19,
    triviality: 0.25,
    clutter: 0.12,
};

/// Shape of the proven solution tree, as opposed to the whole search graph.
///
/// [`crate::features`] counts drops, promotions and branching over *every* node
/// the search visited, because that is what the TypeScript pipeline publishes as
/// difficulty features.  Quality needs the much smaller solution tree instead: a
/// puzzle whose *solution* contains a sacrifice is interesting, a puzzle whose
/// refuted side lines happen to contain thousands of drops is not.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SolutionShape {
    pub attacker_moves: usize,
    pub drops: usize,
    pub promotions: usize,
    /// Attacker moves the defender can legally capture (offered material).
    pub sacrifices: usize,
    /// Defence nodes with at least one legal reply.
    pub defender_nodes: usize,
    pub defender_branches: usize,
    pub max_defender_branching: usize,
    /// Squares that appear as the origin or target of a solution-tree move.
    pub touched: u128,
    /// Attacker drops per piece type, indexed by `PT_*`.
    pub drops_by_type: [usize; 7],
}

impl SolutionShape {
    pub fn mean_defender_branching(&self) -> f64 {
        if self.defender_nodes == 0 {
            0.0
        } else {
            self.defender_branches as f64 / self.defender_nodes as f64
        }
    }
}

/// Walks the proven solution tree of a mating search graph.
///
/// Attacker nodes keep only the minimum-distance mating moves (the same rule
/// [`crate::exact::prune_solver_graph`] uses), defender nodes keep every reply.
pub fn solution_shape(graph: &Graph) -> SolutionShape {
    let mut shape = SolutionShape::default();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![graph.root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = graph.node(id) else {
            continue;
        };
        if node.outcome() != Outcome::Mate {
            continue;
        }
        if crate::exact::is_mated_leaf(node) {
            // Terminal mated nodes: no reply at all, or (rule F, classical) only
            // futile interposition drops, which are not defences.  Counting them
            // would dilute the branching factor of the nodes where the defender
            // actually chooses, and the futile drops are no part of the solution.
            continue;
        }
        if !node.attack() {
            shape.defender_nodes += 1;
            shape.defender_branches += node.moves().len();
            shape.max_defender_branching = shape.max_defender_branching.max(node.moves().len());
        }
        for edge in node.moves().iter() {
            let solution = if node.attack() {
                edge.outcome() == Outcome::Mate && edge.distance() == node.distance()
            } else {
                edge.outcome() == Outcome::Mate
            };
            if !solution {
                continue;
            }
            let mv = edge.mv();
            shape.touched |= 1u128 << mv.to;
            if !mv.is_drop() {
                shape.touched |= 1u128 << mv.from;
            }
            if node.attack() {
                shape.attacker_moves += 1;
                if mv.is_drop() {
                    shape.drops += 1;
                    if (mv.piece as usize) < shape.drops_by_type.len() {
                        shape.drops_by_type[mv.piece as usize] += 1;
                    }
                }
                if mv.promote {
                    shape.promotions += 1;
                }
                if let Some(child) = edge.next().and_then(|next| graph.node(next)) {
                    // A sacrifice offers material the defender may legally take.
                    // Being merely *attacked* is not enough: every contact mate
                    // puts a piece on a square the king attacks, and the mate
                    // exists precisely because taking it is illegal.  A defence
                    // node lists every legal reply, so a reply that lands on the
                    // same square is a real capture of the piece just played.
                    let captured = child.moves().iter().any(|reply| reply.mv().to == mv.to);
                    if captured {
                        shape.sacrifices += 1;
                    }
                }
            }
            if let Some(next) = edge.next() {
                stack.push(next);
            }
        }
    }
    shape
}

/// Squares the piece on `square` attacks, as a bitmask over board indices.
///
/// Used to tell a guarding piece apart from clutter: a rook that never moves but
/// covers the mating square is essential, even though the solution never touches
/// it.
pub fn piece_attack_mask(position: &Position, square: usize) -> u128 {
    let c = position.board[square];
    if c == EMPTY {
        return 0;
    }
    let tables = &TABLES;
    let (file, rank) = (file_of(square) as i8, rank_of(square) as i8);
    let mut mask = 0u128;
    for &(dx, dy, slides) in &tables.dirs[c as usize][..tables.dirs_len[c as usize] as usize] {
        let (mut f, mut r) = (file + dx, rank + dy);
        while (0..9).contains(&f) && (0..9).contains(&r) {
            let target = (r * 9 + f) as usize;
            mask |= 1u128 << target;
            if position.board[target] != EMPTY || !slides {
                break;
            }
            f += dx;
            r += dy;
        }
    }
    mask
}

/// Everything [`score`] needs.
pub struct QualityInput<'a> {
    /// The puzzle diagram (strict-normalised, attacker to move).
    pub position: &'a Position,
    /// A `MATE` result whose graph was built.
    pub result: &'a SolveResult,
    pub features: &'a DifficultyFeatures,
    /// Number of board pieces whose removal alone still leaves a strict-valid
    /// puzzle of the same length.  `None` uses a piece-count proxy instead, which
    /// is what the inner search loop does before it knows which variants survive.
    pub irrelevant_pieces: Option<usize>,
    pub weights: &'a QualityWeights,
}

/// A scored variant, with every component kept for the record.
#[derive(Clone, Debug)]
pub struct Quality {
    pub score: f64,
    pub raw_score: f64,
    pub uniqueness: f64,
    pub false_move_interest: f64,
    pub compactness: f64,
    pub solution_tree_interest: f64,
    pub readability: f64,
    pub triviality_penalty: f64,
    pub clutter_penalty: f64,
    /// `None` when no other root move mates at all.
    pub second_best_mate_ply: Option<u32>,
    pub irrelevant_pieces: Option<usize>,
    pub clutter_pieces: usize,
    /// Attacker hand pieces the solution never drops.
    pub hand_clutter_pieces: usize,
    pub shape: SolutionShape,
}

impl Quality {
    /// The record-level `distill.quality` breakdown.
    pub fn to_json(&self) -> Value {
        json!({
            "score": self.score,
            "rawScore": self.raw_score,
            "uniqueness": self.uniqueness,
            "falseMoveInterest": self.false_move_interest,
            "compactness": self.compactness,
            "solutionTreeInterest": self.solution_tree_interest,
            "readability": self.readability,
            "trivialityPenalty": self.triviality_penalty,
            "clutterPenalty": self.clutter_penalty,
            "secondBestMatePly": self.second_best_mate_ply,
            "irrelevantPieces": self.irrelevant_pieces,
            "clutterPieces": self.clutter_pieces,
            "handClutterPieces": self.hand_clutter_pieces,
            "solutionAttackerMoves": self.shape.attacker_moves,
            "solutionDrops": self.shape.drops,
            "solutionPromotions": self.shape.promotions,
            "solutionSacrifices": self.shape.sacrifices,
            "solutionDefenderNodes": self.shape.defender_nodes,
            "solutionMeanDefenderBranching": self.shape.mean_defender_branching(),
            "solutionMaxDefenderBranching": self.shape.max_defender_branching,
        })
    }

    /// `solution.quality` in the shape `tools/puzzle-pipeline/src/types.ts`
    /// declares (`QualityMetadata`).  `compile` recomputes this field with its own
    /// placeholder heuristic; the full breakdown survives in the `distill` block.
    pub fn to_metadata(&self) -> Value {
        json!({
            "score": self.score,
            "uniqueness": self.uniqueness,
            "visualClutter": self.clutter_penalty,
            "novelty": self.solution_tree_interest,
        })
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Scores one strict-valid variant.
pub fn score(input: QualityInput<'_>) -> Quality {
    let features = input.features;
    let distance = input.result.distance.unwrap_or(0);
    let shape = match input.result.graph.as_ref() {
        Some(graph) => solution_shape(graph),
        None => SolutionShape::default(),
    };

    // Uniqueness margin: how much worse the best *other* mating root move is.
    // Mate distances at an attacker root are odd, so the gap is at least 2.
    let second_best_mate_ply = input
        .result
        .moves
        .iter()
        .filter(|m| m.outcome() == Outcome::Mate && m.distance() != Some(distance))
        .filter_map(|m| m.distance())
        .min();
    let uniqueness = match second_best_mate_ply {
        None => 1.0,
        Some(other) => clamp01(other.saturating_sub(distance) as f64 / 4.0),
    };

    // False-move interest: how many plausible wrong checks there are and how hard
    // each is to refute.  `falseMoveDifficulty` is the spec 14.3 sum of
    // log1p(refutation nodes), so its mean is a per-move depth in log units;
    // ln(3000) is about 8, which is the normalising scale.
    let wrong = features.plausible_wrong_checks as f64;
    let count_term = clamp01(wrong.min(8.0) / 8.0);
    let mean_refutation = if wrong > 0.0 {
        features.false_move_difficulty / wrong
    } else {
        0.0
    };
    let refutation_term = clamp01(mean_refutation / 8.0);
    let false_move_interest = 0.5 * count_term + 0.5 * refutation_term;

    // Compactness (spec 10.6): penalise only provably irrelevant pieces.
    let pieces = features.board_piece_count as f64;
    let compactness = match input.irrelevant_pieces {
        Some(count) => 1.0 / (1.0 + count as f64),
        None => clamp01(1.0 - (pieces - 6.0) / 18.0),
    };

    // Solution-tree interest: defender choice plus the classic motifs.
    let branching_term = clamp01(shape.mean_defender_branching() / 4.0);
    let motif_term = clamp01(
        (shape.sacrifices as f64 * 2.0 + shape.drops as f64 + shape.promotions as f64) / 4.0,
    );
    let solution_tree_interest = 0.6 * branching_term + 0.4 * motif_term;

    // Readability: a small board, a king zone that is not packed, and the edge and
    // corner shapes tsume players read fastest.
    let piece_term = clamp01((20.0 - pieces) / 14.0);
    let density_term = clamp01(1.0 - (features.visual_density_near_king as f64 - 3.0) / 6.0);
    let edge_term = 1.0 - (features.edge_distance.min(4) as f64 / 4.0) * 0.5;
    let readability = 0.4 * piece_term + 0.35 * density_term + 0.25 * edge_term;

    // Triviality: one legal check at the root is a non-puzzle, two is thin.
    let triviality_penalty = match features.legal_checking_moves_at_root {
        0 | 1 => 1.0,
        2 => 0.35,
        _ => 0.0,
    };

    // Clutter: board pieces far from the king that the solution never touches.
    let defender = opponent(input.position.side);
    let king = input
        .position
        .board
        .iter()
        .position(|&c| c != EMPTY && color_of(c) == defender && ptype(c) == PT_K);
    let clutter_pieces = match king {
        None => 0,
        Some(king) => {
            let (kf, kr) = (file_of(king) as i32, rank_of(king) as i32);
            input
                .position
                .board
                .iter()
                .enumerate()
                .filter(|&(square, &c)| {
                    if c == EMPTY || ptype(c) == PT_K {
                        return false;
                    }
                    let distance = (file_of(square) as i32 - kf)
                        .abs()
                        .max((rank_of(square) as i32 - kr).abs());
                    if distance <= 3 || shape.touched & (1u128 << square) != 0 {
                        return false;
                    }
                    // A distant piece that covers a square of the solution is a
                    // guard, not clutter.
                    piece_attack_mask(input.position, square) & shape.touched == 0
                })
                .count()
        }
    };
    // Attacker hand material the solution never drops is clutter too: a hand full
    // of spare pieces makes the diagram harder to read and inflates the number of
    // meaningless checks.  Defender hand material is *not* counted: under the
    // strict convention it is a consequence of the diagram, not a choice.
    let attacker = input.position.side as usize;
    let hand_clutter: usize = (0..7)
        .map(|piece| {
            let held = input.position.hands[attacker][piece] as usize;
            held.saturating_sub(shape.drops_by_type[piece].min(held))
        })
        .sum();
    let clutter_pieces = clutter_pieces + hand_clutter;
    let clutter_penalty = clamp01(clutter_pieces as f64 / 6.0);

    let weights = input.weights;
    let raw_score = weights.uniqueness * uniqueness
        + weights.false_move * false_move_interest
        + weights.compactness * compactness
        + weights.solution_tree * solution_tree_interest
        + weights.readability * readability
        - weights.triviality * triviality_penalty
        - weights.clutter * clutter_penalty;
    Quality {
        score: clamp01(raw_score),
        raw_score,
        uniqueness,
        false_move_interest,
        compactness,
        solution_tree_interest,
        readability,
        triviality_penalty,
        clutter_penalty,
        second_best_mate_ply,
        irrelevant_pieces: input.irrelevant_pieces,
        clutter_pieces,
        hand_clutter_pieces: hand_clutter,
        shape,
    }
}
