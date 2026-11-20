//! Rule D's candidates (a board piece, a unit of attacker hand), applying
//! them, and the inert-piece sweep (rule G).

use super::*;

/// First square index that encodes attacker hand material instead of a board
/// square (rule D entry 3).
pub const HAND_SQUARE_BASE: usize = 81;

/// Rule D v3: the inert sweep iterates to a fixpoint, but never forever.
pub const INERT_SWEEP_ITERATIONS: usize = 16;

/// `true` for rule G's rejection reasons -- the ones the inert sweep looks
/// past: the two inert reasons and clause (e)'s budget-out
/// (`rule_g_budget_exhausted`, `PIPELINE_VERSION` 0.15.0).
pub(super) fn is_rule_g_reason(reason: &str) -> bool {
    reason == "inert_attacker_piece"
        || reason == "inert_defender_piece"
        || reason == crate::inert::RULE_G_BUDGET_EXHAUSTED
}

/// One reduction candidate (rule D).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidate {
    /// Delete the attacker's king from this square.
    AttackerKing(usize),
    /// Remove the board piece at this square; the defender gains it.
    Board(usize),
    /// Move one piece of this type from the attacker's hand to the defender's.
    Hand(u8),
}

impl Candidate {
    /// The `reduction.removed[].square` value: a real square, or `81 + k`.
    pub const fn square(self) -> usize {
        match self {
            Candidate::AttackerKing(square) | Candidate::Board(square) => square,
            Candidate::Hand(piece) => HAND_SQUARE_BASE + piece as usize,
        }
    }
}

/// Rule D candidate list, in the frozen pre-shuffle order.
pub fn reduction_candidates(position: &Position, attacker: u8) -> Vec<Candidate> {
    let mut out = Vec::new();
    if let Some(square) = attacker_king_square(position, attacker) {
        out.push(Candidate::AttackerKing(square));
    }
    for (square, &c) in position.board.iter().enumerate() {
        if c != EMPTY && ptype(c) != PT_K {
            out.push(Candidate::Board(square));
        }
    }
    for &piece in HAND_PIECES.iter() {
        for _ in 0..position.hands[attacker as usize][piece as usize] {
            out.push(Candidate::Hand(piece));
        }
    }
    out
}

/// Moves one piece of type `piece` from the attacker's hand to the defender's.
///
/// The defender's hand is the strict remainder, so dropping the attacker's copy is
/// enough: [`crate::strict_material::apply`] hands it straight back to the
/// defender.
pub(super) fn transfer_hand_piece(
    position: &Position,
    attacker: u8,
    piece: u8,
) -> Option<Position> {
    let held = position.hands[attacker as usize][piece as usize];
    if held == 0 {
        return None;
    }
    let mut out = position.clone();
    out.set_hand(attacker, piece, held as usize - 1);
    strict_material::apply(&mut out, attacker);
    Some(out)
}

/// Total attacker hand material, the rule D secondary tie-break.
pub(super) fn attacker_hand_total(position: &Position, attacker: u8) -> usize {
    position.hands[attacker as usize]
        .iter()
        .map(|&count| count as usize)
        .sum()
}

/// Applies one rule D candidate, or `None` when it no longer applies.
pub(super) fn apply_candidate(
    position: &Position,
    candidate: Candidate,
    attacker: u8,
) -> Option<(Position, RemovedPiece)> {
    match candidate {
        // A king is deleted outright; every other board piece becomes defender
        // hand material, which is what the strict convention forces.
        Candidate::AttackerKing(square) => {
            if position.board[square] == EMPTY {
                return None;
            }
            let label = piece_label(position, square);
            let trial = without_attacker_king(position, attacker)?;
            Some((
                trial,
                RemovedPiece {
                    square,
                    piece: label,
                },
            ))
        }
        Candidate::Board(square) => {
            let c = position.board[square];
            if c == EMPTY || ptype(c) == PT_K {
                return None;
            }
            let label = piece_label(position, square);
            Some((
                remove_board_piece_to_defender(position, square),
                RemovedPiece {
                    square,
                    piece: label,
                },
            ))
        }
        Candidate::Hand(piece) => Some((
            transfer_hand_piece(position, attacker, piece)?,
            RemovedPiece {
                square: candidate.square(),
                piece: format!("hand:{}", PIECE_LETTERS[piece as usize] as char),
            },
        )),
    }
}

/// The outcome of one inert sweep, with the verdict on the diagram it returns.
#[derive(Clone, Debug)]
pub struct SweepResult {
    pub position: Position,
    pub removed: Vec<RemovedPiece>,
    /// The acceptance test of [`SweepResult::position`], so no caller has to solve
    /// it again.
    pub check: ReductionCheck,
}

/// Rule D v3, the **inert sweep**: takes the diagram's inert board pieces (rule G,
/// 無駄駒) off until none is left or none can go.  See the module docs.
///
/// `entry` is the verdict on `position`, which every caller already has.  Every
/// round is judged by `rule` with rule G set aside ([`StepRule::accepts_round`]):
/// under rule D v5's main-line rule, against the line and paths of the diagram
/// the step started from.
pub fn sweep_inert_pieces(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: u32,
    options: &ReduceOptions,
    entry: ReductionCheck,
    rule: &StepRule,
) -> SweepResult {
    let attacker = position.side;
    let want_paths = rule.wants_paths();
    let mut current = position.clone();
    let mut check = entry;
    let mut removed: Vec<RemovedPiece> = Vec::new();
    for _ in 0..INERT_SWEEP_ITERATIONS {
        // The early stop ([`stopped`]): a check of this step (or of the
        // record) ran out of budget, so the step's result is never used.
        if check.inert_pieces.is_empty() || stopped() {
            break;
        }
        let mut squares: Vec<usize> = check
            .inert_pieces
            .iter()
            .map(|piece| piece.square)
            .collect();
        squares.sort_unstable();
        // All of them together first.  Inert pieces cover for each other --
        // shedding one of two leaves the other, which is the rule C valley in rule
        // G's clothing -- so a round that could only step one piece at a time would
        // stall exactly where the rule bites hardest.
        if let Some((trial, removals)) = apply_candidates(&current, &squares, attacker) {
            let _stage = Stage::enter("sweep", None, removals.clone());
            let bulk = check_reduction(solver, &trial, target_mate_ply, options, want_paths);
            if stopped() {
                break;
            }
            if rule.accepts_round(&bulk) {
                current = trial;
                check = bulk;
                removed.extend(removals);
                continue;
            }
        }
        // A single inert piece has already had its one and only trial above.
        if squares.len() == 1 {
            break;
        }
        // Otherwise one piece at a time, ascending, each judged on the diagram the
        // previously accepted removals left behind.
        let mut progressed = false;
        for square in squares {
            let Some((trial, removal)) =
                apply_candidate(&current, Candidate::Board(square), attacker)
            else {
                continue;
            };
            let _stage = Stage::enter("sweep", None, vec![removal.clone()]);
            let trial_check = check_reduction(solver, &trial, target_mate_ply, options, want_paths);
            if stopped() {
                break;
            }
            if !rule.accepts_round(&trial_check) {
                continue;
            }
            current = trial;
            check = trial_check;
            removed.push(removal);
            progressed = true;
        }
        if !progressed || stopped() {
            break;
        }
    }
    SweepResult {
        position: current,
        removed,
        check,
    }
}

/// Applies several board-square candidates in the given order, for the inert
/// sweep's all-at-once step; `None` when any of the squares is already empty.
///
/// Rule G never reports a king (`analyze_inert_pieces` skips them), so treating
/// every square as a [`Candidate::Board`] is exactly what `applyCandidates` does
/// in TypeScript.
pub(super) fn apply_candidates(
    position: &Position,
    squares: &[usize],
    attacker: u8,
) -> Option<(Position, Vec<RemovedPiece>)> {
    let mut trial = position.clone();
    let mut removals: Vec<RemovedPiece> = Vec::new();
    for &square in squares {
        let (next, removal) = apply_candidate(&trial, Candidate::Board(square), attacker)?;
        trial = next;
        removals.push(removal);
    }
    if removals.is_empty() {
        None
    } else {
        Some((trial, removals))
    }
}
