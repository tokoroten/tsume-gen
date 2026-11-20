//! Prestrip: greedy removal of far board pieces while a mate of a known length
//! survives (2026-10-01, `docs/experiments/mine-speedup/README.md`).
//!
//! A crowded diagram -- a `uniform` mining board, a practical mate -- carries a
//! dozen or more pieces the mate never touches, and `distill` pays its full gate
//! (uniqueness, yozume, rule C, ...) on every step of deleting them, until its
//! per-source node budget runs out.  Prestrip asks a much cheaper question on each
//! step -- "is this still a mate of exactly `k` plies?" -- and deletes pieces
//! farthest from the defender king first, so what reaches the expensive search is
//! a small diagram.  It decides nothing about publication: every later gate still
//! runs on the result.
//!
//! The order is deterministic (Chebyshev distance from the defender king,
//! descending, ties by square index) and a removed piece goes to the defender, as
//! the strict material convention wants, so the result is a pure function of the
//! position and of the caller's mate test.

use crate::piece::{color_of, file_of, ptype, rank_of, EMPTY, PT_K};
use crate::position::{Position, NO_SQUARE};
use crate::validate::{validate_position, ValidateOptions};

/// One removed board piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stripped {
    pub square: usize,
    /// The board code that stood there.
    pub code: u8,
}

/// What the caller's test says about one trial diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// The mate survives: keep the removal.
    Yes,
    /// It does not (or the trial is not usable): put the piece back.
    No,
    /// Stop stripping altogether, e.g. because a budget ran out.
    Stop,
}

/// Which board pieces are tried, in which order (2026-10-02).
///
/// [`Order::Distance`] is the original order (`mine --prestrip` and earlier
/// `distill --prestrip` runs).  [`Order::AttackerFirst`] keeps the defender's
/// guards for longer: deleting a guard only ever weakens the defence, so the mate almost
/// always survives it and the distance order strips guards eagerly -- measured,
/// 5-ply puzzles kept 0.9 defender board pieces on average with the distance
/// order against 1.6 without prestrip.
///
/// [`Order::AttackerFirst`] is the default since 2026-10-02 (owner decision): on
/// the uniform candidates it was cheaper per puzzle and found slightly more
/// puzzles than the distance order (`docs/experiments/mine-speedup/README.md`).
/// (A third order, `attacker-only`, and the unused text parser went on
/// 2026-10-27.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Order {
    /// Farthest from the defender king first, attacker and defender alike.
    Distance,
    /// Every attacker piece (farthest first), then every defender piece
    /// (farthest first).
    #[default]
    AttackerFirst,
}

impl Order {
    pub const fn as_str(self) -> &'static str {
        match self {
            Order::Distance => "distance",
            Order::AttackerFirst => "attacker-first",
        }
    }
}

/// Board squares of non-king pieces in prestrip order (see [`Order`]): within a
/// group, farthest from `king` first, ties by square index.
pub fn order(position: &Position, king: usize, order: Order) -> Vec<usize> {
    let distance = |index: usize| {
        let (kf, kr) = (file_of(king) as i32, rank_of(king) as i32);
        let (f, r) = (file_of(index) as i32, rank_of(index) as i32);
        (kf - f).abs().max((kr - r).abs())
    };
    let attacker = position.side;
    let is_attacker = |index: usize| color_of(position.board[index]) == attacker;
    let mut squares: Vec<usize> = (0..81)
        .filter(|&index| {
            let c = position.board[index];
            c != EMPTY && ptype(c) != PT_K
        })
        .collect();
    let group = |index: usize| match order {
        Order::AttackerFirst => u8::from(!is_attacker(index)),
        Order::Distance => 0,
    };
    squares.sort_by_key(|&index| (group(index), std::cmp::Reverse(distance(index)), index));
    squares
}

/// Strips `position` (attacker to move).  Each trial is structurally validated
/// and must leave the defender out of check before `keep` is asked about it.
/// Returns the stripped position and the removals in the order they were made;
/// an empty list means nothing could go.
pub fn strip(
    position: &Position,
    order_by: Order,
    mut keep: impl FnMut(&Position) -> Keep,
) -> (Position, Vec<Stripped>) {
    let defender = crate::piece::opponent(position.side);
    let king = position.king_square(defender);
    let mut current = position.clone();
    let mut removed = Vec::new();
    if king == NO_SQUARE {
        return (current, removed);
    }
    for square in order(position, king as usize, order_by) {
        let mut trial = current.clone();
        let code = trial.board[square];
        trial.board[square] = EMPTY;
        trial.refresh();
        if !validate_position(&trial, ValidateOptions::pipeline()).is_empty()
            || trial.is_in_check(defender)
        {
            continue;
        }
        match keep(&trial) {
            Keep::Yes => {
                current = trial;
                removed.push(Stripped { square, code });
            }
            Keep::No => {}
            Keep::Stop => break,
        }
    }
    (current, removed)
}
