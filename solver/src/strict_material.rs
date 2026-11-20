//! The frozen strict-tsume material convention (`docs/rules.md`).

use crate::piece::*;
use crate::position::Position;

/// Adds every non-king piece absent from the diagram and both hands to the
/// defender's hand.  Port of `withStrictDefenderMaterial`.
///
/// Promoted board pieces count as their unpromoted base piece and no king is
/// ever supplied to a hand.  The function is idempotent: applying it to an
/// already-normalised position changes nothing.
pub fn normalize(position: &Position, attacker: u8) -> Position {
    let mut out = position.clone();
    apply(&mut out, attacker);
    out
}

/// In-place [`normalize`].
pub fn apply(position: &mut Position, attacker: u8) {
    let defender = opponent(attacker);
    let mut seen = [0u32; 7];
    for &c in position.board.iter() {
        if c != EMPTY && ptype(c) != PT_K {
            seen[ptype(c) as usize] += 1;
        }
    }
    for hand in position.hands.iter() {
        for (piece, &count) in hand.iter().enumerate() {
            seen[piece] += count as u32;
        }
    }
    for (piece, &total) in MATERIAL_TOTAL.iter().enumerate() {
        let missing = (total as u32).saturating_sub(seen[piece]);
        if missing > 0 {
            let count = position.hands[defender as usize][piece] as usize + missing as usize;
            position.set_hand(defender, piece as u8, count);
        }
    }
}

/// `true` when the position already carries the full defender remainder, i.e.
/// `defenderRemainderInvariant` in `tools/puzzle-pipeline/src/canonical.ts`.
pub fn remainder_invariant(position: &Position) -> bool {
    normalize(position, position.side) == *position
}
