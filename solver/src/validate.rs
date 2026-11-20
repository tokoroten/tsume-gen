//! Structural validation of a strict-tsume diagram.
//!
//! Port of `validatePosition`.  Issue order is part of the contract: the pipeline
//! copies `issues.map(issue => issue.code)` into `rejection.reasons`, so the
//! checks run in exactly the TypeScript order.

use crate::piece::*;
use crate::position::{Position, NO_SQUARE};
use serde::{Deserialize, Serialize};

/// One validation problem, serialised as `{ code, message }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Issue {
    pub code: String,
    pub message: String,
}

impl Issue {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

/// Options mirroring the TypeScript parameter object.
#[derive(Clone, Copy, Debug)]
pub struct ValidateOptions {
    pub strict_material: bool,
    /// Defaults to the side to move.
    pub attacker: Option<u8>,
    /// `requireDefenderKing !== false` in TypeScript, i.e. `true` by default.
    pub require_defender_king: bool,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            strict_material: false,
            attacker: None,
            require_defender_king: true,
        }
    }
}

impl ValidateOptions {
    /// The options the pipeline uses: `{ strictMaterial: true, attacker: "b" }`.
    pub fn pipeline() -> Self {
        Self {
            strict_material: true,
            attacker: Some(BLACK),
            require_defender_king: true,
        }
    }
}

fn color_letter(color: u8) -> char {
    if color == BLACK {
        'b'
    } else {
        'w'
    }
}

/// Validates a diagram, returning the issues in TypeScript order.
///
/// The `board_size` and non-integer `hand` checks of the TypeScript version
/// cannot trigger here: the board is a fixed 81-entry array and hand counts are
/// unsigned bytes.
pub fn validate_position(position: &Position, options: ValidateOptions) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut kings = [0u32; 2];
    for (index, &c) in position.board.iter().enumerate() {
        if c == EMPTY {
            continue;
        }
        let t = ptype(c);
        let color = color_of(c);
        if t == PT_K {
            kings[color as usize] += 1;
        }
        if is_promoted(c) && !promotable_type(t) {
            issues.push(Issue::new("bad_promotion", "Invalid promoted piece"));
        }
        let rank = rank_of(index);
        if !is_promoted(c) && (t == PT_P || t == PT_L) && rank == if color == BLACK { 0 } else { 8 }
        {
            issues.push(Issue::new(
                "dead_piece",
                "Unpromoted pawn/lance on final rank",
            ));
        }
        if !is_promoted(c) && t == PT_N && if color == BLACK { rank <= 1 } else { rank >= 7 } {
            issues.push(Issue::new(
                "dead_piece",
                "Unpromoted knight on final two ranks",
            ));
        }
    }
    for color in [BLACK, WHITE] {
        if kings[color as usize] > 1 {
            issues.push(Issue::new(
                "king_count",
                format!("Too many {} kings", color_letter(color)),
            ));
        }
    }
    let attacker = options.attacker.unwrap_or(position.side);
    let defender = opponent(attacker);
    if options.require_defender_king && kings[defender as usize] != 1 {
        issues.push(Issue::new(
            "defender_king",
            "Strict tsume requires exactly one defender king",
        ));
    }
    for color in [BLACK, WHITE] {
        for file in 0..9 {
            let pawns = (0..9)
                .filter(|&rank| {
                    let c = position.board[sq(file, rank)];
                    c != EMPTY && color_of(c) == color && ptype(c) == PT_P && !is_promoted(c)
                })
                .count();
            if pawns > 1 {
                issues.push(Issue::new(
                    "nifu",
                    format!("Two unpromoted pawns in a file for {}", color_letter(color)),
                ));
            }
        }
    }
    if options.strict_material {
        for (piece, &total) in MATERIAL_TOTAL.iter().enumerate() {
            let mut n = position.hands[0][piece] as u32 + position.hands[1][piece] as u32;
            for &c in position.board.iter() {
                if c != EMPTY && ptype(c) == piece as u8 {
                    n += 1;
                }
            }
            if n > total as u32 {
                issues.push(Issue::new(
                    "material",
                    format!("Too many {} pieces", PIECE_LETTERS[piece] as char),
                ));
            }
        }
    }
    let black_king = position.king_square(BLACK);
    let white_king = position.king_square(WHITE);
    if black_king != NO_SQUARE && white_king != NO_SQUARE {
        let (bf, br) = (
            file_of(black_king as usize) as i32,
            rank_of(black_king as usize) as i32,
        );
        let (wf, wr) = (
            file_of(white_king as usize) as i32,
            rank_of(white_king as usize) as i32,
        );
        if (bf - wf).abs() <= 1 && (br - wr).abs() <= 1 {
            issues.push(Issue::new("adjacent_kings", "Kings may not be adjacent"));
        }
    }
    if position.king_square(defender) != NO_SQUARE && position.is_in_check(defender) {
        issues.push(Issue::new(
            "defender_already_checked",
            "Defender may not start a tsume already in check",
        ));
    }
    issues
}
