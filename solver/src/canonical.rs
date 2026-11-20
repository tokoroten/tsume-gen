//! Canonical orientation, mirroring, runtime encoding and the exact hash.
//!
//! Port of `tools/puzzle-pipeline/src/canonical.ts`, using the same runtime piece
//! codes as `packages/puzzle-runtime` (`PIECE_CODE`): `EMPTY 0`, `P..R 1..7`,
//! `K 8`, `+8` for a promoted piece, `+32` for a defender piece.

use sha2::{Digest, Sha256};

use crate::piece::*;
use crate::position::Position;
use crate::sfen::to_sfen;
use crate::strict_material;

/// Offset added to a base code for a promoted piece.
pub const PROMOTED_OFFSET: u8 = 8;
/// Flag added for a piece owned by the defender.
pub const DEFENDER_FLAG: u8 = 32;

/// 81 board cells plus 14 hand counts, exactly as `EncodedPosition`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPosition {
    pub board: [u8; 81],
    pub hands: [u8; 14],
}

impl EncodedPosition {
    /// The 95 canonical bytes hashed by [`canonicalize`].
    pub fn bytes(&self) -> [u8; 95] {
        let mut out = [0u8; 95];
        out[..81].copy_from_slice(&self.board);
        out[81..].copy_from_slice(&self.hands);
        out
    }
}

/// Rotates a white-to-move diagram so the attacker is always black, then applies
/// the strict defender remainder.  Port of `normalizeAttacker`.
pub fn normalize_attacker(position: &Position) -> Position {
    if position.side == BLACK {
        return strict_material::normalize(position, BLACK);
    }
    let mut rotated = Position::empty();
    for index in 0..81usize {
        let c = position.board[80 - index];
        if c != EMPTY {
            rotated.board[index] = c ^ FLAG_WHITE;
        }
    }
    rotated.side = BLACK;
    for piece in 0..7usize {
        rotated.hands[BLACK as usize][piece] = position.hands[WHITE as usize][piece];
        rotated.hands[WHITE as usize][piece] = position.hands[BLACK as usize][piece];
    }
    rotated.refresh();
    strict_material::normalize(&rotated, BLACK)
}

/// Mirrors the board horizontally (file 9 becomes file 1).
pub fn mirror_position(position: &Position) -> Position {
    let mut out = position.clone();
    for rank in 0..9usize {
        for file in 0..9usize {
            out.board[sq(file, rank)] = position.board[sq(8 - file, rank)];
        }
    }
    out.refresh();
    out
}

/// Runtime piece-code encoding of the attacker-normalised position.
pub fn encode_position(position: &Position) -> EncodedPosition {
    let normalized = normalize_attacker(position);
    let mut board = [0u8; 81];
    for (index, &c) in normalized.board.iter().enumerate() {
        if c == EMPTY {
            continue;
        }
        let base = ptype(c) + 1;
        let promotion = if is_promoted(c) { PROMOTED_OFFSET } else { 0 };
        let owner = if color_of(c) == WHITE {
            DEFENDER_FLAG
        } else {
            0
        };
        board[index] = base + promotion + owner;
    }
    let mut hands = [0u8; 14];
    for color in 0..2usize {
        for piece in 0..7usize {
            hands[color * 7 + piece] = normalized.hands[color][piece];
        }
    }
    EncodedPosition { board, hands }
}

/// Result of [`canonicalize`].
#[derive(Clone, Debug)]
pub struct Canonical {
    pub position: Position,
    pub sfen: String,
    pub mirrored: bool,
    /// Lowercase hex SHA-256 of [`Canonical::bytes`].
    pub hash: String,
    pub bytes: [u8; 95],
}

/// Picks the lexicographically smaller of the normalised and mirrored encodings
/// and hashes it.  Port of `canonicalize`.
pub fn canonicalize(position: &Position) -> Canonical {
    let normalized = normalize_attacker(position);
    let mirrored_position = mirror_position(&normalized);
    let original = encode_position(&normalized).bytes();
    let mirrored = encode_position(&mirrored_position).bytes();
    let use_mirror = mirrored < original;
    let position = if use_mirror {
        mirrored_position
    } else {
        normalized
    };
    let bytes = if use_mirror { mirrored } else { original };
    Canonical {
        sfen: to_sfen(&position),
        position,
        mirrored: use_mirror,
        hash: sha256_hex(&bytes),
        bytes,
    }
}

/// Lowercase hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((byte & 15) as u32, 16).unwrap());
    }
    out
}

/// Label used by `reduction.removed[].piece`, e.g. `b+P`, `wG`, `empty`.
pub fn piece_label(position: &Position, index: usize) -> String {
    let c = position.board[index];
    if c == EMPTY {
        return "empty".to_string();
    }
    format!(
        "{}{}{}",
        if color_of(c) == BLACK { 'b' } else { 'w' },
        if is_promoted(c) { "+" } else { "" },
        PIECE_LETTERS[ptype(c) as usize] as char
    )
}

/// Removes a board piece and hands every missing piece back to the defender.
///
/// # Panics
/// Panics when the square is empty or holds a king, like the TypeScript version.
pub fn remove_board_piece_to_defender(position: &Position, index: usize) -> Position {
    let c = position.board[index];
    assert!(
        c != EMPTY && ptype(c) != PT_K,
        "Square does not contain a removable piece"
    );
    let mut copy = position.clone();
    copy.board[index] = EMPTY;
    copy.refresh();
    // Formal strict-tsume semantics: every missing physical piece belongs to the
    // defender.
    strict_material::apply(&mut copy, position.side);
    copy
}
