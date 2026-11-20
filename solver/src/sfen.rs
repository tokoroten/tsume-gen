//! SFEN parsing and serialisation.
//!
//! A byte-for-byte port of `parseSfen` / `toSfen` in
//! `packages/core-rules/src/index.ts`, including the order in which the
//! TypeScript version validates its input and the exact error messages, because
//! `validateStage` copies the message into `validation.issues[0].message`.

use crate::piece::*;
use crate::position::Position;

/// A rules-level error carrying the TypeScript message verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SfenError(pub String);

impl std::fmt::Display for SfenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SfenError {}

fn err<T>(message: impl Into<String>) -> Result<T, SfenError> {
    Err(SfenError(message.into()))
}

/// Largest value JavaScript treats as a safe integer.
const MAX_SAFE_INTEGER: u128 = 9_007_199_254_740_991;

fn piece_from_letter(letter: u8) -> Option<(u8, u8)> {
    let upper = letter.to_ascii_uppercase();
    let index = PIECE_LETTERS.iter().position(|&l| l == upper)?;
    let color = if letter.is_ascii_uppercase() {
        BLACK
    } else {
        WHITE
    };
    Some((color, index as u8))
}

/// Parses a four-field SFEN string.
pub fn parse_sfen(sfen: &str) -> Result<Position, SfenError> {
    let parts: Vec<&str> = sfen.split_whitespace().collect();
    if parts.len() != 4 {
        return err("SFEN must contain board, side, hands, and move number");
    }
    let number = parts[3].as_bytes();
    if number.is_empty()
        || !(b'1'..=b'9').contains(&number[0])
        || !number.iter().all(|c| c.is_ascii_digit())
    {
        return err("SFEN move number must be a positive integer");
    }
    let ranks: Vec<&str> = parts[0].split('/').collect();
    if ranks.len() != 9 {
        return err("SFEN board must contain nine ranks");
    }
    let mut position = Position::empty();
    for (rank, row) in ranks.iter().enumerate() {
        let bytes = row.as_bytes();
        let mut file = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let mut c = bytes[i];
            if (b'1'..=b'9').contains(&c) {
                file += (c - b'0') as usize;
                i += 1;
                continue;
            }
            let promoted = c == b'+';
            if promoted {
                i += 1;
                c = bytes.get(i).copied().unwrap_or(0);
            }
            let Some((color, t)) = piece_from_letter(c) else {
                return err(format!("Invalid SFEN board rank {}", rank + 1));
            };
            if file >= 9 {
                return err(format!("Invalid SFEN board rank {}", rank + 1));
            }
            if promoted && !promotable_type(t) {
                return err("Only P/L/N/S/B/R may be promoted");
            }
            position.board[rank * 9 + file] = code(color, t, promoted);
            file += 1;
            i += 1;
        }
        if file != 9 {
            return err(format!("SFEN rank {} has {file} files", rank + 1));
        }
    }
    position.side = match parts[1] {
        "b" => BLACK,
        "w" => WHITE,
        _ => return err("SFEN side must be b or w"),
    };
    if parts[2] != "-" {
        let bytes = parts[2].as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = &parts[2][start..i];
            let letter = bytes.get(i).copied().unwrap_or(0);
            i += 1;
            let Some((color, t)) = piece_from_letter(letter) else {
                return err("Invalid SFEN hand");
            };
            if t == PT_K {
                return err("Invalid SFEN hand");
            }
            let count: u128 = if digits.is_empty() {
                1
            } else {
                digits.parse().unwrap_or(u128::MAX)
            };
            if !(1..=MAX_SAFE_INTEGER).contains(&count) {
                return err("Invalid SFEN hand count");
            }
            let total = position.hands[color as usize][t as usize] as u128 + count;
            if total > crate::zobrist::MAX_HAND_COUNT as u128 {
                // Documented Rust-side limit: hand counts above 255 cannot occur in
                // any reachable strict-tsume position and are rejected instead of
                // silently truncated.
                return err("Invalid SFEN hand count");
            }
            position.hands[color as usize][t as usize] = total as u8;
        }
    }
    position.refresh();
    Ok(position)
}

/// Serialises a position, defaulting to move number 1 like `toSfen`.
pub fn to_sfen(position: &Position) -> String {
    to_sfen_with_move_number(position, 1)
}

/// Serialises a position with an explicit move number.  Hands are written in
/// [`SFEN_HAND_ORDER`] (R B G S N L P), black before white.
pub fn to_sfen_with_move_number(position: &Position, move_number: u64) -> String {
    write_sfen(position, move_number, &SFEN_HAND_ORDER)
}

/// The SFEN spelling used before 2026-10-04: hands in `HAND_PIECES` order
/// (P L N S G B R), move number 1.  Port of core-rules `toLegacySfenKey`.  Not for
/// output: it is kept only as a comparison / hashing key where the old spelling fixed
/// a choice (the "smaller SFEN" reduction tie-break, the predecessor grouping order of
/// `unmove`, source ids hashed from a position), so the written order does not change
/// which puzzle is produced or what id it gets.
pub fn to_legacy_sfen_key(position: &Position) -> String {
    write_sfen(position, 1, &HAND_PIECES)
}

/// [`to_legacy_sfen_key`] of an SFEN text, whatever hand order the text uses.  Text
/// that does not parse is returned unchanged.
pub fn legacy_sfen_key_of(sfen: &str) -> String {
    match parse_sfen(sfen) {
        Ok(position) => to_legacy_sfen_key(&position),
        Err(_) => sfen.to_string(),
    }
}

fn write_sfen(position: &Position, move_number: u64, hand_order: &[u8; 7]) -> String {
    let mut out = String::with_capacity(96);
    for rank in 0..9 {
        if rank > 0 {
            out.push('/');
        }
        let mut empty = 0u32;
        for file in 0..9 {
            let c = position.board[sq(file, rank)];
            if c == EMPTY {
                empty += 1;
                continue;
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
                empty = 0;
            }
            if is_promoted(c) {
                out.push('+');
            }
            let letter = PIECE_LETTERS[ptype(c) as usize];
            out.push(if color_of(c) == BLACK {
                letter as char
            } else {
                letter.to_ascii_lowercase() as char
            });
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
    }
    out.push(' ');
    out.push(if position.side == BLACK { 'b' } else { 'w' });
    out.push(' ');
    let mut hands = String::new();
    for color in [BLACK, WHITE] {
        for &t in hand_order.iter() {
            let n = position.hands[color as usize][t as usize];
            if n == 0 {
                continue;
            }
            if n != 1 {
                hands.push_str(&n.to_string());
            }
            let letter = PIECE_LETTERS[t as usize];
            hands.push(if color == BLACK {
                letter as char
            } else {
                letter.to_ascii_lowercase() as char
            });
        }
    }
    out.push_str(if hands.is_empty() { "-" } else { &hands });
    out.push(' ');
    out.push_str(&move_number.to_string());
    out
}

/// USI text for a square index, e.g. `5b`.
pub fn square_to_usi(square: usize) -> String {
    format!(
        "{}{}",
        9 - file_of(square),
        (b'a' + rank_of(square) as u8) as char
    )
}

/// Parses USI square text, e.g. `5b`.
pub fn square_from_usi(text: &str) -> Result<usize, SfenError> {
    let b = text.as_bytes();
    if b.len() != 2 || !(b'1'..=b'9').contains(&b[0]) || !(b'a'..=b'i').contains(&b[1]) {
        return err(format!("Invalid USI square: {text}"));
    }
    Ok((b[1] - b'a') as usize * 9 + (9 - (b[0] - b'0') as usize))
}

impl Position {
    /// Parses a four-field SFEN string.
    pub fn from_sfen(sfen: &str) -> Result<Position, SfenError> {
        parse_sfen(sfen)
    }
    /// Serialises with move number 1.
    pub fn to_sfen(&self) -> String {
        to_sfen(self)
    }
    /// Serialises with an explicit move number.
    pub fn to_sfen_with_move_number(&self, move_number: u64) -> String {
        to_sfen_with_move_number(self, move_number)
    }
}
