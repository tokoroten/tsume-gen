//! Piece encoding, direction tables, and attack masks.
//!
//! Mirrors `packages/core-rules/src/index.ts` exactly: piece types are ordered
//! `P L N S G B R K`, hand pieces are the first seven, files run 9->1 left to
//! right (file index 0 is USI file 9) and ranks run a->i top to bottom (rank
//! index 0 is USI rank `a`).  Black (`b`) moves toward rank `a`, i.e. toward
//! decreasing rank index.

/// Colour index. Matches SFEN `b` / `w`.
pub const BLACK: u8 = 0;
/// Colour index for white.
pub const WHITE: u8 = 1;

pub const PT_P: u8 = 0;
pub const PT_L: u8 = 1;
pub const PT_N: u8 = 2;
pub const PT_S: u8 = 3;
pub const PT_G: u8 = 4;
pub const PT_B: u8 = 5;
pub const PT_R: u8 = 6;
pub const PT_K: u8 = 7;

/// Bit set on a promoted board piece.
pub const FLAG_PROMOTED: u8 = 8;
/// Bit set on a white piece.
pub const FLAG_WHITE: u8 = 16;
/// Bit that distinguishes an occupied square from the empty code `0`.
pub const FLAG_PRESENT: u8 = 32;

/// The seven hand piece types in the order of the binary encodings (canonical hash
/// bytes, reducer candidates; `HAND_PIECES` in core-rules).  Frozen: part of puzzle
/// identity.  SFEN text does not use it; see [`SFEN_HAND_ORDER`].
pub const HAND_PIECES: [u8; 7] = [PT_P, PT_L, PT_N, PT_S, PT_G, PT_B, PT_R];
/// The order `to_sfen` writes hands in: USI standard rook, bishop, gold, silver,
/// knight, lance, pawn (core-rules `SFEN_HAND_ORDER`, owner decision 2026-10-04).
pub const SFEN_HAND_ORDER: [u8; 7] = [PT_R, PT_B, PT_G, PT_S, PT_N, PT_L, PT_P];
/// Canonical piece totals of a shogi set (`MATERIAL_TOTAL`).
pub const MATERIAL_TOTAL: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];
/// Piece letters indexed by piece type.
pub const PIECE_LETTERS: [u8; 8] = *b"PLNSGBRK";

/// The empty-square code.
pub const EMPTY: u8 = 0;

#[inline]
pub const fn code(color: u8, ptype: u8, promoted: bool) -> u8 {
    FLAG_PRESENT | (color << 4) | if promoted { FLAG_PROMOTED } else { 0 } | ptype
}
#[inline]
pub const fn ptype(c: u8) -> u8 {
    c & 7
}
#[inline]
pub const fn is_promoted(c: u8) -> bool {
    c & FLAG_PROMOTED != 0
}
#[inline]
pub const fn color_of(c: u8) -> u8 {
    (c >> 4) & 1
}
#[inline]
pub const fn opponent(color: u8) -> u8 {
    color ^ 1
}
#[inline]
pub const fn base_code(c: u8) -> u8 {
    c & !FLAG_PROMOTED
}
/// `true` when the letter may carry a `+` prefix (`PLNSBR`).
#[inline]
pub const fn promotable_type(t: u8) -> bool {
    t != PT_G && t != PT_K
}

#[inline]
pub const fn sq(file: usize, rank: usize) -> usize {
    rank * 9 + file
}
#[inline]
pub const fn file_of(square: usize) -> usize {
    square % 9
}
#[inline]
pub const fn rank_of(square: usize) -> usize {
    square / 9
}

/// The eight king directions as `(file delta, rank delta)`.  Index `i` and
/// `7 - i` are always opposites, which the attack scan relies on.
pub const DIRS8: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// A single movement ray: `(file delta, rank delta, slides)`.
pub type Dir = (i8, i8, bool);

/// Per-piece-code movement tables built once at start-up.
pub struct Tables {
    /// Movement rays per piece code, in the exact order of `dirs()` in the
    /// TypeScript rules, so pseudo-move generation order matches.
    pub dirs: [[Dir; 8]; 64],
    pub dirs_len: [u8; 64],
    /// Directions (indices into [`DIRS8`]) the piece attacks with a single step.
    pub step_mask: [u8; 64],
    /// Directions (indices into [`DIRS8`]) the piece slides along.
    pub slide_mask: [u8; 64],
    /// `true` for an unpromoted knight (handled separately by the attack scan).
    pub is_knight: [bool; 64],
    /// Squares reachable from `square` by one king step, as `(direction, target)`.
    pub neighbours: [[(u8, u8); 8]; 81],
    pub neighbours_len: [u8; 81],
}

/// `d` padded to eight entries, with its length.
const fn dir_list<const N: usize>(d: [Dir; N]) -> ([Dir; 8], usize) {
    let mut out = [(0, 0, false); 8];
    let mut i = 0;
    while i < N {
        out[i] = d[i];
        i += 1;
    }
    (out, N)
}

/// The movement rays of a piece, in the order of `dirs()` in the TypeScript
/// rules.  A `const fn` (2026-10-27) so that [`TABLES`] is built at compile
/// time and read without the lazy initialisation check.
const fn dirs_for(color: u8, t: u8, promoted: bool) -> ([Dir; 8], usize) {
    let f: i8 = if color == BLACK { -1 } else { 1 };
    let gold = dir_list([
        (-1, f, false),
        (0, f, false),
        (1, f, false),
        (-1, 0, false),
        (1, 0, false),
        (0, -f, false),
    ]);
    if promoted {
        return match t {
            PT_P | PT_L | PT_N | PT_S => gold,
            PT_B => dir_list([
                (-1, -1, true),
                (1, -1, true),
                (-1, 1, true),
                (1, 1, true),
                (-1, 0, false),
                (1, 0, false),
                (0, -1, false),
                (0, 1, false),
            ]),
            PT_R => dir_list([
                (-1, 0, true),
                (1, 0, true),
                (0, -1, true),
                (0, 1, true),
                (-1, -1, false),
                (1, -1, false),
                (-1, 1, false),
                (1, 1, false),
            ]),
            _ => dir_list::<0>([]),
        };
    }
    match t {
        PT_P => dir_list([(0, f, false)]),
        PT_L => dir_list([(0, f, true)]),
        PT_N => dir_list([(-1, 2 * f, false), (1, 2 * f, false)]),
        PT_S => dir_list([
            (-1, f, false),
            (0, f, false),
            (1, f, false),
            (-1, -f, false),
            (1, -f, false),
        ]),
        PT_G => gold,
        PT_B => dir_list([(-1, -1, true), (1, -1, true), (-1, 1, true), (1, 1, true)]),
        PT_R => dir_list([(-1, 0, true), (1, 0, true), (0, -1, true), (0, 1, true)]),
        PT_K => dir_list([
            (-1, -1, false),
            (0, -1, false),
            (1, -1, false),
            (-1, 0, false),
            (1, 0, false),
            (-1, 1, false),
            (0, 1, false),
            (1, 1, false),
        ]),
        _ => dir_list::<0>([]),
    }
}

const fn build_tables() -> Tables {
    let mut tables = Tables {
        dirs: [[(0, 0, false); 8]; 64],
        dirs_len: [0; 64],
        step_mask: [0; 64],
        slide_mask: [0; 64],
        is_knight: [false; 64],
        neighbours: [[(0, 0); 8]; 81],
        neighbours_len: [0; 81],
    };
    let colors = [BLACK, WHITE];
    let mut ci = 0;
    while ci < 2 {
        let color = colors[ci];
        let mut t = 0u8;
        while t < 8 {
            let mut pi = 0;
            while pi < 2 {
                let promoted = pi == 1;
                if !promoted || promotable_type(t) {
                    let c = code(color, t, promoted) as usize;
                    let (list, len) = dirs_for(color, t, promoted);
                    tables.dirs_len[c] = len as u8;
                    let mut i = 0;
                    while i < len {
                        let d = list[i];
                        tables.dirs[c][i] = d;
                        let mut index = 0;
                        while index < 8 {
                            if DIRS8[index].0 == d.0 && DIRS8[index].1 == d.1 {
                                if d.2 {
                                    tables.slide_mask[c] |= 1 << index;
                                } else {
                                    tables.step_mask[c] |= 1 << index;
                                }
                                break;
                            }
                            index += 1;
                        }
                        i += 1;
                    }
                    tables.is_knight[c] = t == PT_N && !promoted;
                }
                pi += 1;
            }
            t += 1;
        }
        ci += 1;
    }
    let mut square = 0usize;
    while square < 81 {
        let (file, rank) = (file_of(square) as i8, rank_of(square) as i8);
        let mut len = 0usize;
        let mut index = 0;
        while index < 8 {
            let (dx, dy) = DIRS8[index];
            let (f, r) = (file + dx, rank + dy);
            if f >= 0 && f < 9 && r >= 0 && r < 9 {
                tables.neighbours[square][len] = (index as u8, (r * 9 + f) as u8);
                len += 1;
            }
            index += 1;
        }
        tables.neighbours_len[square] = len as u8;
        square += 1;
    }
    tables
}

/// Shared movement tables.
pub static TABLES: Tables = build_tables();

/// `true` when `rank` lies in `color`'s promotion zone.
#[inline]
pub fn promotion_zone(color: u8, rank: usize) -> bool {
    if color == BLACK {
        rank <= 2
    } else {
        rank >= 6
    }
}

/// `true` when a piece of this type landing on `rank` must promote.
#[inline]
pub fn must_promote(t: u8, color: u8, rank: usize) -> bool {
    match t {
        PT_P | PT_L => rank == if color == BLACK { 0 } else { 8 },
        PT_N => {
            if color == BLACK {
                rank <= 1
            } else {
                rank >= 7
            }
        }
        _ => false,
    }
}
