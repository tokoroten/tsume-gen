//! Bitboard geometry: 81 squares in the low bits of a `u128` (bit `square`,
//! square `rank * 9 + file` as everywhere in this crate).
//!
//! The tables answer attack questions by bit operations instead of walking the
//! mailbox square by square (2026-10-26, `docs/experiments/solver-scaling/`).
//! They change no answer: every query here is the same predicate the walking
//! code computed, so the legal-move sets and generation order stay the same.

use crate::piece::*;

/// Every square of the board.
pub const FULL_BOARD: u128 = (1u128 << 81) - 1;

/// The bit of `square`.
#[inline]
pub const fn bit(square: usize) -> u128 {
    1u128 << square
}

/// [`DIRS8`] indices 4..8 step towards higher square numbers (`+1`, `+8`,
/// `+9`, `+10`), 0..4 towards lower ones.
#[inline]
const fn ascending(dir: usize) -> bool {
    dir >= 4
}

pub struct Geometry {
    /// `ray[square][dir]`: the squares along [`DIRS8`]`[dir]` from `square`,
    /// `square` itself excluded.
    pub ray: [[u128; 8]; 81],
    /// `step[code][square]`: the squares a piece of `code` on `square` reaches
    /// by its non-sliding movements (knight jumps included).
    pub step: [[u128; 81]; 64],
    /// `neighbours[square]`: the eight surrounding squares on the board.
    pub neighbours: [u128; 81],
    /// `file[file]`: the nine squares of a file.
    pub file: [u128; 9],
    /// `between[a * 81 + b]`: the squares strictly between `a` and `b` on a
    /// shared rank, file or diagonal, else `0`.
    pub between: [u128; 81 * 81],
}

/// The squares on the board within one step of `(f, r)`: `true` when on it.
const fn on_board(f: i8, r: i8) -> bool {
    f >= 0 && f < 9 && r >= 0 && r < 9
}

/// Builds the tables at compile time (2026-10-27: a plain `static`, read
/// without the lazy initialisation check of a `LazyLock`).
const fn build() -> Geometry {
    let tables = &TABLES;
    let mut ray = [[0u128; 8]; 81];
    let mut neighbours = [0u128; 81];
    let mut square = 0usize;
    while square < 81 {
        let mut dir = 0usize;
        while dir < 8 {
            let (dx, dy) = DIRS8[dir];
            let (mut f, mut r) = (file_of(square) as i8 + dx, rank_of(square) as i8 + dy);
            let mut first = true;
            while on_board(f, r) {
                let to = (r * 9 + f) as usize;
                ray[square][dir] |= bit(to);
                if first {
                    neighbours[square] |= bit(to);
                    first = false;
                }
                f += dx;
                r += dy;
            }
            dir += 1;
        }
        square += 1;
    }
    let mut step = [[0u128; 81]; 64];
    let mut c = 0usize;
    while c < 64 {
        let mut square = 0usize;
        while square < 81 {
            let mut d = 0usize;
            while d < tables.dirs_len[c] as usize {
                let (dx, dy, slide) = tables.dirs[c][d];
                if !slide {
                    let (f, r) = (file_of(square) as i8 + dx, rank_of(square) as i8 + dy);
                    if on_board(f, r) {
                        step[c][square] |= bit((r * 9 + f) as usize);
                    }
                }
                d += 1;
            }
            square += 1;
        }
        c += 1;
    }
    let mut file = [0u128; 9];
    let mut f = 0usize;
    while f < 9 {
        let mut r = 0usize;
        while r < 9 {
            file[f] |= bit(sq(f, r));
            r += 1;
        }
        f += 1;
    }
    let mut between = [0u128; 81 * 81];
    let mut a = 0usize;
    while a < 81 {
        let mut dir = 0usize;
        while dir < 8 {
            let mut path = 0u128;
            let mut rest = ray[a][dir];
            while rest != 0 {
                let b = if ascending(dir) {
                    rest.trailing_zeros() as usize
                } else {
                    127 - rest.leading_zeros() as usize
                };
                rest &= !bit(b);
                between[a * 81 + b] = path;
                path |= bit(b);
            }
            dir += 1;
        }
        a += 1;
    }
    Geometry {
        ray,
        step,
        neighbours,
        file,
        between,
    }
}

/// The shared geometry tables.
pub static GEOMETRY: Geometry = build();

/// The square of the piece nearest to the ray's origin among `blockers`, all
/// on the ray [`DIRS8`]`[dir]` (non-empty).
#[inline]
pub fn nearest(blockers: u128, dir: usize) -> usize {
    debug_assert!(blockers != 0);
    if ascending(dir) {
        blockers.trailing_zeros() as usize
    } else {
        127 - blockers.leading_zeros() as usize
    }
}

/// The squares a slider on `square` reaches along `dir` with `occupied`
/// pieces on the board: up to and including the first piece.
#[inline]
pub fn ray_attack(g: &Geometry, square: usize, dir: usize, occupied: u128) -> u128 {
    let ray = g.ray[square][dir];
    let blockers = ray & occupied;
    if blockers == 0 {
        ray
    } else {
        ray ^ g.ray[nearest(blockers, dir)][dir]
    }
}

/// Diagonal slides from `square`.
#[inline]
pub fn bishop_attack(g: &Geometry, square: usize, occupied: u128) -> u128 {
    ray_attack(g, square, 0, occupied)
        | ray_attack(g, square, 2, occupied)
        | ray_attack(g, square, 5, occupied)
        | ray_attack(g, square, 7, occupied)
}

/// Orthogonal slides from `square`.
#[inline]
pub fn rook_attack(g: &Geometry, square: usize, occupied: u128) -> u128 {
    ray_attack(g, square, 1, occupied)
        | ray_attack(g, square, 3, occupied)
        | ray_attack(g, square, 4, occupied)
        | ray_attack(g, square, 6, occupied)
}

/// The index of a board piece in [`crate::position::Position`]'s per-kind
/// bitboards: its type with the promotion flag (`0..16`), colour dropped.
#[inline]
pub const fn kind(c: u8) -> usize {
    (c & (7 | FLAG_PROMOTED)) as usize
}

/// [`kind`] of a promoted piece of type `t`.
#[inline]
pub const fn promoted_kind(t: u8) -> usize {
    (t | FLAG_PROMOTED) as usize
}
