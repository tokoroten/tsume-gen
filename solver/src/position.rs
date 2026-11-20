//! Board state with an incremental Zobrist key and O(rays) attack detection.
//!
//! The TypeScript reference clones an 81-entry array of heap objects for every
//! pseudo-move and answers `isInCheck` by scanning all 81 squares.  This module
//! keeps a single mutable mailbox with make/unmake, and answers attack queries by
//! walking outward from the target square, which is what makes the Rust search
//! orders of magnitude cheaper while producing identical results.

use crate::bitboard::{
    bishop_attack, bit, kind, nearest, promoted_kind, ray_attack, rook_attack, GEOMETRY,
};
use crate::piece::*;
use crate::zobrist::{KEYS, MAX_HAND_COUNT};

/// Sentinel for "no square".
pub const NO_SQUARE: u8 = 0xFF;

/// A move in the same shape as the TypeScript `Move` interface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    /// Origin square, or [`NO_SQUARE`] for a drop.
    pub from: u8,
    pub to: u8,
    /// Piece type (`PT_*`).  For board moves this is the mover base type.
    pub piece: u8,
    pub promote: bool,
}

impl Move {
    #[inline]
    pub const fn board(from: u8, to: u8, piece: u8, promote: bool) -> Self {
        Self {
            from,
            to,
            piece,
            promote,
        }
    }
    #[inline]
    pub const fn drop(piece: u8, to: u8) -> Self {
        Self {
            from: NO_SQUARE,
            to,
            piece,
            promote: false,
        }
    }
    #[inline]
    pub const fn is_drop(&self) -> bool {
        self.from == NO_SQUARE
    }

    /// USI text, e.g. `7g7f`, `7g7f+`, `G*5b`.
    pub fn to_usi(&self) -> Usi {
        let mut buf = [0u8; 5];
        let mut len;
        if self.is_drop() {
            buf[0] = PIECE_LETTERS[self.piece as usize];
            buf[1] = b'*';
            len = 2;
        } else {
            buf[0] = b'9' - file_of(self.from as usize) as u8;
            buf[1] = b'a' + rank_of(self.from as usize) as u8;
            len = 2;
        }
        buf[len] = b'9' - file_of(self.to as usize) as u8;
        buf[len + 1] = b'a' + rank_of(self.to as usize) as u8;
        len += 2;
        if self.promote {
            buf[len] = b'+';
            len += 1;
        }
        Usi {
            buf,
            len: len as u8,
        }
    }

    /// A number whose order is the bytewise order of [`Move::to_usi`], built
    /// without the text.
    ///
    /// The five USI bytes go most significant first, one byte per 8 bits, and a
    /// missing fifth byte (no `+`) counts as `0`, below every real byte -- so a
    /// prefix sorts first (`7g7f` < `7g7f+`) exactly as the byte comparison of
    /// the strings does.  Built from shifts and ORs only, so the value is the
    /// same on every platform.
    #[inline]
    pub fn usi_key(&self) -> u64 {
        // Two bytes per square (`USI_SQUARE`) or per dropped piece letter and
        // `*` (`USI_DROP`), placed with shifts.
        let head = if self.is_drop() {
            USI_DROP[self.piece as usize & 7]
        } else {
            USI_SQUARE[self.from as usize]
        };
        let tail = USI_SQUARE[self.to as usize];
        let plus = if self.promote { b'+' } else { 0 };
        (u64::from(head) << 24) | (u64::from(tail) << 8) | u64::from(plus)
    }

    /// Sorts `moves` by [`Move::usi_key`] (the same order as
    /// `sort_unstable_by_key(Move::usi_key)`: the keys of distinct moves
    /// differ), sorting plain numbers -- each key with the move's index in its
    /// low 16 bits -- instead of calling the key on every comparison
    /// (2026-10-27).
    pub fn sort_usi(moves: &mut [Move]) {
        const STACK: usize = 128;
        let n = moves.len();
        if n < 2 {
            return;
        }
        if n > STACK {
            moves.sort_unstable_by_key(Move::usi_key);
            return;
        }
        let mut keys = [0u64; STACK];
        let mut copy = [Move::drop(0, 0); STACK];
        for (index, m) in moves.iter().enumerate() {
            keys[index] = m.usi_key() << 16 | index as u64;
            copy[index] = *m;
        }
        keys[..n].sort_unstable();
        for (slot, key) in moves.iter_mut().zip(&keys[..n]) {
            *slot = copy[(key & 0xFFFF) as usize];
        }
    }

    /// The movers whose promotion df-pn tries before the non-promoting twin
    /// ([`Move::promotion_first`]): bishop, rook and pawn, whose promoted piece
    /// keeps every move of the unpromoted one, so the promotion mates whenever
    /// the twin does (owner 2026-11-16, solver 1.7.0).  Silver, knight and lance
    /// lose moves when they promote and keep the plain order.
    pub const PROMOTION_FIRST: [u8; 3] =
        [crate::piece::PT_P, crate::piece::PT_B, crate::piece::PT_R];

    /// Puts the promotion of a bishop, rook or pawn board move right before
    /// its non-promoting twin wherever the two stand side by side with the
    /// non-promotion first -- which is how both USI order and the move
    /// generator's order place them -- and leaves every other move where it
    /// is.  A search-order change only (solver 1.7.0, owner 2026-11-16): the
    /// df-pn provers expand an attack node's checks in this order, so a mate
    /// that both twins give is usually proven through the stronger promoted
    /// piece first.  Verdicts never depend on it; node counts do.  Mirrors
    /// `promotionFirst` in `packages/solver/src/cdfpn.ts`.
    pub fn promotion_first(moves: &mut [Move]) {
        let mut index = 0;
        while index + 1 < moves.len() {
            let (a, b) = (moves[index], moves[index + 1]);
            if !a.is_drop()
                && !a.promote
                && b.promote
                && a.from == b.from
                && a.to == b.to
                && Self::PROMOTION_FIRST.contains(&a.piece)
            {
                moves.swap(index, index + 1);
                index += 2;
            } else {
                index += 1;
            }
        }
    }

    /// Parses USI text.  Board moves keep `piece = PT_P` exactly like
    /// `moveFromUsi` in the TypeScript rules: the mover is read from the board.
    pub fn from_usi(text: &str) -> Result<Move, String> {
        let b = text.as_bytes();
        let square = |file: u8, rank: u8| -> Result<u8, String> {
            if !(b'1'..=b'9').contains(&file) || !(b'a'..=b'i').contains(&rank) {
                return Err(format!("Invalid USI square: {text}"));
            }
            Ok((rank - b'a') * 9 + (9 - (file - b'0')))
        };
        if b.len() == 4 && b[1] == b'*' {
            let piece = PIECE_LETTERS
                .iter()
                .position(|&letter| letter == b[0])
                .ok_or_else(|| format!("Invalid USI move: {text}"))?;
            return Ok(Move::drop(piece as u8, square(b[2], b[3])?));
        }
        if b.len() == 4 || (b.len() == 5 && b[4] == b'+') {
            return Ok(Move::board(
                square(b[0], b[1])?,
                square(b[2], b[3])?,
                PT_P,
                b.len() == 5,
            ));
        }
        Err(format!("Invalid USI move: {text}"))
    }
}

/// The two USI bytes of each square (file digit, rank letter) as
/// `first << 8 | second`.
const USI_SQUARE: [u16; 81] = {
    let mut table = [0u16; 81];
    let mut square = 0;
    while square < 81 {
        let file = b'9' - (square % 9) as u8;
        let rank = b'a' + (square / 9) as u8;
        table[square] = ((file as u16) << 8) | rank as u16;
        square += 1;
    }
    table
};

/// The first two USI bytes of a drop of each piece type (letter, `*`) as
/// `letter << 8 | b'*'`.
const USI_DROP: [u16; 8] = {
    let mut table = [0u16; 8];
    let mut piece = 0;
    while piece < 8 {
        table[piece] = ((PIECE_LETTERS[piece] as u16) << 8) | b'*' as u16;
        piece += 1;
    }
    table
};

/// A stack-allocated USI string.  Ordering is bytewise, matching the JavaScript
/// `<` comparison the TypeScript solver sorts with.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Usi {
    buf: [u8; 5],
    len: u8,
}

impl Usi {
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len as usize]
    }
    #[inline]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).expect("ascii")
    }
}

impl std::fmt::Display for Usi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the king of the side to move stands relative to its attackers.
#[derive(Clone, Copy, Debug)]
pub struct CheckInfo {
    /// Number of enemy pieces attacking the king (0 when there is no king).
    pub checkers: u32,
    /// Empty squares strictly between the king and a single sliding checker.
    pub between: u128,
}

impl CheckInfo {
    #[inline]
    pub const fn in_check(&self) -> bool {
        self.checkers > 0
    }
}

/// The king of one colour against the other colour's pieces
/// ([`Position::king_scan`]).
#[derive(Clone, Copy, Debug)]
pub struct KingScan {
    /// Square of the king ([`Position::king_square`]).
    pub king: usize,
    /// Enemy pieces attacking the king.
    pub checkers: u32,
    /// Their squares.
    pub checker_squares: u128,
    /// [`CheckInfo::between`].
    pub between: u128,
    /// Own pieces that alone stand between the king and an enemy slider aimed
    /// at it (only when asked for).
    pub pinned: u128,
}

/// Information required to undo a move.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    moved: u8,
    captured: u8,
    /// The hash before the move: [`Position::unmake`] restores it instead of
    /// undoing each key (2026-10-27).
    hash: u128,
}

/// A shogi position: 81-square mailbox, two hands, side to move.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Position {
    pub board: [u8; 81],
    /// `[colour][hand piece]` counts in `P L N S G B R` order.
    pub hands: [[u8; 7]; 2],
    pub side: u8,
    king: [u8; 2],
    /// Set when a colour has more than one king; king lookups then rescan so
    /// behaviour matches the TypeScript `findKing`, which returns the first one.
    multi_king: bool,
    hash: u128,
    /// The squares of each colour's pieces (2026-10-26, kept with `board`).
    occ: [u128; 2],
    /// The squares of each piece kind ([`crate::bitboard::kind`]), both colours.
    kinds: [u128; 16],
}

impl Default for Position {
    fn default() -> Self {
        Self::empty()
    }
}

impl Position {
    pub fn empty() -> Self {
        Self {
            board: [EMPTY; 81],
            hands: [[0; 7]; 2],
            side: BLACK,
            king: [NO_SQUARE; 2],
            multi_king: false,
            hash: 0,
            occ: [0; 2],
            kinds: [0; 16],
        }
    }

    /// Rebuilds derived state (king squares, hash) after direct board edits.
    pub fn refresh(&mut self) {
        self.king = [NO_SQUARE; 2];
        self.multi_king = false;
        self.hash = 0;
        self.occ = [0; 2];
        self.kinds = [0; 16];
        let keys = &KEYS;
        for (square, &c) in self.board.iter().enumerate() {
            if c == EMPTY {
                continue;
            }
            self.hash ^= keys.piece[c as usize][square];
            self.occ[color_of(c) as usize] |= bit(square);
            self.kinds[kind(c)] |= bit(square);
            if ptype(c) == PT_K {
                let color = color_of(c) as usize;
                if self.king[color] == NO_SQUARE {
                    self.king[color] = square as u8;
                } else {
                    self.multi_king = true;
                }
            }
        }
        for color in 0..2 {
            for piece in 0..7 {
                self.hash ^= keys.hand[color][piece][self.hands[color][piece] as usize];
            }
        }
        if self.side == WHITE {
            self.hash ^= keys.white_to_move;
        }
    }

    #[inline]
    pub fn hash(&self) -> u128 {
        self.hash
    }

    /// Square of the king of `color`, or [`NO_SQUARE`].
    #[inline]
    pub fn king_square(&self, color: u8) -> u8 {
        if self.multi_king {
            self.board
                .iter()
                .position(|&c| c != EMPTY && ptype(c) == PT_K && color_of(c) == color)
                .map_or(NO_SQUARE, |i| i as u8)
        } else {
            self.king[color as usize]
        }
    }

    /// `true` when `color` has more than one king on the board.
    #[inline]
    pub fn has_several_kings(&self, color: u8) -> bool {
        self.multi_king
            && self
                .board
                .iter()
                .filter(|&&c| c != EMPTY && ptype(c) == PT_K && color_of(c) == color)
                .count()
                > 1
    }

    /// Every occupied square.
    #[inline]
    pub fn occupied(&self) -> u128 {
        self.occ[0] | self.occ[1]
    }

    /// The squares of `color`'s pieces.
    #[inline]
    pub fn pieces(&self, color: u8) -> u128 {
        self.occ[color as usize]
    }

    /// The squares of `color`'s pieces of kind `k` ([`crate::bitboard::kind`]).
    #[inline]
    pub fn kind_of(&self, color: u8, k: usize) -> u128 {
        self.kinds[k] & self.occ[color as usize]
    }

    /// `true` when the bitboards agree with `board` (tests).
    pub fn bitboards_consistent(&self) -> bool {
        let mut copy = self.clone();
        copy.refresh();
        copy.occ == self.occ && copy.kinds == self.kinds
    }

    /// Sets a hand count directly (used by SFEN parsing and normalisation).
    ///
    /// # Panics
    /// Panics above [`MAX_HAND_COUNT`], which no legal shogi set can reach.
    pub fn set_hand(&mut self, color: u8, piece: u8, count: usize) {
        assert!(count <= MAX_HAND_COUNT, "hand count out of range");
        let keys = &KEYS;
        self.hash ^= keys.hand[color as usize][piece as usize]
            [self.hands[color as usize][piece as usize] as usize];
        self.hands[color as usize][piece as usize] = count as u8;
        self.hash ^= keys.hand[color as usize][piece as usize][count];
    }

    /// Applies a move generated for this position.  The caller guarantees the
    /// move came from [`crate::movegen`], exactly like `applyGeneratedMove`.
    pub fn make(&mut self, m: Move) -> Undo {
        let color = self.side;
        let mut undo = Undo {
            moved: EMPTY,
            captured: EMPTY,
            hash: self.hash,
        };
        // The new hash in one pass over the keys (2026-10-27; it toggled
        // each key as the pieces moved).
        let hash = self.hash_after(m);
        if m.is_drop() {
            self.hands[color as usize][m.piece as usize] =
                self.hands[color as usize][m.piece as usize].wrapping_sub(1);
            self.place(m.to as usize, code(color, m.piece, false));
        } else {
            let moved = self.board[m.from as usize];
            self.lift(m.from as usize);
            undo.moved = moved;
            let target = self.board[m.to as usize];
            if target != EMPTY {
                undo.captured = target;
                self.lift(m.to as usize);
                // A captured promoted piece reverts to its base type in hand.
                self.hands[color as usize][ptype(target) as usize] =
                    self.hands[color as usize][ptype(target) as usize].wrapping_add(1);
            }
            let placed = if m.promote {
                moved | FLAG_PROMOTED
            } else {
                moved
            };
            self.place(m.to as usize, placed);
        }
        self.side = opponent(self.side);
        self.hash = hash;
        undo
    }

    /// [`Position::hash`] after [`Position::make`]`(m)`, without making it
    /// (2026-10-27: the prober keys every child it pushes).
    #[inline]
    pub fn hash_after(&self, m: Move) -> u128 {
        let keys = &KEYS;
        let color = self.side as usize;
        let mut hash = self.hash ^ keys.white_to_move;
        let to = m.to as usize;
        if m.is_drop() {
            let piece = m.piece as usize;
            // A hand count wraps as a `u8`, exactly as `make` updates it
            // (a caller may drop from an empty hand: the prestrip's probes).
            let count = self.hands[color][piece];
            hash ^= keys.hand[color][piece][count as usize]
                ^ keys.hand[color][piece][count.wrapping_sub(1) as usize];
            hash ^= keys.piece[code(self.side, m.piece, false) as usize][to];
        } else {
            let moved = self.board[m.from as usize];
            hash ^= keys.piece[moved as usize][m.from as usize];
            let target = self.board[to];
            if target != EMPTY {
                hash ^= keys.piece[target as usize][to];
                let piece = ptype(target) as usize;
                let count = self.hands[color][piece];
                hash ^= keys.hand[color][piece][count as usize]
                    ^ keys.hand[color][piece][count.wrapping_add(1) as usize];
            }
            let placed = if m.promote {
                moved | FLAG_PROMOTED
            } else {
                moved
            };
            hash ^= keys.piece[placed as usize][to];
        }
        hash
    }

    /// Exact inverse of [`Position::make`].
    pub fn unmake(&mut self, m: Move, undo: Undo) {
        self.side = opponent(self.side);
        let color = self.side;
        if m.is_drop() {
            self.lift(m.to as usize);
            self.hands[color as usize][m.piece as usize] =
                self.hands[color as usize][m.piece as usize].wrapping_add(1);
        } else {
            self.lift(m.to as usize);
            if undo.captured != EMPTY {
                self.hands[color as usize][ptype(undo.captured) as usize] =
                    self.hands[color as usize][ptype(undo.captured) as usize].wrapping_sub(1);
                self.place(m.to as usize, undo.captured);
            }
            self.place(m.from as usize, undo.moved);
        }
        self.hash = undo.hash;
    }

    /// Puts `c` on `square` (board and bitboards; the callers set the hash).
    #[inline]
    fn place(&mut self, square: usize, c: u8) {
        self.board[square] = c;
        self.occ[color_of(c) as usize] ^= bit(square);
        self.kinds[kind(c)] ^= bit(square);
        if ptype(c) == PT_K {
            self.king[color_of(c) as usize] = square as u8;
        }
    }

    /// Empties `square` (board and bitboards; the callers set the hash).
    #[inline]
    fn lift(&mut self, square: usize) {
        let c = self.board[square];
        self.board[square] = EMPTY;
        self.occ[color_of(c) as usize] ^= bit(square);
        self.kinds[kind(c)] ^= bit(square);
    }

    /// Immutable apply, for callers that prefer value semantics.
    pub fn applied(&self, m: Move) -> Position {
        let mut next = self.clone();
        next.make(m);
        next
    }

    /// `true` when `by` attacks `target`.  Mirrors `isSquareAttacked`, including
    /// its treatment of the occupant of the target square as irrelevant.
    #[inline]
    pub fn is_attacked(&self, target: u8, by: u8) -> bool {
        self.is_attacked_with(target as usize, by, self.occupied())
    }

    /// [`Position::is_attacked`] with `occupied` as the board's pieces (a
    /// square left out counts as empty; `by`'s pieces are still read from the
    /// board, so leave out only a square of the other colour).
    ///
    /// A piece of `by` on `s` steps onto `target` exactly when the same piece
    /// of the other colour on `target` steps onto `s`, so the step attackers
    /// are read off the other colour's step tables at `target`; the sliders off
    /// the rays from `target` to their first piece.
    pub fn is_attacked_with(&self, target: usize, by: u8, occupied: u128) -> bool {
        let g = &GEOMETRY;
        let own = self.occ[by as usize];
        let me = opponent(by);
        let k = &self.kinds;
        let step = |t: u8, promoted: bool| g.step[code(me, t, promoted) as usize][target];
        let golds = k[PT_G as usize]
            | k[promoted_kind(PT_P)]
            | k[promoted_kind(PT_L)]
            | k[promoted_kind(PT_N)]
            | k[promoted_kind(PT_S)];
        if (step(PT_P, false) & k[PT_P as usize]
            | step(PT_N, false) & k[PT_N as usize]
            | step(PT_S, false) & k[PT_S as usize]
            | step(PT_G, false) & golds
            | step(PT_K, false) & k[PT_K as usize]
            | step(PT_B, true) & k[promoted_kind(PT_B)]
            | step(PT_R, true) & k[promoted_kind(PT_R)])
            & own
            != 0
        {
            return true;
        }
        let lances = k[PT_L as usize] & own;
        // A lance of `by` reaches `target` from behind it: from higher ranks
        // for black ([`DIRS8`] index 6, `(0, 1)`), from lower ones for white
        // (index 1).
        if lances != 0
            && ray_attack(g, target, if by == BLACK { 6 } else { 1 }, occupied) & lances != 0
        {
            return true;
        }
        let diagonal = (k[PT_B as usize] | k[promoted_kind(PT_B)]) & own;
        if diagonal != 0 && bishop_attack(g, target, occupied) & diagonal != 0 {
            return true;
        }
        let orthogonal = (k[PT_R as usize] | k[promoted_kind(PT_R)]) & own;
        orthogonal != 0 && rook_attack(g, target, occupied) & orthogonal != 0
    }

    /// The squares of `mask` that `by` attacks with `occupied` as the board's
    /// pieces: [`Position::is_attacked_with`] for each square of `mask` at
    /// once, by gathering the attacks of `by`'s pieces (sliders only along the
    /// lines that reach `mask`).
    pub fn attacked_in(&self, by: u8, mask: u128, occupied: u128) -> u128 {
        self.attacked_in_by(self.occ[by as usize], mask, occupied)
    }

    /// [`Position::attacked_in`] by the pieces standing on `pieces` (each
    /// read from the board).
    pub fn attacked_in_by(&self, mut pieces: u128, mask: u128, occupied: u128) -> u128 {
        let g = &GEOMETRY;
        let tables = &TABLES;
        let mut out = 0u128;
        while pieces != 0 {
            let square = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let c = self.board[square] as usize;
            out |= g.step[c][square] & mask;
            let slides = tables.slide_mask[c];
            if slides != 0 {
                for dir in 0..8 {
                    if slides & (1 << dir) != 0 && g.ray[square][dir] & mask & !out != 0 {
                        out |= ray_attack(g, square, dir, occupied) & mask;
                    }
                }
            }
            if out == mask {
                break;
            }
        }
        out
    }

    /// The king of `color` against the other colour: its checkers, the empty
    /// squares between it and a single sliding checker, and (with `pins`) the
    /// own pieces pinned to it.  `None` without a king.
    ///
    /// Along each line from the king the first piece decides a check exactly
    /// as the square-by-square scan did (a slider aimed back at the king, or a
    /// step piece next to it), and an own first piece is pinned when the
    /// second is an enemy slider aimed at the king.
    pub fn king_scan(&self, color: u8, pins: bool) -> Option<KingScan> {
        let king = self.king_square(color);
        if king == NO_SQUARE {
            return None;
        }
        let king = king as usize;
        let by = opponent(color);
        let g = &GEOMETRY;
        let tables = &TABLES;
        let occupied = self.occupied();
        // A knight of `by` checks from where one of `color` on the king's
        // square would jump to.
        let knights =
            g.step[code(color, PT_N, false) as usize][king] & self.kind_of(by, PT_N as usize);
        let mut scan = KingScan {
            king,
            checkers: knights.count_ones(),
            checker_squares: knights,
            between: 0,
            pinned: 0,
        };
        for dir in 0..8 {
            let blockers = g.ray[king][dir] & occupied;
            if blockers == 0 {
                continue;
            }
            let first = nearest(blockers, dir);
            let c = self.board[first];
            let toward = 1u8 << (7 - dir);
            if color_of(c) == by {
                if tables.slide_mask[c as usize] & toward != 0
                    || (g.neighbours[king] & bit(first) != 0
                        && tables.step_mask[c as usize] & toward != 0)
                {
                    scan.checkers += 1;
                    scan.checker_squares |= bit(first);
                    scan.between = if scan.checkers == 1 {
                        g.between[king * 81 + first]
                    } else {
                        0
                    };
                }
            } else if pins {
                let rest = blockers & !bit(first);
                if rest != 0 {
                    let second = self.board[nearest(rest, dir)];
                    if color_of(second) == by && tables.slide_mask[second as usize] & toward != 0 {
                        scan.pinned |= bit(first);
                    }
                }
            }
        }
        if scan.checkers != 1 {
            scan.between = 0;
        }
        Some(scan)
    }

    /// Which enemy pieces attack the king of `color`, and where a drop could
    /// block a single sliding check.
    ///
    /// The solver leans on two facts this makes cheap to exploit:
    ///
    /// * A drop never exposes the king of the mover, because it adds a blocker and
    ///   moves nothing, so when the mover is not in check every drop is
    ///   self-check legal without testing.
    /// * When the mover is in check, only a drop strictly between the king and a
    ///   single sliding checker can answer the check, and such a drop always
    ///   answers it.  Two checkers, or one non-sliding checker, leave no legal
    ///   drop at all.
    ///
    /// Both facts are properties of shogi, not of the reference implementation, so
    /// the resulting legal-move set is identical to the clone-and-test version in
    /// `packages/core-rules`.
    pub fn check_info(&self, color: u8) -> CheckInfo {
        match self.king_scan(color, false) {
            None => CheckInfo {
                checkers: 0,
                between: 0,
            },
            Some(scan) => CheckInfo {
                checkers: scan.checkers,
                between: scan.between,
            },
        }
    }

    /// `true` when the king of `color` exists and is attacked.
    #[inline]
    pub fn is_in_check(&self, color: u8) -> bool {
        let king = self.king_square(color);
        king != NO_SQUARE && self.is_attacked(king, opponent(color))
    }

    /// Number of occupied squares.
    pub fn board_piece_count(&self) -> usize {
        self.occupied().count_ones() as usize
    }
}
