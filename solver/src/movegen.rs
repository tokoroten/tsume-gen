//! Pseudo-move generation and legality filtering.
//!
//! Generation order is the same as the TypeScript `pseudoMoves`: board squares in
//! SFEN order, movement directions in piece order, then drops by `P L N S G B R`
//! and square order.  Legality rejects self-check, then `uchi-fuzume` (a pawn
//! drop that gives check with no legal reply), matching the order documented in
//! `docs/rules.md`.

use crate::bitboard::{bit, nearest, ray_attack, Geometry, FULL_BOARD, GEOMETRY};
use crate::piece::*;
use crate::position::{CheckInfo, Move, Position, NO_SQUARE};

/// A reusable move stack.  One buffer serves a whole search: each node appends
/// its moves, uses them, then truncates, so movegen never allocates in the hot
/// loop.
#[derive(Default)]
pub struct MoveBuf {
    pub buf: Vec<Move>,
}

impl MoveBuf {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(1024),
        }
    }

    /// Appends every pseudo-legal move for the side to move.
    ///
    /// `info` restricts drops to the squares that can answer a check; see
    /// [`Position::check_info`].
    pub fn gen_pseudo(&mut self, pos: &Position, info: &CheckInfo) {
        self.gen_pseudo_masked(pos, info, FULL_BOARD);
    }

    /// [`MoveBuf::gen_pseudo`] leaving out the board moves of pieces other than
    /// the king (the one [`Position::king_square`] returns) whose destination is
    /// outside `targets`.
    fn gen_pseudo_masked(&mut self, pos: &Position, info: &CheckInfo, targets: u128) {
        let color = pos.side;
        let tables = &TABLES;
        let king = pos.king_square(color) as usize;
        let g = &GEOMETRY;
        let occupied = pos.occupied();
        let own_or_kings = pos.pieces(color) | pos.kind_of(opponent(color), PT_K as usize);
        let mut own = pos.pieces(color);
        while own != 0 {
            let from = own.trailing_zeros() as usize;
            own &= own - 1;
            let c = pos.board[from];
            let reach = if from == king || ptype(c) == PT_K {
                FULL_BOARD
            } else {
                targets
            };
            // A king is never a capturable piece; mate ends the game first.
            let wanted = reach & !own_or_kings;
            if wanted == 0 {
                continue;
            }
            let promoted = is_promoted(c);
            let t = ptype(c);
            let from_zone = promotion_zone(color, rank_of(from));
            let mut push = |to: usize| {
                let r = rank_of(to);
                if !promoted && must_promote(t, color, r) {
                    self.buf.push(Move::board(from as u8, to as u8, t, true));
                } else {
                    self.buf.push(Move::board(from as u8, to as u8, t, false));
                    if !promoted && promotable_type(t) && (from_zone || promotion_zone(color, r)) {
                        self.buf.push(Move::board(from as u8, to as u8, t, true));
                    }
                }
            };
            if reach != FULL_BOARD {
                // Under check only a few squares are wanted: take the walk's
                // squares among them as bitboards.
                for d in 0..tables.dirs_len[c as usize] as usize {
                    let (dir, reached) = walk(g, tables, c, from, d, occupied);
                    let mut hits = reached & wanted;
                    while hits != 0 {
                        let to = nearest(hits, dir);
                        hits &= !bit(to);
                        push(to);
                    }
                }
                continue;
            }
            // Every square wanted: the plain walk is cheaper.
            let (ff, fr) = (file_of(from) as i8, rank_of(from) as i8);
            for &(dx, dy, slide) in &tables.dirs[c as usize][..tables.dirs_len[c as usize] as usize]
            {
                let (mut f, mut r) = (ff + dx, fr + dy);
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let to = (r * 9 + f) as usize;
                    let occupant = pos.board[to];
                    if occupant != EMPTY && color_of(occupant) == color {
                        break;
                    }
                    if occupant == EMPTY || ptype(occupant) != PT_K {
                        push(to);
                    }
                    if occupant != EMPTY || !slide {
                        break;
                    }
                    f += dx;
                    r += dy;
                }
            }
        }
        if info.checkers > 1 {
            // Nothing dropped can answer a double check.
            return;
        }
        for &t in HAND_PIECES.iter() {
            if pos.hands[color as usize][t as usize] == 0 {
                continue;
            }
            if info.in_check() {
                let mut candidates = info.between;
                while candidates != 0 {
                    let to = candidates.trailing_zeros() as usize;
                    candidates &= candidates - 1;
                    self.push_drop(pos, color, t, to);
                }
            } else {
                let mut empty = !pos.occupied() & FULL_BOARD;
                while empty != 0 {
                    let to = empty.trailing_zeros() as usize;
                    empty &= empty - 1;
                    self.push_drop(pos, color, t, to);
                }
            }
        }
    }

    /// Pushes a drop unless it lands on an immobile rank or doubles a pawn.
    #[inline]
    fn push_drop(&mut self, pos: &Position, color: u8, t: u8, to: usize) {
        debug_assert_eq!(pos.board[to], EMPTY);
        if must_promote(t, color, rank_of(to)) {
            return;
        }
        if t == PT_P && has_unpromoted_pawn(pos, color, file_of(to)) {
            return;
        }
        self.buf.push(Move::drop(t, to as u8));
    }

    /// Appends every legal move for the side to move and returns its range in
    /// [`MoveBuf::buf`].  The caller truncates back to `range.start` when done.
    ///
    /// `enforce_pawn_drop_mate` mirrors `GenerateOptions.enforcePawnDropMate`; the
    /// recursive call that tests for `uchi-fuzume` passes `false`.
    pub fn gen_legal(
        &mut self,
        pos: &mut Position,
        enforce_pawn_drop_mate: bool,
    ) -> std::ops::Range<usize> {
        let safety = Safety::new(pos);
        self.gen_legal_with(pos, enforce_pawn_drop_mate, &safety)
    }

    /// [`MoveBuf::gen_legal`] with `safety`, which must be [`Safety::new`]`(pos)`.
    pub fn gen_legal_with(
        &mut self,
        pos: &mut Position,
        enforce_pawn_drop_mate: bool,
        safety: &Safety,
    ) -> std::ops::Range<usize> {
        let base = self.buf.len();
        let color = pos.side;
        let info = safety.check_info();
        self.gen_pseudo_masked(pos, &info, safety.targets);
        let end = self.buf.len();
        let mut write = base;
        // Whether the opponent is in check before any drop (asked once, at the
        // first pawn drop): a position that is not a legal game position.
        let mut already_in_check: Option<bool> = None;
        for read in base..end {
            let m = self.buf[read];
            let mut legal = true;
            if m.is_drop() {
                // Generated drops are always self-check legal (see `check_info`);
                // only `uchi-fuzume` can still reject one.
                if enforce_pawn_drop_mate && m.piece == PT_P {
                    let already =
                        *already_in_check.get_or_insert_with(|| pos.is_in_check(opponent(color)));
                    let undo = pos.make(m);
                    if pos.is_in_check(opponent(color)) {
                        legal = if already {
                            // The check need not be the pawn's: every legal
                            // answer counts, drops included (Codex review of
                            // stage2c, 2026-10-08).
                            let inner = self.gen_legal(pos, false);
                            let any = inner.end > inner.start;
                            self.buf.truncate(inner.start);
                            any
                        } else {
                            self.has_legal_reply_to_pawn_drop(pos)
                        };
                    }
                    pos.unmake(m, undo);
                }
            } else {
                legal = safety.is_legal(pos, m);
            }
            if legal {
                self.buf[write] = m;
                write += 1;
            }
        }
        self.buf.truncate(write);
        base..write
    }

    /// Whether the side to move, just checked by a pawn drop, has any legal
    /// move -- `gen_legal(pos, false)` non-empty.  Only when it was not in check
    /// before the drop: a drop opens no line, so the pawn is then the only
    /// checker, it checks from the adjacent square, nothing can be dropped
    /// between it and the king, and only board moves can answer.
    fn has_legal_reply_to_pawn_drop(&mut self, pos: &mut Position) -> bool {
        self.has_legal_board_move(pos)
    }

    /// `true` when [`MoveBuf::gen_legal`] would return at least one board move
    /// (not a drop) for the side to move.  Stops at the first one found, trying
    /// the king's steps first; never touches `self.buf`.
    ///
    /// When the mover is in check, a move of a piece other than its king can
    /// only be legal by capturing an enemy piece or landing between the king and
    /// a single sliding checker, so other destinations are not tried (every
    /// candidate is still played to test legality).  A second king of the same
    /// colour is tried everywhere, because moving it can change which king
    /// counts.
    pub fn has_legal_board_move(&mut self, pos: &mut Position) -> bool {
        let safety = Safety::new(pos);
        self.has_legal_board_move_with(pos, &safety)
    }

    /// [`MoveBuf::has_legal_board_move`] with `safety`, which must be
    /// [`Safety::new`]`(pos)`.
    pub fn has_legal_board_move_with(&mut self, pos: &mut Position, safety: &Safety) -> bool {
        let color = pos.side;
        let tables = &TABLES;
        let king = safety.king;
        if king != NONE {
            let c = pos.board[king];
            let (ff, fr) = (file_of(king) as i8, rank_of(king) as i8);
            for &(dx, dy, _) in &tables.dirs[c as usize][..tables.dirs_len[c as usize] as usize] {
                let (f, r) = (ff + dx, fr + dy);
                if !(0..9).contains(&f) || !(0..9).contains(&r) {
                    continue;
                }
                let to = (r * 9 + f) as usize;
                let occupant = pos.board[to];
                if occupant != EMPTY && (color_of(occupant) == color || ptype(occupant) == PT_K) {
                    continue;
                }
                if safety.is_legal(pos, Move::board(king as u8, to as u8, PT_K, false)) {
                    return true;
                }
            }
        }
        let mut own = pos.pieces(color);
        if king != NONE {
            own &= !bit(king);
        }
        while own != 0 {
            let from = own.trailing_zeros() as usize;
            own &= own - 1;
            let c = pos.board[from];
            let targets = if ptype(c) == PT_K {
                FULL_BOARD
            } else {
                safety.targets
            };
            if targets == 0 {
                continue;
            }
            let promoted = is_promoted(c);
            let t = ptype(c);
            let (ff, fr) = (file_of(from) as i8, rank_of(from) as i8);
            for &(dx, dy, slide) in &tables.dirs[c as usize][..tables.dirs_len[c as usize] as usize]
            {
                let (mut f, mut r) = (ff + dx, fr + dy);
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let to = (r * 9 + f) as usize;
                    let occupant = pos.board[to];
                    if occupant != EMPTY && color_of(occupant) == color {
                        break;
                    }
                    if (occupant == EMPTY || ptype(occupant) != PT_K)
                        && targets & (1u128 << to) != 0
                    {
                        // Legality does not depend on promotion; a form that
                        // must promote is the only form generated.
                        let promote = !promoted && must_promote(t, color, r as usize);
                        if safety.is_legal(pos, Move::board(from as u8, to as u8, t, promote)) {
                            return true;
                        }
                    }
                    if occupant != EMPTY || !slide {
                        break;
                    }
                    f += dx;
                    r += dy;
                }
            }
        }
        false
    }

    /// Appends every legal move of the side to move that **gives check**, and
    /// returns its range in [`MoveBuf::buf`].
    ///
    /// The result is exactly the subsequence of [`MoveBuf::gen_legal`] whose
    /// moves leave the opponent in check -- same set, same relative order
    /// (board moves by origin square and piece direction, then drops by
    /// `P L N S G B R` and square) -- but it never generates the other moves and
    /// never plays a move to find out whether it checks:
    ///
    /// * a board move checks when the moved piece (promoted or not) attacks the
    ///   enemy king from its destination, the origin square counted as vacated,
    ///   or when it uncovers a slider of the mover standing behind it on a line
    ///   through the king and leaves that line;
    /// * a drop checks when the dropped piece attacks the king, so only those
    ///   squares are tried.
    ///
    /// Self-check is tested (make / unmake) only on these candidates and only
    /// when the mover has a king; `uchi-fuzume`, double pawns, immobile pieces
    /// and promotion follow [`MoveBuf::gen_legal`] verbatim.
    pub fn gen_checks(
        &mut self,
        pos: &mut Position,
        enforce_pawn_drop_mate: bool,
    ) -> std::ops::Range<usize> {
        let base = self.buf.len();
        let ctx = match CheckCtx::scan(pos) {
            Scan::NoKing => return base..base,
            Scan::AlreadyInCheck => {
                return self.gen_checks_by_playing(pos, enforce_pawn_drop_mate);
            }
            Scan::Ready(ctx) => ctx,
        };
        let mut own = pos.pieces(ctx.color);
        while own != 0 {
            let from = own.trailing_zeros() as usize;
            own &= own - 1;
            self.push_piece_checks(pos, &ctx, from, FULL_BOARD);
        }
        let color = ctx.color;
        let info = pos.check_info(color);
        if info.checkers > 1 {
            // Nothing dropped can answer a double check.
            return base..self.buf.len();
        }
        // `drop_check_squares` are empty squares already.
        let allowed = if info.in_check() {
            info.between
        } else {
            FULL_BOARD
        };
        for &t in HAND_PIECES.iter() {
            if pos.hands[color as usize][t as usize] == 0 {
                continue;
            }
            let mut targets = ctx.drop_check_squares(pos, code(color, t, false)) & allowed;
            while targets != 0 {
                let to = targets.trailing_zeros() as usize;
                targets &= targets - 1;
                if must_promote(t, color, rank_of(to)) {
                    continue;
                }
                if t == PT_P && has_unpromoted_pawn(pos, color, file_of(to)) {
                    continue;
                }
                let m = Move::drop(t, to as u8);
                if enforce_pawn_drop_mate && t == PT_P {
                    // The drop checks by construction; reject it when the
                    // opponent then has no legal move (`uchi-fuzume`).
                    let undo = pos.make(m);
                    let legal = self.has_legal_reply_to_pawn_drop(pos);
                    pos.unmake(m, undo);
                    if !legal {
                        continue;
                    }
                }
                self.buf.push(m);
            }
        }
        base..self.buf.len()
    }

    /// The reference definition, for a position whose side *not* to move is
    /// already in check (never reached from a legal search): every legal move,
    /// played, kept when the opponent is in check afterwards.
    fn gen_checks_by_playing(
        &mut self,
        pos: &mut Position,
        enforce_pawn_drop_mate: bool,
    ) -> std::ops::Range<usize> {
        let color = pos.side;
        let range = self.gen_legal(pos, enforce_pawn_drop_mate);
        let mut write = range.start;
        for read in range.clone() {
            let m = self.buf[read];
            let undo = pos.make(m);
            let checks = pos.is_in_check(opponent(color));
            pos.unmake(m, undo);
            if checks {
                self.buf[write] = m;
                write += 1;
            }
        }
        self.buf.truncate(write);
        range.start..write
    }

    /// Appends the legal checking board moves of the piece on `from` whose
    /// destination lies in `to_mask`, in generation order.
    fn push_piece_checks(
        &mut self,
        pos: &mut Position,
        ctx: &CheckCtx,
        from: usize,
        to_mask: u128,
    ) {
        let tables = &TABLES;
        let c = pos.board[from];
        let color = ctx.color;
        let promoted = is_promoted(c);
        let t = ptype(c);
        // Squares that keep the uncovered line blocked, when `from` is a
        // discoverer: every other destination checks.
        let block = ctx.discoverer_block(from);
        if block.is_none() && !may_check_directly(c, from, ctx.king) {
            return;
        }
        // Where each form attacks the king from, `from` vacated.
        let vacated = pos.occupied() & !bit(from);
        let checks = [
            ctx.checks_from(c, vacated),
            if !promoted && promotable_type(t) {
                ctx.checks_from(c | FLAG_PROMOTED, vacated)
            } else {
                0
            },
        ];
        // The destinations that can check: every one off the uncovered line
        // of a discoverer, else those where some form attacks the king; never
        // an own piece or a king (mate ends the game first).
        let occupied = pos.occupied();
        let wanted = to_mask
            & if block.is_some() {
                FULL_BOARD
            } else {
                checks[0] | checks[1]
            }
            & !pos.pieces(color)
            & !pos.kind_of(opponent(color), PT_K as usize);
        if wanted == 0 {
            return;
        }
        let g = &GEOMETRY;
        let from_zone = promotion_zone(color, rank_of(from));
        for d in 0..tables.dirs_len[c as usize] as usize {
            let (dir, reached) = walk(g, tables, c, from, d, occupied);
            let mut hits = reached & wanted;
            while hits != 0 {
                let to = nearest(hits, dir);
                hits &= !bit(to);
                let r = rank_of(to);
                let discovered = block.is_some_and(|b| b & bit(to) == 0);
                let direct = |promote: bool| checks[usize::from(promote)] & bit(to) != 0;
                if !promoted && must_promote(t, color, r) {
                    self.push_checked(pos, ctx, from, to, t, true, discovered || direct(true));
                } else {
                    self.push_checked(pos, ctx, from, to, t, false, discovered || direct(false));
                    if !promoted && promotable_type(t) && (from_zone || promotion_zone(color, r)) {
                        self.push_checked(pos, ctx, from, to, t, true, discovered || direct(true));
                    }
                }
            }
        }
    }

    /// Pushes the board move `from -> to` when it `checks` and is legal.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn push_checked(
        &mut self,
        pos: &mut Position,
        ctx: &CheckCtx,
        from: usize,
        to: usize,
        t: u8,
        promote: bool,
        checks: bool,
    ) {
        if !checks {
            return;
        }
        let m = Move::board(from as u8, to as u8, t, promote);
        if ctx.own_king {
            let undo = pos.make(m);
            let legal = !pos.is_in_check(ctx.color);
            pos.unmake(m, undo);
            if !legal {
                return;
            }
        }
        self.buf.push(m);
    }
}

/// "No square" as an index.
const NONE: usize = NO_SQUARE as usize;

/// The safety of the side to move's king, enough to decide whether a board move
/// leaves it in check without playing the move -- the verdict of make /
/// [`Position::is_in_check`] / unmake, which is still used for the cases this
/// does not cover (pinned pieces, several kings of one colour).
///
/// Built once per position ([`Safety::new`]), it can be handed to
/// [`MoveBuf::gen_legal_with`], [`MoveBuf::has_legal_board_move_with`] and
/// [`Safety::check_line`] for that same position, so a defence node scans its
/// king once (2026-10-27).
#[derive(Clone, Copy, Debug)]
pub struct Safety {
    /// Square of the king ([`Position::king_square`]), or [`NONE`].
    king: usize,
    /// The mover has more than one king: which one counts can change with a
    /// move, so every board move is played.
    multi: bool,
    /// Enemy pieces attacking the king, counted like [`Position::check_info`].
    checkers: u32,
    /// [`CheckInfo::between`].
    between: u128,
    /// The checkers' squares.
    checker_squares: u128,
    /// The destinations a piece other than the king may have in a legal move:
    /// every square when not in check, the checker's square and `between` under
    /// a single check, none under a double check (any other move leaves a
    /// checker's attack in place).
    targets: u128,
    /// Own pieces that alone stand between the king and an enemy slider aimed
    /// at it.
    pinned: u128,
    /// The king's neighbours the enemy attacks once the king has left its
    /// square ([`Position::attacked_in`]): where it cannot step.
    covered: u128,
}

impl Safety {
    /// The safety of `pos.side`'s king in `pos`.
    pub fn new(pos: &Position) -> Self {
        let color = pos.side;
        let multi = pos.has_several_kings(color);
        let Some(scan) = pos.king_scan(color, true) else {
            return Safety {
                king: NONE,
                multi,
                checkers: 0,
                between: 0,
                checker_squares: 0,
                targets: FULL_BOARD,
                pinned: 0,
                covered: 0,
            };
        };
        let g = &GEOMETRY;
        let king = scan.king;
        let covered = pos.attacked_in(
            opponent(color),
            g.neighbours[king] & !pos.pieces(color),
            pos.occupied() & !bit(king),
        );
        Safety {
            king: scan.king,
            multi,
            checkers: scan.checkers,
            between: scan.between,
            checker_squares: scan.checker_squares,
            targets: match scan.checkers {
                0 => FULL_BOARD,
                1 => scan.between | scan.checker_squares,
                _ => 0,
            },
            pinned: scan.pinned,
            covered,
        }
    }

    /// [`crate::solver::sliding_check_line`] of the position, the mover being
    /// the defender: one checker with empty squares between it and the king.
    pub fn check_line(&self) -> Option<crate::solver::CheckLine> {
        (self.checkers == 1 && self.between != 0).then(|| crate::solver::CheckLine {
            checker: self.checker_squares.trailing_zeros() as u8,
            between: self.between,
        })
    }

    /// The same value as [`Position::check_info`].
    fn check_info(&self) -> CheckInfo {
        CheckInfo {
            checkers: self.checkers,
            between: if self.checkers == 1 { self.between } else { 0 },
        }
    }

    /// Whether the generated board move `m` leaves the mover's king safe.
    /// Moves of pieces other than the king must already lie in `targets`.
    #[inline]
    fn is_legal(&self, pos: &mut Position, m: Move) -> bool {
        if self.king == NONE {
            return true;
        }
        let from = m.from as usize;
        if self.multi || self.pinned & (1u128 << from) != 0 {
            let color = pos.side;
            let undo = pos.make(m);
            let legal = !pos.is_in_check(color);
            pos.unmake(m, undo);
            return legal;
        }
        if from == self.king {
            // The king steps to `to`: attacked there once it has left `from`
            // (the occupant of `to` never matters to the attack test).
            debug_assert_eq!(
                self.covered & bit(m.to as usize) != 0,
                pos.is_attacked_with(
                    m.to as usize,
                    opponent(pos.side),
                    pos.occupied() & !bit(from)
                )
            );
            return self.covered & bit(m.to as usize) == 0;
        }
        debug_assert!(self.targets & (1u128 << m.to) != 0);
        // Not pinned: moving it uncovers nothing, and under check it lands on
        // the checker or between it and the king.
        true
    }
}

/// The squares the piece `c` on `from` reaches along its `d`-th movement
/// ([`Tables::dirs`]) with `occupied` pieces on the board, up to and including
/// the first piece, and the [`DIRS8`] index to take them nearest first by
/// ([`nearest`]; a knight's jump is one square, any index does).  The walk of
/// the square-by-square generator, as a bitboard.
#[inline]
fn walk(
    g: &Geometry,
    tables: &Tables,
    c: u8,
    from: usize,
    d: usize,
    occupied: u128,
) -> (usize, u128) {
    let (dx, dy, slide) = tables.dirs[c as usize][d];
    match dir_index(dx, dy) {
        Some(dir) if slide => (dir, ray_attack(g, from, dir, occupied)),
        Some(dir) => (dir, g.ray[from][dir] & g.neighbours[from]),
        None => {
            let (f, r) = (file_of(from) as i8 + dx, rank_of(from) as i8 + dy);
            let to = if (0..9).contains(&f) && (0..9).contains(&r) {
                bit((r * 9 + f) as usize)
            } else {
                0
            };
            (7, to)
        }
    }
}

/// The [`DIRS8`] index of the step `(dx, dy)`, `None` for a knight's jump.
#[inline]
const fn dir_index(dx: i8, dy: i8) -> Option<usize> {
    if dy < -1 || dy > 1 {
        return None;
    }
    // Row-major over `dy`, `dx` in -1..=1, the centre left out.
    const INDEX: [u8; 9] = [0, 1, 2, 3, 0xFF, 4, 5, 6, 7];
    let index = INDEX[((dy + 1) * 3 + dx + 1) as usize];
    if index == 0xFF {
        None
    } else {
        Some(index as usize)
    }
}

/// A cheap necessary condition for the piece `c` on `from` to give a direct
/// check (its own attack, not a discovered one) with some move.  Step pieces
/// travel at most two ranks and check from at most two ranks away, so one
/// standing farther than four files or ranks from the king never can.
#[inline]
fn may_check_directly(c: u8, from: usize, king: usize) -> bool {
    let tables = &TABLES;
    if tables.slide_mask[c as usize] != 0
        || (!is_promoted(c)
            && promotable_type(ptype(c))
            && tables.slide_mask[(c | FLAG_PROMOTED) as usize] != 0)
    {
        return true;
    }
    let df = (file_of(from) as i8 - file_of(king) as i8).abs();
    let dr = (rank_of(from) as i8 - rank_of(king) as i8).abs();
    df <= 4 && dr <= 4
}

/// What [`MoveBuf::gen_checks`] needs to know about the enemy king.
struct CheckCtx {
    /// The side to move (the checking side).
    color: u8,
    /// Square of the enemy king.
    king: usize,
    /// The empty squares seen from the king along its eight lines (up to the
    /// first piece on each).
    visible: u128,
    /// `true` when the mover has a king, so a move may expose it.
    own_king: bool,
    /// Pieces of the mover that alone block a slider of the mover from the
    /// king: `(square, squares strictly between the king and that slider)`.
    discoverers: [(u8, u128); 8],
    discoverers_len: usize,
}

/// The outcome of [`CheckCtx::scan`].  Built on the stack and consumed at
/// once, so the size difference of the variants costs nothing.
#[allow(clippy::large_enum_variant)]
enum Scan {
    /// The opponent has no king: nothing can check it.
    NoKing,
    /// The opponent is in check already, with the mover to move.  Not a legal
    /// game position; [`MoveBuf::gen_checks`] then falls back to playing every
    /// legal move.
    AlreadyInCheck,
    Ready(CheckCtx),
}

impl CheckCtx {
    fn scan(pos: &Position) -> Scan {
        let color = pos.side;
        let king = pos.king_square(opponent(color));
        if king == NO_SQUARE {
            return Scan::NoKing;
        }
        let king = king as usize;
        let tables = &TABLES;
        let mut ctx = CheckCtx {
            color,
            king,
            visible: 0,
            own_king: pos.king_square(color) != NO_SQUARE,
            discoverers: [(0, 0); 8],
            discoverers_len: 0,
        };
        let g = &GEOMETRY;
        // A knight of the mover checks from where one of the other colour on
        // the king's square would jump to.
        if g.step[code(opponent(color), PT_N, false) as usize][king]
            & pos.kind_of(color, PT_N as usize)
            != 0
        {
            return Scan::AlreadyInCheck;
        }
        let occupied = pos.occupied();
        for dir in 0..8 {
            let ray = g.ray[king][dir];
            let blockers = ray & occupied;
            if blockers == 0 {
                ctx.visible |= ray;
                continue;
            }
            let first = nearest(blockers, dir);
            ctx.visible |= g.between[king * 81 + first];
            let c = pos.board[first];
            if color_of(c) != color {
                continue;
            }
            let toward = 1u8 << (7 - dir);
            if tables.slide_mask[c as usize] & toward != 0
                || (g.neighbours[king] & bit(first) != 0
                    && tables.step_mask[c as usize] & toward != 0)
            {
                return Scan::AlreadyInCheck;
            }
            let rest = blockers & !bit(first);
            if rest != 0 {
                let second = nearest(rest, dir);
                let c = pos.board[second];
                if color_of(c) == color && tables.slide_mask[c as usize] & toward != 0 {
                    ctx.discoverers[ctx.discoverers_len] =
                        (first as u8, g.between[king * 81 + second]);
                    ctx.discoverers_len += 1;
                }
            }
        }
        Scan::Ready(ctx)
    }

    /// The squares from which a piece of code `c` (of the mover) attacks the
    /// king, with `occupied` as the board's pieces: the squares it steps onto
    /// the king from (read off the other colour's step table at the king), and
    /// along each of its sliding directions the king's ray the other way up to
    /// and including its first piece.  The verdict of
    /// [`Position::is_attacked`] with the piece placed there.
    #[inline]
    fn checks_from(&self, c: u8, occupied: u128) -> u128 {
        let g = &GEOMETRY;
        let slides = TABLES.slide_mask[c as usize];
        let mut squares = g.step[(c ^ FLAG_WHITE) as usize][self.king];
        if slides != 0 {
            for dir in 0..8 {
                if slides & (1 << dir) != 0 {
                    squares |= ray_attack(g, self.king, 7 - dir, occupied);
                }
            }
        }
        squares
    }

    /// The empty squares a piece of code `c` could be dropped on to attack the
    /// king.
    fn drop_check_squares(&self, pos: &Position, c: u8) -> u128 {
        let tables = &TABLES;
        let g = &GEOMETRY;
        let king = self.king;
        if tables.is_knight[c as usize] {
            // A knight checks from where one of the other colour on the king's
            // square would jump to.
            return g.step[code(opponent(color_of(c)), PT_N, false) as usize][king]
                & !pos.occupied();
        }
        let mut squares = 0u128;
        for dir in 0..8 {
            // A piece on this ray reaches the king moving the opposite way.
            let toward = 1u8 << (7 - dir);
            // The squares seen from the king along the ray (`visible` holds
            // them up to the first piece).
            let line = g.ray[king][dir] & self.visible;
            if tables.slide_mask[c as usize] & toward != 0 {
                squares |= line;
            } else if tables.step_mask[c as usize] & toward != 0 {
                squares |= line & g.neighbours[king];
            }
        }
        squares
    }

    #[inline]
    fn discoverer_block(&self, from: usize) -> Option<u128> {
        self.discoverers[..self.discoverers_len]
            .iter()
            .find(|(square, _)| *square as usize == from)
            .map(|&(_, block)| block)
    }
}

/// The legal checking moves of the piece on `from` (of the side to move) onto
/// `to`, as the non-promoting (`[0]`) and promoting (`[1]`) forms -- the same
/// moves `gen_legal` would offer from `from` to `to` that leave the opponent in
/// check.  `buf` is used as scratch above its current end and restored.
pub fn checks_from_to(
    buf: &mut MoveBuf,
    pos: &mut Position,
    from: u8,
    to: u8,
) -> [Option<Move>; 2] {
    let mut forms = [None; 2];
    let c = pos.board[from as usize];
    if c == EMPTY || color_of(c) != pos.side {
        return forms;
    }
    let base = buf.buf.len();
    match CheckCtx::scan(pos) {
        Scan::NoKing => {}
        Scan::AlreadyInCheck => {
            buf.gen_checks_by_playing(pos, true);
        }
        Scan::Ready(ctx) => buf.push_piece_checks(pos, &ctx, from as usize, 1u128 << to),
    }
    for &m in &buf.buf[base..] {
        if m.from == from && m.to == to {
            forms[usize::from(m.promote)] = Some(m);
        }
    }
    buf.buf.truncate(base);
    forms
}

fn has_unpromoted_pawn(pos: &Position, color: u8, file: usize) -> bool {
    pos.kind_of(color, PT_P as usize) & GEOMETRY.file[file] != 0
}

/// `true` when, after the generated move `m` of the side to move, the other
/// side's king has a step to a square the mover does not attack -- a legal
/// board move, so [`MoveBuf::has_legal_board_move`] there is `true` -- found
/// without playing `m`: the board after `m` is rebuilt on the occupancy
/// bitboards only (2026-10-27).  `false` means "not found this way" (no such
/// step, no king, or several kings of that colour, where which king counts
/// can change), never "no legal move".
pub fn king_escapes_after(pos: &Position, m: Move) -> bool {
    let mover = pos.side;
    let defender = opponent(mover);
    let king = pos.king_square(defender);
    if king == NO_SQUARE || pos.has_several_kings(defender) {
        return false;
    }
    let king = king as usize;
    let g = &GEOMETRY;
    let to = m.to as usize;
    let (placed, vacated) = if m.is_drop() {
        (code(mover, m.piece, false), 0u128)
    } else {
        let c = pos.board[m.from as usize];
        (
            if m.promote { c | FLAG_PROMOTED } else { c },
            bit(m.from as usize),
        )
    };
    let occupied = (pos.occupied() & !vacated) | bit(to);
    // The king may step onto any neighbour but its own pieces (a captured one
    // on `to` is gone) and a king of the mover.
    let mover_kings = (pos.kind_of(mover, PT_K as usize) & !vacated)
        | if ptype(placed) == PT_K { bit(to) } else { 0 };
    let steps = g.neighbours[king] & !(pos.pieces(defender) & !bit(to)) & !mover_kings;
    if steps == 0 {
        return false;
    }
    // Attacked once the king has left its square: the mover's other pieces
    // and the moved one on `to`.
    let without_king = occupied & !bit(king);
    // The moved piece first (one piece), then the others only on the squares
    // it leaves open (2026-10-27: the same answer, fewer squares to cover).
    let mut covered = g.step[placed as usize][to] & steps;
    let slides = TABLES.slide_mask[placed as usize];
    if slides != 0 {
        for dir in 0..8 {
            if slides & (1 << dir) != 0 {
                covered |= ray_attack(g, to, dir, without_king) & steps;
            }
        }
    }
    let open = steps & !covered;
    if open == 0 {
        return false;
    }
    let others = pos.pieces(mover) & !vacated & !bit(to);
    open & !pos.attacked_in_by(others, open, without_king) != 0
}

/// Every legal move for the side to move, in TypeScript generation order.
pub fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut buf = MoveBuf::new();
    let mut work = pos.clone();
    let range = buf.gen_legal(&mut work, true);
    debug_assert_eq!(&work, pos, "gen_legal must restore the position");
    buf.buf[range].to_vec()
}

/// Every legal move for the side to move, sorted by USI bytes (solver order).
pub fn legal_moves_sorted(pos: &Position) -> Vec<Move> {
    let mut moves = legal_moves(pos);
    moves.sort_unstable_by_key(Move::usi_key);
    moves
}

/// Every legal checking move for the side to move, in generation order (the
/// order of [`legal_moves`] restricted to checks).
pub fn checking_moves(pos: &Position) -> Vec<Move> {
    let mut buf = MoveBuf::new();
    let mut work = pos.clone();
    let range = buf.gen_checks(&mut work, true);
    debug_assert_eq!(&work, pos, "gen_checks must restore the position");
    buf.buf[range].to_vec()
}

/// Every legal checking move for the side to move, sorted by USI bytes.
pub fn checking_moves_sorted(pos: &Position) -> Vec<Move> {
    let mut moves = checking_moves(pos);
    moves.sort_unstable_by_key(Move::usi_key);
    moves
}

/// `true` when `m` leaves the opponent of the mover in check.
pub fn gives_check(pos: &Position, m: Move) -> bool {
    let mut work = pos.clone();
    let mover = work.side;
    work.make(m);
    work.is_in_check(opponent(mover))
}

/// `true` when the side to move has a legal move whose USI text is `usi`.
pub fn is_legal_usi(pos: &Position, usi: &str) -> bool {
    legal_moves(pos).iter().any(|m| m.to_usi().as_str() == usi)
}

/// Counts legal move sequences of exactly `depth` plies.  Used to cross-check
/// movegen against the TypeScript engine.
pub fn perft(pos: &Position, depth: u32) -> u64 {
    let mut buf = MoveBuf::new();
    let mut work = pos.clone();
    perft_inner(&mut work, &mut buf, depth)
}

fn perft_inner(pos: &mut Position, buf: &mut MoveBuf, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let range = buf.gen_legal(pos, true);
    if depth == 1 {
        let count = range.len() as u64;
        buf.buf.truncate(range.start);
        return count;
    }
    let mut total = 0u64;
    for index in range.clone() {
        let m = buf.buf[index];
        let undo = pos.make(m);
        total += perft_inner(pos, buf, depth - 1);
        pos.unmake(m, undo);
    }
    buf.buf.truncate(range.start);
    total
}
