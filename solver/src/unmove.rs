//! Exact enumeration of 2-ply predecessors: the reverse method (逆算法).
//!
//! Given a verified strict tsume `P_k` (attacker = black to move, forced mate in
//! `k`), [`predecessors`] enumerates every position `P_{k+2}` from which one legal
//! attacker **check** `c` followed by one legal defender reply `r` reaches exactly
//! `P_k`.  This is shape (a) of `docs/research/learned-reverse-method.md` §3: the
//! enumerator is exhaustive and model-free, and a classifier's only job would be to
//! order what it returns.
//!
//! ## The two retractions
//!
//! 1. **Retract the defender's reply `r`** from `P_k`, giving `P'` (defender to
//!    move, in check).  The reply's *destination* is a defender piece on the board
//!    of `P_k`, so the sweep is over defender board pieces; the origin is walked
//!    backwards along the mover's own rays ([`reverse_origins`]).
//! 2. **Retract the attacker's check `c`** from `P'`, giving `P_{k+2}` (attacker to
//!    move).  The mover's *destination* is an attacker piece on the board of `P'`.
//!    Every attacker piece is tried, not only the ones that check: a discovered
//!    check is produced by a piece that ends up nowhere near the king.
//!
//! Both steps generate pseudo-retractions and the result is then filtered by
//! **forward legality** — `c` must be a legal check of `P_{k+2}`, `r` a legal reply
//! to it, and the two together must reproduce `P_k` — exactly as Hwa's Retractor
//! does (`papers/generation/2023-hwa-retractor-retro-chess.md`) and as Irving 2014
//! recommends validating retrograde code.  The forward filter, not the generator,
//! is the authority: over-generation only costs time.
//!
//! ## Hands under the strict convention
//!
//! `docs/rules.md` gives every non-king piece absent from the board and the
//! attacker's hand to the defender.  That invariant —
//! `defender_hand = MATERIAL_TOTAL − board − attacker_hand` — is preserved by
//! *every* legal move: a capture moves one unit from the board into the capturer's
//! hand, a drop the other way, and both sides of the equation move together.  So
//! this module never writes a defender hand count: it clears the defender's hand,
//! edits the board and the attacker's hand, and lets
//! [`strict_material::apply`] derive the rest ([`finish`]).  Two consequences:
//!
//! * **Un-capturing a defender piece** (the attacker's check took something) costs
//!   one unit of the attacker's hand, so the type must be one the attacker actually
//!   holds in `P'`.
//! * **Retracting a defender capture** (the defender's reply took an attacker
//!   piece) puts an attacker piece back on the board, which *removes* one unit from
//!   the defender's implicit hand.  Every type the defender holds in `P_k` is a
//!   candidate, promoted and unpromoted alike, because a captured promoted piece
//!   enters a hand as its base type.

use std::collections::BTreeMap;

use crate::canonical::canonicalize;
use crate::movegen::{gives_check, legal_moves};
use crate::piece::*;
use crate::position::{Move, Position, NO_SQUARE};
use crate::sfen::{to_legacy_sfen_key, to_sfen};
use crate::strict_material;
use crate::validate::{validate_position, ValidateOptions};

/// How the attacker's check was retracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnCheckKind {
    /// A board move back to an empty origin square.
    BoardMove,
    /// A board move that also promoted; the piece is unpromoted at the origin.
    BoardMoveUnpromote,
    /// The check captured a defender piece, which goes back on the destination and
    /// leaves the attacker's hand.
    UnCapture,
    /// [`UnCheckKind::UnCapture`] where the attacker also promoted.
    UnCaptureUnpromote,
    /// 打ち戻し: the checking piece was dropped, so it returns to the attacker's hand.
    UnDrop,
}

/// How the defender's reply was retracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnReplyKind {
    /// The king stepped to an empty square.
    KingMove,
    /// The king captured the checker (or another attacker piece).
    KingCapture,
    /// A non-king board move to an empty square.
    BoardMove,
    /// A non-king board move that also promoted.
    BoardMoveUnpromote,
    /// A non-king board move that captured an attacker piece.
    BoardCapture,
    /// A non-king board move that captured *and* promoted.
    BoardCaptureUnpromote,
    /// A drop — an interposition (合駒) or any other drop; the piece returns to the
    /// defender's implicit hand automatically.
    Drop,
}

/// Every [`UnCheckKind`], in enumeration order.
pub const UN_CHECK_KINDS: [UnCheckKind; 5] = [
    UnCheckKind::BoardMove,
    UnCheckKind::BoardMoveUnpromote,
    UnCheckKind::UnCapture,
    UnCheckKind::UnCaptureUnpromote,
    UnCheckKind::UnDrop,
];

/// Every [`UnReplyKind`], in enumeration order.
pub const UN_REPLY_KINDS: [UnReplyKind; 7] = [
    UnReplyKind::KingMove,
    UnReplyKind::KingCapture,
    UnReplyKind::BoardMove,
    UnReplyKind::BoardMoveUnpromote,
    UnReplyKind::BoardCapture,
    UnReplyKind::BoardCaptureUnpromote,
    UnReplyKind::Drop,
];

impl UnCheckKind {
    /// Dense index into a per-kind counter array.
    pub const fn index(self) -> usize {
        match self {
            UnCheckKind::BoardMove => 0,
            UnCheckKind::BoardMoveUnpromote => 1,
            UnCheckKind::UnCapture => 2,
            UnCheckKind::UnCaptureUnpromote => 3,
            UnCheckKind::UnDrop => 4,
        }
    }

    /// Stable name used in `--stats-out` and record metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnCheckKind::BoardMove => "board_move",
            UnCheckKind::BoardMoveUnpromote => "board_move_unpromote",
            UnCheckKind::UnCapture => "uncapture",
            UnCheckKind::UnCaptureUnpromote => "uncapture_unpromote",
            UnCheckKind::UnDrop => "undrop",
        }
    }
}

impl UnReplyKind {
    /// Dense index into a per-kind counter array.
    pub const fn index(self) -> usize {
        match self {
            UnReplyKind::KingMove => 0,
            UnReplyKind::KingCapture => 1,
            UnReplyKind::BoardMove => 2,
            UnReplyKind::BoardMoveUnpromote => 3,
            UnReplyKind::BoardCapture => 4,
            UnReplyKind::BoardCaptureUnpromote => 5,
            UnReplyKind::Drop => 6,
        }
    }

    /// Stable name used in `--stats-out` and record metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnReplyKind::KingMove => "king_move",
            UnReplyKind::KingCapture => "king_capture",
            UnReplyKind::BoardMove => "board_move",
            UnReplyKind::BoardMoveUnpromote => "board_move_unpromote",
            UnReplyKind::BoardCapture => "board_capture",
            UnReplyKind::BoardCaptureUnpromote => "board_capture_unpromote",
            UnReplyKind::Drop => "drop",
        }
    }
}

/// One verified 2-ply predecessor.
///
/// `position` is the strict-normalised `P_{k+2}` with the attacker to move;
/// playing `check` then `reply` from it reproduces the source position exactly.
#[derive(Clone, Debug)]
pub struct Predecessor {
    pub position: Position,
    pub check: Move,
    pub reply: Move,
    pub kind: (UnCheckKind, UnReplyKind),
}

/// One distinct predecessor position and every `(check, reply)` pair that reaches
/// the source through it.
#[derive(Clone, Debug)]
pub struct PredecessorGroup {
    pub position: Position,
    /// SFEN of [`PredecessorGroup::position`] (hands in `SFEN_HAND_ORDER`).  Groups are
    /// deduplicated and ordered by the legacy spelling (`to_legacy_sfen_key`).
    pub sfen: String,
    /// `canonicalize` hash, i.e. mirror-insensitive puzzle identity.
    pub canonical_hash: String,
    /// `(check, reply, kind)` triples, ascending by `(check USI, reply USI)`.
    pub moves: Vec<UnMovePair>,
}

/// Per-kind counters over a list of predecessors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredecessorCounts {
    /// Indexed by [`UnCheckKind::index`].
    pub check: [usize; 5],
    /// Indexed by [`UnReplyKind::index`].
    pub reply: [usize; 7],
    /// `(check, reply)` pairs in total.
    pub pairs: usize,
}

impl PredecessorCounts {
    /// Counts every pair of `list`.
    pub fn of(list: &[Predecessor]) -> Self {
        let mut out = Self::default();
        for item in list {
            out.check[item.kind.0.index()] += 1;
            out.reply[item.kind.1.index()] += 1;
            out.pairs += 1;
        }
        out
    }

    /// Adds `other` into `self`, for aggregating over many sources.
    pub fn merge(&mut self, other: &Self) {
        for (slot, value) in self.check.iter_mut().zip(other.check.iter()) {
            *slot += value;
        }
        for (slot, value) in self.reply.iter_mut().zip(other.reply.iter()) {
            *slot += value;
        }
        self.pairs += other.pairs;
    }
}

/// One `(check, reply, kind)` triple of a [`PredecessorGroup`].
pub type UnMovePair = (Move, Move, (UnCheckKind, UnReplyKind));

/// A pseudo-retraction before the forward filter has seen it.
struct Raw {
    position: Position,
    check: Move,
    reply: Move,
    kind: (UnCheckKind, UnReplyKind),
}

/// Clears the defender's hand, sets the side to move and derives the strict
/// defender remainder from the board and the attacker's hand.
///
/// The defender's hand is *never* written directly: clearing it and calling
/// [`strict_material::apply`] is what makes the material bookkeeping of an
/// un-capture or a retracted capture exact.
fn finish(mut position: Position, side: u8) -> Position {
    position.hands[WHITE as usize] = [0; 7];
    position.side = side;
    position.refresh();
    strict_material::apply(&mut position, BLACK);
    position
}

/// Squares a piece with code `pre_code` could have stood on to reach `to`.
///
/// Walks each of the piece's own rays backwards from `to`.  A candidate origin
/// must be empty — one ply ago nothing had moved into it — and for a sliding ray
/// every square strictly between origin and destination must be empty too, which
/// is why the walk stops at the first occupied square.  Results are appended in
/// direction-table order, then by increasing distance, so the enumeration order is
/// fixed.
fn reverse_origins(board: &[u8; 81], to: usize, pre_code: u8, out: &mut Vec<usize>) {
    let tables = &TABLES;
    let (tf, tr) = (file_of(to) as i8, rank_of(to) as i8);
    for &(dx, dy, slide) in
        &tables.dirs[pre_code as usize][..tables.dirs_len[pre_code as usize] as usize]
    {
        let (mut f, mut r) = (tf - dx, tr - dy);
        while (0..9).contains(&f) && (0..9).contains(&r) {
            let square = (r * 9 + f) as usize;
            if board[square] != EMPTY {
                break;
            }
            out.push(square);
            if !slide {
                break;
            }
            f -= dx;
            r -= dy;
        }
    }
}

/// Piece codes an attacker piece captured on `square` could have been, in
/// `P L N S G B R` order with the unpromoted form first.
///
/// `held` is the hand the piece has to come out of: the defender's implicit hand
/// when retracting a defender capture, the attacker's explicit hand when
/// un-capturing.  A promoted piece enters a hand as its base type, so both forms
/// draw on the same count.
fn capture_codes(held: &[u8; 7], color: u8, out: &mut Vec<u8>) {
    out.clear();
    for &t in HAND_PIECES.iter() {
        if held[t as usize] == 0 {
            continue;
        }
        out.push(code(color, t, false));
        if promotable_type(t) {
            out.push(code(color, t, true));
        }
    }
}

/// Step 1: every pseudo-retraction of the defender's reply.
///
/// The sweep is over board squares in SFEN order; for each defender piece the
/// order is board moves (plain, then unpromoting), each with its captures in piece
/// order, and finally the drop retraction.
fn retract_replies(source: &Position) -> Vec<(Position, Move, UnReplyKind)> {
    let mut out = Vec::new();
    let mut origins: Vec<usize> = Vec::new();
    let mut captures: Vec<u8> = Vec::new();
    // Every type the defender holds in the source is a candidate for a retracted
    // capture; the piece goes back on the board and leaves the implicit hand.
    capture_codes(&source.hands[WHITE as usize], BLACK, &mut captures);
    let captures = captures;

    for to in 0..81usize {
        let occupant = source.board[to];
        if occupant == EMPTY || color_of(occupant) != WHITE {
            continue;
        }
        let t = ptype(occupant);
        let promoted = is_promoted(occupant);
        let is_king = t == PT_K;

        // A promoted piece may have been promoted already, or promoted *on* this
        // move; the second reading retracts to the unpromoted code.
        let mut variants: Vec<(u8, bool)> = vec![(occupant, false)];
        if promoted && promotable_type(t) {
            variants.push((base_code(occupant), true));
        }

        for (pre_code, unpromote) in variants {
            origins.clear();
            reverse_origins(&source.board, to, pre_code, &mut origins);
            for &from in origins.iter() {
                if unpromote
                    && !(promotion_zone(WHITE, rank_of(from)) || promotion_zone(WHITE, rank_of(to)))
                {
                    continue;
                }
                let reply = Move::board(from as u8, to as u8, t, unpromote);
                // (a) the reply moved to an empty square.
                let mut candidate = source.clone();
                candidate.board[from] = pre_code;
                candidate.board[to] = EMPTY;
                let kind = match (is_king, unpromote) {
                    (true, _) => UnReplyKind::KingMove,
                    (false, false) => UnReplyKind::BoardMove,
                    (false, true) => UnReplyKind::BoardMoveUnpromote,
                };
                push_reply(&mut out, candidate, reply, kind);
                // (b) the reply captured an attacker piece standing on `to`.
                for &captured in captures.iter() {
                    let mut candidate = source.clone();
                    candidate.board[from] = pre_code;
                    candidate.board[to] = captured;
                    let kind = match (is_king, unpromote) {
                        (true, _) => UnReplyKind::KingCapture,
                        (false, false) => UnReplyKind::BoardCapture,
                        (false, true) => UnReplyKind::BoardCaptureUnpromote,
                    };
                    push_reply(&mut out, candidate, reply, kind);
                }
            }
        }

        // (c) the reply was a drop.  Only an unpromoted non-king piece can be one,
        // and it returns to the defender's implicit hand by construction.
        if !promoted && !is_king {
            let mut candidate = source.clone();
            candidate.board[to] = EMPTY;
            let reply = Move::drop(t, to as u8);
            push_reply(&mut out, candidate, reply, UnReplyKind::Drop);
        }
    }
    out
}

/// Keeps a defender-reply retraction only if the defender really is in check in it,
/// which every position reached by an attacker's check is.
fn push_reply(
    out: &mut Vec<(Position, Move, UnReplyKind)>,
    candidate: Position,
    reply: Move,
    kind: UnReplyKind,
) {
    let position = finish(candidate, WHITE);
    if position.king_square(WHITE) == NO_SQUARE || !position.is_in_check(WHITE) {
        return;
    }
    out.push((position, reply, kind));
}

/// Step 2: every pseudo-retraction of the attacker's check.
///
/// Every attacker board piece is tried as the mover, not only the checkers: a
/// discovered check is given by a piece the mover left behind.  Rule A means the
/// attacker has no king, so there are no king retractions here.
fn retract_checks(prior: &Position) -> Vec<(Position, Move, UnCheckKind)> {
    let mut out = Vec::new();
    let mut origins: Vec<usize> = Vec::new();
    let mut captures: Vec<u8> = Vec::new();
    // An un-capture pays one unit of the attacker's hand, so only types the
    // attacker holds here can be put back on the board.
    capture_codes(&prior.hands[BLACK as usize], WHITE, &mut captures);
    let captures = captures;

    for to in 0..81usize {
        let occupant = prior.board[to];
        if occupant == EMPTY || color_of(occupant) != BLACK {
            continue;
        }
        let t = ptype(occupant);
        if t == PT_K {
            // Rule A: a published tsume has no attacker king, and `predecessors`
            // refuses a source that carries one, so this is unreachable in practice.
            continue;
        }
        let promoted = is_promoted(occupant);

        let mut variants: Vec<(u8, bool)> = vec![(occupant, false)];
        if promoted && promotable_type(t) {
            variants.push((base_code(occupant), true));
        }

        for (pre_code, unpromote) in variants {
            origins.clear();
            reverse_origins(&prior.board, to, pre_code, &mut origins);
            for &from in origins.iter() {
                if unpromote
                    && !(promotion_zone(BLACK, rank_of(from)) || promotion_zone(BLACK, rank_of(to)))
                {
                    continue;
                }
                let check = Move::board(from as u8, to as u8, t, unpromote);
                // (a) the check moved to an empty square.
                let mut candidate = prior.clone();
                candidate.board[from] = pre_code;
                candidate.board[to] = EMPTY;
                let kind = if unpromote {
                    UnCheckKind::BoardMoveUnpromote
                } else {
                    UnCheckKind::BoardMove
                };
                push_check(&mut out, candidate, check, kind);
                // (b) the check captured a defender piece, which returns to the
                // board while one unit leaves the attacker's hand.
                for &captured in captures.iter() {
                    let mut candidate = prior.clone();
                    candidate.board[from] = pre_code;
                    candidate.board[to] = captured;
                    candidate.hands[BLACK as usize][ptype(captured) as usize] -= 1;
                    let kind = if unpromote {
                        UnCheckKind::UnCaptureUnpromote
                    } else {
                        UnCheckKind::UnCapture
                    };
                    push_check(&mut out, candidate, check, kind);
                }
            }
        }

        // (c) 打ち戻し: the checking piece was dropped and returns to the hand.
        if !promoted {
            let mut candidate = prior.clone();
            candidate.board[to] = EMPTY;
            candidate.hands[BLACK as usize][t as usize] += 1;
            let check = Move::drop(t, to as u8);
            push_check(&mut out, candidate, check, UnCheckKind::UnDrop);
        }
    }
    out
}

/// Keeps an attacker-check retraction only if the defender is *not* in check in it,
/// which a tsume root never is.
fn push_check(
    out: &mut Vec<(Position, Move, UnCheckKind)>,
    candidate: Position,
    check: Move,
    kind: UnCheckKind,
) {
    let position = finish(candidate, BLACK);
    if position.king_square(WHITE) == NO_SQUARE || position.is_in_check(WHITE) {
        return;
    }
    out.push((position, check, kind));
}

/// Every verified 2-ply predecessor of `position`, ascending by
/// `(predecessor SFEN, check USI, reply USI)`.
///
/// `position` must be a strict tsume root: attacker (black) to move, exactly one
/// defender king, no attacker king (rule A) and the defender not already in check.
/// A source that fails any of those yields an empty list.
///
/// Each returned `P_{k+2}` is itself a valid tsume *start* — attacker to move,
/// defender not in check, no attacker king, structurally valid under
/// [`ValidateOptions::pipeline`] — but whether it is a *puzzle* (a forced,
/// minimal, unique-first-move mate in `k + 2` satisfying rules A–G) is for
/// `exact_validate_with_rules` to decide, not for this enumerator.
pub fn predecessors(position: &Position) -> Vec<Predecessor> {
    grouped_predecessors(position)
        .into_iter()
        .flat_map(|group| {
            group
                .moves
                .into_iter()
                .map(move |(check, reply, kind)| Predecessor {
                    position: group.position.clone(),
                    check,
                    reply,
                    kind,
                })
        })
        .collect()
}

/// [`predecessors`], one entry per distinct predecessor position.
///
/// Positions are deduplicated by SFEN first — that is the key the `(check, reply)`
/// pairs belong to — and the resulting groups are then deduplicated by canonical
/// hash, keeping the first in SFEN order.  The second pass only ever fires when the
/// source is its own mirror image, and dropping a mirror twin keeps every retained
/// pair reproducible against the position it is recorded with.
pub fn grouped_predecessors(position: &Position) -> Vec<PredecessorGroup> {
    let source = strict_material::normalize(position, BLACK);
    if source.side != BLACK
        || source.king_square(BLACK) != NO_SQUARE
        || source.king_square(WHITE) == NO_SQUARE
        || source.is_in_check(WHITE)
    {
        return Vec::new();
    }
    let source_key = to_sfen(&source);

    let mut raw: Vec<Raw> = Vec::new();
    for (prior, reply, reply_kind) in retract_replies(&source) {
        for (candidate, check, check_kind) in retract_checks(&prior) {
            raw.push(Raw {
                position: candidate,
                check,
                reply,
                kind: (check_kind, reply_kind),
            });
        }
    }

    // Group by exact SFEN: the forward filter only has to build the legal-move list
    // once per distinct position, and the BTreeMap fixes the output order.  The key is
    // the frozen pre-2026-10-04 spelling so the order (and which predecessor survives
    // the canonical dedup below) does not depend on the written hand order.
    let mut grouped: BTreeMap<String, (Position, Vec<UnMovePair>)> = BTreeMap::new();
    for item in raw {
        let sfen = to_legacy_sfen_key(&item.position);
        grouped
            .entry(sfen)
            .or_insert_with(|| (item.position, Vec::new()))
            .1
            .push((item.check, item.reply, item.kind));
    }

    let mut out: Vec<PredecessorGroup> = Vec::new();
    let mut seen_canonical: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_legacy_key, (candidate, mut pairs)) in grouped {
        if !validate_position(&candidate, ValidateOptions::pipeline()).is_empty() {
            continue;
        }
        pairs.sort_unstable_by_key(|(check, reply, _)| (check.to_usi(), reply.to_usi()));
        pairs.dedup_by_key(|(check, reply, _)| (check.to_usi(), reply.to_usi()));
        let legal = legal_moves(&candidate);
        let mut verified: Vec<UnMovePair> = Vec::new();
        // Pairs are sorted by check first, so one `applied(check)` and one
        // legal-move list serves every reply that follows it.
        let mut current: Option<(Move, Vec<Move>, Position)> = None;
        for (check, reply, kind) in pairs {
            if !legal.contains(&check) || !gives_check(&candidate, check) {
                continue;
            }
            let reuse = matches!(&current, Some((previous, _, _)) if *previous == check);
            if !reuse {
                let after = candidate.applied(check);
                current = Some((check, legal_moves(&after), after));
            }
            let Some((_, replies, after)) = current.as_ref() else {
                continue;
            };
            if !replies.contains(&reply) {
                continue;
            }
            let reached = after.applied(reply);
            if to_sfen(&strict_material::normalize(&reached, BLACK)) != source_key {
                continue;
            }
            verified.push((check, reply, kind));
        }
        if verified.is_empty() {
            continue;
        }
        let canonical_hash = canonicalize(&candidate).hash;
        if !seen_canonical.insert(canonical_hash.clone()) {
            continue;
        }
        out.push(PredecessorGroup {
            sfen: to_sfen(&candidate),
            position: candidate,
            canonical_hash,
            moves: verified,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::parse_sfen;

    #[test]
    fn reverse_origins_stop_at_blockers() {
        // A black rook on 5e reaches 5e from the whole file and rank, but a piece on
        // 5c blocks everything beyond it.
        let position = parse_sfen("4k4/9/4P4/9/4R4/9/9/9/9 b - 1").unwrap();
        let to = sq(4, 4);
        let mut out = Vec::new();
        reverse_origins(&position.board, to, code(BLACK, PT_R, false), &mut out);
        // Upwards the walk stops below the pawn on 5c (rank index 2): only 5d.
        assert!(out.contains(&sq(4, 3)));
        assert!(!out.contains(&sq(4, 2)));
        assert!(!out.contains(&sq(4, 1)));
        // Downwards and sideways the file and rank are empty.
        assert!(out.contains(&sq(4, 8)));
        assert!(out.contains(&sq(0, 4)));
        assert!(out.contains(&sq(8, 4)));
    }

    #[test]
    fn a_source_with_an_attacker_king_has_no_predecessors() {
        // Rule A: the enumerator refuses a diagram that carries an attacker king.
        let position = parse_sfen("6K1k/9/6S2/9/9/9/9/9/9 b - 1").unwrap();
        assert!(predecessors(&position).is_empty());
    }

    #[test]
    fn enumeration_is_deterministic() {
        let position = parse_sfen("8k/9/9/9/7R1/9/9/9/9 b G 1").unwrap();
        let first = predecessors(&position);
        let second = predecessors(&position);
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(to_sfen(&a.position), to_sfen(&b.position));
            assert_eq!(a.check.to_usi(), b.check.to_usi());
            assert_eq!(a.reply.to_usi(), b.reply.to_usi());
        }
    }
}
