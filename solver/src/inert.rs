//! Rule G of `docs/rules.md`: no inert board piece (無駄駒).
//!
//! Port of the rule G block of `tools/puzzle-pipeline/src/solve.ts`
//! (`analyzeInertPieces`).  A composed tsume may not carry a board piece that
//! takes no part in the solution, and "takes part" is a statement about the
//! **solution tree** and about the wrong checks the defender has to refute, never
//! about the diagram alone.
//!
//! Two things deliberately do *not* count as taking part: keeping a piece out of
//! the defender's hand (the strict remainder convention hands every absent piece
//! to the defender, so every spare board piece denies a drop - if that were
//! activity, nothing could ever be inert), and the reducer having failed to
//! remove it (rule D is greedy and all-or-nothing, so "irreducible" and
//! "necessary" are different properties).
//!
//! Piece identity is followed through the tree: a piece *is* the root-diagram
//! square it started on, and moving, promoting or being captured never changes
//! which piece it is.  The analysis is therefore a walk of the solution tree
//! carrying one map from the current square to the root square.

use std::collections::{HashMap, HashSet};

use crate::canonical::piece_label;
use crate::cdfpn::ClassicalProber;
use crate::jsonl::InertPiece;
use crate::movegen::{checking_moves_sorted, legal_moves};
use crate::piece::{color_of, file_of, opponent, ptype, rank_of, EMPTY, PT_K, TABLES};
use crate::position::{Move, Position, NO_SQUARE};
use crate::solver::{Graph, Outcome};

/// Rule G's budget-out (clause e, `PIPELINE_VERSION` 0.15.0): a budget-dependent
/// reason, like `solver_budget_exhausted`.
pub const RULE_G_BUDGET_EXHAUSTED: &str = "rule_g_budget_exhausted";

/// The node budget of each clause (e) question when the caller names none
/// (`exactValidate`'s default node budget).  The gates pass their own.
pub const DEFAULT_RULE_G_BUDGET: u64 = crate::exact::DEFAULT_NODE_BUDGET;

/// What rule G found on one proven mate graph.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InertReport {
    /// Inert attacker pieces, ascending by square (`inert_attacker_piece`).
    pub attacker: Vec<InertPiece>,
    /// Inert defender pieces, ascending by square (`inert_defender_piece`).
    pub defender: Vec<InertPiece>,
    /// Pieces no clause made active whose clause (d) or (e) questions ran out
    /// of budget (`rule_g_budget_exhausted`): neither active nor inert.
    /// Attacker entries first, each side ascending by square.
    pub undecided: Vec<InertPiece>,
}

impl InertReport {
    /// Attacker entries first, then defender entries, each ascending by square -
    /// the order the record publishes as `rejection.inertPieces`.
    pub fn all(&self) -> Vec<InertPiece> {
        let mut out = self.attacker.clone();
        out.extend(self.defender.iter().cloned());
        out
    }

    /// Published as the difficulty feature `inertPieceCount`.
    pub fn count(&self) -> usize {
        self.attacker.len() + self.defender.len()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Every non-king board piece of `root` undecided: rule G could not be
    /// asked at all because the solution tree at the proven distance ran out
    /// of budget (`rule_g_budget_exhausted`).
    pub fn all_undecided(root: &Position, attacker: u8) -> Self {
        let mut report = InertReport::default();
        let mut defender = Vec::new();
        for square in 0..81 {
            let code = root.board[square];
            if code == EMPTY || ptype(code) == PT_K {
                continue;
            }
            let entry = InertPiece {
                square,
                piece: piece_label(root, square),
            };
            if color_of(code) == attacker {
                report.undecided.push(entry);
            } else {
                defender.push(entry);
            }
        }
        report.undecided.extend(defender);
        report
    }
}

/// The eight king directions.  Board geometry only, never piece movement.
const RAYS: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
/// The four squares a knight of either colour could attack a square from.
const KNIGHT_SQUARES: [(i8, i8); 4] = [(-1, -2), (1, -2), (-1, 2), (1, 2)];

#[inline]
fn on_board(file: i8, rank: i8) -> bool {
    (0..9).contains(&file) && (0..9).contains(&rank)
}

#[inline]
fn chebyshev(a: usize, b: usize) -> usize {
    let df = (file_of(a) as i32 - file_of(b) as i32).unsigned_abs() as usize;
    let dr = (rank_of(a) as i32 - rank_of(b) as i32).unsigned_abs() as usize;
    df.max(dr)
}

/// `true` when the piece standing on `from` attacks `target` given the blockers of
/// `board`.  Walks the mover's own rays, exactly as `isSquareAttacked` does for a
/// single piece in `packages/core-rules`.
fn piece_attacks(board: &[u8; 81], from: usize, target: usize) -> bool {
    let code = board[from] as usize;
    let tables = &TABLES;
    let (origin_file, origin_rank) = (file_of(from) as i8, rank_of(from) as i8);
    for &(dx, dy, slide) in &tables.dirs[code][..tables.dirs_len[code] as usize] {
        let (mut file, mut rank) = (origin_file + dx, origin_rank + dy);
        while on_board(file, rank) {
            let square = (rank * 9 + file) as usize;
            if square == target {
                return true;
            }
            if board[square] != EMPTY || !slide {
                break;
            }
            file += dx;
            rank += dy;
        }
    }
    false
}

/// Root squares of the tracked pieces still on the board, by current square.
type Origins = HashMap<usize, usize>;

/// What one edge did to the identity map.
struct OriginStep {
    origins: Origins,
    /// Root square of the piece that moved, when a tracked piece moved.
    mover: Option<usize>,
    /// Root square of the piece that was captured, when a tracked piece was.
    captured: Option<usize>,
}

/// Applies one move to the identity map.  No board is needed: the map already
/// says which tracked piece stands where, a drop always lands on an empty square,
/// and a king is neither tracked nor capturable.
fn advance_origins(origins: &Origins, m: Move) -> OriginStep {
    let to = m.to as usize;
    let captured = origins.get(&to).copied();
    let mut next = origins.clone();
    if captured.is_some() {
        next.remove(&to);
    }
    let mut mover = None;
    if !m.is_drop() {
        let from = m.from as usize;
        mover = origins.get(&from).copied();
        next.remove(&from);
        if let Some(origin) = mover {
            next.insert(to, origin);
        }
    }
    OriginStep {
        origins: next,
        mover,
        captured,
    }
}

/// The root squares of the tracked piece that moves on `m` and of the tracked
/// piece it captures (the [`advance_origins`] step without the new map).
fn involved(origins: &Origins, m: Move) -> (Option<usize>, Option<usize>) {
    let captured = origins.get(&(m.to as usize)).copied();
    let mover = if m.is_drop() {
        None
    } else {
        origins.get(&(m.from as usize)).copied()
    };
    (mover, captured)
}

/// A stable key for one identity map, so a transposition is walked once per map.
fn origin_key(origins: &Origins) -> Vec<(usize, usize)> {
    let mut entries: Vec<(usize, usize)> = origins.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable();
    entries
}

/// Rule G clauses (b) and (c) at one node of the solution tree: which tracked
/// pieces press on the defender king's zone there.
///
/// * (b) the piece attacks the king's square or a square adjacent to it.  A
///   *defender* piece also qualifies by standing on an adjacent square, because it
///   seals that escape; an attacker piece standing there does not, since the king
///   may simply take it.
/// * (c) the piece is the single blocker between an attacker slider and a square
///   of that zone, so removing it would extend the slider into the zone.
fn classify_king_zone(
    position: &Position,
    attacker: u8,
    origins: &Origins,
    active: &mut HashSet<usize>,
) {
    if origins.values().all(|origin| active.contains(origin)) {
        return;
    }
    let king = position.king_square(opponent(attacker));
    if king == NO_SQUARE {
        return;
    }
    let king = king as usize;
    let (king_file, king_rank) = (file_of(king) as i8, rank_of(king) as i8);
    let mut zone: Vec<usize> = vec![king];
    for (dx, dy) in RAYS {
        let (file, rank) = (king_file + dx, king_rank + dy);
        if on_board(file, rank) {
            zone.push((rank * 9 + file) as usize);
        }
    }

    // (b): everything that attacks the zone, plus defender pieces standing in it.
    let mut attacking: HashSet<usize> = HashSet::new();
    for &target in zone.iter() {
        collect_attackers(position, target, &mut attacking);
    }
    for (&square, &origin) in origins.iter() {
        if active.contains(&origin) {
            continue;
        }
        if attacking.contains(&square) {
            active.insert(origin);
            continue;
        }
        let code = position.board[square];
        if code != EMPTY && color_of(code) != attacker && chebyshev(square, king) == 1 {
            active.insert(origin);
        }
    }

    // (c): the single blocker on an attacker slider's line into the zone.
    let tables = &TABLES;
    for square in 0..81usize {
        let code = position.board[square];
        if code == EMPTY || color_of(code) != attacker {
            continue;
        }
        let (origin_file, origin_rank) = (file_of(square) as i8, rank_of(square) as i8);
        for &(dx, dy, slide) in
            &tables.dirs[code as usize][..tables.dirs_len[code as usize] as usize]
        {
            if !slide {
                continue;
            }
            let (mut file, mut rank) = (origin_file + dx, origin_rank + dy);
            let mut blockers = 0usize;
            let mut blocker = 0usize;
            while on_board(file, rank) {
                let target = (rank * 9 + file) as usize;
                if blockers == 1 && zone.contains(&target) {
                    if let Some(&origin) = origins.get(&blocker) {
                        active.insert(origin);
                    }
                    break;
                }
                if position.board[target] != EMPTY {
                    blockers += 1;
                    if blockers > 1 {
                        break;
                    }
                    blocker = target;
                }
                file += dx;
                rank += dy;
            }
        }
    }
}

/// Adds every square whose occupant attacks `target` **with the blockers of this
/// position** to `into`.  Walking outward from the target means only the first
/// piece on each ray can be an attacker.
fn collect_attackers(position: &Position, target: usize, into: &mut HashSet<usize>) {
    let (target_file, target_rank) = (file_of(target) as i8, rank_of(target) as i8);
    for (dx, dy) in RAYS {
        let (mut file, mut rank) = (target_file + dx, target_rank + dy);
        while on_board(file, rank) {
            let square = (rank * 9 + file) as usize;
            if position.board[square] != EMPTY {
                if piece_attacks(&position.board, square, target) {
                    into.insert(square);
                }
                break;
            }
            file += dx;
            rank += dy;
        }
    }
    for (dx, dy) in KNIGHT_SQUARES {
        let (file, rank) = (target_file + dx, target_rank + dy);
        if !on_board(file, rank) {
            continue;
        }
        let square = (rank * 9 + file) as usize;
        if position.board[square] != EMPTY && piece_attacks(&position.board, square, target) {
            into.insert(square);
        }
    }
}

/// Rule G: classifies every non-king board piece of the diagram as active or
/// inert, on the proven mate graph of that diagram.
///
/// The solution tree T is walked from the root: at an attack node the correct
/// moves (`MATE` with `distance == node.distance`), at a defense node every legal
/// reply rule F did not flag 無駄合.  A piece is active when it moves, captures or
/// is captured on an edge of T (clause a), when it presses on the defender king's
/// zone at some node of T (clauses b and c), or when it takes part in one of the
/// defender's refutations of a wrong check (clause d) - which is what keeps a
/// piece that exists only to prevent 余詰 on the board.  Since
/// `PIPELINE_VERSION` 0.16.0 clause (d) reads **every** reply that refutes the
/// wrong check within the attack node's distance (asked of the classical
/// prober, [`Occupancy::reply_holds`]), not the one reply the graph recorded,
/// so neither the USI order (which the left-right mirror reverses) nor the
/// bound the graph was searched at enters the verdict.  The caller passes the
/// graph searched at the proven mate length.
///
/// A piece (a)-(d) leave inert is still active under clause (e) when it stands
/// on a square that alone keeps out a check that would mate in time
/// ([`Occupancy::activity`]).
///
/// `graph` must be a proven mate graph and `attacker` the attacking colour.
/// `budget` is the node budget of each clause (d) and (e) question.
pub fn analyze_inert_pieces(graph: &Graph, attacker: u8, budget: u64) -> InertReport {
    analyze_inert_pieces_counted(graph, attacker, budget).0
}

/// [`analyze_inert_pieces`] with the prober nodes its clause (d) and (e)
/// questions expanded (for the callers' node accounting; not part of the verdict).
pub fn analyze_inert_pieces_counted(
    graph: &Graph,
    attacker: u8,
    budget: u64,
) -> (InertReport, u64) {
    let Some(root) = graph.node(graph.root) else {
        return (InertReport::default(), 0);
    };
    let root_position = root.position();
    let tracked: Vec<usize> = (0..81)
        .filter(|&square| {
            let code = root_position.board[square];
            code != EMPTY && ptype(code) != PT_K
        })
        .collect();
    if tracked.is_empty() {
        return (InertReport::default(), 0);
    }
    let mut active: HashSet<usize> = HashSet::new();
    let mut seen: HashSet<(u32, Vec<(usize, usize)>)> = HashSet::new();
    let mut attack_nodes: Vec<(u32, Origins)> = Vec::new();
    let origins: Origins = tracked.iter().map(|&square| (square, square)).collect();
    visit(
        graph,
        attacker,
        graph.root,
        &origins,
        &tracked,
        &mut active,
        &mut seen,
        &mut attack_nodes,
    );

    let mut asker = Occupancy {
        prober: None,
        memo: HashMap::new(),
        replies: HashMap::new(),
        nodes: 0,
        attacker,
        budget,
    };

    // Clause (d) (`PIPELINE_VERSION` 0.16.0): at every attack node N of T, of
    // distance d, every legal check that is not a correct move, answered by
    // **every** reply that refutes it within N's distance.  Only the replies
    // that move or capture a piece (a)-(c) left inert are asked about, so a
    // diagram whose pieces T already uses asks nothing.
    let mut undecided_d: HashSet<usize> = HashSet::new();
    for (id, origins) in attack_nodes.iter() {
        if active.len() == tracked.len() {
            break;
        }
        let Some(node) = graph.node(*id) else {
            continue;
        };
        let (Outcome::Mate, Some(distance)) = (node.outcome(), node.distance()) else {
            continue;
        };
        if !origins.values().any(|origin| !active.contains(origin)) {
            continue;
        }
        let position = node.position();
        let correct: HashSet<u64> = node
            .moves()
            .iter()
            .filter(|m| m.outcome() == Outcome::Mate && m.distance() == Some(distance))
            .map(|m| m.mv().usi_key())
            .collect();
        for m in checking_moves_sorted(&position) {
            if correct.contains(&m.usi_key()) {
                continue;
            }
            // The wrong check itself proves nothing - it is not part of the
            // solution - so only its answers mark pieces active.
            let after = advance_origins(origins, m);
            if !after
                .origins
                .values()
                .any(|origin| !active.contains(origin))
            {
                continue;
            }
            let mut defense = position.applied(m);
            for reply in legal_moves(&defense).iter().copied() {
                let (mover, captured) = involved(&after.origins, reply);
                let pieces: Vec<usize> = [mover, captured]
                    .into_iter()
                    .flatten()
                    .filter(|origin| !active.contains(origin))
                    .collect();
                if pieces.is_empty() {
                    continue;
                }
                match asker.reply_holds(&mut defense, distance - 1, reply) {
                    Outcome::Refuted => active.extend(pieces),
                    Outcome::Unknown => undecided_d.extend(pieces),
                    Outcome::Mate => {}
                }
            }
        }
    }

    // Clause (e), on the pieces (a)-(d) left inert only.
    let mut nodes: Vec<AttackNode> = Vec::new();
    if tracked.iter().any(|square| !active.contains(square)) {
        let mut taken: HashSet<u32> = HashSet::new();
        for (id, _) in attack_nodes.iter() {
            if !taken.insert(*id) {
                continue;
            }
            let Some(node) = graph.node(*id) else {
                continue;
            };
            if node.outcome() != Outcome::Mate {
                continue;
            }
            if let Some(distance) = node.distance() {
                nodes.push(AttackNode {
                    position: node.position(),
                    distance,
                    legal: None,
                });
            }
        }
    }
    let mut report = InertReport::default();
    let mut undecided_defender = Vec::new();
    for &square in tracked.iter() {
        if active.contains(&square) {
            continue;
        }
        let verdict = match asker.activity(&mut nodes, square) {
            // A clause (d) question that ran out of budget leaves the piece
            // undecided unless clause (e) finds it active.
            Activity::Inert if undecided_d.contains(&square) => Activity::Undecided,
            verdict => verdict,
        };
        let entry = InertPiece {
            square,
            piece: piece_label(&root_position, square),
        };
        let own = color_of(root_position.board[square]) == attacker;
        match (verdict, own) {
            (Activity::Active, _) => {}
            (Activity::Inert, true) => report.attacker.push(entry),
            (Activity::Inert, false) => report.defender.push(entry),
            (Activity::Undecided, true) => report.undecided.push(entry),
            (Activity::Undecided, false) => undecided_defender.push(entry),
        }
    }
    report.undecided.extend(undecided_defender);
    (report, asker.nodes)
}

/// Clause (e)'s verdict on one piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activity {
    Active,
    Inert,
    Undecided,
}

/// An attack node of T that clause (e) asks at.
struct AttackNode {
    position: Position,
    distance: u32,
    /// The USI keys of the node's legal moves, made on the first piece that
    /// asks here and kept for the others.
    legal: Option<HashSet<u64>>,
}

/// The clause (e) questions of one analysis: a prober made on the first
/// question (none when no piece needs one) and the answers so far.
struct Occupancy {
    prober: Option<ClassicalProber>,
    /// Answers by question (the SFEN after the check, the bound), across
    /// pieces: an identical question has one answer.
    memo: HashMap<(String, u32), Outcome>,
    /// Clause (d)'s answers by question (the SFEN of the defense node after
    /// the wrong check, its bound, the reply).
    replies: HashMap<(String, u32, u64), Outcome>,
    nodes: u64,
    attacker: u8,
    budget: u64,
}

impl Occupancy {
    /// Rule G clause (d): does `reply` refute the wrong check that reached the
    /// defense node `defense`, within `bound` (the attack node's distance less
    /// one)?  `REFUTED` when it does, `MATE` when it does not (the check still
    /// mates in time against it, or rule F2 proves it futile), `UNKNOWN` when
    /// the question ran out of budget.  A fresh classical df-pn prober on
    /// `budget` nodes ([`ClassicalProber::reply_holds`]) answers each question
    /// once, so no answer depends on the order of the questions.  Mirrors
    /// `replyHolds` in `tools/puzzle-pipeline/src/solve.ts`.
    fn reply_holds(&mut self, defense: &mut Position, bound: u32, reply: Move) -> Outcome {
        let key = (crate::sfen::to_sfen(defense), bound, reply.usi_key());
        if let Some(&verdict) = self.replies.get(&key) {
            return verdict;
        }
        let prober = self.prober.get_or_insert_with(ClassicalProber::new);
        prober.reset(self.attacker);
        let verdict = prober.reply_holds(defense, bound, reply, self.budget);
        self.nodes += prober.stats().nodes;
        self.replies.insert(key, verdict);
        verdict
    }

    /// Rule G clause (e) for the piece on root square `square` that clauses
    /// (a)-(d) left inert, asked at every attack node of T (`nodes`, each with
    /// its mate distance): active when some check that the presence of the
    /// piece alone makes illegal mates within the distance of the node once the
    /// piece is gone, inert when every such check is refuted, undecided when
    /// none mates but one ran out of budget.
    ///
    /// The piece is taken off and handed, unpromoted, to the defender (rule D's
    /// removal, the strict-tsume convention).  A check of that position whose
    /// USI no legal move of the node has is a move the piece forbids; each is
    /// asked "mate within distance - 1?" of a fresh classical df-pn prober on
    /// `budget` nodes, so no answer depends on the order of the questions; an
    /// identical question (the same position after the check, the same bound)
    /// is asked once.  Mirrors `occupancyActivity` in
    /// `tools/puzzle-pipeline/src/solve.ts`.
    fn activity(&mut self, nodes: &mut [AttackNode], square: usize) -> Activity {
        let attacker = self.attacker;
        let defender = opponent(attacker);
        let mut undecided = false;
        for node in nodes.iter_mut() {
            let (position, distance) = (&node.position, node.distance);
            // The piece never moves on T, so it stands here at every node of T.
            if position.board[square] == EMPTY || position.side != attacker {
                continue;
            }
            let mut without = position.clone();
            without.board[square] = EMPTY;
            without.refresh();
            crate::strict_material::apply(&mut without, attacker);
            // Unreachable for a piece (c) left inert (it would be the single
            // blocker of a check), but never ask about an illegal position.
            if without.is_in_check(defender) {
                continue;
            }
            let legal = node
                .legal
                .get_or_insert_with(|| legal_moves(position).iter().map(Move::usi_key).collect());
            let bound = distance.saturating_sub(1);
            for m in checking_moves_sorted(&without) {
                if legal.contains(&m.usi_key()) {
                    continue;
                }
                let mut after = without.applied(m);
                let key = (crate::sfen::to_sfen(&after), bound);
                let verdict = match self.memo.get(&key) {
                    Some(&verdict) => verdict,
                    None => {
                        let prober = self.prober.get_or_insert_with(ClassicalProber::new);
                        prober.reset(attacker);
                        let verdict = prober.mate_within(&mut after, bound, self.budget);
                        self.nodes += prober.stats().nodes;
                        self.memo.insert(key, verdict);
                        verdict
                    }
                };
                match verdict {
                    Outcome::Mate => return Activity::Active,
                    Outcome::Unknown => undecided = true,
                    Outcome::Refuted => {}
                }
            }
        }
        if undecided {
            Activity::Undecided
        } else {
            Activity::Inert
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn visit(
    graph: &Graph,
    attacker: u8,
    id: u32,
    origins: &Origins,
    tracked: &[usize],
    active: &mut HashSet<usize>,
    seen: &mut HashSet<(u32, Vec<(usize, usize)>)>,
    attack_nodes: &mut Vec<(u32, Origins)>,
) {
    // Every piece already accounted for: nothing deeper can change the verdict.
    if active.len() == tracked.len() {
        return;
    }
    let Some(node) = graph.node(id) else {
        return;
    };
    if !seen.insert((id, origin_key(origins))) {
        return;
    }
    if node.attack() {
        attack_nodes.push((id, origins.clone()));
    }
    classify_king_zone(&node.position(), attacker, origins, active);
    let solution: Vec<usize> = node
        .moves()
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            if node.attack() {
                m.outcome() == Outcome::Mate && m.distance() == node.distance()
            } else {
                m.outcome() == Outcome::Mate && !m.useless()
            }
        })
        .map(|(index, _)| index)
        .collect();
    for index in solution {
        let m = &node.moves()[index];
        let step = advance_origins(origins, m.mv());
        // Clause (a).
        if let Some(origin) = step.mover {
            active.insert(origin);
        }
        if let Some(origin) = step.captured {
            active.insert(origin);
        }
        if let Some(next) = m.next() {
            visit(
                graph,
                attacker,
                next,
                &step.origins,
                tracked,
                active,
                seen,
                attack_nodes,
            );
        }
    }
}
