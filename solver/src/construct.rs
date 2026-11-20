//! Solver v3 item 1: **prove first, then build only what is needed, exactly.**
//!
//! The exact gate ([`crate::exact::exact_validate_with_rules`]) searches every
//! node within the ply bound full-width and then reads a small part of that
//! graph: the root's mate length and correct moves, the shortest solution graph
//! `S` (rules B and C, [`crate::exact::designated_main_line`]) and the solution
//! tree `T` with the defender's recorded answer to every wrong check (rule G
//! walks `T`, [`crate::inert::analyze_inert_pieces`]; since `PIPELINE_VERSION`
//! 0.16.0 it asks the prober for the refuting replies itself, and the recorded
//! answer is the runtime's).  This module builds exactly that
//! part, node for node as the full-width search records it, from bounded
//! "mate within `b`?" questions to the classical df-pn prober
//! ([`crate::cdfpn::ClassicalProber`], whose verdicts are the credit oracle's
//! and so the main search's at the same bound), and hands it to the same two
//! analyses.  See `docs/experiments/solver-v3/README.md` for the argument that
//! every recorded field it builds is the one the full-width search records.
//!
//! What is built, with `r` the node's remaining bound (`max_ply` minus the
//! plies above it, as in the full-width search's table key):
//!
//! * an attack node of `T` with mate length `d`: every legal check in USI order.
//!   A check whose child mates within `d - 1` is correct (`MATE`, `d`, its
//!   child built).  Any other check is recorded `MATE` with its exact length or
//!   `REFUTED` at `r - 1`, without a child, carrying the reply the full-width
//!   graph's `selected_reply` ([`crate::exact`]) would pick as its refutation (the
//!   first `REFUTED` reply, or the longest resistance of a longer mate).
//! * a defense node of `T`: every legal reply.  Rule F's flags are the
//!   full-width search's -- F2 futility at an all-drop node, and at a credit
//!   line (drops and other replies) a drop is 無駄合 exactly when a capture form
//!   mates within `dist_other`.  Every other reply mates; its child is built.
//!
//! Nodes are shared by `(r, position)`, the full-width search's key, so the
//! region counters see the same transpositions.  Running out of the node budget
//! is not a verdict: [`build`] returns `None` and the caller falls back to the
//! full-width search.
//!
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::cdfpn::ClassicalProber;
use crate::movegen::MoveBuf;
use crate::position::{Move, Position};
use crate::solver::{
    capture_forms_of, sliding_check_line, table_key, Futility, Graph, GraphMove, KeyHasher,
    NodeSpec, Outcome,
};

/// The part of the full-width graph the gate reads, and what it cost.
pub struct Built {
    /// The root's mate length.
    pub distance: u32,
    /// `T` and the answers to its wrong checks; node 0 is the root.
    pub graph: Graph,
    /// Prober nodes spent.
    pub nodes: u64,
    /// Nodes found already built under the full-width key `(r, position)`:
    /// the transpositions of the built graph (the `cacheHits` item 1 publishes
    /// in the `solve` stage).
    pub transpositions: u64,
}

/// Out of budget (or an inconsistency the caller must not trust).
struct Abort;

type Memo = HashMap<[u64; 2], u32, BuildHasherDefault<KeyHasher>>;

struct Builder<'p> {
    prober: &'p mut ClassicalProber,
    attacker: u8,
    budget: u64,
    spent: u64,
    transpositions: u64,
    buf: MoveBuf,
    futility: Futility,
    futile: Vec<Option<u32>>,
    specs: Vec<NodeSpec>,
    memo: Memo,
}

/// Builds `T` for the normalised attack position `root` with `attacker` to
/// move, under rule F classical, at bound `max_ply`.  `None` when there is no
/// mate within `max_ply` or the prober would need more than `budget` nodes.
/// `prober` must have been reset for `attacker`; its tables are reused.
pub fn build(
    prober: &mut ClassicalProber,
    root: &Position,
    attacker: u8,
    max_ply: u32,
    budget: u64,
) -> Option<Built> {
    let mut builder = Builder::new(prober, attacker, budget);
    let mut pos = root.clone();
    let distance = builder.attack_distance(&mut pos, max_ply).ok()??;
    builder.attack(&mut pos, max_ply, distance).ok()?;
    let nodes = builder.spent;
    Some(Built {
        distance,
        graph: Graph::from_specs(0, builder.specs),
        nodes,
        transpositions: builder.transpositions,
    })
}

impl<'p> Builder<'p> {
    fn new(prober: &'p mut ClassicalProber, attacker: u8, budget: u64) -> Self {
        Builder {
            prober,
            attacker,
            budget,
            spent: 0,
            transpositions: 0,
            buf: MoveBuf::new(),
            futility: Futility::new(),
            futile: Vec::new(),
            specs: Vec::new(),
            memo: Memo::default(),
        }
    }

    /// Whether `pos` mates within `bound` (rule-F plies).
    fn mate(&mut self, pos: &mut Position, bound: u32) -> Result<bool, Abort> {
        let prober = &mut *self.prober;
        let before = prober.stats().nodes;
        let left = self.budget.saturating_sub(self.spent);
        let started = crate::posprof::start();
        let verdict = prober.mate_within(pos, bound, left);
        crate::posprof::charge(crate::posprof::K::MainProberNs, started);
        crate::posprof::add(crate::posprof::K::MainCalls, 1);
        crate::posprof::bound(bound);
        crate::posprof::add(
            crate::posprof::K::MainProberNodes,
            prober.stats().nodes - before,
        );
        self.spent += prober.stats().nodes - before;
        match verdict {
            Outcome::Mate => Ok(true),
            Outcome::Refuted => Ok(false),
            Outcome::Unknown => Err(Abort),
        }
    }

    /// The mate length of a position within `upper`: the smallest bound of the
    /// side's parity that mates (attack nodes odd, defense nodes even).
    fn distance(&mut self, pos: &mut Position, upper: u32) -> Result<Option<u32>, Abort> {
        if !self.mate(pos, upper)? {
            return Ok(None);
        }
        let mut bound = if pos.side == self.attacker { 1 } else { 0 };
        while bound < upper {
            if self.mate(pos, bound)? {
                return Ok(Some(bound));
            }
            bound += 2;
        }
        Ok(Some(upper))
    }

    fn attack_distance(&mut self, pos: &mut Position, upper: u32) -> Result<Option<u32>, Abort> {
        if upper == 0 {
            return Ok(None);
        }
        self.distance(pos, upper)
    }

    /// The legal replies of the defense node `pos`, USI order.
    fn replies(&mut self, pos: &mut Position) -> Vec<Move> {
        let range = self.buf.gen_legal(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let moves = self.buf.buf[range.clone()].to_vec();
        self.buf.buf.truncate(range.start);
        moves
    }

    /// F2 verdicts of the replies of `pos` (`moves`, USI order): `Some` per
    /// reply when every reply is an interposition drop, `None` otherwise.
    fn futile_flags(&mut self, pos: &mut Position, moves: &[Move]) -> Option<Vec<Option<u32>>> {
        let start = self.buf.buf.len();
        self.buf.buf.extend_from_slice(moves);
        let range = start..start + moves.len();
        let base = self.futile.len();
        let classical =
            self.futility
                .classify(&mut self.buf, self.attacker, pos, range, &mut self.futile);
        self.buf.buf.truncate(start);
        let flags = classical.then(|| self.futile[base..].to_vec());
        self.futile.truncate(base);
        flags
    }

    /// Whether the reply `m` of `pos` leads to a position mating within `bound`.
    fn child_mates(&mut self, pos: &mut Position, m: Move, bound: u32) -> Result<bool, Abort> {
        let undo = pos.make(m);
        let result = self.mate(pos, bound);
        pos.unmake(m, undo);
        result
    }

    /// The exact mate length of the child after `m`, within `upper` (`None`
    /// for no bound at all).
    fn child_distance(
        &mut self,
        pos: &mut Position,
        m: Move,
        upper: Option<u32>,
    ) -> Result<Option<u32>, Abort> {
        let Some(upper) = upper else {
            return Ok(None);
        };
        let undo = pos.make(m);
        let result = if pos.side == self.attacker {
            self.attack_distance(pos, upper)
        } else {
            self.distance(pos, upper)
        };
        pos.unmake(m, undo);
        result
    }

    /// Whether some capture form of the checker on the drop `m` (at the credit
    /// line `pos`) leads to a defense node mating within `bound`.
    fn capture_mates(
        &mut self,
        pos: &mut Position,
        m: Move,
        checker: u8,
        bound: u32,
    ) -> Result<bool, Abort> {
        let undo = pos.make(m);
        let forms = capture_forms_of(&mut self.buf, self.attacker, pos, checker, m.to);
        let mut found = Ok(false);
        for form in forms.iter().flatten() {
            let capture = pos.make(*form);
            let mates = self.mate(pos, bound);
            pos.unmake(*form, capture);
            match mates {
                Ok(true) => {
                    found = Ok(true);
                    break;
                }
                Ok(false) => {}
                Err(abort) => {
                    found = Err(abort);
                    break;
                }
            }
        }
        pos.unmake(m, undo);
        found
    }

    /// Builds the attack node `pos` (remaining `r`, mate length `d`) and returns
    /// its id.
    fn attack(&mut self, pos: &mut Position, r: u32, d: u32) -> Result<u32, Abort> {
        let key = table_key(r, pos.hash());
        if let Some(&id) = self.memo.get(&key) {
            self.transpositions += 1;
            return Ok(id);
        }
        let id = self.specs.len() as u32;
        self.memo.insert(key, id);
        self.specs.push(NodeSpec {
            attack: true,
            position: pos.clone(),
            outcome: Outcome::Mate,
            distance: Some(d),
            moves: Vec::new(),
        });
        let range = self.buf.gen_checks(pos, true);
        self.buf.buf[range.clone()].sort_unstable_by_key(Move::usi_key);
        let checks = self.buf.buf[range.clone()].to_vec();
        self.buf.buf.truncate(range.start);
        let mut moves = Vec::with_capacity(checks.len());
        for m in checks {
            let undo = pos.make(m);
            let built = self.check(pos, r - 1, d);
            pos.unmake(m, undo);
            let (outcome, distance, next, reply) = built?;
            moves.push(GraphMove::new(m, outcome, distance, next, reply, false));
        }
        self.specs[id as usize].moves = moves;
        Ok(id)
    }

    /// The defense node `pos` reached by a check from an attack node of length
    /// `d`, at remaining `s`: `(outcome, distance of the check, child, reply)`.
    #[allow(clippy::type_complexity)]
    fn check(
        &mut self,
        pos: &mut Position,
        s: u32,
        d: u32,
    ) -> Result<(Outcome, Option<u32>, Option<u32>, Option<Move>), Abort> {
        // Correct: the child mates within d - 1 (and not sooner, d being the
        // attack node's length).
        if self.mate(pos, d - 1)? {
            let child = self.defense(pos, s, d - 1)?;
            return Ok((Outcome::Mate, Some(d), Some(child), None));
        }
        match self.distance(pos, s)? {
            None => {
                let reply = self.escape(pos, s)?;
                Ok((Outcome::Refuted, None, None, reply))
            }
            Some(delta) => {
                let reply = self.longest_resistance(pos, delta)?;
                Ok((Outcome::Mate, Some(delta + 1), None, reply))
            }
        }
    }

    /// The rule-F reading of the MATE defense node `pos` (length `delta`): per
    /// reply in USI order, `(useless, distance as recorded)`, and the credit
    /// line's `dist_other` when it has one.  Rule F flags exactly as the
    /// full-width search does: F2 at an all-drop node; at a credit line a drop
    /// whose capture form mates within `dist_other` (bound-independent, see the
    /// module docs).  The recorded distance of a reply that is not useless is
    /// its child's length plus one.
    fn mate_replies(
        &mut self,
        pos: &mut Position,
        moves: &[Move],
        upper: u32,
    ) -> Result<Vec<(bool, Option<u32>)>, Abort> {
        let futile = self.futile_flags(pos, moves);
        if let Some(flags) = futile {
            let mut out = Vec::with_capacity(moves.len());
            for (&m, futile) in moves.iter().zip(flags) {
                match futile {
                    Some(plies) => out.push((true, Some(plies))),
                    None => {
                        let child = self.child_distance(pos, m, upper.checked_sub(1))?;
                        out.push((false, child.map(|c| c + 1)));
                    }
                }
            }
            return Ok(out);
        }
        let line = credit_line(pos, self.attacker, moves);
        let Some(line) = line else {
            let mut out = Vec::with_capacity(moves.len());
            for &m in moves {
                let child = self.child_distance(pos, m, upper.checked_sub(1))?;
                out.push((false, child.map(|c| c + 1)));
            }
            return Ok(out);
        };
        let mut out: Vec<(bool, Option<u32>)> = vec![(false, None); moves.len()];
        let mut dist_other = 0u32;
        for (index, &m) in moves.iter().enumerate() {
            if line.interposes(m) {
                continue;
            }
            let child = self.child_distance(pos, m, upper.checked_sub(1))?;
            let distance = child.map(|c| c + 1);
            dist_other = dist_other.max(distance.unwrap_or(0));
            out[index] = (false, distance);
        }
        for (index, &m) in moves.iter().enumerate() {
            if !line.interposes(m) {
                continue;
            }
            let child = self.child_distance(pos, m, upper.checked_sub(1))?;
            let useless = self.capture_mates(pos, m, line.checker, dist_other)?;
            out[index] = match (useless, child) {
                (false, child) => (false, child.map(|c| c + 1)),
                (true, Some(c)) => (true, Some(c + 1)),
                // First pass: a useless drop whose child does not mate is
                // recorded with the capture line, `2 + dist_other`.
                (true, None) => (true, Some(dist_other + 2)),
            };
        }
        Ok(out)
    }

    /// Builds the MATE defense node `pos` (remaining `s`, length `delta`).
    fn defense(&mut self, pos: &mut Position, s: u32, delta: u32) -> Result<u32, Abort> {
        let key = table_key(s, pos.hash());
        if let Some(&id) = self.memo.get(&key) {
            self.transpositions += 1;
            return Ok(id);
        }
        let id = self.specs.len() as u32;
        self.memo.insert(key, id);
        self.specs.push(NodeSpec {
            attack: false,
            position: pos.clone(),
            outcome: Outcome::Mate,
            distance: Some(delta),
            moves: Vec::new(),
        });
        let replies = self.replies(pos);
        let mut moves = Vec::with_capacity(replies.len());
        if !replies.is_empty() {
            let read = self.mate_replies(pos, &replies, s)?;
            for (&m, (useless, distance)) in replies.iter().zip(read) {
                if useless {
                    // A flagged reply is outside T; the full-width graph keeps
                    // its child (or none, for an F2 drop), which no reader of T
                    // follows through a flagged reply.
                    moves.push(GraphMove::new(m, Outcome::Mate, distance, None, None, true));
                    continue;
                }
                let Some(distance) = distance else {
                    // A reply that does not mate at a MATE node: inconsistent.
                    return Err(Abort);
                };
                let undo = pos.make(m);
                let child = self.attack(pos, s - 1, distance - 1);
                pos.unmake(m, undo);
                moves.push(GraphMove::new(
                    m,
                    Outcome::Mate,
                    Some(distance),
                    Some(child?),
                    None,
                    false,
                ));
            }
        }
        self.specs[id as usize].moves = moves;
        Ok(id)
    }

    /// The reply the full-width graph records first as `REFUTED` at the
    /// REFUTED defense node `pos` (remaining `s`), which is what
    /// [`crate::exact::selected_reply`] reads after a refuted check.
    fn escape(&mut self, pos: &mut Position, s: u32) -> Result<Option<Move>, Abort> {
        let replies = self.replies(pos);
        if replies.is_empty() {
            // Stalemate: no recorded reply, no refutation.
            return Ok(None);
        }
        let futile = self.futile_flags(pos, &replies);
        let is_futile = |index: usize| futile.as_ref().is_some_and(|f| f[index].is_some());
        if s == 0 {
            // Horizon escape: the first reply that is not futile.
            return Ok((0..replies.len())
                .find(|&index| !is_futile(index))
                .map(|index| replies[index]));
        }
        let line = if futile.is_some() {
            None
        } else {
            credit_line(pos, self.attacker, &replies)
        };
        let Some(line) = line else {
            for (index, &m) in replies.iter().enumerate() {
                if is_futile(index) {
                    continue;
                }
                if !self.child_mates(pos, m, s - 1)? {
                    return Ok(Some(m));
                }
            }
            return Err(Abort);
        };
        // A credit line: with a refuted other reply the replies keep their
        // searched outcomes, so the first refuted one in USI order.
        let mut child_mates: Vec<bool> = Vec::with_capacity(replies.len());
        let mut other_refuted = false;
        for &m in replies.iter() {
            let mates = self.child_mates(pos, m, s - 1)?;
            if !line.interposes(m) && !mates {
                other_refuted = true;
            }
            child_mates.push(mates);
        }
        if other_refuted {
            let first = replies
                .iter()
                .zip(child_mates.iter())
                .find(|(_, &mates)| !mates)
                .map(|(&m, _)| m);
            return Ok(first);
        }
        // Every other reply mates: the first effective drop escapes.
        let mut dist_other = 0u32;
        for &m in replies.iter() {
            if line.interposes(m) {
                continue;
            }
            let child = self.child_distance(pos, m, s.checked_sub(1))?;
            dist_other = dist_other.max(child.map_or(0, |c| c + 1));
        }
        for (index, &m) in replies.iter().enumerate() {
            if !line.interposes(m) || child_mates[index] {
                continue;
            }
            if dist_other + 2 <= s || !self.capture_mates(pos, m, line.checker, dist_other)? {
                return Ok(Some(m));
            }
        }
        Err(Abort)
    }

    /// [`crate::exact::longest_resistance`] of the MATE defense node `pos`
    /// (length `delta`): the reply that is not 無駄合 with the longest mate,
    /// ties on the smallest USI.
    fn longest_resistance(
        &mut self,
        pos: &mut Position,
        delta: u32,
    ) -> Result<Option<Move>, Abort> {
        let replies = self.replies(pos);
        if replies.is_empty() {
            return Ok(None);
        }
        let read = self.mate_replies(pos, &replies, delta)?;
        let mut best: Option<(Move, i64)> = None;
        for (&m, (useless, distance)) in replies.iter().zip(read) {
            if useless {
                continue;
            }
            let candidate = distance.map_or(-1i64, i64::from);
            best = match best {
                None => Some((m, candidate)),
                Some((incumbent, current)) => {
                    if candidate > current
                        || (candidate == current && m.to_usi() < incumbent.to_usi())
                    {
                        Some((m, candidate))
                    } else {
                        Some((incumbent, current))
                    }
                }
            };
        }
        Ok(best.map(|(m, _)| m))
    }
}

/// The credit line of a defense node whose legal replies are `moves`: an
/// interposable check with both interposition drops and other replies
/// ([`crate::solver`]'s `credit_line`).
fn credit_line(pos: &Position, attacker: u8, moves: &[Move]) -> Option<crate::solver::CheckLine> {
    if !moves.iter().any(|m| m.is_drop()) || moves.iter().all(|m| m.is_drop()) {
        return None;
    }
    let line = sliding_check_line(pos, attacker)?;
    let drops = moves.iter().filter(|&&m| line.interposes(m)).count();
    (drops > 0 && drops < moves.len()).then_some(line)
}
