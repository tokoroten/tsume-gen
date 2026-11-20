//! The kind of a disproof (`probe --engine cdfpn`'s `disproofKind`, owner
//! request 2026-11-16): after the prober refuted "mate within `b`?", was the
//! refutation independent of the bound (**checks exhausted**: no bound, however
//! large, gives a mate) or did some checking sequence reach the bound
//! (**horizon**: a deeper mate may exist)?
//!
//! # How
//!
//! A walk *after* the search, which reads the prober's tables and generates
//! moves but never writes a table, never spends the budget and never touches
//! the counters: every verdict, length and node count is the search's, and
//! the walk runs only when the probe command asks for it.
//!
//! The walk proves a stronger fact than the search did: "**never mate**", at
//! any bound.  By induction on the walk (each claim built only from claims
//! already proven):
//!
//! * an attack node with no legal check is never mate;
//! * an attack node is never mate when every check's defense node is never
//!   mate;
//! * a defense node is never mate when one of its replies that the
//!   classical AND node counts (not a drop rule F2 proves futile; on a credit
//!   line, a reply other than the interposition drops) leads to an attack
//!   node that is never mate -- in particular a reply after which the
//!   attacker has no check;
//! * on a credit line, a drop `r` refutes at every bound when `C_r` (the
//!   attack node after the drop) is never mate and `Sub(r, q)` fails at every
//!   bound: either every capture form's `E` is never mate (or there is no
//!   capture form), or `Witness(q)` is disproven in the table (every other
//!   reply's attack node mates within `q - 2`, which stays true at every
//!   larger bound) **and** the defense node is refuted in the table at the
//!   walk's bound (monotonicity covers the bounds below).
//!
//! The verdicts rule F2 and the credit line depend on are facts about the
//! position, not the bound, so "never mate" is a fact about the position.
//! The tables only steer the walk: at a defense node it tries the replies
//! whose attack node the tables hold refuted first (near the leaves, where
//! the leaf search keeps no entries, every reply), then the other replies (the
//! refutation the tables hold may have used the bound where another reply
//! refutes for good).  The walk's bound follows the tables' refuted bounds,
//! so its proofs are found within about the probed depth.  Its steps are
//! capped at a quarter of the search's node count plus 2^20 (measured on the
//! ply-17 study's first 200 lines: the same split as a cap of the whole node
//! count, at about half the extra CPU).
//!
//! The answer is **conservative**: `ChecksExhausted` is a proof that raising
//! the bound can never give a mate; `Horizon` means only that the walk found
//! no such proof -- some checking line reached the bound, or the refutation
//! the tables hold used the bound somewhere (another defence that refutes
//! for good may exist), or an entry the walk needed was evicted, or the
//! walk's own step cap ran out, or the walk met a repetition (a position
//! already on its path: treated as reaching the horizon, never as a
//! refutation).  With `--prober-threads > 1` the tables differ from one
//! thread's, so the kind may differ in those conservative cases (the
//! verdicts never do).

use std::collections::HashSet;

use super::*;

/// The kind of a `REFUTED` answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisproofKind {
    /// Not mate at any bound: every checking line ends in a position where
    /// the attacker has no check (or every check is refuted for good).
    ChecksExhausted,
    /// The walk found no bound-free refutation: a deeper mate may exist.
    Horizon,
}

impl DisproofKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DisproofKind::ChecksExhausted => "checksExhausted",
            DisproofKind::Horizon => "horizon",
        }
    }
}

/// The walk's recursion depth cap (a deeper claim is not attempted).
const WALK_DEPTH_MAX: u32 = 96;

/// The walk's step cap: this share of the search's node count, plus [`WALK_STEPS_EXTRA`].
const WALK_STEPS_SHARE: u64 = 4;

/// The walk's step cap beyond its share of the search's node count.
const WALK_STEPS_EXTRA: u64 = 1 << 20;

/// The replies of a defense node as the classical AND node counts them.
enum Replies {
    /// No legal reply: mate in check, never mate out of check.
    None { in_check: bool },
    /// Every reply an F2-futile drop: mate at every bound.
    AllFutile,
    /// The replies whose attack node is a child, and the interposition drops
    /// of a credit line (empty off one).
    Moves { plain: Vec<Move>, drops: Vec<Move> },
}

#[derive(Default)]
struct Walk {
    /// `(kind tag, position hash)` proven never mate.
    free: HashSet<(u64, u128)>,
    /// `(kind tag, position hash, bound)` the walk could not prove.
    failed: HashSet<(u64, u128, u32)>,
    /// On the walk's path (a repetition is not a refutation).
    open: HashSet<(u64, u128)>,
    steps: u64,
    limit: u64,
    /// Times the walk met a position on its path; a failure that met one is
    /// not remembered (it depends on the path, not the position).
    cycles: u64,
}

impl ClassicalProber {
    /// The kind of the refutation the last question left in the tables:
    /// call right after `mate_within(position, bound)` answered `REFUTED`
    /// (before a `reset`), with `root` the position that was asked.
    pub fn disproof_kind(&mut self, root: &Position, bound: u32) -> DisproofKind {
        let mut walk = Walk {
            limit: (self.stats.nodes / WALK_STEPS_SHARE).saturating_add(WALK_STEPS_EXTRA),
            ..Walk::default()
        };
        let mut pos = root.clone();
        let free = if pos.side == self.attacker {
            self.walk_attack(&mut pos, bound, 0, &mut walk)
        } else {
            self.walk_defense(&mut pos, bound, 0, &mut walk)
        };
        if free {
            DisproofKind::ChecksExhausted
        } else {
            DisproofKind::Horizon
        }
    }

    /// The largest bound `>= bound` at which the tables hold the real node
    /// `kind` at `pos` refuted, if any (read only).
    fn refuted_at(&self, pos: &Position, kind: Kind, bound: u32) -> Option<u32> {
        let hash = pos.hash();
        let oracle = crate::solver::oracle_key_of(hash);
        let below = match self.par.as_ref() {
            None => self.intervals.get(&oracle).map(|entry| entry.refuted_below),
            Some(par) => par.facts.get(&oracle).map(|hit| unpack_fact(hit.d).1),
        };
        if let Some(below) = below {
            if below > 0 && u32::from(below) > bound {
                return Some(u32::from(below) - 1);
            }
        }
        let node = Node {
            kind,
            bound,
            drop: Move::drop(0, 0),
        };
        let (_, dn, _, _) = self.read(&node_key(&node, hash));
        (dn == 0).then_some(bound)
    }

    /// The checks of `pos` (USI order).
    fn walk_checks(&mut self, pos: &mut Position) -> Vec<Move> {
        let range = self.buf.gen_checks(pos, true);
        let mut checks = self.buf.buf[range.clone()].to_vec();
        self.buf.buf.truncate(range.start);
        Move::sort_usi(&mut checks);
        checks
    }

    /// Whether the attacker has a check in `pos`.
    fn has_check(&mut self, pos: &mut Position) -> bool {
        let range = self.buf.gen_checks(pos, true);
        let any = !range.is_empty();
        self.buf.buf.truncate(range.start);
        any
    }

    /// The defense node's replies as [`ClassicalProber::expand_defense_into`]
    /// counts them (at a bound of 2 or more).
    fn walk_replies(&mut self, pos: &mut Position) -> Replies {
        let safety = Safety::new(pos);
        let range = self.buf.gen_legal_with(pos, true, &safety);
        Move::sort_usi(&mut self.buf.buf[range.clone()]);
        let count = range.len();
        if count == 0 {
            self.buf.buf.truncate(range.start);
            return Replies::None {
                in_check: pos.is_in_check(pos.side),
            };
        }
        let fbase = self.futile_buf.len();
        let classical = self.futility.classify_on(
            &mut self.buf,
            self.attacker,
            pos,
            range.clone(),
            &mut self.futile_buf,
            safety.check_line(),
        );
        let moves = self.buf.buf[range.clone()].to_vec();
        let futile: Vec<bool> = if classical {
            self.futile_buf[fbase..fbase + count]
                .iter()
                .map(Option::is_some)
                .collect()
        } else {
            Vec::new()
        };
        self.futile_buf.truncate(fbase);
        self.buf.buf.truncate(range.start);
        if classical {
            if futile.iter().all(|&f| f) {
                return Replies::AllFutile;
            }
            let plain = moves
                .iter()
                .zip(futile.iter())
                .filter(|(_, &f)| !f)
                .map(|(&m, _)| m)
                .collect();
            return Replies::Moves {
                plain,
                drops: Vec::new(),
            };
        }
        match search::credit_line(pos, self.attacker, &moves) {
            Some(line) => {
                let (drops, plain) = moves.iter().partition(|&&m| line.interposes(m));
                Replies::Moves { plain, drops }
            }
            None => Replies::Moves {
                plain: moves,
                drops: Vec::new(),
            },
        }
    }

    /// Whether the walk may claim anything more (steps and depth left);
    /// counts a step.
    fn walk_step(walk: &mut Walk, depth: u32) -> bool {
        walk.steps += 1;
        walk.steps <= walk.limit && depth <= WALK_DEPTH_MAX
    }

    /// Whether the attack node `pos` (walked at `bound`) is never mate.
    fn walk_attack(&mut self, pos: &mut Position, bound: u32, depth: u32, walk: &mut Walk) -> bool {
        let id = (Kind::Attack.tag(), pos.hash());
        if walk.free.contains(&id) {
            return true;
        }
        if walk.failed.contains(&(id.0, id.1, bound)) {
            return false;
        }
        if walk.open.contains(&id) {
            walk.cycles += 1;
            return false;
        }
        if !Self::walk_step(walk, depth) {
            return false;
        }
        let checks = self.walk_checks(pos);
        if checks.is_empty() {
            walk.free.insert(id);
            return true;
        }
        if bound == 0 {
            walk.failed.insert((id.0, id.1, bound));
            return false;
        }
        let at = self.refuted_at(pos, Kind::Attack, bound).unwrap_or(bound);
        let cycles = walk.cycles;
        walk.open.insert(id);
        let mut free = true;
        for m in checks {
            let undo = pos.make(m);
            let child = self.walk_defense(pos, at - 1, depth + 1, walk);
            pos.unmake(m, undo);
            if !child {
                free = false;
                break;
            }
        }
        walk.open.remove(&id);
        if free {
            walk.free.insert(id);
        } else if walk.cycles == cycles {
            walk.failed.insert((id.0, id.1, bound));
        }
        free
    }

    /// Whether the defense node `pos` (walked at `bound`) is never mate.
    fn walk_defense(
        &mut self,
        pos: &mut Position,
        bound: u32,
        depth: u32,
        walk: &mut Walk,
    ) -> bool {
        let id = (Kind::Defense.tag(), pos.hash());
        if walk.free.contains(&id) {
            return true;
        }
        if walk.failed.contains(&(id.0, id.1, bound)) {
            return false;
        }
        if walk.open.contains(&id) {
            walk.cycles += 1;
            return false;
        }
        if !Self::walk_step(walk, depth) {
            return false;
        }
        let (plain, drops) = match self.walk_replies(pos) {
            Replies::None { in_check } => {
                if !in_check {
                    walk.free.insert(id);
                } else {
                    walk.failed.insert((id.0, id.1, bound));
                }
                return !in_check;
            }
            Replies::AllFutile => {
                walk.failed.insert((id.0, id.1, bound));
                return false;
            }
            Replies::Moves { plain, drops } => (plain, drops),
        };
        // A reply after which the attacker has no check refutes for good.
        for &m in &plain {
            let undo = pos.make(m);
            let none = !self.has_check(pos);
            pos.unmake(m, undo);
            if none {
                walk.free.insert(id);
                return true;
            }
        }
        if bound == 0 {
            walk.failed.insert((id.0, id.1, bound));
            return false;
        }
        let refuted = self.refuted_at(pos, Kind::Defense, bound);
        let at = refuted.unwrap_or(bound);
        let cycles = walk.cycles;
        walk.open.insert(id);
        let mut free = false;
        for &m in &plain {
            let undo = pos.make(m);
            // Below bound 2 the leaf search keeps no entries: try every reply.
            let child = if at - 1 <= 1 {
                Some(at - 1)
            } else {
                self.refuted_at(pos, Kind::Attack, at - 1)
            };
            let child_free = child.is_some_and(|b| self.walk_attack(pos, b, depth + 1, walk));
            pos.unmake(m, undo);
            if child_free {
                free = true;
                break;
            }
        }
        // Then the replies the tables do not hold refuted (the refutation
        // they hold may have used the bound where another reply refutes for
        // good; every claim is a proof, the step cap bounds the cost).
        if !free && at - 1 > 1 {
            for &m in &plain {
                let undo = pos.make(m);
                let child_free = self.refuted_at(pos, Kind::Attack, at - 1).is_none()
                    && self.walk_attack(pos, at - 1, depth + 1, walk);
                pos.unmake(m, undo);
                if child_free {
                    free = true;
                    break;
                }
            }
        }
        if !free && at >= 2 {
            for &r in &drops {
                let gate = Node {
                    kind: Kind::Gate,
                    bound: at,
                    drop: r,
                };
                let (_, dn, _, _) = self.read(&node_key(&gate, pos.hash()));
                if dn != 0 {
                    continue;
                }
                if self.walk_gate(pos, at, r, refuted.is_some(), depth, walk) {
                    free = true;
                    break;
                }
            }
        }
        walk.open.remove(&id);
        if free {
            walk.free.insert(id);
        } else if walk.cycles == cycles {
            walk.failed.insert((id.0, id.1, bound));
        }
        free
    }

    /// Whether the credit-line drop `r` of the defense node `pos` refutes at
    /// every bound (`Gate(r, b)` false for every `b`); `refuted` when the
    /// tables hold the defense node refuted at `bound`.
    fn walk_gate(
        &mut self,
        pos: &mut Position,
        bound: u32,
        r: Move,
        refuted: bool,
        depth: u32,
        walk: &mut Walk,
    ) -> bool {
        let undo = pos.make(r);
        let c_free = self.walk_attack(pos, bound - 1, depth + 1, walk);
        pos.unmake(r, undo);
        if !c_free {
            return false;
        }
        let q = bound & !1;
        if refuted {
            let witness = Node {
                kind: Kind::Witness,
                bound: q,
                drop: Move::drop(0, 0),
            };
            let (_, dn, _, _) = self.read(&node_key(&witness, pos.hash()));
            if dn == 0 {
                return true;
            }
        }
        // Every capture form's `E` never mate (none: `Ecap` fails at every bound).
        let Some(line) = sliding_check_line(pos, self.attacker) else {
            return false;
        };
        let undo = pos.make(r);
        let forms = capture_forms_of(&mut self.buf, self.attacker, pos, line.checker, r.to);
        let mut free = true;
        for form in forms.iter().flatten() {
            let undo_form = pos.make(*form);
            let e_free = self.walk_defense(pos, q, depth + 1, walk);
            pos.unmake(*form, undo_form);
            if !e_free {
                free = false;
                break;
            }
        }
        pos.unmake(r, undo);
        free
    }
}
