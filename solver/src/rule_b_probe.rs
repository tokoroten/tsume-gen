//! A df-pn **rule B pre-filter**: reject yozume cheaply, never accept.
//!
//! # Why this exists
//!
//! Rule B is the dominant wall of the composition gate, and `distill --probe
//! dfpn` pays a full exact search for every trial that fails only it.  A df-pn
//! *proof* settles part of that question cheaply: a prober proof of "a forced mate
//! within `N` plies" is a mate within the bound `N` for the exact gate too, and a
//! mate's rule-F distance never exceeds the bound that proves it.  The prober runs
//! in the rule-F mode of the rules ([`crate::exact::dfpn_useless_mode`]): under
//! the classical reading it applies F2, so it also proves the distant checks whose
//! only answers are futile drops -- exactly the extra correct moves (and yozume) F2
//! creates.  Two such proofs at one attack node of the main line are rule B's
//! defect (or a mate shorter than the target), and that is what this module
//! looks for.
//!
//! **Reject-only.**  [`rule_b_reject`] answers either "the exact gate rejects this
//! diagram at this target" or "don't know".  It never accepts and it never decides
//! a published verdict; a caller that gets [`RuleBOutcome::Inconclusive`] runs the
//! exact gate exactly as it would have without this module.  Like [`crate::dfpn`]
//! it is Rust-only tooling with no TypeScript counterpart.  Its caller is the
//! `rule-b-probe` command (verification); `distill`'s opt-in use of it,
//! `DistillOptions::yozume_prefilter`, went on 2026-10-27.
//!
//! # The walk
//!
//! Write `D` for the exact gate's mate distance (rule F applied), `target` for the
//! length the caller demands, and suppose -- for contradiction -- that the diagram
//! **passes** the exact gate at `target`.  Then `D == target`, the root has exactly
//! one correct move, and (with the composition rules on) every attack node of the
//! main-line region `R` at distance `>= 3` has exactly one correct move
//! ([`crate::exact::designated_main_line`]).
//!
//! 1. At an attack node `n` of `R` with mate distance `r`, a checking move whose
//!    child df-pn proves "mate within `r - 1`" has a rule-F distance `<= r`, and
//!    `r` is the minimum over `n`'s moves, so the move is **correct**.  Two proven
//!    moves are therefore two correct moves: `first_move_not_unique` at the root,
//!    `yozume_in_main_line` deeper (`r >= 3`; the final move is exempt) --
//!    unless they are exactly a promotion twin (the same from- and to-squares),
//!    which rule B counts as one when the twins merge (`PIPELINE_VERSION`
//!    0.17.0): that walk stops inconclusive (`promotion_twin`), since merging
//!    reads the replies' main-line candidates, which only the exact gate has.
//! 2. With exactly one proven move `A`, `A` is *the* unique correct move (every
//!    proven move is correct, and a passing diagram has only one).  If the
//!    defender then has exactly **one** legal reply, that reply is in `R`: it is
//!    not a futile 無駄合 (a defence node whose only reply is futile is a mate leaf
//!    of distance 0, but this one has distance `r - 1 >= 2`), so it is the only
//!    longest reply and the only main-line candidate.  The next attack node is in
//!    `R` with distance `r - 2`, and step 1 applies to it again.
//! 3. If `A` leaves the defender **no** legal reply, `A` mates at once.  This one
//!    needs no assumption: each walked attack node mates at most two plies later
//!    than the next one (the proven move, then the forced reply -- or sooner, if
//!    that reply is futile), so `D <= ply + 1`, which is `< target` while
//!    `r = target - ply >= 3`: `target_mate_ply_changed`.
//!
//! Every reject is such a contradiction, so the exact gate rejects the diagram --
//! with the reason named here whenever the levels above the reject really were as
//! the walk assumed (`D == target` and the moves walked were correct), and with
//! some other reason (`target_mate_ply_changed`, a budget) otherwise.  The walk
//! stops, inconclusive, as soon as the defender has a choice (the region may branch
//! there, and which branch is `R` depends on `Amari`/`Clean`), no checking move is
//! proven, `r` drops below 3, or the node budget is spent.
//!
//! # What it never touches
//!
//! The exact solver, the gate, rule F and every verdict.  Rule B is applied exactly
//! where the exact gate applies it: the root test at every target (whether or not
//! the composition rules are on -- the root code is part of the bare contract),
//! the deeper tests only with the composition rules on.  Nothing at all happens
//! under [`TsumeRules::allow_yozume`].  (Until `PIPELINE_VERSION` 0.12.0 nothing
//! happened from 9 plies up either, unless `--strict-rule-b` was set.)
//!
//! # Budget and determinism
//!
//! One node budget covers every df-pn probe of a call (spec §35: nodes, never
//! time).  The probes at one attack node run in rounds with a growing per-probe
//! cap -- [`FIRST_ROUND_CAP`], then four times that, and so on -- so a cheap
//! alternative mate is found before an expensive one eats the budget; each probe
//! starts from a cleared table ([`Prober::prove`]), so the result depends only on
//! the position, the target, the rules and the budget.

use crate::dfpn::{ProbeOptions, Prober, Verdict};
use crate::exact::{dfpn_useless_mode, TsumeRules};
use crate::movegen::{gives_check, legal_moves_sorted};
use crate::position::{Move, Position};
use crate::solver::UselessInterposition;
use crate::strict_material;

/// Per-probe node cap of the first round at one attack node.
pub const FIRST_ROUND_CAP: u64 = 1_024;
/// Growth of the per-probe cap from one round to the next.
const ROUND_GROWTH: u64 = 4;

/// The exact gate's reason a reject stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleBReason {
    /// Two checking moves at the root both mate within the target.
    FirstMoveNotUnique,
    /// Two checking moves at a forced main-line attack node of distance `>= 3`.
    YozumeInMainLine,
    /// The only proven move of a forced main-line node mates at once, so the mate
    /// is shorter than the target.
    TargetMatePlyChanged,
}

impl RuleBReason {
    /// The exact gate's spelling of the reason.
    pub const fn as_str(self) -> &'static str {
        match self {
            RuleBReason::FirstMoveNotUnique => "first_move_not_unique",
            RuleBReason::YozumeInMainLine => "yozume_in_main_line",
            RuleBReason::TargetMatePlyChanged => "target_mate_ply_changed",
        }
    }
}

/// Why a walk stopped without a reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inconclusive {
    /// Rule B is not enforced under these rules ([`TsumeRules::allow_yozume`]),
    /// or the target is 0.
    NotEnforced,
    /// The node budget ran out.
    Budget,
    /// The defender has more than one legal reply after the single proven move.
    DefenderChoice,
    /// No checking move of the attack node was proven within the bound.
    NoProvenMove,
    /// The next attack node would be at distance `< 3` (rule B's final-move
    /// exemption), or deeper tests are off because the composition rules are.
    Horizon,
    /// Exactly two checking moves were proven and they are a promotion twin
    /// (the same from- and to-squares): rule B counts them as one correct move
    /// when they merge (`PIPELINE_VERSION` 0.17.0,
    /// [`crate::exact::is_merged_promotion_twin`]), which reads the solution
    /// tree, so the exact gate decides.
    PromotionTwin,
}

impl Inconclusive {
    pub const fn as_str(self) -> &'static str {
        match self {
            Inconclusive::NotEnforced => "rule_b_not_enforced",
            Inconclusive::Budget => "budget_exhausted",
            Inconclusive::DefenderChoice => "defender_choice",
            Inconclusive::NoProvenMove => "no_proven_move",
            Inconclusive::Horizon => "horizon",
            Inconclusive::PromotionTwin => "promotion_twin",
        }
    }
}

/// What [`rule_b_reject`] concluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleBOutcome {
    /// The exact gate rejects this diagram at this target.
    Reject(RuleBReason),
    /// Nothing settled; the exact gate decides.
    Inconclusive(Inconclusive),
}

/// One [`rule_b_reject`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleBProbe {
    pub outcome: RuleBOutcome,
    /// df-pn nodes spent over every probe of the call.  A different unit from
    /// `SolveStats::nodes`, and never published.
    pub nodes: u64,
    /// Ply of the deepest attack node examined: `0` for the root, then `2`, `4`, ….
    pub ply_reached: u32,
    /// The most probe rounds one attack node needed: `1` when every checking move
    /// was settled under [`FIRST_ROUND_CAP`], more when the cap had to widen;
    /// `0` when nothing was probed.
    pub max_round: u32,
}

impl RuleBProbe {
    pub fn is_reject(&self) -> bool {
        matches!(self.outcome, RuleBOutcome::Reject(_))
    }
    /// The reject reason, `None` when inconclusive.
    pub fn reason(&self) -> Option<RuleBReason> {
        match self.outcome {
            RuleBOutcome::Reject(reason) => Some(reason),
            RuleBOutcome::Inconclusive(_) => None,
        }
    }
}

/// Whether the exact gate enforces rule B under `rules`: the negation of
/// `rule_b_skipped` in [`crate::exact::exact_validate_with_rules`].  Rule B has no
/// length condition since `PIPELINE_VERSION` 0.12.0, so the target does not enter.
pub fn rule_b_applies(rules: TsumeRules) -> bool {
    !rules.allow_yozume
}

/// What the probes at one attack node found.
enum Scan {
    /// No checking move proven; every one was disproven.
    None,
    /// Exactly one proven, every other one disproven.
    One(Move),
    /// At least two proven, and not just one promotion twin.
    Two,
    /// Exactly two proven, a promotion twin; every other one disproven.
    Twin,
    /// The budget ran out with the question open.
    Exhausted,
}

/// The rule B pre-filter; see the module docs for the walk and why a reject is
/// sound.
///
/// `position` is the diagram with the attacker to move; it is put under the strict
/// material convention first, exactly as the exact gate and [`Prober::prove`] do
/// with their defaults.  `target` is the mate length the caller demands
/// (`exact_validate_with_rules(.., Some(target), target, ..)` in `distill`), and
/// `rules` the rule set that gate runs under.
pub fn rule_b_reject(
    prober: &mut Prober,
    position: &Position,
    target: u32,
    rules: TsumeRules,
    node_budget: u64,
) -> RuleBProbe {
    rule_b_reject_with_first_cap(
        prober,
        position,
        target,
        rules,
        node_budget,
        FIRST_ROUND_CAP,
    )
}

/// [`rule_b_reject`] with an explicit first-round per-probe cap instead of
/// [`FIRST_ROUND_CAP`] (`>= 1`).
///
/// The cap only orders the work.  Given a budget that settles every probe, the
/// outcome and `ply_reached` do not depend on it: two proven moves are found
/// whenever two exist, and a single proven move is the same move whichever round
/// proves it.  Only `nodes` and `max_round` move.  Exposed so tests can drive the
/// widening rounds on small diagrams.
pub fn rule_b_reject_with_first_cap(
    prober: &mut Prober,
    position: &Position,
    target: u32,
    rules: TsumeRules,
    node_budget: u64,
    first_cap: u64,
) -> RuleBProbe {
    let first_cap = first_cap.max(1);
    let mut probe = RuleBProbe {
        outcome: RuleBOutcome::Inconclusive(Inconclusive::NotEnforced),
        nodes: 0,
        ply_reached: 0,
        max_round: 0,
    };
    if target == 0 || !rule_b_applies(rules) {
        return probe;
    }
    let attacker = position.side;
    let mode = dfpn_useless_mode(rules);
    let mut node = strict_material::normalize(position, attacker);
    let mut remaining = target;
    let mut ply = 0u32;
    loop {
        probe.ply_reached = ply;
        let checks: Vec<Move> = legal_moves_sorted(&node)
            .into_iter()
            .filter(|&m| gives_check(&node, m))
            .collect();
        let scan = scan_checks(
            prober,
            &node,
            &checks,
            remaining - 1,
            attacker,
            mode,
            node_budget,
            &mut probe.nodes,
            &mut probe.max_round,
            first_cap,
        );
        let chosen = match scan {
            Scan::Two => {
                probe.outcome = RuleBOutcome::Reject(if ply == 0 {
                    RuleBReason::FirstMoveNotUnique
                } else {
                    RuleBReason::YozumeInMainLine
                });
                return probe;
            }
            Scan::Twin => {
                // A merged promotion twin is one correct move (0.17.0); at the
                // root only while `ROOT_TWIN_MERGES` is on.
                probe.outcome = if ply == 0 && !crate::exact::ROOT_TWIN_MERGES {
                    RuleBOutcome::Reject(RuleBReason::FirstMoveNotUnique)
                } else {
                    RuleBOutcome::Inconclusive(Inconclusive::PromotionTwin)
                };
                return probe;
            }
            Scan::Exhausted => {
                probe.outcome = RuleBOutcome::Inconclusive(Inconclusive::Budget);
                return probe;
            }
            Scan::None => {
                probe.outcome = RuleBOutcome::Inconclusive(Inconclusive::NoProvenMove);
                return probe;
            }
            Scan::One(m) => m,
        };
        let mut child = node.clone();
        child.make(chosen);
        let replies = legal_moves_sorted(&child);
        if replies.is_empty() {
            // `chosen` is a check, so this is mate in one (step 3 of the docs).
            probe.outcome = RuleBOutcome::Inconclusive(Inconclusive::Horizon);
            if remaining >= 3 {
                probe.outcome = RuleBOutcome::Reject(RuleBReason::TargetMatePlyChanged);
            }
            return probe;
        }
        if replies.len() > 1 {
            probe.outcome = RuleBOutcome::Inconclusive(Inconclusive::DefenderChoice);
            return probe;
        }
        // Below the root rule B is a composition rule, and the final move (and
        // anything shorter) is exempt.
        if !rules.composition_rules || remaining < 5 {
            probe.outcome = RuleBOutcome::Inconclusive(Inconclusive::Horizon);
            return probe;
        }
        child.make(replies[0]);
        node = child;
        remaining -= 2;
        ply += 2;
    }
}

/// Probes every checking move of one attack node for "mate within `bound` plies
/// from the child", in rounds of growing per-probe caps, until two are proven,
/// every one is settled, or the shared budget is gone.
#[allow(clippy::too_many_arguments)]
fn scan_checks(
    prober: &mut Prober,
    node: &Position,
    checks: &[Move],
    bound: u32,
    attacker: u8,
    mode: UselessInterposition,
    budget: u64,
    spent: &mut u64,
    max_round: &mut u32,
    first_cap: u64,
) -> Scan {
    let mut open: Vec<Move> = checks.to_vec();
    let mut proven: Vec<Move> = Vec::new();
    let mut cap = first_cap;
    let mut round = 0u32;
    loop {
        round += 1;
        *max_round = (*max_round).max(round);
        let mut unsettled = Vec::new();
        for &m in open.iter() {
            let left = budget.saturating_sub(*spent);
            if left == 0 {
                return Scan::Exhausted;
            }
            let mut child = node.clone();
            child.make(m);
            // The child is already strict-normalised (material is conserved by a
            // move), and its side to move is the defender, so the attacker is
            // named explicitly.
            let result = prober.prove(
                &child,
                &ProbeOptions::default()
                    .max_ply(bound)
                    .node_budget(cap.min(left))
                    .attacker(attacker)
                    .strict_material(false)
                    .useless(mode),
            );
            *spent += result.nodes;
            match result.verdict {
                Verdict::Proven => {
                    // A second proven move is rule B's defect unless it is the
                    // promotion twin of the first (which may merge).
                    match proven.as_slice() {
                        [] => {}
                        [first] if crate::exact::is_promotion_twin(*first, m) => {}
                        _ => return Scan::Two,
                    }
                    proven.push(m);
                }
                Verdict::Disproven => {}
                Verdict::Unknown => unsettled.push(m),
            }
        }
        if unsettled.is_empty() {
            return match proven.as_slice() {
                [] => Scan::None,
                [m] => Scan::One(*m),
                _ => Scan::Twin,
            };
        }
        if *spent >= budget {
            return Scan::Exhausted;
        }
        open = unsettled;
        cap = cap.saturating_mul(ROUND_GROWTH);
    }
}
