//! One chain of greedy steps: the step rules, the repair and greedy passes,
//! each step's acceptance test (`check_reduction`), and the negatives a
//! refused step records (`--emit-negatives`).

use super::*;

/// How one reduction step is judged (rule D v5, `PIPELINE_VERSION` 0.13.0,
/// `docs/rules.md` "Reduction v5"; the main-line condition as of 0.14.0).
///
/// * [`StepRule::Length`]: the acceptance test alone, `accepted(y)` -- the only
///   judgement before 0.13.0.  The main line may change.
/// * [`StepRule::MainLine`]: `accepted(y)` and every main-line path of `y` in
///   `paths` (`Π(y) ⊆ Π(z)`).  `paths` are those of the diagram `z` the step
///   starts from; after an accepted step they are the new diagram's
///   ([`StepRule::after`]) and can only shrink.  Through 0.13.0 the displayed main
///   line of `y` also had to equal that of `z`; the owner dropped that condition
///   on 2026-10-27 because the displayed line's tie-break (the smallest USI)
///   depends on the board's orientation, so a mirrored diagram could get the
///   other verdict.  Since `λ(y) ∈ Π(y)`, the step now keeps *some* main line
///   of `z`: a piece that only supports an alternative mate can come off.
///
/// * [`StepRule::Repair`] (the opt-in C/G entry, [`TwoKindsOptions::cg_entry`];
///   not part of rule D v5): the acceptance test with the input's own defects
///   set aside -- every reason the verdict reports is a rule C or G reason, or
///   (`allow_rule_b`) rule B's.  The length, the mate, its minimality and
///   uniqueness of everything else stay enforced.  Once a step reaches a
///   diagram that passes the full test, the chain steps by
///   [`StepRule::Length`] from then on ([`StepRule::after`]).
#[derive(Clone, Debug)]
pub enum StepRule {
    Length,
    MainLine {
        /// Sorted, as [`main_line_paths`] returns them.
        paths: Arc<Vec<Vec<Usi>>>,
    },
    Repair {
        /// The input also failed rule B, so rule B's reasons are set aside too.
        allow_rule_b: bool,
    },
}

impl StepRule {
    /// Whether the checks this rule reads must enumerate `Π`.
    pub(super) fn wants_paths(&self) -> bool {
        matches!(self, StepRule::MainLine { .. })
    }

    /// The repair rule's verdict: the full test, or only the defects it sets
    /// aside.  A structurally invalid diagram reports its issue codes, so it
    /// never qualifies.
    pub(super) fn repair_accepts(check: &ReductionCheck, allow_rule_b: bool) -> bool {
        check.accepted
            || (!check.reasons.is_empty()
                && check.reasons.iter().all(|reason| {
                    RULE_CG_REASONS.contains(&reason.as_str())
                        || (allow_rule_b
                            && crate::stages::RULE_B_REASONS.contains(&reason.as_str()))
                }))
    }

    /// The main-line conditions on the verdict `check` of `y`.  A diagram whose
    /// paths were not enumerated (over the cap) fails them: the safe side.
    pub(super) fn keeps_main_line(&self, check: &ReductionCheck) -> bool {
        match self {
            StepRule::Length | StepRule::Repair { .. } => true,
            StepRule::MainLine { paths } => match &check.main_line_paths {
                MainLinePaths::Paths(found) => {
                    found.iter().all(|path| paths.binary_search(path).is_ok())
                }
                _ => false,
            },
        }
    }

    /// One greedy step (a removal and the sweep it makes possible) is accepted.
    pub(super) fn accepts_step(&self, check: &ReductionCheck) -> bool {
        match self {
            StepRule::Repair { allow_rule_b } => Self::repair_accepts(check, *allow_rule_b),
            _ => check.accepted && self.keeps_main_line(check),
        }
    }

    /// One round of the inert sweep is accepted: the same with rule G set aside
    /// (the repair rule sets it aside already).
    pub(super) fn accepts_round(&self, check: &ReductionCheck) -> bool {
        match self {
            StepRule::Repair { allow_rule_b } => Self::repair_accepts(check, *allow_rule_b),
            _ => check.accepted_without_rule_g && self.keeps_main_line(check),
        }
    }

    /// The rule the next step is judged by, once a step to the diagram `check`
    /// judged has been accepted.
    pub(super) fn after(&self, check: &ReductionCheck) -> StepRule {
        match (self, &check.main_line_paths) {
            (StepRule::MainLine { .. }, MainLinePaths::Paths(paths)) => StepRule::MainLine {
                paths: paths.clone(),
            },
            (StepRule::Repair { .. }, _) if check.accepted => StepRule::Length,
            _ => self.clone(),
        }
    }
}

/// One greedy chain: its diagram, what came off it, the rule its next step is
/// judged by, and what the reducer needs to know about the diagram.
#[derive(Clone, Debug)]
pub(super) struct ChainState {
    pub(super) board: Position,
    pub(super) removed: Vec<RemovedPiece>,
    pub(super) rule: StepRule,
    /// The displayed main line of `board` (rule D v5 groups the
    /// length-preserving results by it).
    pub(super) line: Option<Vec<Usi>>,
    /// `board` passes the acceptance test: always, except on the input of a
    /// relaxed entry (rule B, C/G) before a step reaches the full test.
    pub(super) accepted: bool,
    /// The greedy passes' record, kept only with [`set_trace_passes`] on.
    pub(super) trace: Option<Box<ChainTrace>>,
    /// A whole greedy pass over the reduction's candidate list (every pass is a
    /// permutation of it) took nothing on `board` under `rule`: the next pass
    /// would try the same steps on the same diagram under the same rule, and
    /// every step is a pure function of those (`evaluate_step`), so it would
    /// take nothing either.  [`greedy_pass`] then returns at once.  Cleared by
    /// every change of `board` (`commit`, `absorb`).
    pub(super) settled: bool,
    /// The chain's number in the negatives' order (`--emit-negatives`,
    /// [`crate::negatives`]); never read by a verdict.
    pub(super) neg_chain: u32,
}

impl ChainState {
    pub(super) fn new(
        board: Position,
        removed: Vec<RemovedPiece>,
        rule: StepRule,
        line: Option<Vec<Usi>>,
        accepted: bool,
    ) -> Self {
        Self {
            board,
            removed,
            rule,
            line,
            accepted,
            trace: TRACE_PASSES
                .load(Ordering::Relaxed)
                .then(|| Box::new(ChainTrace::default())),
            settled: false,
            neg_chain: 0,
        }
    }

    /// Whether the next greedy pass on this chain is skipped: it is settled, the
    /// skip is on, and no trace is kept (with [`set_trace_passes`] every pass
    /// runs, so the measurement records every refusal as before).  Off while a
    /// negatives' collection is open (`--emit-negatives`): the repeated pass
    /// reports its refusals again, so the events (and the per-chain caps that
    /// count them) are those of the reducer before the skip.
    pub(super) fn skips_pass(&self) -> bool {
        let skips = self.settled
            && self.trace.is_none()
            && SETTLED_SKIP.load(Ordering::Relaxed)
            && !negatives_open();
        if skips {
            SPEEDUP_HITS[0].fetch_add(1, Ordering::Relaxed);
        }
        skips
    }

    /// `best` stands on the same diagram as `chain`, which took nothing in a
    /// whole pass: then `best` took nothing either (no commit leaves a diagram
    /// unchanged, so neither chain committed and both still read the start's
    /// rule), and its next pass would take nothing.
    pub(super) fn inherit_settled(&mut self, chain: &ChainState) {
        if chain.settled && self.board == chain.board && self.removed.len() == chain.removed.len() {
            self.settled = true;
        }
    }

    /// A copy of this (start) chain for attempt `attempt`, numbered for the
    /// negatives' order.  Everything else is the plain clone.
    pub(super) fn for_attempt(&self, attempt: usize) -> Self {
        let mut chain = self.clone();
        chain.neg_chain = self.neg_chain + 1 + attempt as u32;
        chain
    }

    /// Reports a step `self.rule` refused on this chain's diagram
    /// (`--emit-negatives`; a no-op otherwise).
    pub(super) fn note_refused(&self, removal: &RemovedPiece, swept: &SweepResult, target: u32) {
        negative_event(self.neg_chain, || crate::negatives::Event {
            kind: crate::negatives::EventKind::Refused,
            tool: "reduce",
            position: swept.position.clone(),
            target: Some(target),
            reasons: swept.check.reasons.clone(),
            accepted: swept.check.accepted,
            parent_sound: self.accepted,
            step: Some(removal.clone()),
            path: self.removed.clone(),
            chain: self.neg_chain,
            parent: Some(self.board.hash()),
            keeps_line: false,
        });
    }

    /// Reports the diagram an accepted step leads to, before it is applied.
    pub(super) fn note_step(&self, removal: Option<&RemovedPiece>, swept: &SweepResult) {
        negative_event(self.neg_chain, || crate::negatives::Event {
            kind: crate::negatives::EventKind::Step,
            tool: "reduce",
            position: swept.position.clone(),
            target: None,
            reasons: swept.check.reasons.clone(),
            accepted: swept.check.accepted,
            parent_sound: self.accepted,
            step: removal.cloned(),
            path: self.removed.clone(),
            chain: self.neg_chain,
            parent: Some(self.board.hash()),
            keeps_line: swept.check.accepted && matches!(self.rule, StepRule::MainLine { .. }),
        });
    }

    /// Applies an accepted step: the candidate's removal and its sweep.
    ///
    /// Every rule but the repair rule accepts only steps that pass the full
    /// test, so `accepted` becomes `true` there exactly as before; a repair
    /// step may leave it `false`.
    pub(super) fn commit(&mut self, entry: RemovedPiece, swept: SweepResult) {
        self.settled = false;
        self.note_step(Some(&entry), &swept);
        self.rule = self.rule.after(&swept.check);
        self.line = swept.check.main_line;
        self.accepted = swept.check.accepted;
        self.board = swept.position;
        self.removed.push(entry);
        self.removed.extend(swept.removed);
    }

    /// Applies an inert sweep's result in place (the repair pre-pass): its
    /// removals and its verdict, with no candidate removal of its own.
    pub(super) fn absorb(&mut self, swept: SweepResult) {
        self.settled = false;
        if !swept.removed.is_empty() {
            self.note_step(None, &swept);
        }
        self.rule = self.rule.after(&swept.check);
        self.line = swept.check.main_line;
        self.accepted = swept.check.accepted;
        self.board = swept.position;
        self.removed.extend(swept.removed);
    }
}

/// The C/G entry's repair pre-pass ([`TwoKindsOptions::cg_entry`]): on the
/// chain's diagram, in a fixed order, until it passes the full test or a round
/// takes nothing (at most [`INERT_SWEEP_ITERATIONS`] rounds):
///
/// 1. the inert sweep, every round judged by the chain's repair rule;
/// 2. while `koma_amari` is reported, spare attacker hand units, piece types in
///    `P L N S G B R` order, one unit at a time, each judged by the repair rule
///    (a type whose next unit is refused is not tried again in the round).
///
/// This is what the strict reducer's valley (two spare hand pieces: handing one
/// back still leaves koma-amari, so the full test refuses it) needs: each unit
/// is judged with rule C set aside, and the full test only at the end.
/// Removals land in `removed` in the order they happened; the shuffled passes
/// follow from the diagram it leaves.
pub(super) fn repair_pass(
    solver: &mut Solver,
    chain: &mut ChainState,
    target: u32,
    options: &ReduceOptions,
    attacker: u8,
) {
    let mut check = {
        let _stage = Stage::enter("repair", None, Vec::new());
        check_reduction(solver, &chain.board, target, options, false)
    };
    if stopped() {
        return;
    }
    if check.accepted {
        chain.absorb(SweepResult {
            position: chain.board.clone(),
            removed: Vec::new(),
            check,
        });
        return;
    }
    for _ in 0..INERT_SWEEP_ITERATIONS {
        if chain.accepted || stopped() {
            return;
        }
        let mut progressed = false;
        if !check.inert_pieces.is_empty() {
            let swept = sweep_inert_pieces(
                solver,
                &chain.board,
                target,
                options,
                check.clone(),
                &chain.rule,
            );
            if stopped() {
                return;
            }
            if !swept.removed.is_empty() {
                check = swept.check.clone();
                chain.absorb(swept);
                progressed = true;
                if chain.accepted {
                    return;
                }
            }
        }
        if check.reasons.iter().any(|reason| reason == "koma_amari") {
            'types: for &piece in HAND_PIECES.iter() {
                while chain.board.hands[attacker as usize][piece as usize] > 0 {
                    let Some((trial, removal)) =
                        apply_candidate(&chain.board, Candidate::Hand(piece), attacker)
                    else {
                        break;
                    };
                    let stage = Stage::enter("repair", None, vec![removal.clone()]);
                    let trial_check = check_reduction(solver, &trial, target, options, false);
                    drop(stage);
                    if stopped() {
                        return;
                    }
                    if !chain.rule.accepts_step(&trial_check) {
                        if negatives_open() {
                            chain.note_refused(
                                &removal,
                                &SweepResult {
                                    position: trial,
                                    removed: Vec::new(),
                                    check: trial_check,
                                },
                                target,
                            );
                        }
                        break;
                    }
                    check = trial_check.clone();
                    chain.commit(
                        removal,
                        SweepResult {
                            position: trial,
                            removed: Vec::new(),
                            check: trial_check,
                        },
                    );
                    progressed = true;
                    if chain.accepted || !check.reasons.iter().any(|r| r == "koma_amari") {
                        break 'types;
                    }
                }
            }
            if chain.accepted {
                return;
            }
        }
        if !progressed {
            return;
        }
    }
}

/// One greedy pass over `order`: every candidate whose removal (plus the inert
/// sweep it makes possible) the chain's rule accepts is applied and logged.
/// Returns whether anything came off.
pub(super) fn greedy_pass(
    solver: &mut Solver,
    chain: &mut ChainState,
    order: &[Candidate],
    target: u32,
    options: &ReduceOptions,
    attacker: u8,
) -> bool {
    if chain.skips_pass() {
        // The pass this one repeats took nothing (`ChainState::settled`).
        return false;
    }
    if stopped() {
        // The early stop: the record is a budget-out, its chains are dropped.
        return false;
    }
    let mut changed = false;
    let before = chain.removed.len();
    let first = chain
        .trace
        .as_ref()
        .is_some_and(|trace| trace.pass_removed.is_empty());
    for &step in order {
        let Some((trial, entry)) = apply_candidate(&chain.board, step, attacker) else {
            continue;
        };
        let structural = structural_test(&chain.board, &trial, &chain.rule);
        if structural == Some(true) {
            // Refused without a search (the test is off under
            // `set_trace_passes`, so no trace misses it).
            continue;
        }
        let (swept, log) = evaluate_step(
            solver,
            &trial,
            target,
            options,
            &chain.rule,
            &entry,
            structural.is_some(),
        );
        absorb_budget_outs(log);
        if stopped() {
            // The early stop: nothing more of this record is checked, and
            // the step is not applied (the chain is dropped).
            return false;
        }
        if !chain.rule.accepts_step(&swept.check) {
            chain.note_refused(&entry, &swept, target);
            if first {
                let why = refusal(&chain.rule, &swept.check);
                if let Some(trace) = chain.trace.as_mut() {
                    trace.first_pass_refused.push(RefusedStep {
                        removal: entry,
                        trial_sfen: crate::sfen::to_sfen(&trial),
                        why,
                    });
                }
            }
            continue;
        }
        chain.commit(entry, swept);
        changed = true;
    }
    if let Some(trace) = chain.trace.as_mut() {
        trace.pass_removed.push(chain.removed.len() - before);
    }
    chain.settled = !changed;
    changed
}

/// The negatives' chain numbers ([`ChainState::neg_chain`]): the entry is 0,
/// the main-line kind's start and attempts from here, the length kind's from
/// [`NEG_LENGTH_CHAINS`].  Only the order of events reads them.
pub(super) const NEG_MAIN_CHAINS: u32 = 1;

pub(super) const NEG_LENGTH_CHAINS: u32 = 1 << 20;

thread_local! {
    /// One [`reduce_two_kinds`] call's negative events (`--emit-negatives`),
    /// `Some` only while a [`NegativeScope`] is open on this thread.
    pub(super) static NEG_EVENTS: RefCell<Option<NegBuffer>> = const { RefCell::new(None) };
}

/// One [`reduce_two_kinds`] call's buffered events, keyed by `(chain, n)`
/// with `n` the event's position in its chain: the chains' own orders are
/// the same on every `--jobs` path, only their interleaving differs, so a cut
/// by key is the same everywhere.  It keeps at most
/// [`crate::negatives::MAX_EVENTS_PER_CHAIN`] per chain and the
/// [`crate::negatives::MAX_EVENTS`] smallest keys overall (memory stays below
/// twice that), counting the rest.
#[derive(Default)]
pub(super) struct NegBuffer {
    pub(super) events: Vec<((u32, usize), crate::negatives::Event)>,
    pub(super) per_chain: std::collections::HashMap<u32, usize>,
    /// Every key above this one is already out of the smallest `MAX_EVENTS`.
    pub(super) threshold: Option<(u32, usize)>,
    pub(super) dropped: u64,
}

impl NegBuffer {
    /// Keeps the `MAX_EVENTS` smallest keys, in key order.
    pub(super) fn compact(&mut self) {
        let max = crate::negatives::MAX_EVENTS;
        self.events.sort_by_key(|(key, _)| *key);
        if self.events.len() > max {
            self.dropped += (self.events.len() - max) as u64;
            self.events.truncate(max);
            self.threshold = self.events.last().map(|(key, _)| *key);
        }
    }
}

/// A negatives' collection is open, so events are worth building.
pub(super) fn negatives_open() -> bool {
    NEG_EVENTS.with(|events| events.borrow().is_some())
}

/// Buffers one event when a [`NegativeScope`] is open; `make` runs only then.
pub(super) fn negative_event(chain: u32, make: impl FnOnce() -> crate::negatives::Event) {
    NEG_EVENTS.with(|buffer| {
        if let Some(buffer) = buffer.borrow_mut().as_mut() {
            let seen = buffer.per_chain.entry(chain).or_insert(0);
            let key = (chain, *seen);
            *seen += 1;
            if key.1 >= crate::negatives::MAX_EVENTS_PER_CHAIN
                || buffer.threshold.is_some_and(|threshold| key > threshold)
            {
                buffer.dropped += 1;
                return;
            }
            buffer.events.push((key, make()));
            if buffer.events.len() >= 2 * crate::negatives::MAX_EVENTS {
                buffer.compact();
            }
        }
    });
}

/// `--emit-negatives` ([`crate::negatives`]): while open, the events of the
/// chains of one [`reduce_two_kinds`] call are buffered; on close they go to
/// the caller's collection in key order (by chain, each chain in its own
/// order; see [`NegBuffer`]).  The chains are numbered the same way and each sees the same
/// refusals in the same order on both paths (`jobs` 1 and the pool's
/// coordinator, which commits in chain order), so the events do not depend on
/// `--jobs`.  Open only when the caller opened a collection: otherwise no
/// event is ever built.
pub(super) struct NegativeScope {
    pub(super) open: bool,
    /// The call's target length, for the events that do not carry it.
    pub(super) target: Option<u32>,
}

impl NegativeScope {
    pub(super) fn open() -> Self {
        let open = crate::negatives::active();
        if open {
            NEG_EVENTS.with(|buffer| *buffer.borrow_mut() = Some(NegBuffer::default()));
        }
        Self { open, target: None }
    }

    /// Drops the call's buffered events (the early stop, [`stopped`]): how
    /// far each chain of `reduce --jobs` got before the stop depends on the
    /// schedule, so a stopped record hands over no event at all.
    pub(super) fn discard(&self) {
        if self.open {
            NEG_EVENTS.with(|buffer| *buffer.borrow_mut() = Some(NegBuffer::default()));
        }
    }
}

impl Drop for NegativeScope {
    fn drop(&mut self) {
        if !self.open {
            return;
        }
        let mut buffer = NEG_EVENTS
            .with(|buffer| buffer.borrow_mut().take())
            .unwrap_or_default();
        buffer.compact();
        let target = self.target;
        let events = buffer.events.into_iter().map(|(_, mut event)| {
            event.target = event.target.or(target);
            event
        });
        crate::negatives::record_all(events, buffer.dropped);
    }
}

/// The greedy passes of one chain (measurement only, [`set_trace_passes`]).
#[derive(Clone, Debug, Default)]
pub struct ChainTrace {
    /// Pieces each pass took off (candidates and their sweeps), in pass order:
    /// the attempt's shuffled pass first, then the fixpoint passes (the last
    /// always 0).
    pub pass_removed: Vec<usize>,
    /// The first pass's refused steps, in the order tried.
    pub first_pass_refused: Vec<RefusedStep>,
}

/// One refused greedy step of a first pass.
#[derive(Clone, Debug)]
pub struct RefusedStep {
    pub removal: RemovedPiece,
    /// The diagram the candidate's removal left (before any sweep).
    pub trial_sfen: String,
    /// `acceptance` (rules A-C, F, B or the length), `rule_g` (only inert pieces
    /// left after the sweep) or `main_line_paths` (`main_line_changed` went with
    /// the displayed-line condition at 0.14.0).
    pub why: &'static str,
}

/// The passes of the outputs of one [`reduce_two_kinds`] call.
#[derive(Clone, Debug, Default)]
pub struct TwoKindsTrace {
    /// The main-line-preserving output's chain (its winning attempt).
    pub main_line: Option<ChainTrace>,
    /// Every length-preserving attempt, in attempt order.
    pub length_attempts: Vec<ChainTrace>,
    /// The emitted length-preserving outputs' chains, rank order.
    pub length: Vec<ChainTrace>,
}

/// Why `rule` refused a step whose verdict is `check`.
pub(super) fn refusal(rule: &StepRule, check: &ReductionCheck) -> &'static str {
    if !check.accepted_without_rule_g {
        "acceptance"
    } else if !check.accepted {
        "rule_g"
    } else if let StepRule::MainLine { .. } = rule {
        "main_line_paths"
    } else {
        "acceptance"
    }
}

/// One greedy step's verdict on `trial` (a candidate already applied): the
/// removal plus the sweep it makes possible, judged as a whole with rule G back
/// on -- a step that only trades one inert piece for another is no step at all.
/// The caller reads the verdict with [`StepRule::accepts_step`].
///
/// A pure function of its arguments: [`Solver::solve`] clears the table per
/// call, so `solver` carries allocations but no information between steps.
/// This is what lets [`workers`] run steps on different solvers.
pub(super) fn evaluate_step(
    solver: &mut Solver,
    trial: &Position,
    target: u32,
    options: &ReduceOptions,
    rule: &StepRule,
    removal: &RemovedPiece,
    structural: bool,
) -> (SweepResult, StepLog) {
    // The step's budget-outs go with its result to whoever consumes it (the
    // sequential pass, or the coordinator of `reduce --jobs`, which drops a
    // discarded speculative step's), so the log does not depend on `--jobs`.
    // A step the structural main-line test refuses anyway (searched only
    // because the test is off, [`structural_test`]) counts nothing: its
    // budget-outs are marked, and none of them stops its sweep ([`stopped`]).
    let (swept, mut log) = with_frame(|| {
        let _mute = structural.then(Mute::on);
        let _stage = Stage::enter("greedy", Some(removal), vec![removal.clone()]);
        evaluate_step_searched(solver, trial, target, options, rule)
    });
    if structural {
        mark_structural(&mut log.entries);
    }
    (swept, log)
}

/// [`evaluate_step`]'s searches.
pub(super) fn evaluate_step_searched(
    solver: &mut Solver,
    trial: &Position,
    target: u32,
    options: &ReduceOptions,
    rule: &StepRule,
) -> SweepResult {
    // A trial with inert pieces goes on to the sweep, which reads them.
    let trial_check = check_reduction(solver, trial, target, options, rule.wants_paths());
    sweep_inert_pieces(solver, trial, target, options, trial_check, rule)
}

/// [`main_line_unreachable`] under [`main_line_replay_on`]: `Some(true)` skips
/// the step, `Some(false)` searches it and marks its budget-outs
/// ([`mark_structural`]), `None` searches it as any other step.
pub(super) fn structural_test(board: &Position, trial: &Position, rule: &StepRule) -> Option<bool> {
    if !main_line_unreachable(board, trial, rule) {
        return None;
    }
    let skip = main_line_replay_on();
    if skip {
        SPEEDUP_HITS[1].fetch_add(1, Ordering::Relaxed);
    }
    Some(skip)
}

/// Whether a main-line-preserving step to `trial` (a candidate's removal from
/// `board`) is refused without a search, because no main-line path the rule
/// keeps (`Π(z)`, [`StepRule::MainLine`]) can be played on `trial` nor on
/// anything the inert sweep could make of it.
///
/// An accepted `y` has a displayed main line, `λ(y) ∈ Π(y) ⊆ Π(z)`, playable on
/// `y`; so when every path of `Π(z)` is structurally unplayable on `trial`, the
/// step is refused whatever the search would say.  Until 0.14.0 the test
/// replayed the displayed main line alone, which the rule then kept.
///
/// Each path is replayed on `board` (where it is a main-line path, so every
/// move is legal) and on `trial` side by side.  At the first move that is not
/// legal on `trial`, the path is unplayable when the move is *structurally*
/// impossible: a board move whose origin square is empty, or an attacker drop
/// of a piece the attacker does not hold.  No sweep repairs that: the sweep
/// only takes board pieces off, into the defender's hand, so every square
/// empty in the replay on `trial` is empty in the replay on any swept diagram
/// (as long as both replays go on, they apply the same moves), and the
/// attacker's hand there is never larger.  Any other illegality (a pin, a
/// check, a defender drop) could be undone by a sweep and is left to the
/// search; so is a path that plays to its end on `trial`.
///
/// The verdict alone, whatever the switches: a step it refuses is skipped only
/// while [`main_line_replay_on`]; otherwise it is searched and its budget-outs
/// marked ([`BudgetOut::structural_refusal`]), so the record's tally is the
/// same either way ([`structural_test`]).
pub(super) fn main_line_unreachable(board: &Position, trial: &Position, rule: &StepRule) -> bool {
    let StepRule::MainLine { paths } = rule else {
        return false;
    };
    let attacker = board.side;
    let start_board = strict_material::normalize(board, attacker);
    let start_trial = strict_material::normalize(trial, attacker);
    // A path that shares the moves up to an unplayable one with a path already
    // found unplayable is unplayable too (the replay is the same up to there);
    // `paths` is sorted, so such paths are neighbours.
    let mut unplayable_prefix: Option<&[Usi]> = None;
    for path in paths.iter() {
        if unplayable_prefix.is_some_and(|prefix| path.starts_with(prefix)) {
            continue;
        }
        match unplayable_at(&start_board, &start_trial, path, attacker) {
            Some(at) => unplayable_prefix = Some(&path[..=at]),
            None => return false,
        }
    }
    true
}

/// One path of [`main_line_unreachable`] (both positions already normalised):
/// the index of the move at which it is structurally unplayable on `trial`, or
/// `None` when it is not.
pub(super) fn unplayable_at(
    board: &Position,
    trial: &Position,
    path: &[Usi],
    attacker: u8,
) -> Option<usize> {
    let mut on_board = board.clone();
    let mut on_trial = trial.clone();
    let find = |position: &Position, usi: &Usi| {
        crate::movegen::legal_moves(position)
            .into_iter()
            .find(|m| m.to_usi() == *usi)
    };
    for (index, usi) in path.iter().enumerate() {
        let played = find(&on_board, usi)?;
        match find(&on_trial, usi) {
            Some(same) => {
                on_board.make(played);
                on_trial.make(same);
            }
            None => {
                let structural = if played.is_drop() {
                    on_trial.side == attacker
                        && on_trial.hands[attacker as usize][played.piece as usize] == 0
                } else {
                    on_trial.board[played.from as usize] == EMPTY
                };
                return structural.then_some(index);
            }
        }
    }
    None
}

/// Fewer board pieces, then fewer attacker hand pieces, then the smaller SFEN.
pub(super) fn beats_best(candidate: &Position, best: &Position, attacker: u8) -> bool {
    let pieces = candidate.board_piece_count();
    let best_pieces = best.board_piece_count();
    if pieces != best_pieces {
        return pieces < best_pieces;
    }
    let hand = attacker_hand_total(candidate, attacker);
    let best_hand = attacker_hand_total(best, attacker);
    if hand != best_hand {
        return hand < best_hand;
    }
    // "Smaller SFEN" is the frozen pre-2026-10-04 spelling (hands P L N S G B R), so
    // the written hand order does not move the tie-break (`docs/rules.md` D).
    to_legacy_sfen_key(candidate) < to_legacy_sfen_key(best)
}

/// `Π` of a diagram, as far as one acceptance test measured it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MainLinePaths {
    /// Not asked for, or no main line to enumerate (the diagram failed the test
    /// for a reason other than rule G).
    #[default]
    Unmeasured,
    /// More paths than the cap.
    Capped,
    /// Sorted, without repeats ([`main_line_paths`]).
    Paths(Arc<Vec<Vec<Usi>>>),
}

/// Everything the reducer learns about one candidate diagram from the single
/// solve its acceptance test performs.
///
/// Port of `ReductionCheck` in `tools/puzzle-pipeline/src/stages.ts`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReductionCheck {
    /// The reducer's acceptance test (rule D): structural validity plus an exact,
    /// minimum, unique-first-move mate of the target length that also satisfies
    /// the orthodox composition rules A-C, F **and** G of `docs/rules.md`.
    pub accepted: bool,
    /// The same test with rule G alone excluded, which is what the inert sweep of
    /// rule D v3 steps on: a sweep round may hand back a diagram that still
    /// carries an inert piece, because the *next* round is what takes that one
    /// off.  `true` whenever the verdict is accepted, since no reason at all
    /// trivially satisfies "every reason is a rule G reason".
    pub accepted_without_rule_g: bool,
    /// Rule G's evidence: the board pieces this diagram's solution never uses.
    pub inert_pieces: Vec<InertPiece>,
    /// The displayed main line (`λ`, both sides' moves), when the composition
    /// rules read one off a proven mate (rule D v5).
    pub main_line: Option<Vec<Usi>>,
    /// The main-line paths (`Π`), measured only when the caller asked for them
    /// and the diagram passed the test with rule G set aside (rule D v5).
    pub main_line_paths: MainLinePaths,
    /// The test's reasons (empty exactly when `accepted`), which the repair rule
    /// of the C/G entry reads ([`StepRule::Repair`]); a structurally invalid
    /// diagram reports its issue codes.
    pub reasons: Vec<String>,
}

impl ReductionCheck {
    /// The verdict on a diagram that is not even structurally valid.
    pub(super) fn rejected(issues: Vec<String>) -> Self {
        Self {
            reasons: issues,
            ..Self::default()
        }
    }
}

/// One acceptance test, read both with and without rule G (`checkReduction`).
/// `want_paths` also enumerates `Π` (rule D v5); the verdict is the same either
/// way.
pub(super) fn check_reduction(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: u32,
    options: &ReduceOptions,
    want_paths: bool,
) -> ReductionCheck {
    crate::posprof::add(crate::posprof::K::ReduceChecks, 1);
    let started = crate::posprof::start();
    let check = check_reduction_laddered(solver, position, target_mate_ply, options, want_paths);
    crate::posprof::charge(crate::posprof::K::ReduceNs, started);
    check
}

/// [`check_reduction`] on the budget ladder.
pub(super) fn check_reduction_laddered(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: u32,
    options: &ReduceOptions,
    want_paths: bool,
) -> ReductionCheck {
    on_ladder(
        options.node_budget,
        solver,
        |check: &ReductionCheck| &check.reasons,
        Some(|| ReductionCheck::rejected(vec!["solver_budget_exhausted".to_string()])),
        Checked {
            kind: "check",
            position,
            target: Some(target_mate_ply),
            paths: want_paths,
        },
        |budget, solver| {
            check_reduction_at(
                solver,
                position,
                target_mate_ply,
                options,
                want_paths,
                budget,
            )
        },
    )
}

/// [`check_reduction`] on the node budget `budget`.
pub(super) fn check_reduction_at(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: u32,
    options: &ReduceOptions,
    want_paths: bool,
    budget: u64,
) -> ReductionCheck {
    let issues = validate_position(position, ValidateOptions::pipeline());
    if !issues.is_empty() {
        return ReductionCheck::rejected(issues.into_iter().map(|issue| issue.code).collect());
    }
    let trace = crate::cdfpn::trace_on();
    let traced = trace.then(crate::clock::Instant::now);
    let fast = fast_validate(
        position,
        Some(target_mate_ply),
        options.rules,
        options.max_ply,
        budget,
        CONSTRUCT_ON.load(Ordering::Relaxed),
        false,
    );
    if let Fast::Decided {
        reasons,
        inert_pieces,
        built,
        ..
    } = fast
    {
        let accepted_without_rule_g = reasons.iter().all(|reason| is_rule_g_reason(reason));
        let (main_line, graph) = built_line_and_graph(built);
        return ReductionCheck {
            accepted: reasons.is_empty(),
            accepted_without_rule_g,
            inert_pieces,
            main_line,
            main_line_paths: if want_paths && accepted_without_rule_g {
                measure_paths(graph.as_ref(), position.side)
            } else {
                MainLinePaths::Unmeasured
            },
            reasons: reasons.into_iter().map(str::to_string).collect(),
        };
    }
    // The fast path ran out of budget (the construction on): a budget-out,
    // which the ladder takes to its next rung and the last rung logs
    // (`--emit-budget-outs`, `reduction.minimality` `unverified`).  The
    // one-thread full-width search is not run again on this budget (solver
    // 1.4.0, owner 2026-11-05).
    if let Fast::BudgetOut { .. } = fast {
        if let Some(traced) = traced {
            eprintln!(
                "reduce trace: check fast path budget-out in {:.3} s (budget {budget})",
                traced.elapsed().as_secs_f64()
            );
        }
        return ReductionCheck::rejected(fast_budget_out_reasons());
    }
    // A proven mate within the bound means the exact gate will analyse the
    // solution graph, which the graph-free search would buy with a second,
    // identical search at the same bound: build it in the first one instead
    // (the verdict, reasons and inert pieces are the same either way).  Rule D
    // v5 reads its paths off that graph too.
    if let Some(traced) = traced {
        eprintln!(
            "reduce trace: check fast path {:?} in {:.3} s (budget {budget}); exact search next",
            match &fast {
                Fast::Mate => "mate",
                Fast::Unknown => "unknown",
                Fast::BudgetOut { .. } => "budget-out",
                Fast::Decided { .. } => "decided",
            },
            traced.elapsed().as_secs_f64()
        );
    }
    // The browser gate: free the prober's tables before the exact search
    // (`release_prober`; wasm32's memory never shrinks).
    #[cfg(target_arch = "wasm32")]
    release_prober();
    count_full_width_gate();
    let validation = exact_validate_with_rules(
        solver,
        position,
        Some(target_mate_ply),
        options.max_ply,
        budget,
        matches!(fast, Fast::Mate) || want_paths,
        options.rules,
    );
    charge_spent(validation.result.stats.nodes + validation.extra_nodes + validation.shorter_nodes);
    let accepted_without_rule_g = validation
        .reasons
        .iter()
        .all(|reason| is_rule_g_reason(reason));
    let main_line_paths = if want_paths && accepted_without_rule_g {
        measure_paths(validation.result.graph.as_ref(), validation.result.attacker)
    } else {
        MainLinePaths::Unmeasured
    };
    ReductionCheck {
        accepted: validation.accepted,
        accepted_without_rule_g,
        inert_pieces: validation.inert_pieces,
        main_line: validation.main_line.map(|line| line.moves),
        main_line_paths,
        reasons: validation.reasons,
    }
}

/// The reducer's acceptance test (rules A-C, F and G), as `acceptedReduction`.
///
/// Kept as a predicate for the mandatory attacker-king pre-pass of the
/// length-preserving reducer, which is judged on the full rule set.
pub(super) fn accepted_reduction(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: u32,
    options: &ReduceOptions,
) -> bool {
    check_reduction(solver, position, target_mate_ply, options, false).accepted
}
