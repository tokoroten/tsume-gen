//! Budgets: the ladder of node budgets a check climbs, the budget-outs it
//! logs (`--emit-budget-outs`), the early stop of a sweep, and the stage
//! frames the log is kept in.

use super::*;

/// The budget ladder of the reducer's checks (`reduce --reduce-budget-ladder`,
/// owner 2026-10-15):
/// up to [`LADDER_RUNGS`] smaller node budgets, ascending, `0` for none.  Every
/// acceptance test ([`check_reduction`]) and entry check ([`entry_verdict`]) is
/// first run on the smallest rung below `ReduceOptions::node_budget`; an answer
/// with no budget-dependent reason ([`budget_out`]) is final, otherwise the test
/// is run again on the next rung, and last on the full budget.
///
/// On one prober thread the output is byte-identical to running every test on
/// the full budget alone (under a parallel prober, where the full-budget run
/// itself may resolve a check on one run and not on another, the ladder
/// changes nothing either: every resolved answer is the exact one, and a
/// full-budget run that ran out stays a budget-out -- Codex review
/// 2026-10-15): a test's answer (`ReductionCheck`, `EntryVerdict`) holds no node
/// count, and every budget-dependent part of it is flagged by a budget-out
/// reason or `minimum_length_not_proven` -- the prober and the serial construction take the budget only as a
/// cut-off (a fresh prober per test, so the same trajectory up to the cut), the
/// exact search sets `stats.budgetExhausted` (and so
/// `incomplete_response_graph`) whenever its node budget or its credit oracle
/// (four times the node budget) ran out, and the fast path and the exact gate
/// give the same reasons, length, main line, paths and inert pieces whenever
/// neither ran out (`tests/fast_reduce.rs`).  So an answer decided on a small
/// rung is the answer of the full budget.  Process-wide; set it before any
/// reduction starts.
pub const LADDER_RUNGS: usize = 3;

pub(super) static BUDGET_LADDER: [AtomicU64; LADDER_RUNGS] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Sets the budget ladder ([`BUDGET_LADDER`]): the smaller rungs, ascending
/// (at most [`LADDER_RUNGS`]); an empty slice turns it off.
pub fn set_budget_ladder(rungs: &[u64]) {
    assert!(
        rungs.len() <= LADDER_RUNGS,
        "at most {LADDER_RUNGS} smaller rungs"
    );
    assert!(
        rungs.windows(2).all(|pair| pair[0] < pair[1]),
        "the rungs must ascend"
    );
    for (index, slot) in BUDGET_LADDER.iter().enumerate() {
        slot.store(rungs.get(index).copied().unwrap_or(0), Ordering::Relaxed);
    }
}

/// The budgets a test is run on, in order: the ladder's rungs below `full`,
/// then `full`.
pub(super) fn ladder(full: u64) -> impl Iterator<Item = u64> {
    BUDGET_LADDER
        .iter()
        .map(|slot| slot.load(Ordering::Relaxed))
        .filter(move |&rung| rung != 0 && rung < full)
        .chain(std::iter::once(full))
}

/// `true` when `reasons` hold a budget-out (`solver_budget_exhausted`,
/// `incomplete_response_graph`, rule G clause (e)'s `rule_g_budget_exhausted`
/// since `PIPELINE_VERSION` 0.15.0) or `minimum_length_not_proven`, which the exact
/// gate gives when its second search (no mate one ply shorter) did not refute
/// -- after a minimal-length mate, only by running out of budget, and that
/// search reports no budget-out reason of its own (Codex review 2026-10-15):
/// the answer may change on a larger budget.
pub(super) fn budget_out<S: AsRef<str>>(reasons: &[S]) -> bool {
    reasons.iter().any(|reason| {
        matches!(
            reason.as_ref(),
            "solver_budget_exhausted"
                | "incomplete_response_graph"
                | "minimum_length_not_proven"
                | crate::inert::RULE_G_BUDGET_EXHAUSTED
        )
    })
}

/// Runs `test` on every budget of [`ladder`] until one answers without a
/// budget-out (`reasons` reads them off an answer), counting the rung that
/// decided in [`LADDER_STATS`].  Without a ladder: `test(full, solver)`,
/// nothing else.
///
/// Every rung runs on the same threads: one thread, or the parallel prober
/// (`--prober-threads`), whose resolved answers are one thread's (owner
/// 2026-10-23; the position
/// pool that ran some rungs on other threads went on 2026-10-27).
///
/// The full budget's answer, when it ran out, is final: a budget-out, logged
/// ([`log_budget_out`]) and counted in the record's tally ([`BudgetOutTally`],
/// `reduction.minimality` `unverified`); it is never re-run on one thread
/// (owner 2026-10-23).
/// `forced_out` is the budget-out answer a test may force ([`FORCE_OUT`]).
pub(super) fn on_ladder<T>(
    full: u64,
    solver: &mut Solver,
    reasons: impl Fn(&T) -> &[String],
    forced_out: Option<fn() -> T>,
    checked: Checked<'_>,
    mut test: impl FnMut(u64, &mut Solver) -> T,
) -> T {
    let mut rungs = ladder(full).enumerate().peekable();
    loop {
        let (index, budget) = rungs.next().expect("the ladder ends with the full budget");
        let last = rungs.peek().is_none();
        let forced =
            forced_out.filter(|_| last && FORCE_OUT.load(Ordering::Relaxed) && verdicts_may_vary());
        let spent_before = spent();
        let answer = match forced {
            Some(forced) => forced(),
            None => test(budget, solver),
        };
        if last || !budget_out(reasons(&answer)) {
            if index > 0 || !last {
                LADDER_STATS[index.min(LADDER_RUNGS)].fetch_add(1, Ordering::Relaxed);
            }
            note_rung(budget);
            if !(last && budget_out(reasons(&answer))) {
                return answer;
            }
            // The final answer of the last rung ran out of budget: measured
            // for `--emit-budget-outs` ([`budget_outs_begin`]), which never
            // changes the answer.
            let nodes = spent().wrapping_sub(spent_before);
            let first = BudgetOutRun {
                reasons: budget_reasons(reasons(&answer)),
                nodes,
            };
            FINAL_BUDGET_OUTS.fetch_add(1, Ordering::Relaxed);
            log_budget_out(&checked, full, ThreadMode::current(), first);
            return answer;
        }
    }
}

/// `true` when whether a check runs out of its budget may differ from the
/// one-thread search's: a parallel prober (`--prober-threads`, whose nodes
/// are every thread's).  A resolved answer is the one-thread answer; only a
/// budget-out may not be (kept as a budget-out).
pub(super) fn verdicts_may_vary() -> bool {
    PROBER.with(|cell| cell.borrow().threads()) > 1
}

/// Checks refused for lack of budget (a final answer on the last rung that
/// ran out), counted whether or not a log is
/// open and on every thread, speculative `--jobs` steps included
/// (measurement and tests: the log's `still-out` entries of a one-worker run
/// are exactly these).
pub static FINAL_BUDGET_OUTS: AtomicU64 = AtomicU64::new(0);

/// What a check judged, for the budget-out log: the entry check or an
/// acceptance test, its diagram, its target and whether it measured `Π`.
#[derive(Clone, Copy)]
pub(super) struct Checked<'a> {
    pub(super) kind: &'static str,
    pub(super) position: &'a Position,
    pub(super) target: Option<u32>,
    pub(super) paths: bool,
}

/// How the search behind a logged check ran ([`BudgetOut::mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadMode {
    /// One prober thread: the budget-out is one thread's.
    Serial,
    /// A parallel prober (`--prober-threads`): whether the check ran out may
    /// differ from one thread's.  Kept as it was, never re-run on one thread.
    Parallel,
}

impl ThreadMode {
    /// The mode of a budget-out on this thread's prober.
    pub(super) fn current() -> Self {
        if verdicts_may_vary() {
            Self::Parallel
        } else {
            Self::Serial
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Parallel => "parallel",
        }
    }
}

/// One search of a logged check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetOutRun {
    /// Its budget-dependent reasons ([`budget_out`]'s set; empty when it
    /// resolved the check).
    pub reasons: Vec<String>,
    /// The nodes it counted: the prober's (prefilter and construction, a
    /// fast-path answer reused from the thread's memo counting its nodes
    /// again) and the full-width search's (`stats.nodes` plus its extra
    /// nodes).  Under a parallel prober every thread's.
    pub nodes: u64,
}

/// One acceptance check or entry check whose answer on the last rung of its
/// budget ladder carried a budget-dependent reason (`--emit-budget-outs`).
#[derive(Clone, Debug)]
pub struct BudgetOut {
    /// `entry` (the entry check) or `check` (an acceptance test).
    pub kind: &'static str,
    /// The reducer's step it belonged to ([`Stage`]).
    pub stage: &'static str,
    /// The greedy step it belonged to, when it did.
    pub step: Option<RemovedPiece>,
    /// The pieces the checked diagram took off its parent (the step's own
    /// removal, a sweep round's pieces, a repair's hand unit, the attacker
    /// king); empty for the entry check.
    pub removed: Vec<RemovedPiece>,
    /// The diagram checked.
    pub sfen: String,
    pub target: Option<u32>,
    /// The check measured the main-line paths `Π` (rule D v5).
    pub paths: bool,
    /// The full budget (the ladder's last rung).
    pub budget: u64,
    pub mode: ThreadMode,
    /// The search on the full budget.
    pub first: BudgetOutRun,
    /// The check belonged to a main-line-preserving step the structural
    /// main-line test refuses without a search ([`main_line_unreachable`]),
    /// searched only because the test was off (`--emit-negatives`, the pass
    /// trace, the test switch): the step is refused whatever the search says,
    /// so the budget-out kept no piece and the record's tally
    /// ([`BudgetOutTally`]) leaves it out.
    pub structural_refusal: bool,
}

impl BudgetOut {
    /// The check counts against the record's minimality ([`BudgetOutTally`]):
    /// an acceptance check (not the entry check, whose budget-out rejects the
    /// input), and not of a step the structural main-line test refuses anyway.
    /// The first one stops the reducer ([`stopped`]).
    pub fn counts(&self) -> bool {
        self.kind == "check" && !self.structural_refusal
    }

    /// The order of [`budget_outs_take`]: by stage, diagram and the rest, so
    /// the log does not depend on which `--jobs` worker ran which check (the
    /// set of checks it holds may, on an input the early stop ended).
    pub(super) fn sort_key(&self) -> impl Ord + '_ {
        (
            self.stage,
            self.kind,
            &self.sfen,
            self.target,
            self.paths,
            self.step
                .as_ref()
                .map(|piece| (piece.square, piece.piece.clone())),
            self.removed
                .iter()
                .map(|piece| (piece.square, piece.piece.clone()))
                .collect::<Vec<_>>(),
            self.mode.label(),
            (&self.first.reasons, self.first.nodes),
        )
    }
}

/// The budget-dependent reasons of `reasons` ([`budget_out`]'s set), in order.
pub(super) fn budget_reasons(reasons: &[String]) -> Vec<String> {
    reasons
        .iter()
        .filter(|reason| budget_out(std::slice::from_ref(reason)))
        .cloned()
        .collect()
}

thread_local! {
    /// The budget-out log the thread writes to: the record's
    /// ([`budget_outs_begin`]) or one greedy step's ([`evaluate_step`], which
    /// hands its entries to whoever consumes the step's result); `None`
    /// when off.
    pub(super) static BUDGET_OUTS: RefCell<Option<Vec<BudgetOut>>> = const { RefCell::new(None) };
    /// What the checks run now belong to ([`Stage`]).
    pub(super) static STAGE: RefCell<StageInfo> = const {
        RefCell::new(StageInfo {
            stage: "check",
            step: None,
            removed: Vec::new(),
        })
    };
    /// Search nodes this thread's checks counted, for the log: a check reads
    /// the difference across its search.  Wraps; only differences are read.
    pub(super) static SPENT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The early stop is armed on this thread ([`StopScope`]): set by
    /// [`reduce_two_kinds`] and inherited by its `--jobs` workers.
    pub(super) static STOP_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A check that counts ([`BudgetOut::counts`]) reached the current log
    /// frame ([`with_budget_out_frame`]): the record's, or one greedy step's.
    pub(super) static STOP_HIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A step that counts nothing is being evaluated ([`Mute`]): its own
    /// budget-outs ([`STOP_HIT`]) stop nothing, the pool's abort still does.
    pub(super) static STOP_MUTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// On a `reduce --jobs` worker: its pool's abort flag, set once the
    /// coordinator stopped the record, so a step in flight stops at its next
    /// check too (its answer is never read).
    pub(super) static POOL_ABORT: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
    /// The largest ladder budget a check of the current log frame needed to be
    /// decided ([`note_rung`], [`with_frame`]; 0 none yet): the record's
    /// `reduce.budget` (`--emit-record`).  Measurement only.
    pub(super) static RUNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Notes that a check was decided on the ladder budget `budget` (its last
/// rung when it ran out), unless the step it belongs to counts nothing
/// ([`Mute`]: a step the structural main-line test refuses anyway, searched
/// only with `--emit-negatives` or the pass trace), so the largest one of a
/// reduction does not depend on those switches.
pub(super) fn note_rung(budget: u64) {
    if !STOP_MUTED.with(std::cell::Cell::get) {
        RUNG.with(|cell| cell.set(cell.get().max(budget)));
    }
}

/// The early stop of [`reduce_two_kinds`] (owner 2026-10-25): once an acceptance check
/// that counts against the record's minimality ([`BudgetOut::counts`]) ends out
/// of budget, the reducer runs no further check on that record (an inert sweep
/// stops at its round, a greedy pass at its step, every chain and the pool at
/// once, a `--jobs` worker's step in flight at its next check through
/// [`POOL_ABORT`]) and returns it as a budget-out ([`ReducedSet::stopped`]).  `true`
/// only while armed: the length-preserving reducer [`reduce_with`] never
/// stops.
pub(super) fn stopped() -> bool {
    STOP_ARMED.with(std::cell::Cell::get)
        && EARLY_STOP.load(Ordering::Relaxed)
        && ((STOP_HIT.with(std::cell::Cell::get) && !STOP_MUTED.with(std::cell::Cell::get))
            || POOL_ABORT.with(|abort| {
                abort
                    .borrow()
                    .as_ref()
                    .is_some_and(|abort| abort.load(Ordering::Relaxed))
            }))
}

/// Arms the early stop on this thread while it lives, restoring the outer
/// state (armed or not, hit or not) on drop.
pub(super) struct StopScope(pub(super) bool, pub(super) bool);

impl StopScope {
    pub(super) fn arm() -> Self {
        let armed = STOP_ARMED.with(|cell| cell.replace(true));
        let hit = STOP_HIT.with(|cell| cell.replace(false));
        Self(armed, hit)
    }
}

/// While it lives, the budget-outs of the current frame stop nothing (a step
/// that counts nothing, [`evaluate_step`]); a `--jobs` pool's abort still
/// does ([`POOL_ABORT`]).
pub(super) struct Mute(pub(super) bool);

impl Mute {
    pub(super) fn on() -> Self {
        Self(STOP_MUTED.with(|cell| cell.replace(true)))
    }
}

impl Drop for Mute {
    fn drop(&mut self) {
        STOP_MUTED.with(|cell| cell.set(self.0));
    }
}

impl Drop for StopScope {
    fn drop(&mut self) {
        STOP_ARMED.with(|cell| cell.set(self.0));
        STOP_HIT.with(|cell| cell.set(self.1));
    }
}

pub(super) fn spent() -> u64 {
    SPENT.with(std::cell::Cell::get)
}

pub(super) fn charge_spent(nodes: u64) {
    SPENT.with(|cell| cell.set(cell.get().wrapping_add(nodes)));
}

/// Opens the budget-out log on this thread, discarding any left open: every
/// reducer check run from now on (on this thread, or on a `reduce --jobs`
/// worker for a step this thread commits or refuses) whose final answer ran
/// out of budget is logged until [`budget_outs_take`].  Speculative steps
/// the committed order discards are not.  Measurement only: the reducer's
/// output is the same with the log open or not.
pub fn budget_outs_begin() {
    BUDGET_OUTS.with(|log| *log.borrow_mut() = Some(Vec::new()));
}

/// Closes the log and returns its entries, sorted by stage, diagram and the
/// rest ([`BudgetOut::sort_key`]): on one position thread the same for any
/// `--jobs`, except on an input the early stop ended ([`stopped`]), whose
/// log holds the checks consumed up to the stop, and which check stopped it
/// may depend on `--jobs`.
pub fn budget_outs_take() -> Vec<BudgetOut> {
    let mut entries = BUDGET_OUTS.with(|log| log.borrow_mut().take().unwrap_or_default());
    entries.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    entries
}

/// Runs `body` with a log of its own on this thread (a greedy step's,
/// [`evaluate_step`]) and returns its entries; the thread's log is restored
/// afterwards, also when `body` unwinds ([`LogFrame`]).
#[doc(hidden)]
pub fn with_budget_out_frame<R>(body: impl FnOnce() -> R) -> (R, Vec<BudgetOut>) {
    let (out, log) = with_frame(body);
    (out, log.entries)
}

/// What one log frame ([`with_frame`]) collected: its budget-outs and the
/// largest ladder budget its checks needed ([`note_rung`]).
pub(super) struct StepLog {
    pub(super) entries: Vec<BudgetOut>,
    pub(super) rung: u64,
}

/// [`with_budget_out_frame`], with the frame's largest rung.
pub(super) fn with_frame<R>(body: impl FnOnce() -> R) -> (R, StepLog) {
    let frame = LogFrame::open();
    let out = body();
    (out, frame.close())
}

/// The log [`with_budget_out_frame`] replaced, put back on drop.
/// The frame also has its own [`STOP_HIT`] (the outer one is restored with
/// the log): a budget-out inside a greedy step stops that step's sweep, and
/// reaches the record only when the step's entries are absorbed
/// ([`absorb_budget_outs`]).
/// The frame has its own [`RUNG`] too, handed on the same way.
pub(super) struct LogFrame(
    pub(super) Option<Option<Vec<BudgetOut>>>,
    pub(super) bool,
    pub(super) u64,
);

impl LogFrame {
    pub(super) fn open() -> Self {
        let hit = STOP_HIT.with(|cell| cell.replace(false));
        let rung = RUNG.with(|cell| cell.replace(0));
        Self(
            Some(BUDGET_OUTS.with(|log| log.replace(Some(Vec::new())))),
            hit,
            rung,
        )
    }

    /// Restores the outer log and returns the frame's entries and rung.
    pub(super) fn close(mut self) -> StepLog {
        let outer = self.0.take().expect("an open frame");
        STOP_HIT.with(|cell| cell.set(self.1));
        let rung = RUNG.with(|cell| cell.replace(self.2));
        let entries = BUDGET_OUTS
            .with(|log| std::mem::replace(&mut *log.borrow_mut(), outer))
            .unwrap_or_default();
        StepLog { entries, rung }
    }
}

impl Drop for LogFrame {
    fn drop(&mut self) {
        if let Some(outer) = self.0.take() {
            STOP_HIT.with(|cell| cell.set(self.1));
            RUNG.with(|cell| cell.set(self.2));
            BUDGET_OUTS.with(|log| *log.borrow_mut() = outer);
        }
    }
}

/// `true` while a budget-out log is open on this thread (tests).
#[doc(hidden)]
pub fn budget_outs_active() -> bool {
    BUDGET_OUTS.with(|log| log.borrow().is_some())
}

/// Adds the entries of a consumed greedy step ([`evaluate_step`]) to the
/// thread's log, and its rung to the frame's.
pub(super) fn absorb_budget_outs(log: StepLog) {
    let StepLog { entries, rung } = log;
    if rung > 0 {
        RUNG.with(|cell| cell.set(cell.get().max(rung)));
    }
    if entries.is_empty() {
        return;
    }
    if entries.iter().any(BudgetOut::counts) {
        STOP_HIT.with(|cell| cell.set(true));
    }
    BUDGET_OUTS.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.extend(entries);
        }
    });
}

/// The budget-outs behind one output of [`reduce_two_kinds`] (owner
/// 2026-10-22: the reducer reports them on every
/// record, not only in the `--emit-budget-outs` side file): the distinct
/// diagrams whose acceptance check ended out of budget on its full budget
/// (the ladder's last rung), so a step on them was refused for lack of budget and may have kept a piece.  `checks` 0 is
/// the record's `reduction.minimality` `"verified"`: every refusal behind the
/// output was decided; otherwise `"unverified"`.
///
/// Which checks count ([`BudgetOutTally::of`]): acceptance checks (`kind`
/// `check`; the entry check's budget-out rejects the input), not refused by the structural
/// main-line test anyway ([`BudgetOut::structural_refusal`]), of the output's
/// kind (the main-line kind's checks measure `Π`, the length kind's do not; the
/// attacker-king deletion is both kinds'), counted once per diagram (the SFEN,
/// the target and whether `Π` was measured): a pass the settled skip leaves
/// out repeats checks already counted, so the tally is the same with every
/// speed-up on or off, with `--emit-negatives` or the pass trace, and for
/// every `--jobs`.  On a parallel prober (no one-thread re-run since
/// 2026-10-23) the set of checks that ran out, and so the tally, may depend
/// on the threads and the schedule; a check counted is always one that really
/// ran out, so a tally of 0 (`verified`) is exact whatever the threads.  The
/// TypeScript `reduceStageAll` counts the same.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BudgetOutTally {
    /// Distinct diagrams whose check ended out of budget.
    pub checks: u64,
    /// The same by stage (`greedy`, `sweep`, `king`, `repair`); a diagram
    /// checked in two stages counts under the first in byte order.
    pub stages: std::collections::BTreeMap<&'static str, u64>,
}

impl BudgetOutTally {
    /// The tally of a record the early stop ended ([`ReducedSet::stopped`]):
    /// one check, no stage.  Which check stopped it, and in which stage, may
    /// depend on `--jobs` (the pool's chains run side by side), so the record
    /// says only that one did (`--emit-budget-outs` names it).
    pub fn stopped() -> Self {
        Self {
            checks: 1,
            stages: std::collections::BTreeMap::new(),
        }
    }

    /// The tally of the main-line kind (`main_line`) or the length kind from
    /// the log of one [`reduce_two_kinds`] call.
    pub fn of(entries: &[BudgetOut], main_line: bool) -> Self {
        let mut first: std::collections::BTreeMap<(&str, Option<u32>, bool), &'static str> =
            std::collections::BTreeMap::new();
        for entry in entries {
            let ours = if main_line {
                entry.paths
            } else {
                !entry.paths || entry.stage == "king"
            };
            if entry.kind != "check" || entry.structural_refusal || !ours {
                continue;
            }
            first
                .entry((entry.sfen.as_str(), entry.target, entry.paths))
                .and_modify(|stage| *stage = (*stage).min(entry.stage))
                .or_insert(entry.stage);
        }
        let mut tally = Self::default();
        for stage in first.into_values() {
            tally.checks += 1;
            *tally.stages.entry(stage).or_insert(0) += 1;
        }
        tally
    }

    /// `reduction.minimality`: `"verified"` without a budget-out, else
    /// `"unverified"`.
    pub fn minimality(&self) -> &'static str {
        if self.checks == 0 {
            "verified"
        } else {
            "unverified"
        }
    }
}

/// Marks the entries of a step the structural main-line test refuses
/// ([`BudgetOut::structural_refusal`]).
pub(super) fn mark_structural(entries: &mut [BudgetOut]) {
    for entry in entries {
        entry.structural_refusal = true;
    }
}

/// Logs a check whose final answer on the last rung ran out of budget.
pub(super) fn log_budget_out(
    checked: &Checked<'_>,
    budget: u64,
    mode: ThreadMode,
    first: BudgetOutRun,
) {
    BUDGET_OUTS.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            let (stage, step, removed) = STAGE.with(|info| {
                let info = info.borrow();
                if checked.kind == "entry" {
                    ("entry", None, Vec::new())
                } else {
                    (info.stage, info.step.clone(), info.removed.clone())
                }
            });
            let entry = BudgetOut {
                kind: checked.kind,
                stage,
                step,
                removed,
                sfen: crate::sfen::to_sfen(checked.position),
                target: checked.target,
                paths: checked.paths,
                budget,
                mode,
                first,
                structural_refusal: false,
            };
            if entry.counts() {
                STOP_HIT.with(|cell| cell.set(true));
            }
            log.push(entry);
        }
    });
}

/// What the checks run now belong to.
#[derive(Clone)]
pub(super) struct StageInfo {
    pub(super) stage: &'static str,
    pub(super) step: Option<RemovedPiece>,
    pub(super) removed: Vec<RemovedPiece>,
}

/// Labels the checks run while it lives (the budget-out log): `greedy` (a
/// greedy step's own check), `sweep` (an inert-sweep round), `repair` (the
/// C/G entry's checks), `king` (the attacker-king deletion), `start` (the
/// length-preserving reducer's check of its start diagram); `check`
/// otherwise.  `removed` is what the checked diagram took off; `step`, when
/// given, the greedy step it belongs to (a sweep inside a step keeps it).
pub(super) struct Stage(pub(super) Option<StageInfo>);

impl Stage {
    pub(super) fn enter(
        stage: &'static str,
        step: Option<&RemovedPiece>,
        removed: Vec<RemovedPiece>,
    ) -> Self {
        Self(Some(STAGE.with(|info| {
            let mut info = info.borrow_mut();
            let step = step.cloned().or_else(|| info.step.clone());
            std::mem::replace(
                &mut *info,
                StageInfo {
                    stage,
                    step,
                    removed,
                },
            )
        })))
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(saved) = self.0.take() {
            STAGE.with(|info| *info.borrow_mut() = saved);
        }
    }
}
