//! The reducer's process-wide switches (set once by the CLI or a test,
//! before any reduction starts).

use super::*;

/// The classical df-pn prefilter of the acceptance test (solver v3 item 2):
/// on by default; [`set_prefilter`] turns it off for measurement (and with it
/// the construction, which runs on the prefilter's prober).
pub(super) static PREFILTER_ON: AtomicBool = AtomicBool::new(true);

/// The construction of the acceptance test (solver v3 item 1).  **On by
/// default** (owner decision 2026-10-09, `docs/experiments/solver-v3/README.md`
/// §5-1): a gate the full-width search would leave undecided for lack of
/// budget is decided instead.  The TypeScript reducer
/// (`tools/puzzle-pipeline/src/stages.ts`, `fastValidate` in `solve.ts`) runs
/// the same prober and the same construction, so the two reducers stay byte
/// for byte equal at every budget (since solver 1.4.0 only with
/// [`set_exact_fallback`], where the fast path runs out of budget: the Rust
/// reducer then reports a budget-out, the TypeScript one runs the exact
/// search).  [`set_construct`] turns it off, which the
/// CLI does when the environment variable `TSUME_REDUCE_CONSTRUCT=0` is set
/// (as the TypeScript reducer does); off, every output is byte-identical to
/// the reducer without solver v3.
pub(super) static CONSTRUCT_ON: AtomicBool = AtomicBool::new(true);

/// Tests only (`TSUME_FAST_EXACT_FALLBACK=1`, off by default): a fast-path
/// budget-out ([`Fast::BudgetOut`]) leaves the gate to the one-thread
/// full-width search on the same budget, as solver 1.3.0 and earlier did.
///
/// Since solver 1.4.0 (owner 2026-11-05, in line with
/// the owner's rule of 2026-10-23: no re-run on one thread) a fast path that runs out
/// is a budget-out.  The TypeScript reducer and `solveStage`
/// (`fastValidate` in `tools/puzzle-pipeline/src/solve.ts`) still fall back,
/// and `diff-reduce`'s `BUDGET` and `TALLY` records pin how the two run out of
/// budget byte for byte (with `TSUME_CDFPN_LEAF=0`), so the script turns this
/// on for them.  It is kept for that parity check alone; no production run
/// sets it (the catalog records of a run that does are tagged `+x`).
pub(super) static EXACT_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Turns `EXACT_FALLBACK` on or off (tests only): a fast-path budget-out
/// leaves the gate to the full-width search, as solver 1.3.0 did.  Set it
/// before any search starts.
#[doc(hidden)]
pub fn set_exact_fallback(on: bool) {
    EXACT_FALLBACK.store(on, Ordering::Relaxed);
}

/// Whether [`EXACT_FALLBACK`] is on.
pub(super) fn exact_fallback_on() -> bool {
    EXACT_FALLBACK.load(Ordering::Relaxed)
}

/// Lowered caps of the prober's tables (tests only, `0` = the default):
/// [`set_cdfpn_caps`], which the CLI calls for `TSUME_CDFPN_CAPS=a,b,c` so the
/// differential test can drive both reducers through the garbage collection.
pub(super) static CDFPN_CAPS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

/// Tests only: every full-budget acceptance check on a search whose
/// budget-outs may vary ([`verdicts_may_vary`]) is answered as if it ran out,
/// so a test sees the kept budget-out on every check regardless of the
/// schedule.
pub(super) static FORCE_OUT: AtomicBool = AtomicBool::new(false);

/// Turns [`FORCE_OUT`] on or off (tests only).
#[doc(hidden)]
pub fn set_force_parallel_budget_out(on: bool) {
    FORCE_OUT.store(on, Ordering::Relaxed);
}

/// [`stopped`] is read.  On by default; the switch exists only so a test can
/// count the checks the reducer ran before the early stop (2026-10-25).
/// Process-wide.
pub(super) static EARLY_STOP: AtomicBool = AtomicBool::new(true);

/// Turns the early stop on or off (tests only).
#[doc(hidden)]
pub fn set_early_stop(on: bool) {
    EARLY_STOP.store(on, Ordering::Relaxed);
}

/// Turns the construction of `check_reduction` and of the entry check on or
/// off (solver v3 item 1, on by default; see `CONSTRUCT_ON`).  Set it before
/// any reduction starts.
pub fn set_construct(on: bool) {
    CONSTRUCT_ON.store(on, Ordering::Relaxed);
}

/// Lowers the caps of the reducer's prober tables (tests only; `[0; 3]`
/// restores the defaults).  Set it before any reduction starts.
#[doc(hidden)]
pub fn set_cdfpn_caps(caps: [usize; 3]) {
    for (slot, cap) in CDFPN_CAPS.iter().zip(caps) {
        slot.store(cap, Ordering::Relaxed);
    }
}

/// Turns the prefilter of `check_reduction` on or off (measurement only: the
/// verdicts, and so every output byte, are the same either way).
pub fn set_prefilter(on: bool) {
    PREFILTER_ON.store(on, Ordering::Relaxed);
}

/// The path cap of rule D v5 ([`MAIN_LINE_PATHS_CAP`]), lowered only by tests
/// and the differential test ([`set_main_line_paths_cap`]).
pub(super) static PATHS_CAP: AtomicUsize = AtomicUsize::new(MAIN_LINE_PATHS_CAP);

/// Lowers (or restores) the path cap of the main-line-preserving reduction.
/// Tests only: the CLI calls it for `TSUME_REDUCE_PATHS_CAP=n`, which the
/// TypeScript reducer reads too, so `diff-reduce` can drive both past the cap.
/// Set it before any reduction starts.
#[doc(hidden)]
pub fn set_main_line_paths_cap(cap: usize) {
    PATHS_CAP.store(cap, Ordering::Relaxed);
}

pub(super) fn paths_cap() -> usize {
    PATHS_CAP.load(Ordering::Relaxed)
}

/// The largest `TSUME_REDUCE_PATHS_CAP` both reducers read exactly
/// (`Number.MAX_SAFE_INTEGER`, 2^53 - 1).
pub const PATHS_CAP_MAX: u64 = 9_007_199_254_740_991;

/// `TSUME_REDUCE_PATHS_CAP` spelled as a number, read as the TypeScript reducer
/// reads it (`parsePathsCap` in `tools/puzzle-pipeline/src/stages.ts`): ASCII
/// digits with ASCII spaces and tabs around them and nothing else (no sign,
/// no decimal point, no other whitespace), at most [`PATHS_CAP_MAX`]; `None`
/// otherwise (1.0.0: `+1`, values past `usize` and some Unicode whitespace
/// were read differently on the two sides).  `tests/reduce_paths_cap.rs` and
/// `tests/pipeline.test.ts` check the same spellings.
pub fn parse_paths_cap(spelled: &str) -> Option<usize> {
    let digits = spelled.trim_matches(|c| c == ' ' || c == '\t');
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits
        .parse::<u64>()
        .ok()
        .filter(|&cap| cap <= PATHS_CAP_MAX)
        .and_then(|cap| usize::try_from(cap).ok())
}

/// [`fast_validate`] reuses its last answer for the same question
/// ([`LAST_FAST`]).  On by default; the switch exists so a test can show the
/// output is the same without it.  Process-wide.
pub(super) static FAST_MEMO: AtomicBool = AtomicBool::new(true);

/// Turns the memo of [`fast_validate`] on or off (tests and measurements only:
/// the output is the same either way).
#[doc(hidden)]
pub fn set_fast_memo(on: bool) {
    FAST_MEMO.store(on, Ordering::Relaxed);
}

pub(super) static RULE_B_EARLY_STOP: AtomicBool = AtomicBool::new(true);

/// Turns off the rule B entry's early stop (the first attempt took no step, so
/// the others are not run), for the test that pins it changes no output
/// (`docs/experiments/yozume-reduce/REPORT.md` §3.6).  Process-wide; on by default.
pub fn set_rule_b_early_stop(on: bool) {
    RULE_B_EARLY_STOP.store(on, Ordering::Relaxed);
}

pub(super) static TRACE_PASSES: AtomicBool = AtomicBool::new(false);

/// [`ChainState::settled`] is read (a pass that would repeat a whole pass that
/// took nothing is not run).  On by default; the switch exists so a test can
/// show the records are the same without it.  Process-wide.
pub(super) static SETTLED_SKIP: AtomicBool = AtomicBool::new(true);

/// Turns the settled-pass skip ([`ChainState::settled`]) on or off.  Tests and
/// measurements only: the output is the same either way.
#[doc(hidden)]
pub fn set_settled_skip(on: bool) {
    SETTLED_SKIP.store(on, Ordering::Relaxed);
}

/// Keeps the record of the greedy passes ([`ReducedSet::trace`]) for the
/// measurement of how often a later pass still removes something (2026-10-09).
/// Process-wide, off by default; the sequential path (`jobs` 1) only.  Never
/// written to a record.
pub fn set_trace_passes(on: bool) {
    TRACE_PASSES.store(on, Ordering::Relaxed);
}

/// The structural main-line test is used ([`main_line_unreachable`]).  On by
/// default; the switch exists so a test can show the output is the same
/// without it.  Process-wide.
pub(super) static MAIN_LINE_REPLAY: AtomicBool = AtomicBool::new(true);

/// Turns the structural main-line test on or off (tests and measurements only:
/// the output is the same either way).
#[doc(hidden)]
pub fn set_main_line_replay(on: bool) {
    MAIN_LINE_REPLAY.store(on, Ordering::Relaxed);
}

/// Whether a step [`main_line_unreachable`] refuses is skipped without a
/// search: the test is on ([`set_main_line_replay`]) and neither the pass
/// trace ([`set_trace_passes`], which records every refusal's reason) nor a
/// negatives' collection is open (`--emit-negatives`: the step is then
/// searched and its refusal reported with the verdict's reasons, as before
/// the test; the refused diagram is a negative candidate).
pub(super) fn main_line_replay_on() -> bool {
    MAIN_LINE_REPLAY.load(Ordering::Relaxed)
        && !TRACE_PASSES.load(Ordering::Relaxed)
        && !negatives_open()
}
