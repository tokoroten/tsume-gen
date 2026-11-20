//! The reducer's measurement counters (`TSUME_V3_STATS`,
//! [`crate::diagnostics`]): read only by the reports, never by a reduction.

use super::*;

/// Counters for measurement: candidates the prefilter asked, rejected without
/// the exact search, `UNKNOWN` (out of budget), prober nodes; candidates the
/// construction decided, out of budget in the construction, its prober nodes.
/// Since solver 1.4.0 the two budget-outs (indices 2 and 5, `unknown=` and
/// `fallback=` in `TSUME_V3_STATS`) are budget-outs of the gate whenever the
/// construction is on; they reach the full-width search only with the
/// construction off or the test-only `set_exact_fallback`.
pub static PREFILTER_STATS: [AtomicU64; 7] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Wall nanoseconds per phase of the gate, summed over threads (measurement
/// only, printed with `TSUME_V3_STATS`): the prefilter's question, the
/// construction, the `nodesProof` recount and the full-width fallback of the
/// publishing gate.
pub static PHASE_NANOS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Adds the time since `started` to [`PHASE_NANOS`]`[phase]`.
pub fn charge_phase(phase: usize, started: crate::clock::Instant) {
    PHASE_NANOS[phase].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Tests decided per rung of the ladder (measurement only, printed with
/// `TSUME_V3_STATS`, counted only while a ladder is set): index `i` counts the
/// tests decided on the `i + 1`-th budget tried (the full budget is the last
/// one tried, so a test the full budget left undecided counts there too).
pub static LADDER_STATS: [AtomicU64; LADDER_RUNGS + 1] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// How often each exact speed-up of `docs/experiments/extend-speedup/` fired:
/// `[settled passes skipped, steps the structural main-line test refused,
/// fast-path memo hits]`.  Process-wide, never reset; tests read the change.
pub(super) static SPEEDUP_HITS: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// [`SPEEDUP_HITS`] (tests only).
#[doc(hidden)]
pub fn speedup_hits() -> [u64; 3] {
    SPEEDUP_HITS
        .each_ref()
        .map(|hits| hits.load(Ordering::Relaxed))
}

/// Full-width gates (`exact_validate_with_rules`) the reducer's checks, its
/// entry check, the publishing gate and extend's parent-yozume measurement
/// (`exact::measure_main_line_yozume`) ran (measurement and tests only: never
/// read by a reduction).  With the construction on, a fast-path budget-out
/// adds nothing here since solver 1.4.0 (owner 2026-11-05); the counter is how
/// `tests/fast_budget_out.rs` sees that no one-thread full-width search ran.
/// Process-wide, never reset; tests read the change.
pub static FULL_WIDTH_GATES: AtomicU64 = AtomicU64::new(0);

/// Counts one full-width gate in [`FULL_WIDTH_GATES`].
pub(crate) fn count_full_width_gate() {
    FULL_WIDTH_GATES.fetch_add(1, Ordering::Relaxed);
}
