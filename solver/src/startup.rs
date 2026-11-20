//! What the process sets up before a command runs: the allocator, the experimental switches read from the environment (each noted on stderr and tagging catalog records `+x`), and the retired variables' warnings.

use super::*;

/// Large allocations on 2 MiB pages, on by default since 1.5.0
/// (`TSUME_LARGE_PAGES=0` turns them off; `src/largepage.rs`,
/// `docs/experiments/solver-scaling/README.md` §3 and §16).
#[global_allocator]
pub(crate) static GLOBAL: largepage::Alloc = largepage::Alloc;

/// Set once an experimental option or switch is in effect: catalog records
/// then carry the tag `+x` after the solver id ([`RecordArgs::setup`]).
pub(crate) static EXPERIMENTAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The solver-id tag of a run with an experimental option or switch
/// (stage 4, 2026-10-22, `docs/experiments/solver-cli-audit/README.md` §7.4).
pub(crate) const EXPERIMENTAL_TAG: &str = "+x";

/// Notes on stderr that the experimental option `name` is in effect (when
/// `used`), and marks the run so its catalog records are tagged `+x`.
pub(crate) fn note_experimental(name: &str, used: bool) {
    if used {
        EXPERIMENTAL.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "tsume-solver: experimental {name} in effect (catalog records are tagged {EXPERIMENTAL_TAG})"
        );
    }
}

/// Environment variables the solver read once and reads no more (the
/// measurement switches retired on 2026-10-27,
/// `docs/experiments/solver-cli-audit/README.md` §8), with what became of each:
/// a run that still sets one is told on stderr instead of silently running
/// with the default.
/// Kept through 1.x (plan 2026-10-28): `(name, since, what instead)`.
pub(crate) const REMOVED_ENV: [(&str, &str, &str); 13] = [
    (
        "TSUME_V3_NO_PREFILTER",
        "2026-10-27",
        "a measurement switch, gone",
    ),
    (
        "TSUME_SOLVE_FALLBACK_BUDGET",
        "2026-10-27",
        "a measurement switch, gone",
    ),
    ("TSUME_PDFPN_JOIN", "2026-10-27", "fixed at PAR_JOIN_AFTER"),
    (
        "TSUME_PDFPN_VIRTUAL",
        "2026-10-27",
        "fixed at PAR_VIRTUAL_UNIT",
    ),
    (
        "TSUME_PDFPN_SWEEP",
        "2026-10-27",
        "fixed at SWEEP_AT_PERCENT",
    ),
    (
        "TSUME_PDFPN_KEEP",
        "2026-10-27",
        "a full bucket replaces its least valuable unvisited slot",
    ),
    (
        "TSUME_PDFPN_WAIT",
        "2026-10-27",
        "the main prober does not wait for the helpers",
    ),
    (
        "TSUME_PDFPN_FORCE_PAR",
        "2026-10-27",
        "one prober thread always searches serially",
    ),
    (
        "TSUME_LAYERED_LAYOUT",
        "2026-10-27",
        "fixed at the default layout",
    ),
    (
        "TSUME_RERUN_TRACE",
        "2026-10-27",
        "the one-thread re-run is gone",
    ),
    (
        "TSUME_COMPACT_SERIAL_FIRST",
        "2026-10-27",
        "the position pool is gone",
    ),
    (
        "TSUME_CDFPN_REREAD",
        "1.0.0",
        "the prober always keeps its children's numbers; build an older commit for prober-leaf3",
    ),
    (
        "TSUME_CDFPN_EPS",
        "1.0.0",
        "the 1+ε threshold is always on (prober-v3); build an older commit for prober-v2",
    ),
];

pub(crate) fn warn_removed_env() {
    for (name, since, instead) in REMOVED_ENV {
        if std::env::var_os(name).is_some() {
            eprintln!("tsume-solver: {name} is no longer read (since {since}): {instead}");
        }
    }
}

/// Reads the environment before the command line: the profiler's period, the
/// retired variables' warnings, `TSUME_RECORD_CLOCK`, and the experimental
/// switches (each noted on stderr; a misspelt value exits 2).
pub(crate) fn configure() {
    tsume_solver::posprof::start_periodic();
    // Solver v3 switches of the reducer (`docs/experiments/solver-v3/README.md`).
    // Item 1, the construction, is on by default (owner decision 2026-10-09):
    // it decides gates the exact search leaves undecided for lack of budget,
    // exactly as the TypeScript reducer does.  `TSUME_REDUCE_CONSTRUCT=0`
    // turns it off in both (the pre-v3 reducer).  `TSUME_CDFPN_CAPS=a,b,c` lowers the prober's
    // table caps (tests only, read by the TypeScript reducer too).
    //
    // Each of these variables, when it takes effect, prints a note on stderr
    // and tags catalog records `+x` (stage 4, 2026-10-22,
    // `docs/experiments/solver-cli-audit/README.md` §7.4).
    warn_removed_env();
    // `TSUME_RECORD_CLOCK` (`fixed`, or unset / `real`): the catalog records'
    // times ([`tsume_solver::record_clock`]); a misspelt value is refused.
    if let Err(message) = tsume_solver::record_clock::check_env() {
        eprintln!("tsume-solver: {message}");
        std::process::exit(2);
    }
    if std::env::var_os("TSUME_REDUCE_CONSTRUCT").is_some_and(|value| value == "0") {
        note_experimental("TSUME_REDUCE_CONSTRUCT=0", true);
        tsume_solver::reduce::set_construct(false);
    }
    // `TSUME_SOLVE_CONSTRUCT=0` turns item 1 off in the publishing gate
    // (`solve`), on by default (owner decision 2026-10-09 23:00 JST).
    if std::env::var_os("TSUME_SOLVE_CONSTRUCT").is_some_and(|value| value == "0") {
        note_experimental("TSUME_SOLVE_CONSTRUCT=0", true);
        tsume_solver::stages::set_solve_construct(false);
    }
    // `TSUME_CDFPN_LEAF=0` turns the prober's leaf search off (tests only:
    // `diff-reduce` runs its budget records so, where the TypeScript prober,
    // which has none, runs out of budget at the same checks).
    if std::env::var_os("TSUME_CDFPN_LEAF").is_some_and(|value| value == "0") {
        note_experimental("TSUME_CDFPN_LEAF=0", true);
        tsume_solver::cdfpn::set_leaf_search(false);
    }
    // `TSUME_FAST_EXACT_FALLBACK=1` (tests only): a fast-path budget-out runs
    // the one-thread full-width gate on the same budget, as solver 1.3.0 did.
    // Since 1.4.0 (owner 2026-11-05) it is a budget-out; `diff-reduce` sets
    // this for its budget records, whose TypeScript side still falls back.
    if std::env::var_os("TSUME_FAST_EXACT_FALLBACK").is_some_and(|value| value == "1") {
        note_experimental("TSUME_FAST_EXACT_FALLBACK=1", true);
        tsume_solver::reduce::set_exact_fallback(true);
    }
    // Unset and empty mean the same (the defaults), as in TypeScript.
    if let Some(spelled) = std::env::var_os("TSUME_CDFPN_CAPS").filter(|value| !value.is_empty()) {
        let spelled = spelled.to_string_lossy().into_owned();
        // Exactly three positive integers, as the TypeScript side demands: decimal
        // digits only, at most 2^53 - 1 (`Number.isSafeInteger`).
        let caps: Vec<Option<usize>> = spelled
            .split(',')
            .map(|part| {
                let part = part.trim();
                if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                part.parse::<u64>()
                    .ok()
                    .filter(|&cap| cap > 0 && cap <= 9_007_199_254_740_991)
                    .and_then(|cap| usize::try_from(cap).ok())
            })
            .collect();
        match caps[..] {
            [Some(table), Some(intervals), Some(dominance)] => {
                note_experimental("TSUME_CDFPN_CAPS", true);
                tsume_solver::reduce::set_cdfpn_caps([table, intervals, dominance]);
            }
            _ => {
                eprintln!("TSUME_CDFPN_CAPS must be three positive integers, got {spelled}");
                std::process::exit(2);
            }
        }
    }
    // `TSUME_REDUCE_PATHS_CAP=n` lowers rule D v5's path cap (tests only; the
    // TypeScript reducer reads the same variable, so `diff-reduce` can drive
    // both past it).
    if let Some(spelled) = std::env::var_os("TSUME_REDUCE_PATHS_CAP") {
        let spelled = spelled.to_string_lossy().into_owned();
        match tsume_solver::reduce::parse_paths_cap(&spelled) {
            Some(cap) => {
                note_experimental("TSUME_REDUCE_PATHS_CAP", true);
                tsume_solver::reduce::set_main_line_paths_cap(cap);
            }
            None => {
                eprintln!("TSUME_REDUCE_PATHS_CAP must be a non-negative integer, got {spelled}");
                std::process::exit(2);
            }
        }
    }
}

/// What the process prints on stderr after the command (`TSUME_V3_STATS`).
pub(crate) fn report() {
    if tsume_solver::diagnostics::v3_stats() {
        eprintln!("{}", largepage::summary());
        let stats = &tsume_solver::reduce::PREFILTER_STATS;
        let load = |index: usize| stats[index].load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "v3-prefilter asked={} rejected={} unknown={} nodes={} constructed={} fallback={} cnodes={}",
            load(0),
            load(1),
            load(2),
            load(3),
            load(4),
            load(5),
            load(6)
        );
        let nanos = &tsume_solver::reduce::PHASE_NANOS;
        let secs =
            |index: usize| nanos[index].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
        eprintln!(
            "v3-phases prefilter={:.1}s construct={:.1}s proof={:.1}s exact={:.1}s",
            secs(0),
            secs(1),
            secs(2),
            secs(3)
        );
        let ladder = &tsume_solver::reduce::LADDER_STATS;
        eprintln!(
            "v3-ladder decided-per-rung={:?}",
            ladder
                .iter()
                .map(|count| count.load(std::sync::atomic::Ordering::Relaxed))
                .collect::<Vec<_>>()
        );
    }
}
