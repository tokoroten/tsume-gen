//! Optional per-phase time accounting for `mine` and `distill` (2026-10-02).
//!
//! Off unless the environment variable `TSUME_PROFILE` is set to a non-empty
//! value other than `0`; when off every call here is one relaxed load and a
//! branch.  When on, `--stats-out` gains a `profile` object.  It is measurement
//! only: wall time per worker thread, summed over threads (so comparable with
//! `cpuSeconds` when the threads are not descheduled), never read by any search
//! and never part of a record, so output bytes do not depend on it.
//!
//! Two axes:
//!
//! * **phase**: where in the stage the thread is.  [`enter`] opens a phase on a
//!   per-thread stack and returns a guard; [`switch`] replaces the phase at the top
//!   of the stack.  Time is charged exclusively to the top of the stack.
//! * **engine**: [`engine`] wraps one search call (the df-pn prober, the exact
//!   solver with or without a graph).  Its time is charged to the phase that is
//!   current when it starts.  Nested engine calls are folded into the outer one.
//!
//! A phase's `otherSeconds` is its exclusive time minus the engine time charged to
//! it: move generation for candidates, quality scoring, JSON, and so on.

use crate::clock::Instant;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Map, Value};

/// `TSUME_PROFILE` ([`crate::diagnostics::profile`]).
#[inline]
pub fn enabled() -> bool {
    crate::diagnostics::profile()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    /// Time outside any phase (thread start-up, I/O between records).
    Unscoped,
    MineGenerate,
    MineMateSearch,
    MinePrestrip,
    MineRecord,
    DistillSource,
    DistillPrestrip,
    DistillBaseline,
    DistillVariants,
    DistillRank,
    DistillFinal,
    /// `--emit-negatives`: the re-judgement of the picked negatives
    /// ([`crate::negatives::finalize`]), kept apart from the stages it observes.
    Negatives,
}

const PHASES: usize = 12;
const PHASE_NAMES: [&str; PHASES] = [
    "unscoped",
    "mine.generate",
    "mine.mateSearch",
    "mine.prestrip",
    "mine.record",
    "distill.source",
    "distill.prestrip",
    "distill.baseline",
    "distill.variants",
    "distill.rank",
    "distill.final",
    "negatives",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Engine {
    /// [`crate::dfpn::Prober`] entry points.
    Dfpn,
    /// [`crate::solver::Solver::solve`] without a solution graph.
    ExactNoGraph,
    /// [`crate::solver::Solver::solve`] building the graph (the search whose
    /// `stats.nodes` is published metadata when the record is final).
    ExactGraph,
}

const ENGINES: usize = 3;
const ENGINE_NAMES: [&str; ENGINES] = ["dfpn", "exactNoGraph", "exactGraph"];

struct Totals {
    phase_ns: [AtomicU64; PHASES],
    engine_ns: [[AtomicU64; ENGINES]; PHASES],
    engine_calls: [[AtomicU64; ENGINES]; PHASES],
}

static TOTALS: Totals = Totals {
    phase_ns: [const { AtomicU64::new(0) }; PHASES],
    engine_ns: [const { [const { AtomicU64::new(0) }; ENGINES] }; PHASES],
    engine_calls: [const { [const { AtomicU64::new(0) }; ENGINES] }; PHASES],
};

thread_local! {
    /// Phase stack; each frame is `(phase, start of its current stretch)`.
    static STACK: RefCell<Vec<(Phase, Instant)>> = const { RefCell::new(Vec::new()) };
    static IN_ENGINE: Cell<bool> = const { Cell::new(false) };
}

fn charge(phase: Phase, since: Instant, now: Instant) {
    let ns = now.duration_since(since).as_nanos() as u64;
    TOTALS.phase_ns[phase as usize].fetch_add(ns, Ordering::Relaxed);
}

fn current_phase() -> Phase {
    STACK.with(|stack| stack.borrow().last().map_or(Phase::Unscoped, |f| f.0))
}

/// Guard of [`enter`]: pops the phase and resumes the one below it.
pub struct PhaseGuard {
    active: bool,
}

/// Opens `phase` on top of the current one until the guard drops.
pub fn enter(phase: Phase) -> PhaseGuard {
    if !enabled() {
        return PhaseGuard { active: false };
    }
    let now = Instant::now();
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(top) = stack.last_mut() {
            charge(top.0, top.1, now);
        }
        stack.push((phase, now));
    });
    PhaseGuard { active: true }
}

/// Replaces the phase at the top of the stack (no-op outside any [`enter`]).
pub fn switch(phase: Phase) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    STACK.with(|stack| {
        if let Some(top) = stack.borrow_mut().last_mut() {
            charge(top.0, top.1, now);
            *top = (phase, now);
        }
    });
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let now = Instant::now();
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((phase, since)) = stack.pop() {
                charge(phase, since, now);
            }
            if let Some(top) = stack.last_mut() {
                top.1 = now;
            }
        });
    }
}

/// Guard of [`engine`].
pub struct EngineGuard {
    start: Option<(Instant, Phase, Engine)>,
}

/// Times one engine call until the guard drops.
pub fn engine(engine: Engine) -> EngineGuard {
    if !enabled() || IN_ENGINE.with(Cell::get) {
        return EngineGuard { start: None };
    }
    IN_ENGINE.with(|flag| flag.set(true));
    EngineGuard {
        start: Some((Instant::now(), current_phase(), engine)),
    }
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        if let Some((start, phase, engine)) = self.start {
            let ns = start.elapsed().as_nanos() as u64;
            TOTALS.engine_ns[phase as usize][engine as usize].fetch_add(ns, Ordering::Relaxed);
            TOTALS.engine_calls[phase as usize][engine as usize].fetch_add(1, Ordering::Relaxed);
            IN_ENGINE.with(|flag| flag.set(false));
        }
    }
}

/// The `profile` object for `--stats-out`, or `None` when profiling is off.
/// Phases with no time are left out.
pub fn report() -> Option<Value> {
    if !enabled() {
        return None;
    }
    let secs = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
    let mut phases = Map::new();
    let mut engine_total = [0f64; ENGINES];
    let mut total = 0f64;
    for (p, phase_name) in PHASE_NAMES.iter().enumerate() {
        let phase_s = secs(&TOTALS.phase_ns[p]);
        let mut engines = Map::new();
        let mut in_engines = 0f64;
        for e in 0..ENGINES {
            let s = secs(&TOTALS.engine_ns[p][e]);
            let calls = TOTALS.engine_calls[p][e].load(Ordering::Relaxed);
            if calls > 0 {
                engines.insert(
                    ENGINE_NAMES[e].to_string(),
                    json!({"seconds": s, "calls": calls}),
                );
                in_engines += s;
                engine_total[e] += s;
            }
        }
        if phase_s == 0.0 && in_engines == 0.0 {
            continue;
        }
        total += phase_s;
        phases.insert(
            phase_name.to_string(),
            json!({
                "seconds": phase_s,
                "engines": engines,
                "otherSeconds": phase_s - in_engines,
            }),
        );
    }
    let mut engines = Map::new();
    for (name, seconds) in ENGINE_NAMES.iter().zip(engine_total) {
        engines.insert(name.to_string(), json!(seconds));
    }
    Some(json!({
        "note": "wall seconds per worker thread, summed over threads (TSUME_PROFILE)",
        "seconds": total,
        "engines": engines,
        "phases": phases,
    }))
}
