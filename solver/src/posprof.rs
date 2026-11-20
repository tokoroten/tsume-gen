//! Per-position profile of the publishing gate and the reducer's checks
//! (owner 2026-10-16).
//!
//! Off unless `TSUME_PROFILE` is set (the switch of [`crate::prof`]); when off
//! every hook is one relaxed load and a branch, and nothing is printed.  When
//! on, `solve` and `reduce` print one `posprof {json}` line per record on
//! stderr after the record is done, and reset the counters.  The counters are
//! process-wide, so run with `--threads 1` to get one position per line.  Measurement only: nothing here is read by a search
//! and nothing reaches stdout or a record.
//!
//! Times are wall nanoseconds summed over the threads that ran the phase (so a
//! phase that ran on 16 threads for 1 s shows about 16 s).

use crate::clock::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Map, Value};

macro_rules! counters {
    ($($name:ident = $key:literal,)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy)]
        #[repr(usize)]
        pub enum K { $($name,)* }
        const KEYS: &[&str] = &[$($key,)*];
    };
}

counters! {
    PrefilterNs = "prefilterNs",
    PrefilterNodes = "prefilterNodes",
    ConstructNs = "constructNs",
    MainCalls = "mainCalls",
    MainProberNs = "mainProberNs",
    MainProberNodes = "mainProberNodes",
    ExactNs = "exactNs",
    ExactNodes = "exactNodes",
    ReduceChecks = "reduceChecks",
    ReduceNs = "reduceNs",
    MainIntervals = "mainIntervalsMax",
    MainTable = "mainTableMax",
}

const N: usize = KEYS.len();
/// Bounds 0..=15 of the "mate within `b`?" questions, then one bucket for more.
const BOUNDS: usize = 17;

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static COUNTERS: [AtomicU64; N] = [ZERO; N];
static MAIN_BOUNDS: [AtomicU64; BOUNDS] = [ZERO; BOUNDS];

/// `true` when profiling is on (`TSUME_PROFILE`).
#[inline]
pub fn on() -> bool {
    crate::prof::enabled()
}

/// Adds `value` to the counter `key`.
#[inline]
pub fn add(key: K, value: u64) {
    if on() {
        COUNTERS[key as usize].fetch_add(value, Ordering::Relaxed);
    }
}

/// Raises the counter `key` to at least `value` (table sizes).
#[inline]
pub fn max(key: K, value: u64) {
    if on() {
        COUNTERS[key as usize].fetch_max(value, Ordering::Relaxed);
    }
}

/// A start time when profiling is on.
#[inline]
pub fn start() -> Option<Instant> {
    on().then(Instant::now)
}

/// Adds the time since `started` to the counter `key`.
#[inline]
pub fn charge(key: K, started: Option<Instant>) {
    if let Some(started) = started {
        COUNTERS[key as usize].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Adds the time since `started` to the counter `key` while profiling is on
/// (the clock is not read otherwise).
#[inline]
pub fn charge_since(key: K, started: Instant) {
    if on() {
        COUNTERS[key as usize].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Counts one question of the construction at `bound`.
#[inline]
pub fn bound(bound: u32) {
    if on() {
        let slot = (bound as usize).min(BOUNDS - 1);
        MAIN_BOUNDS[slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// Prints the counters of the record `id` as one `posprof {json}` line on
/// stderr, with `wallNs` since `started`, and resets them.  Nothing when off.
pub fn flush(id: &str, started: Option<Instant>) {
    if !on() {
        return;
    }
    let mut map = Map::new();
    map.insert("id".into(), json!(id));
    if let Some(started) = started {
        map.insert("wallNs".into(), json!(started.elapsed().as_nanos() as u64));
    }
    for (key, counter) in KEYS.iter().zip(COUNTERS.iter()) {
        map.insert((*key).into(), json!(counter.swap(0, Ordering::Relaxed)));
    }
    let take = |table: &[AtomicU64; BOUNDS]| -> Vec<u64> {
        table
            .iter()
            .map(|count| count.swap(0, Ordering::Relaxed))
            .collect()
    };
    map.insert("mainBounds".into(), json!(take(&MAIN_BOUNDS)));
    eprintln!("posprof {}", Value::Object(map));
}

/// `TSUME_PROFILE_PERIOD=<seconds>` (measurement of long records): a thread
/// that prints the counters so far, without resetting them, as one
/// `posprof-partial {json}` line every period.
pub fn start_periodic() {
    let Some(period) = crate::diagnostics::profile_period().filter(|_| on()) else {
        return;
    };
    std::thread::Builder::new()
        .name("posprof-periodic".into())
        .spawn(move || {
            let started = Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(period));
                let mut map = Map::new();
                map.insert(
                    "elapsedNs".into(),
                    json!(started.elapsed().as_nanos() as u64),
                );
                for (key, counter) in KEYS.iter().zip(COUNTERS.iter()) {
                    map.insert((*key).into(), json!(counter.load(Ordering::Relaxed)));
                }
                eprintln!("posprof-partial {}", Value::Object(map));
            }
        })
        .expect("spawning the profile thread");
}
