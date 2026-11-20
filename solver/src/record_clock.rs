//! When and at what cost a catalog record was judged and reduced
//! (`--emit-record`, owner 2026-10-27: the records go to
//! `tools/catalog/ingest.py` as written, with the times on them;
//! `docs/experiments/emit-record-fields/README.md`).
//!
//! A [`Meter`] measures one judgement (the publishing gate's solve of one
//! diagram, [`crate::stages::solve_stage_reported`]) or one reduction (rule D
//! on one input, every rung of `--reduce-budget-ladder` included,
//! [`crate::stages::reduce_stage`] and [`crate::stages::reduce_accepted`]) and
//! gives its [`Cost`]: the time it finished, its wall time and its CPU time.
//!
//! The CPU time is the worker thread's own ([`crate::cpu::thread_cpu_seconds`])
//! plus the parallel prober's helper threads' (`--prober-threads` > 1,
//! [`crate::reduce::prober_helper_cpu_seconds`]; they serve this worker only,
//! so their time over the span is this record's, idle spinning between
//! questions included) plus the `reduce --jobs` workers' ([`charge`]: each
//! worker thread's whole CPU time, added when the pool closes).  `None` where a
//! clock cannot be read (the helpers' on macOS).
//!
//! **The clock is the real one by default.**  `TSUME_RECORD_CLOCK=fixed`
//! (given explicitly, never implied by `SOURCE_DATE_EPOCH`; composition
//! 2026-10-27 19:2x) fixes every time at `SOURCE_DATE_EPOCH` (or the epoch),
//! every wall and CPU time at 0 and the host at `null`, so a record's bytes
//! are a function of its input again: the tests, golden files and difference
//! checks that compare records set it.  Nothing else in the record depends on
//! the clock, and the main output never does.

use crate::clock::Instant;
use std::cell::Cell;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// The environment variable that fixes the clock.
pub const CLOCK_ENV: &str = "TSUME_RECORD_CLOCK";

/// The commit the binary was built from, `<7 hex>` or `<7 hex>+dirty` (the
/// crate's files differ from the commit), or `unknown` without git
/// (`build.rs`; `TSUME_SOLVER_BUILD` at build time overrides it).
pub const SOLVER_BUILD: &str = env!("TSUME_SOLVER_BUILD");

/// How [`Meter`] tells the time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockMode {
    /// The real clock (the default).
    Real,
    /// `TSUME_RECORD_CLOCK=fixed`.
    Fixed,
}

/// Reads a `TSUME_RECORD_CLOCK` value: unset, empty or `real` is the real
/// clock, `fixed` the fixed one; anything else is refused (a misspelt
/// `fixed` must not silently write real times into a golden file).
pub fn parse_clock(value: Option<&str>) -> Result<ClockMode, String> {
    match value.map(str::trim) {
        None | Some("") | Some("real") => Ok(ClockMode::Real),
        Some("fixed") => Ok(ClockMode::Fixed),
        Some(other) => Err(format!(
            "{CLOCK_ENV} must be `fixed` or `real` (or unset), got {other}"
        )),
    }
}

/// The process's clock mode, read from the environment once.
pub fn mode() -> ClockMode {
    static MODE: OnceLock<ClockMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var(CLOCK_ENV).ok();
        parse_clock(value.as_deref()).unwrap_or(ClockMode::Real)
    })
}

/// Refuses a malformed `TSUME_RECORD_CLOCK` (the CLI calls it at start-up).
pub fn check_env() -> Result<(), String> {
    let value = std::env::var(CLOCK_ENV).ok();
    parse_clock(value.as_deref()).map(|_| ())
}

/// The fixed clock's time: `SOURCE_DATE_EPOCH` (seconds), else the epoch, in JST.
fn fixed_time() -> String {
    let epoch = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|raw| raw.trim().parse::<i64>().ok())
        .unwrap_or(0);
    crate::util::iso_8601_jst_ms(epoch.saturating_mul(1000))
}

/// Now, in JST with milliseconds (the epoch on `wasm32`, which has no clock in
/// `std`: [`crate::clock`]).
fn now() -> String {
    if cfg!(target_arch = "wasm32") {
        return crate::util::iso_8601_jst_ms(0);
    }
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as i64);
    crate::util::iso_8601_jst_ms(millis)
}

/// What one judgement or reduction took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cost {
    /// When it finished, ISO 8601 in JST (`2026-10-27T19:30:12.345+09:00`).
    pub at: String,
    /// Its wall time in milliseconds.
    pub wall_ms: u64,
    /// Its CPU time in milliseconds ([module docs](self)), `None` where not
    /// measurable.
    pub cpu_ms: Option<u64>,
}

thread_local! {
    /// CPU seconds of `reduce --jobs` worker threads, charged to the thread
    /// that ran the pool ([`charge`]).  Only ever grows; [`Meter`] reads the
    /// difference.
    static CHARGED: Cell<f64> = const { Cell::new(0.0) };
}

/// Charges `seconds` of CPU spent on other threads for this thread's work
/// (the `reduce --jobs` pool's workers, which exit before the record ends).
pub fn charge(seconds: f64) {
    if seconds > 0.0 {
        CHARGED.with(|cell| cell.set(cell.get() + seconds));
    }
}

/// CPU seconds of this thread's work so far: its own, its prober's helpers'
/// and the charged ones.
fn work_cpu_seconds() -> Option<f64> {
    let own = crate::cpu::thread_cpu_seconds()?;
    let helpers = crate::reduce::prober_helper_cpu_seconds()?;
    Some(own + helpers + CHARGED.with(Cell::get))
}

/// Measures one judgement or reduction on the calling thread.
#[derive(Clone, Debug)]
pub struct Meter {
    /// `None` on the fixed clock.
    started: Option<(Instant, Option<f64>)>,
}

impl Meter {
    /// Starts measuring now.
    pub fn start() -> Self {
        Self {
            started: (mode() == ClockMode::Real).then(|| (Instant::now(), work_cpu_seconds())),
        }
    }

    /// What it took up to now.
    pub fn finish(&self) -> Cost {
        let Some((started, cpu)) = self.started else {
            return Cost {
                at: fixed_time(),
                wall_ms: 0,
                cpu_ms: Some(0),
            };
        };
        let wall_ms = started.elapsed().as_millis() as u64;
        let cpu_ms = match (cpu, work_cpu_seconds()) {
            (Some(before), Some(after)) => {
                Some(((after - before).max(0.0) * 1000.0).round() as u64)
            }
            _ => None,
        };
        Cost {
            at: now(),
            wall_ms,
            cpu_ms,
        }
    }
}

/// The machine's name (`producer.host`): `COMPUTERNAME` on Windows,
/// `gethostname` elsewhere; `None` on the fixed clock or when it cannot be read.
pub fn host() -> Option<String> {
    static HOST: OnceLock<Option<String>> = OnceLock::new();
    if mode() == ClockMode::Fixed {
        return None;
    }
    HOST.get_or_init(platform_host).clone()
}

#[cfg(windows)]
fn platform_host() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
}

#[cfg(unix)]
fn platform_host() -> Option<String> {
    use std::os::raw::{c_char, c_int};
    extern "C" {
        fn gethostname(name: *mut c_char, len: usize) -> c_int;
    }
    let mut buffer = [0u8; 256];
    // SAFETY: `buffer` is live and writable for its whole length, which is
    // what `len` says; `gethostname` writes at most that many bytes.
    let ok = unsafe { gethostname(buffer.as_mut_ptr().cast::<c_char>(), buffer.len()) };
    if ok != 0 {
        return None;
    }
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    let name = String::from_utf8_lossy(&buffer[..end]).trim().to_string();
    (!name.is_empty()).then_some(name)
}

#[cfg(not(any(windows, unix)))]
fn platform_host() -> Option<String> {
    None
}

/// The default `--record-solver-id`: `S-<SOLVER_BUILD>` when the build knows
/// its commit (`S-1a2b3c4`, `S-1a2b3c4+dirty`), `None` for `unknown`.
pub fn default_solver_id() -> Option<String> {
    let id = format!("S-{SOLVER_BUILD}");
    crate::catalog_record::valid_solver_id(&id).then_some(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_variable_is_read_strictly() {
        assert_eq!(parse_clock(None), Ok(ClockMode::Real));
        assert_eq!(parse_clock(Some("")), Ok(ClockMode::Real));
        assert_eq!(parse_clock(Some("real")), Ok(ClockMode::Real));
        assert_eq!(parse_clock(Some("fixed")), Ok(ClockMode::Fixed));
        assert!(parse_clock(Some("1789257600")).is_err());
        assert!(parse_clock(Some("Fixed")).is_err());
    }

    #[test]
    fn the_build_is_a_commit_or_unknown() {
        let (commit, rest) = SOLVER_BUILD.split_at(SOLVER_BUILD.len().min(7));
        if SOLVER_BUILD == "unknown" {
            assert_eq!(default_solver_id(), None);
        } else {
            assert!(
                commit.bytes().all(|b| b.is_ascii_hexdigit()),
                "{SOLVER_BUILD}"
            );
            assert!(rest.is_empty() || rest == "+dirty", "{SOLVER_BUILD}");
            assert_eq!(default_solver_id(), Some(format!("S-{SOLVER_BUILD}")));
        }
    }

    #[test]
    fn a_real_meter_reports_a_jst_time() {
        if mode() == ClockMode::Fixed {
            return;
        }
        let cost = Meter::start().finish();
        assert!(cost.at.ends_with("+09:00"), "{}", cost.at);
        assert_eq!(cost.at.len(), "2026-10-27T19:30:12.345+09:00".len());
    }
}
