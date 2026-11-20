//! The measurement switches: environment variables, each read once, that make
//! the solver print counters or timings to stderr.  None of them changes an
//! output byte (stdout, the side files, the records), and none tags catalog
//! records `+x`; the switches that do change outputs are read in `main.rs`
//! and noted there as experimental.
//!
//! | variable | what it prints |
//! | --- | --- |
//! | `TSUME_PROFILE` (non-empty, not `0`) | per-phase wall times ([`crate::prof`]) and the per-position counters ([`crate::posprof`]) |
//! | `TSUME_PROFILE_PERIOD=<seconds>` | with `TSUME_PROFILE`, the position counters so far every period |
//! | `TSUME_PDFPN_PROF` | the time split and counters of a parallel prober ([`crate::cdfpn`]) and its shared tables ([`crate::pardfpn`]) |
//! | `TSUME_PDFPN_TRACE` | one line per big question of a prober |
//! | `TSUME_V3_STATS` | the reducer's prefilter and ladder counters at exit ([`crate::reduce::PREFILTER_STATS`]) |
//! | `TSUME_COMPACT_STATS=1` (or `2`) | the compact table's size after every compact search |

use std::sync::LazyLock;

fn set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

static PROFILE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("TSUME_PROFILE")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
});
static PROFILE_PERIOD: LazyLock<Option<u64>> = LazyLock::new(|| {
    std::env::var("TSUME_PROFILE_PERIOD")
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|&secs| secs > 0)
});
static PDFPN_PROF: LazyLock<bool> = LazyLock::new(|| set("TSUME_PDFPN_PROF"));
static PDFPN_TRACE: LazyLock<bool> = LazyLock::new(|| set("TSUME_PDFPN_TRACE"));
static V3_STATS: LazyLock<bool> = LazyLock::new(|| set("TSUME_V3_STATS"));
static COMPACT_STATS: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("TSUME_COMPACT_STATS").is_some_and(|v| v == "1" || v == "2"));

/// `TSUME_PROFILE`.
#[inline]
pub fn profile() -> bool {
    *PROFILE
}

/// `TSUME_PROFILE_PERIOD`, in seconds (whether or not `TSUME_PROFILE` is on).
pub fn profile_period() -> Option<u64> {
    *PROFILE_PERIOD
}

/// `TSUME_PDFPN_PROF`.
pub fn pdfpn_prof() -> bool {
    *PDFPN_PROF
}

/// `TSUME_PDFPN_TRACE`.
pub fn pdfpn_trace() -> bool {
    *PDFPN_TRACE
}

/// `TSUME_V3_STATS`.
pub fn v3_stats() -> bool {
    *V3_STATS
}

/// `TSUME_COMPACT_STATS`.
pub fn compact_stats() -> bool {
    *COMPACT_STATS
}
