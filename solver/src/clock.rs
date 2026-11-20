//! The monotonic clock the library's measurements read (`TSUME_PROFILE`,
//! `TSUME_V3_STATS`, the records' wall time): [`std::time::Instant`] everywhere
//! but `wasm32`, where `std` has no clock and `Instant::now` panics.  There the
//! browser gate (`tools/tsume-solver-wasm`) gets a clock that never moves, so
//! every measured duration is zero.  Measurements only: no verdict, budget or
//! node count reads the clock (budgets are node counts, `docs/generation.md`
//! §4.3).
//!
//! Scope: the gate / reduce / profiling paths the browser gate calls.  `mine`
//! still reads `std::time::Instant` for its `time_limit`; it is not built into
//! the browser gate, and a stopped clock there would make the limit never
//! expire, so it is left alone on purpose.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use stopped::Instant;

#[cfg(target_arch = "wasm32")]
mod stopped {
    use std::time::Duration;

    /// A clock that never moves (`wasm32` has no clock in `std`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant;

    impl Instant {
        pub fn now() -> Self {
            Instant
        }

        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        pub fn duration_since(&self, _earlier: Instant) -> Duration {
            Duration::ZERO
        }
    }
}
