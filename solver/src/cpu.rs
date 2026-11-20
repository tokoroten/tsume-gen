//! Measured process and thread CPU time, for the `--stats-out` counters and
//! the catalog records' `cpuMs` ([`crate::record_clock`]).
//!
//! `--stats-out` used to report `cpuSeconds = wall × threads`, which is the CPU
//! the run was *allocated*, not the CPU it *consumed*: at the 33.1 % utilization
//! measured in `docs/benchmarks/compute-utilization-2026-09-13.md` §B the two
//! differ by 3×, and every "per CPU-minute" figure derived from the allocated
//! number flatters an under-filled machine.  Both are reported now —
//! `cpuSeconds` measured here, `allocatedCoreSeconds` from the clock — so the
//! ratio of the two *is* the utilization.
//!
//! This is user + kernel time of the whole process, every thread included, taken
//! from `GetProcessTimes` on Windows and `getrusage(RUSAGE_SELF)` on Unix.  Both
//! are declared by hand rather than pulled from `windows-sys` or `libc`: the CLI
//! links neither (see the `clap` note in `Cargo.toml` — `windows-sys` needs
//! `dlltool` on the `x86_64-pc-windows-gnu` host this repository builds on), and
//! the two calls used here are stable, tiny and read-only.
//!
//! Nothing in the pipeline *depends* on these numbers: they are reported, never
//! compared against a budget.  Budgets are counted in nodes and validations so
//! runs reproduce (`docs/generation.md` §4.3), and a platform that cannot report
//! CPU time simply yields `None`.

/// User + kernel CPU seconds consumed by this process so far, all threads
/// included, or `None` where the platform cannot report it.
///
/// Take two samples and subtract to measure one phase of a run.
pub fn process_cpu_seconds() -> Option<f64> {
    platform::process_cpu_seconds()
}

/// Seconds of CPU consumed between two [`process_cpu_seconds`] samples, clamped
/// at zero so a missing sample never produces a negative report.
pub fn cpu_seconds_since(before: Option<f64>) -> Option<f64> {
    let (before, now) = (before?, process_cpu_seconds()?);
    Some((now - before).max(0.0))
}

/// User + kernel CPU seconds the **calling thread** consumed since it
/// started, or `None` where the platform cannot report it (the catalog
/// records' `cpuMs`, [`crate::record_clock`]).  `GetThreadTimes` on Windows
/// (counted in scheduler ticks, about 15.6 ms), `clock_gettime` with the
/// thread CPU clock on Linux and macOS.
pub fn thread_cpu_seconds() -> Option<f64> {
    platform::thread_cpu_seconds()
}

/// User + kernel CPU seconds the thread of `handle` (a live thread, not yet
/// joined) consumed since it started: the parallel prober's helper threads
/// ([`crate::cdfpn::ClassicalProber::helper_cpu_seconds`]).  `GetThreadTimes`
/// on Windows, `pthread_getcpuclockid` on Linux; `None` elsewhere (macOS has
/// no `pthread_getcpuclockid`).
pub fn joined_thread_cpu_seconds<T>(handle: &std::thread::JoinHandle<T>) -> Option<f64> {
    platform::joined_thread_cpu_seconds(handle)
}

#[cfg(windows)]
mod platform {
    /// `FILETIME`: 100-nanosecond ticks, split in two halves because the struct
    /// is only 4-byte aligned.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        fn seconds(self) -> f64 {
            ((u64::from(self.high) << 32) | u64::from(self.low)) as f64 * 1e-7
        }
    }

    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessTimes(
            process: *mut std::ffi::c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetThreadTimes(
            thread: *mut std::ffi::c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }

    /// Kernel + user seconds of the thread behind `thread` (the handle of a
    /// live `JoinHandle`, or the current-thread pseudo-handle).
    fn thread_times(thread: *mut std::ffi::c_void) -> Option<f64> {
        let mut creation = FileTime::default();
        let mut exit = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        // SAFETY: `thread` is the current-thread pseudo-handle or the handle of
        // a live `JoinHandle` (full access), valid for the call, and the four
        // out parameters are live, correctly aligned `FILETIME`s.
        let ok =
            unsafe { GetThreadTimes(thread, &mut creation, &mut exit, &mut kernel, &mut user) };
        (ok != 0).then(|| kernel.seconds() + user.seconds())
    }

    pub fn thread_cpu_seconds() -> Option<f64> {
        // SAFETY: `GetCurrentThread` only returns a pseudo-handle, which needs
        // no closing.
        thread_times(unsafe { GetCurrentThread() })
    }

    pub fn joined_thread_cpu_seconds<T>(handle: &std::thread::JoinHandle<T>) -> Option<f64> {
        use std::os::windows::io::AsRawHandle;
        thread_times(handle.as_raw_handle())
    }

    pub fn process_cpu_seconds() -> Option<f64> {
        let mut creation = FileTime::default();
        let mut exit = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        // SAFETY: the pseudo-handle from `GetCurrentProcess` needs no closing and
        // always carries PROCESS_QUERY_INFORMATION rights, and the four out
        // parameters are live, correctly aligned `FILETIME`s.
        let ok = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        };
        (ok != 0).then(|| kernel.seconds() + user.seconds())
    }
}

#[cfg(unix)]
mod platform {
    use std::os::raw::{c_int, c_long};

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TimeVal {
        sec: c_long,
        usec: c_long,
    }

    /// `struct rusage`, of which only the first two members are read.  The tail is
    /// padding: 16 words is at least the 14 `long`s Linux and macOS both place
    /// after the timevals, so the kernel never writes past the buffer.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RUsage {
        utime: TimeVal,
        stime: TimeVal,
        tail: [c_long; 16],
    }

    const RUSAGE_SELF: c_int = 0;

    extern "C" {
        fn getrusage(who: c_int, usage: *mut RUsage) -> c_int;
    }

    fn seconds(value: TimeVal) -> f64 {
        value.sec as f64 + value.usec as f64 * 1e-6
    }

    /// `struct timespec` (`time_t` is a `long` on every target built here).
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TimeSpec {
        sec: c_long,
        nsec: c_long,
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    extern "C" {
        fn clock_gettime(clock: c_int, now: *mut TimeSpec) -> c_int;
    }

    /// `CLOCK_THREAD_CPUTIME_ID`.
    #[cfg(target_os = "linux")]
    const CLOCK_THREAD_CPUTIME_ID: c_int = 3;
    /// `CLOCK_THREAD_CPUTIME_ID`.
    #[cfg(target_os = "macos")]
    const CLOCK_THREAD_CPUTIME_ID: c_int = 16;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn clock_seconds(clock: c_int) -> Option<f64> {
        let mut now = TimeSpec::default();
        // SAFETY: `now` is a live, correctly aligned `struct timespec`, and
        // `clock_gettime` only writes into it.
        let ok = unsafe { clock_gettime(clock, &mut now) };
        (ok == 0).then_some(now.sec as f64 + now.nsec as f64 * 1e-9)
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub fn thread_cpu_seconds() -> Option<f64> {
        clock_seconds(CLOCK_THREAD_CPUTIME_ID)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn thread_cpu_seconds() -> Option<f64> {
        None
    }

    #[cfg(target_os = "linux")]
    pub fn joined_thread_cpu_seconds<T>(handle: &std::thread::JoinHandle<T>) -> Option<f64> {
        use std::os::unix::thread::{JoinHandleExt, RawPthread};
        extern "C" {
            fn pthread_getcpuclockid(thread: RawPthread, clock: *mut c_int) -> c_int;
        }
        let mut clock: c_int = 0;
        // SAFETY: the thread of a live `JoinHandle` is neither joined nor
        // detached, so its `pthread_t` is valid; `clock` is a live `c_int`.
        let ok = unsafe { pthread_getcpuclockid(handle.as_pthread_t(), &mut clock) };
        if ok != 0 {
            return None;
        }
        clock_seconds(clock)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn joined_thread_cpu_seconds<T>(_handle: &std::thread::JoinHandle<T>) -> Option<f64> {
        None
    }

    pub fn process_cpu_seconds() -> Option<f64> {
        let mut usage = RUsage::default();
        // SAFETY: `usage` is a live, correctly aligned buffer at least as large as
        // the platform's `struct rusage`, and `getrusage` only writes into it.
        let ok = unsafe { getrusage(RUSAGE_SELF, &mut usage) };
        (ok == 0).then(|| seconds(usage.utime) + seconds(usage.stime))
    }
}

#[cfg(not(any(windows, unix)))]
mod platform {
    pub fn process_cpu_seconds() -> Option<f64> {
        None
    }

    pub fn thread_cpu_seconds() -> Option<f64> {
        None
    }

    pub fn joined_thread_cpu_seconds<T>(_handle: &std::thread::JoinHandle<T>) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU time must exist and must not run backwards.  Deliberately not an
    /// assertion about *how much* CPU a spin loop costs: that is a benchmark, and
    /// a loaded machine would make it flaky.
    #[test]
    fn cpu_time_is_monotonic() {
        let before = process_cpu_seconds().expect("this platform reports CPU time");
        let mut sink = 0u64;
        for index in 0..5_000_000u64 {
            sink = sink.wrapping_add(index * index);
        }
        assert!(sink > 0);
        let after = process_cpu_seconds().expect("this platform reports CPU time");
        assert!(after >= before, "{after} < {before}");
        assert!(cpu_seconds_since(Some(before)).expect("delta") >= 0.0);
        assert_eq!(cpu_seconds_since(None), None);
    }

    /// The thread clock exists and never runs backwards where the solver is
    /// built (Windows, Linux, macOS); a live thread's clock is readable
    /// through its handle on Windows and Linux.
    #[test]
    fn thread_cpu_time_is_monotonic() {
        if !cfg!(any(windows, target_os = "linux", target_os = "macos")) {
            return;
        }
        let before = thread_cpu_seconds().expect("this platform reports thread CPU time");
        let mut sink = 0u64;
        for index in 0..5_000_000u64 {
            sink = sink.wrapping_add(index * index);
        }
        assert!(sink > 0);
        let after = thread_cpu_seconds().expect("this platform reports thread CPU time");
        assert!(after >= before, "{after} < {before}");
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        let other = joined_thread_cpu_seconds(&handle);
        if cfg!(any(windows, target_os = "linux")) {
            assert!(other.expect("a live thread's CPU time") >= 0.0);
        }
        drop(tx);
        handle.join().expect("the thread ends");
    }
}
