//! Global allocator: large allocations on 2 MiB pages, on by default.
//!
//! Since solver 1.5.0 (owner 2026-11-05, `docs/experiments/solver-scaling/README.md`
//! §3 and §16: about 10 % faster on one thread) every allocation of at least
//! [`THRESHOLD`] bytes is served from `VirtualAlloc` with `MEM_LARGE_PAGES`
//! (Windows; the user needs "Lock pages in memory", `SeLockMemoryPrivilege`,
//! which the process enables on its token) and falls back to ordinary
//! `VirtualAlloc` pages when a large-page allocation fails.  Without the
//! privilege the whole path is off and every allocation goes to `System`, as
//! before, silently.  Smaller allocations always go to `System`.
//!
//! `TSUME_LARGE_PAGES=0` (read once, at the first large allocation) turns the
//! path off; unset or any other value leaves it on (`=1` was the opt-in before
//! 1.5.0 and still means on).  Large pages only change where the bytes live,
//! never what is searched: the output is byte-identical either way
//! (`tests/large_pages.rs`).
//!
//! Linux: `mmap` aligned to 2 MiB + `madvise(MADV_HUGEPAGE)` (transparent huge
//! pages; the kernel may still serve small pages), only when THP takes advice
//! (`[always]` or `[madvise]`; otherwise the path is off).  Other unix (macOS): off,
//! every allocation goes to `System` (no huge-page advice to give).
//!
//! Memory watchers: on Windows large pages are committed and locked at the
//! allocation, count in the process's private bytes (commit) and in the
//! system's committed bytes and used physical memory, but **not in its working
//! set** (`tests` below pin this).  A guard that sizes or kills a solver by
//! its working set does not see them.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Allocations at least this large are page-backed (and so freed by size).
pub const THRESHOLD: usize = 8 << 20;
const LARGE: usize = 2 << 20;
/// The largest alignment the page-backed path guarantees: on Windows the
/// ordinary `VirtualAlloc` fallback starts on the 64 KiB allocation
/// granularity only; the unix path aligns every mapping to [`LARGE`].
#[cfg(windows)]
const MAX_ALIGN: usize = 64 << 10;
#[cfg(not(windows))]
const MAX_ALIGN: usize = LARGE;

/// Whether the page-backed path is on: decided once, by the first large
/// allocation (every thread that races it waits for the one answer), and never
/// changed, so `dealloc` routes each pointer as `alloc` did.
static MODE: OnceLock<bool> = OnceLock::new();
/// Bytes served on large pages (Windows) or on 2 MiB-aligned, huge-page
/// advised mappings (unix).
pub static LARGE_BYTES: AtomicU64 = AtomicU64::new(0);
/// Bytes whose `madvise(MADV_HUGEPAGE)` the kernel accepted (Linux only).
#[cfg(target_os = "linux")]
pub static ADVISED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Bytes served by the ordinary-page fallback (Windows only: the unix path
/// never falls back).
#[cfg(windows)]
pub static FALLBACK_BYTES: AtomicU64 = AtomicU64::new(0);

pub struct Alloc;

/// Whether `TSUME_LARGE_PAGES` (its value, or `None` when unset) asks for
/// the page-backed path: everything but `0` does (on by default since 1.5.0).
fn wanted(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|value| value != "0")
}

/// The mode [`on`] decides: asked for by the variable, and `enable` (the
/// platform's check: the Windows privilege, Linux THP) says yes.  `enable` is
/// not called when the variable says `0`.
fn decide(value: Option<&std::ffi::OsStr>, enable: impl FnOnce() -> bool) -> bool {
    wanted(value) && enable()
}

fn on() -> bool {
    *MODE.get_or_init(|| {
        // `std::env::var_os` allocates only small strings (below THRESHOLD),
        // which never reach this function.
        decide(
            std::env::var_os("TSUME_LARGE_PAGES").as_deref(),
            sys::enable,
        )
    })
}

/// Whether the page-backed path is on: `None` until the first large
/// allocation has decided (`report` prints it under `TSUME_V3_STATS`).
pub fn mode() -> Option<bool> {
    MODE.get().copied()
}

/// The line `report` prints under `TSUME_V3_STATS`: whether the path is on
/// (`undecided` when nothing reached [`THRESHOLD`]), the bytes on large pages,
/// the bytes of the ordinary-page fallback (Windows; 0 elsewhere) and the
/// bytes whose `MADV_HUGEPAGE` the kernel accepted (Linux; 0 elsewhere).
pub fn summary() -> String {
    let mode = match mode() {
        Some(true) => "on",
        Some(false) => "off",
        None => "undecided",
    };
    #[cfg(windows)]
    let fallback = FALLBACK_BYTES.load(Ordering::Relaxed);
    #[cfg(not(windows))]
    let fallback = 0u64;
    #[cfg(target_os = "linux")]
    let advised = ADVISED_BYTES.load(Ordering::Relaxed);
    #[cfg(not(target_os = "linux"))]
    let advised = 0u64;
    format!(
        "large-pages mode={mode} bytes={} fallback={fallback} advised={advised}",
        LARGE_BYTES.load(Ordering::Relaxed)
    )
}

/// Whether a block of `size` bytes aligned to `align` goes to the pages (when
/// the path is on) rather than to `System`.
fn large(size: usize, align: usize) -> bool {
    size >= THRESHOLD && align <= MAX_ALIGN
}

/// Whether a block of `layout` was (or would be) page-backed: only after
/// [`on`] has decided, which every large allocation does first.
fn page_backed(size: usize, align: usize) -> bool {
    large(size, align) && MODE.get() == Some(&true)
}

fn round(size: usize) -> usize {
    (size + LARGE - 1) & !(LARGE - 1)
}

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if large(layout.size(), layout.align()) && on() {
            return sys::alloc(round(layout.size()));
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if large(layout.size(), layout.align()) && on() {
            // Fresh pages are zero.
            return sys::alloc(round(layout.size()));
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if page_backed(layout.size(), layout.align()) {
            sys::free(ptr, round(layout.size()));
            return;
        }
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let big_old = page_backed(layout.size(), layout.align());
        let big_new = large(new_size, layout.align()) && on();
        if !big_old && !big_new {
            return System.realloc(ptr, layout, new_size);
        }
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new = self.alloc(new_layout);
        if !new.is_null() {
            std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        new
    }
}

#[cfg(windows)]
mod sys {
    use super::{Ordering, FALLBACK_BYTES, LARGE_BYTES};
    use std::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const MEM_LARGE_PAGES: u32 = 0x2000_0000;
    const PAGE_READWRITE: u32 = 0x04;
    const TOKEN_ADJUST_PRIVILEGES: u32 = 0x20;
    const TOKEN_QUERY: u32 = 0x08;
    const SE_PRIVILEGE_ENABLED: u32 = 0x2;

    #[repr(C)]
    struct Luid {
        low: u32,
        high: i32,
    }
    #[repr(C)]
    struct TokenPrivileges {
        count: u32,
        luid: Luid,
        attributes: u32,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetLargePageMinimum() -> usize;
        fn CloseHandle(h: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }
    #[link(name = "advapi32")]
    extern "system" {
        fn OpenProcessToken(p: *mut c_void, access: u32, token: *mut *mut c_void) -> i32;
        fn LookupPrivilegeValueW(sys: *const u16, name: *const u16, luid: *mut Luid) -> i32;
        fn AdjustTokenPrivileges(
            token: *mut c_void,
            disable_all: i32,
            new: *const TokenPrivileges,
            len: u32,
            prev: *mut c_void,
            ret: *mut u32,
        ) -> i32;
    }

    pub fn enable() -> bool {
        enable_with("SeLockMemoryPrivilege")
    }

    /// [`enable`] with the privilege named (tests ask for one the user lacks).
    pub fn enable_with(privilege: &str) -> bool {
        // SAFETY: a plain query without arguments.
        if unsafe { GetLargePageMinimum() } == 0 {
            return false;
        }
        enable_privilege(privilege)
    }

    /// Enables the privilege `name` on the process token: false when the
    /// token does not hold it (`ERROR_NOT_ALL_ASSIGNED`), which is what
    /// turns the page-backed path off for a user without "Lock pages in
    /// memory".
    pub fn enable_privilege(name: &str) -> bool {
        unsafe {
            let mut token = std::ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                &mut token,
            ) == 0
            {
                return false;
            }
            let name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            let mut luid = Luid { low: 0, high: 0 };
            let mut ok = LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut luid) != 0;
            if ok {
                let tp = TokenPrivileges {
                    count: 1,
                    luid,
                    attributes: SE_PRIVILEGE_ENABLED,
                };
                ok = AdjustTokenPrivileges(
                    token,
                    0,
                    &tp,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ) != 0
                    && GetLastError() == 0;
            }
            CloseHandle(token);
            ok
        }
    }

    pub unsafe fn alloc(size: usize) -> *mut u8 {
        let p = VirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
            PAGE_READWRITE,
        );
        if !p.is_null() {
            LARGE_BYTES.fetch_add(size as u64, Ordering::Relaxed);
            return p as *mut u8;
        }
        FALLBACK_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        VirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        ) as *mut u8
    }

    pub unsafe fn free(ptr: *mut u8, _size: usize) {
        VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE);
    }
}

#[cfg(unix)]
mod sys {
    use super::{Ordering, LARGE, LARGE_BYTES};
    use std::ffi::c_void;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 2;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANONYMOUS: i32 = 0x1000;
    #[cfg(target_os = "linux")]
    const MADV_HUGEPAGE: i32 = 14;

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        #[cfg(target_os = "linux")]
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    /// Linux: when the kernel takes `MADV_HUGEPAGE` advice, i.e.
    /// `/sys/kernel/mm/transparent_hugepage/enabled` says `[always]` or
    /// `[madvise]` (Codex review of 1.5.0: with THP `never`, or no sysfs in a
    /// container, the path would gain nothing and only copy on every
    /// `realloc`).  Other unix (the Mac): never, for the same reason.
    pub fn enable() -> bool {
        #[cfg(target_os = "linux")]
        {
            // A small read (below THRESHOLD), so it never re-enters `on`.
            std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
                .is_ok_and(|setting| thp_takes_advice(&setting))
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Whether a THP `enabled` setting (`always [madvise] never`) honours
    /// `MADV_HUGEPAGE`.
    #[cfg(target_os = "linux")]
    pub fn thp_takes_advice(setting: &str) -> bool {
        setting.contains("[always]") || setting.contains("[madvise]")
    }

    pub unsafe fn alloc(size: usize) -> *mut u8 {
        // Over-map by one large page so the start can be 2 MiB aligned.
        let len = size + LARGE;
        let p = mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        if p as isize == -1 {
            return std::ptr::null_mut();
        }
        let start = (p as usize + LARGE - 1) & !(LARGE - 1);
        let head = start - p as usize;
        if head > 0 {
            munmap(p, head);
        }
        let tail = LARGE - head;
        if tail > 0 {
            munmap((start + size) as *mut c_void, tail);
        }
        #[cfg(target_os = "linux")]
        if madvise(start as *mut c_void, size, MADV_HUGEPAGE) == 0 {
            super::ADVISED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        LARGE_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        start as *mut u8
    }

    pub unsafe fn free(ptr: *mut u8, size: usize) {
        munmap(ptr as *mut c_void, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::sync::Mutex;

    /// The counters are process-wide: the tests that read them run one at a time.
    /// The only other test in this (binary) crate is the CLI surface check,
    /// which allocates nothing near 8 MiB, so no other test moves them
    /// (Codex review of 1.5.0, low).
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn the_path_is_on_unless_the_variable_says_0() {
        assert!(wanted(None), "unset: on by default since 1.5.0");
        assert!(
            wanted(Some(OsStr::new("1"))),
            "the old opt-in still means on"
        );
        assert!(wanted(Some(OsStr::new(""))));
        assert!(
            !wanted(Some(OsStr::new("0"))),
            "TSUME_LARGE_PAGES=0 turns it off"
        );
    }

    /// No privilege (Windows) or no THP advice (Linux): `enable` says no and
    /// the path is off; `=0` does not even ask.
    #[test]
    fn the_mode_is_on_only_when_asked_and_enabled() {
        assert!(decide(None, || true));
        assert!(!decide(None, || false), "no privilege / no THP: off");
        assert!(!decide(Some(OsStr::new("0")), || panic!("not asked")));
        assert!(decide(Some(OsStr::new("1")), || true));
    }

    #[test]
    fn below_8_mib_goes_to_system() {
        assert_eq!(THRESHOLD, 8 << 20);
        assert!(!large(THRESHOLD - 1, 8));
        assert!(large(THRESHOLD, 8));
        assert!(large(THRESHOLD, MAX_ALIGN));
        assert!(
            !large(THRESHOLD, MAX_ALIGN * 2),
            "over-aligned blocks go to System"
        );
        // With the path off nothing is page-backed, so `dealloc` hands every
        // block back to `System`.
        if MODE.get() != Some(&true) {
            assert!(!page_backed(THRESHOLD, 8));
        }
    }

    fn fallback_bytes() -> u64 {
        #[cfg(windows)]
        return FALLBACK_BYTES.load(Ordering::Relaxed);
        #[cfg(not(windows))]
        0
    }

    /// The global allocator routes by size: a block of exactly 8 MiB comes
    /// from the pages (when the path is on), one byte less from `System`.
    #[test]
    fn the_global_allocator_routes_at_the_threshold() {
        let _serial = SERIAL.lock().unwrap();
        let before = LARGE_BYTES.load(Ordering::Relaxed) + fallback_bytes();
        let small = vec![1u8; THRESHOLD - 1];
        let after_small = LARGE_BYTES.load(Ordering::Relaxed) + fallback_bytes();
        let big = vec![1u8; THRESHOLD];
        let after_big = LARGE_BYTES.load(Ordering::Relaxed) + fallback_bytes();
        assert_eq!(
            small.iter().map(|&b| b as usize).sum::<usize>(),
            THRESHOLD - 1
        );
        assert_eq!(big.iter().map(|&b| b as usize).sum::<usize>(), THRESHOLD);
        assert_eq!(
            after_small, before,
            "8 MiB - 1 byte must not reach the pages"
        );
        match mode() {
            Some(true) => {
                assert!(after_big >= before + THRESHOLD as u64);
                assert_eq!(big.as_ptr() as usize % MAX_ALIGN, 0);
            }
            Some(false) => assert_eq!(after_big, before),
            None => panic!("an 8 MiB allocation decides the mode"),
        }
        #[cfg(target_os = "linux")]
        assert_eq!(
            mode(),
            Some(wanted(std::env::var_os("TSUME_LARGE_PAGES").as_deref()) && sys::enable()),
            "on Linux the path is on by default where THP takes advice"
        );
    }

    /// Linux: a 2 MiB-aligned mapping the kernel accepted `MADV_HUGEPAGE`
    /// for (`hg` in its `VmFlags`), zero-filled, counted in `LARGE_BYTES`.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_maps_2_mib_aligned_and_advises_huge_pages() {
        let _serial = SERIAL.lock().unwrap();
        let size = round(THRESHOLD + 1);
        assert_eq!(size, THRESHOLD + LARGE);
        let large_before = LARGE_BYTES.load(Ordering::Relaxed);
        let advised_before = ADVISED_BYTES.load(Ordering::Relaxed);
        let p = unsafe { sys::alloc(size) };
        assert!(!p.is_null());
        assert_eq!(p as usize % LARGE, 0);
        assert_eq!(
            LARGE_BYTES.load(Ordering::Relaxed) - large_before,
            size as u64
        );
        // The kernel accepted the advice (`madvise` returned 0) whenever THP
        // takes advice, which is when the path is on at all (`sys::enable`).
        if sys::enable() {
            assert_eq!(
                ADVISED_BYTES.load(Ordering::Relaxed) - advised_before,
                size as u64
            );
        } else {
            eprintln!("THP takes no advice here: the path is off (checked: advice only)");
        }
        // And it shows on the mapping, where /proc/self/smaps is readable.
        if let (true, Ok(smaps)) = (sys::enable(), std::fs::read_to_string("/proc/self/smaps")) {
            // The area holding `p` (it may have merged with a neighbouring
            // advised mapping, so it need not start at `p`).
            let holds = |line: &str| {
                let range = line.split_whitespace().next().unwrap_or("");
                let Some((low, high)) = range.split_once('-') else {
                    return false;
                };
                match (
                    usize::from_str_radix(low, 16),
                    usize::from_str_radix(high, 16),
                ) {
                    (Ok(low), Ok(high)) => low <= p as usize && (p as usize) + size <= high,
                    _ => false,
                }
            };
            let flags = smaps
                .split('\n')
                .skip_while(|line| !holds(line))
                .find(|line| line.starts_with("VmFlags:"))
                .expect("the mapping is listed in /proc/self/smaps");
            assert!(flags.split_whitespace().any(|flag| flag == "hg"), "{flags}");
        }
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(p, size);
            assert!(bytes.iter().all(|&b| b == 0), "fresh mappings are zero");
            bytes.fill(7);
            sys::free(p, size);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_thp_settings_that_take_advice() {
        assert!(sys::thp_takes_advice(
            "always [madvise] never
"
        ));
        assert!(sys::thp_takes_advice(
            "[always] madvise never
"
        ));
        assert!(!sys::thp_takes_advice(
            "always madvise [never]
"
        ));
        assert!(!sys::thp_takes_advice(""));
    }

    /// Windows, no privilege: a token without the privilege (here
    /// `SeTcbPrivilege`, which a normal user never holds) is refused, which is
    /// what turns the path off without "Lock pages in memory".
    #[cfg(windows)]
    #[test]
    fn windows_without_the_privilege_the_path_stays_off() {
        assert!(!sys::enable_with("SeTcbPrivilege"));
        assert!(!sys::enable_with("SeNoSuchPrivilege"));
    }

    /// Windows: a large-page allocation that fails (here: a size that is not a
    /// multiple of the large-page minimum) falls back to ordinary pages.
    #[cfg(windows)]
    #[test]
    fn windows_a_failed_large_page_allocation_falls_back_to_ordinary_pages() {
        let _serial = SERIAL.lock().unwrap();
        let size = THRESHOLD + (64 << 10);
        let large_before = LARGE_BYTES.load(Ordering::Relaxed);
        let fallback_before = FALLBACK_BYTES.load(Ordering::Relaxed);
        let p = unsafe { sys::alloc(size) };
        assert!(!p.is_null());
        assert_eq!(p as usize % MAX_ALIGN, 0);
        assert_eq!(LARGE_BYTES.load(Ordering::Relaxed), large_before);
        assert_eq!(
            FALLBACK_BYTES.load(Ordering::Relaxed) - fallback_before,
            size as u64
        );
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(p, size);
            assert!(bytes.iter().all(|&b| b == 0));
            bytes.fill(7);
            sys::free(p, size);
        }
    }

    /// Windows: what a memory guard sees.  Large pages show in the process's
    /// private bytes (commit) at once, and not in its working set.
    #[cfg(windows)]
    #[test]
    fn windows_large_pages_are_private_bytes_not_working_set() {
        let _serial = SERIAL.lock().unwrap();
        if !sys::enable() {
            eprintln!("skipped: no SeLockMemoryPrivilege here");
            return;
        }
        let size = 64 << 20;
        let (private_before, working_before) = counters();
        let large_before = LARGE_BYTES.load(Ordering::Relaxed);
        let p = unsafe { sys::alloc(size) };
        assert!(!p.is_null());
        if LARGE_BYTES.load(Ordering::Relaxed) == large_before {
            unsafe { sys::free(p, size) };
            eprintln!("skipped: no contiguous physical memory for large pages now");
            return;
        }
        unsafe { std::slice::from_raw_parts_mut(p, size).fill(7) };
        let (private_after, working_after) = counters();
        unsafe { sys::free(p, size) };
        assert!(
            private_after >= private_before + size as u64 * 9 / 10,
            "private {private_before} -> {private_after}"
        );
        assert!(
            working_after < working_before + size as u64 / 2,
            "working set {working_before} -> {working_after}"
        );
    }

    #[cfg(windows)]
    fn counters() -> (u64, u64) {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_working_set: usize,
            working_set: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_non_paged: usize,
            quota_non_paged: usize,
            pagefile: usize,
            peak_pagefile: usize,
            private: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                counters: *mut Counters,
                cb: u32,
            ) -> i32;
        }
        let mut c = Counters {
            cb: std::mem::size_of::<Counters>() as u32,
            ..Default::default()
        };
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        assert_ne!(ok, 0);
        (c.private as u64, c.working_set as u64)
    }
}
