//! Order-preserving, memory-bounded parallel driver for the JSONL stages.
//!
//! Every stdin/stdout command that maps records (`validate`, `solve`, `reduce`,
//! `reduce-accepted`, `probe`, `distill`, `extend`, ...) has the same shape:
//! read one record per line, do expensive work on it (on one thread, or on
//! `--prober-threads` threads), write the results in **input order**.  The obvious implementation — read a
//! fixed batch, `par_iter()` it, join, read the next batch — has a barrier at
//! every batch boundary, and one expensive record inside a batch leaves 31 of 32
//! cores idle until it finishes.  That was measured: `distill` over 200 mined
//! records used **33.1 %** of this machine and spent 29 % of its wall clock on
//! exactly one thread (`docs/benchmarks/compute-utilization-2026-09-13.md` §B).
//!
//! This module removes the barrier without giving up determinism:
//!
//! * **no join between records.**  `threads` workers pull from a shared queue the
//!   reader keeps topped up, so a worker never waits for a peer.
//! * **a reorder buffer.**  Results are stored by input ordinal and the reader
//!   thread writes them strictly in ordinal order, so stdout is byte-identical to
//!   a single-threaded run (and to the old batched driver).
//! * **bounded memory.**  At most [`Schedule::in_flight`] records are *outstanding*
//!   (read but not yet written).  That single cap bounds the input queue, the
//!   running work and the reorder buffer together: it is not enough to cap the
//!   buffer alone, because one slow early record would otherwise let the reader
//!   run to EOF and hold every later result in memory.
//! * **per-record work is owned by one worker**, which is what keeps a record's
//!   output independent of how many threads ran or which worker got it
//!   (`Solver` clears its transposition table per solve, so worker reuse cannot
//!   leak state between records either).
//!
//! The window is the one real trade-off: a record that outlives the next
//! `in_flight` records stalls the reader (the alternative is unbounded memory),
//! so the tail of a *window* can still narrow to one thread.  `in_flight`
//! defaults to [`DEFAULT_IN_FLIGHT_PER_THREAD`] × threads, i.e. 1 024 records on a
//! 32-thread machine: eight times the old barrier, a few MiB of buffered JSON, and
//! measured to be the difference between 66 % and 80 % utilization on a
//! 1 000-source `distill` run.
//!
//! (Until 2026-10-08 a worker could also ask for a second, full-budget run of a
//! record, for `distill --first-pass-nodes`; that two-pass schedule measured
//! slower than one pass and had no caller, and was removed,
//! `docs/experiments/rule-f-bound/REFACTOR.md`.)

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Condvar, Mutex, MutexGuard};

use anyhow::Result;

/// Outstanding records per worker thread, when the caller does not say.
///
/// 32 × threads — 1 024 records on this machine — is what it took to keep the
/// machine at 80 % on a 1 000-source `distill` corpus: at 8 × threads a single
/// 20-second source stalled the reader often enough to cost 9 % of the wall clock
/// (`docs/benchmarks/compute-utilization-2026-09-13.md` §B4).  The cost is
/// buffered output, a few MiB of JSON for the stages that do not build graphs.
pub const DEFAULT_IN_FLIGHT_PER_THREAD: usize = 32;

/// One input record handed to a worker.
#[derive(Clone, Debug)]
pub struct Job {
    /// 0-based index of this record among the non-empty input lines.
    pub ordinal: usize,
    line: String,
}

impl Job {
    /// The raw input line, without its newline.
    pub fn line(&self) -> &str {
        &self.line
    }
}

/// Worker count and window size.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Worker threads.  Always at least 1.
    pub threads: usize,
    /// Records read but not yet written.  Always at least `threads`.
    pub in_flight: usize,
}

impl Schedule {
    /// `threads` workers with a window of `in_flight` records; `0` picks the
    /// default for either.
    pub fn new(threads: usize, in_flight: usize) -> Self {
        let threads = threads.max(1);
        let in_flight = if in_flight == 0 {
            threads * DEFAULT_IN_FLIGHT_PER_THREAD
        } else {
            in_flight
        }
        .max(threads);
        Self { threads, in_flight }
    }
}

/// What one run moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Non-empty input lines read.
    pub inputs: usize,
    /// Output lines written.
    pub outputs: usize,
}

/// Completed work, keyed by input ordinal: the reorder buffer.
type Ready = BTreeMap<usize, Result<Vec<String>>>;

struct State {
    /// Jobs not yet taken, in input order.
    pending: VecDeque<Job>,
    ready: Ready,
    running: usize,
    input_done: bool,
    stop: bool,
}

/// What the reader thread should do next.  Decided while holding the lock and
/// acted on after releasing it, so neither stdin nor stdout is read or written
/// under the mutex.
enum Step {
    /// Write these completed records, in ordinal order.
    Write(Vec<Result<Vec<String>>>),
    /// Read one more input line.
    Read,
    /// Nothing to do until a worker reports.
    Wait,
    /// Every record has been written.
    Finish,
}

/// Streams `input` through `work` on `schedule.threads` workers and passes the
/// results to `emit` in input order.
///
/// `init` builds the per-worker state (a [`crate::solver::Solver`], typically)
/// once per thread.  `work` returns a record's output lines, in order, and must
/// be deterministic in the record (not in the worker's state) for the output to
/// be reproducible; `emit` sees every output line of record *n* before any line
/// of record *n + 1*.
///
/// Errors behave like the batched driver they replace: the lines of the records
/// before the failing one are written, then the failure propagates.  Records that
/// are already running finish — a solve cannot be interrupted — but their results
/// are discarded and nothing further is read.
pub fn run_ordered<T, I, W, E>(
    input: I,
    schedule: &Schedule,
    init: impl Fn() -> T + Sync,
    work: W,
    mut emit: E,
) -> Result<StreamStats>
where
    T: Send,
    I: Iterator<Item = std::io::Result<String>>,
    W: Fn(&mut T, &Job) -> Result<Vec<String>> + Sync,
    E: FnMut(&str) -> Result<()>,
{
    let state = Mutex::new(State {
        pending: VecDeque::new(),
        ready: Ready::new(),
        running: 0,
        input_done: false,
        stop: false,
    });
    let progress = Condvar::new();
    let mut stats = StreamStats::default();
    let mut input = input;

    let outcome = std::thread::scope(|scope| -> Result<()> {
        for _ in 0..schedule.threads {
            let state = &state;
            let progress = &progress;
            let init = &init;
            let work = &work;
            scope.spawn(move || worker(state, progress, init, work));
        }

        // The reader thread is also the writer: it reads a line whenever the
        // window has room, and otherwise drains the reorder buffer.
        let mut next_ordinal = 0usize;
        let mut next_out = 0usize;
        let mut input_done = false;
        loop {
            let step = {
                let mut guard = lock(&state);
                let mut batch: Vec<Result<Vec<String>>> = Vec::new();
                while let Some(entry) = guard.ready.remove(&next_out) {
                    next_out += 1;
                    let failed = entry.is_err();
                    batch.push(entry);
                    // Stop at the first failure so the error is reported in input
                    // order, exactly where the batched driver reported it.
                    if failed {
                        break;
                    }
                }
                if !batch.is_empty() {
                    Step::Write(batch)
                } else if input_done && next_out == next_ordinal {
                    Step::Finish
                } else if !input_done && next_ordinal - next_out < schedule.in_flight {
                    Step::Read
                } else {
                    let _unused = progress.wait(guard);
                    Step::Wait
                }
            };
            match step {
                Step::Wait => continue,
                Step::Finish => break,
                Step::Write(batch) => {
                    for entry in batch {
                        match entry {
                            Ok(lines) => {
                                for line in lines {
                                    if let Err(error) = emit(&line) {
                                        return Err(abort(&state, &progress, error));
                                    }
                                    stats.outputs += 1;
                                }
                            }
                            Err(error) => return Err(abort(&state, &progress, error)),
                        }
                    }
                }
                Step::Read => match input.next() {
                    None => {
                        input_done = true;
                        let mut guard = lock(&state);
                        guard.input_done = true;
                        drop(guard);
                        progress.notify_all();
                    }
                    Some(Err(error)) => return Err(abort(&state, &progress, error.into())),
                    Some(Ok(line)) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let job = Job {
                            ordinal: next_ordinal,
                            line,
                        };
                        next_ordinal += 1;
                        stats.inputs += 1;
                        let mut guard = lock(&state);
                        guard.pending.push_back(job);
                        drop(guard);
                        progress.notify_one();
                    }
                },
            }
        }
        Ok(())
    });
    outcome?;
    Ok(stats)
}

/// Poisoning can only happen if a worker panicked, in which case the panic is
/// already propagating out of the scope; the driver takes the guard either way so
/// it cannot deadlock behind a poisoned mutex.
fn lock<'a>(state: &'a Mutex<State>) -> MutexGuard<'a, State> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Tells the workers to stop and hands the error back, so `thread::scope` joins
/// instead of waiting for a queue nobody will drain.
fn abort(state: &Mutex<State>, progress: &Condvar, error: anyhow::Error) -> anyhow::Error {
    let mut guard = lock(state);
    guard.stop = true;
    drop(guard);
    progress.notify_all();
    error
}

fn worker<T, W>(state: &Mutex<State>, progress: &Condvar, init: &(impl Fn() -> T + Sync), work: &W)
where
    W: Fn(&mut T, &Job) -> Result<Vec<String>> + Sync,
{
    let mut local = init();
    loop {
        let job = {
            let mut guard = lock(state);
            loop {
                if guard.stop {
                    return;
                }
                if let Some(job) = guard.pending.pop_front() {
                    guard.running += 1;
                    break Some(job);
                }
                if guard.input_done && guard.running == 0 {
                    break None;
                }
                guard = match progress.wait(guard) {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
        };
        let Some(job) = job else {
            // Nothing can arrive any more: wake the peers that are still waiting
            // on the same condition and leave.
            progress.notify_all();
            return;
        };
        let outcome = work(&mut local, &job);
        let mut guard = lock(state);
        guard.running -= 1;
        guard.ready.insert(job.ordinal, outcome);
        drop(guard);
        progress.notify_all();
    }
}
