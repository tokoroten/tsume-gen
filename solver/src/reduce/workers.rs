//! `reduce --jobs K`: the greedy chains on a pool of K solver threads (scoped
//! threads, each with its own [`Solver`]), with output byte-identical to the
//! sequential loop (2026-09-25, owner-approved).
//!
//! # The unit of work
//!
//! The unit of work is one greedy step's acceptance test ([`evaluate_step`]):
//! the search of one candidate diagram, on the worker's own [`Solver`] whose
//! table [`Solver::solve`] clears per call.  Its answer is a pure function of
//! (diagram, target, options, rule), so which thread computes it, and when,
//! cannot change it ([`ReduceOptions::jobs`]).
//!
//! # How the order is kept
//!
//! * **Chains** are independent: each starts from its own diagram with an
//!   order drawn up front (see [`reduce_with`]), and the caller folds the
//!   results in its own order after all of them finish, exactly as the
//!   sequential loop folds them.
//! * **Within one chain** the next candidates are checked *speculatively* on
//!   the chain's current diagram, under the chain's current rule.  Results are
//!   committed strictly in the chain's order: the first candidate that passes
//!   is applied, exactly as the sequential loop would, and every check already
//!   started past it is discarded (each commit starts a new generation; a
//!   result of an older generation is dropped on arrival).  The candidates
//!   after it are then applied to the new diagram and checked again.
//! * Nothing a check computes reaches the output except through the verdict
//!   that is committed; a discarded check leaves no trace.
//!
//! Scheduling: every chain's next non-speculative check comes first (chain
//! order), and only spare threads take speculative ones, fewest in flight
//! first.  This only affects the wall time.
//!
//! Memory: K solvers, each with its own table, so about K times the peak of
//! one sequential reduce (the caller's solver is worker 0).

use super::{
    apply_candidate, evaluate_step, Candidate, ChainState, ReduceOptions, StepLog, StepRule,
    SweepResult,
};
use crate::jsonl::RemovedPiece;
use crate::position::Position;
use crate::solver::Solver;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// One acceptance test for a worker.
struct StepJob {
    chain: usize,
    generation: u64,
    index: usize,
    trial: Position,
    rule: StepRule,
    /// The step's removal (the budget-out log).
    entry: RemovedPiece,
    /// The structural main-line test refuses the step, searched only
    /// because the test is off ([`super::structural_test`]).
    structural: bool,
}

/// A worker's answer, with the step's budget-outs; `result` is `None`
/// when the check panicked.
struct StepDone {
    chain: usize,
    generation: u64,
    index: usize,
    result: Option<(SweepResult, StepLog)>,
}

/// A submitted check of the chain's current generation (kept in order
/// index order), with its verdict once it arrives.
struct Pending {
    index: usize,
    entry: RemovedPiece,
    result: Option<(SweepResult, StepLog)>,
}

/// One greedy pass (one attempt, or one fixpoint sweep) in progress.
struct Chain<'o> {
    order: &'o [Candidate],
    state: ChainState,
    /// The target length, for the negatives' events only.
    target: u32,
    /// The next order index not yet submitted on the current diagram.
    next: usize,
    generation: u64,
    pending: VecDeque<Pending>,
    changed: bool,
}

impl<'o> Chain<'o> {
    fn new(order: &'o [Candidate], state: ChainState, target: u32) -> Self {
        Self {
            order,
            state,
            target,
            next: 0,
            generation: 0,
            pending: VecDeque::new(),
            changed: false,
        }
    }

    fn done(&self) -> bool {
        self.next >= self.order.len() && self.pending.is_empty()
    }

    /// The next candidate that still applies to the diagram, advancing
    /// `next` past those that do not (the sequential loop skips them the
    /// same way).
    fn next_trial(&mut self, attacker: u8) -> Option<(usize, Position, RemovedPiece, bool)> {
        while self.next < self.order.len() {
            let index = self.next;
            self.next += 1;
            if let Some((trial, entry)) =
                apply_candidate(&self.state.board, self.order[index], attacker)
            {
                // A step the structural main-line test refuses is skipped
                // like one that does not apply: the sequential loop moves
                // past both without committing anything.
                let structural =
                    super::structural_test(&self.state.board, &trial, &self.state.rule);
                if structural == Some(true) {
                    continue;
                }
                return Some((index, trial, entry, structural.is_some()));
            }
        }
        None
    }

    /// Commits every verdict at the front of the queue, in order.
    fn resolve(&mut self, pool: &mut Pool) {
        while self
            .pending
            .front()
            .is_some_and(|pending| pending.result.is_some())
        {
            let pending = self.pending.pop_front().expect("a front entry");
            // A structural step's entries come marked
            // ([`super::evaluate_step`]).
            let (swept, log) = pending.result.expect("a verdict");
            // Exactly the steps the sequential pass consumes.
            super::absorb_budget_outs(log);
            if super::stopped() {
                // The early stop: the record is a budget-out; nothing
                // more is committed ([`Pool::run`] returns).
                return;
            }
            if !self.state.rule.accepts_step(&swept.check) {
                // In the chain's order, exactly the refusals the
                // sequential pass sees (`--emit-negatives`).
                self.state.note_refused(&pending.entry, &swept, self.target);
                continue;
            }
            // The sequential loop applies this step and moves on to the
            // candidate after it, on the new diagram.  Everything checked
            // past it was checked on the old diagram: discard it.
            self.state.commit(pending.entry, swept);
            self.changed = true;
            self.pending.clear();
            self.generation = pool.fresh_generation();
            self.next = pending.index + 1;
        }
    }
}

/// The coordinator's end of the worker pool.
struct Pool {
    jobs: Sender<StepJob>,
    done: Receiver<StepDone>,
    /// Checks submitted and not yet answered, stale ones included, so no
    /// more than `capacity` searches ever run at once.
    busy: usize,
    capacity: usize,
    generation: u64,
    /// Set once the record stopped ([`super::stopped`]): the workers skip
    /// the checks still queued, and no pass runs any more.
    abort: Arc<AtomicBool>,
}

impl Pool {
    fn fresh_generation(&mut self) -> u64 {
        self.generation += 1;
        self.generation
    }

    fn submit(&mut self, id: usize, chain: &mut Chain<'_>, attacker: u8) {
        let Some((index, trial, entry, structural)) = chain.next_trial(attacker) else {
            return;
        };
        chain.pending.push_back(Pending {
            index,
            entry: entry.clone(),
            result: None,
        });
        self.jobs
            .send(StepJob {
                chain: id,
                generation: chain.generation,
                index,
                trial,
                rule: chain.state.rule.clone(),
                entry,
                structural,
            })
            .expect("reduce workers alive");
        self.busy += 1;
    }

    /// Runs `chains` to completion.  Each ends exactly where the
    /// sequential [`super::greedy_pass`] over its order would.
    ///
    /// With the early stop ([`super::stopped`]) it returns as soon as a
    /// committed step's check counts, the chains where they stand (the
    /// caller drops them), and every later call returns at once.
    fn run(&mut self, chains: &mut [Chain<'_>], attacker: u8) {
        for chain in chains.iter_mut() {
            chain.generation = self.fresh_generation();
        }
        loop {
            for chain in chains.iter_mut() {
                if super::stopped() {
                    break;
                }
                chain.resolve(self);
            }
            if super::stopped() {
                // Stale and queued checks are dropped: the coordinator
                // never receives again in this record.
                self.abort.store(true, Ordering::Relaxed);
                return;
            }
            if chains.iter().all(Chain::done) {
                // Stale checks may still be running; their answers are
                // dropped by generation whenever they arrive.
                return;
            }
            // Non-speculative checks first: each chain with nothing in
            // flight gets its next candidate, in chain order.
            for (id, chain) in chains.iter_mut().enumerate() {
                if self.busy >= self.capacity {
                    break;
                }
                if chain.pending.is_empty() {
                    self.submit(id, chain, attacker);
                }
            }
            // Spare threads speculate, on the chain with the fewest checks
            // in flight (lowest id on ties).
            while self.busy < self.capacity {
                let pick = chains
                    .iter()
                    .enumerate()
                    .filter(|(_, chain)| chain.next < chain.order.len())
                    .min_by_key(|(id, chain)| (chain.pending.len(), *id))
                    .map(|(id, _)| id);
                let Some(id) = pick else { break };
                self.submit(id, &mut chains[id], attacker);
            }
            if chains.iter().all(Chain::done) {
                // The last candidates did not apply to their diagrams.
                return;
            }
            // Some chain is not done, so it has a check in flight: every
            // chain was resolved above and one with nothing in flight and
            // candidates left would have been given one (`busy` counts
            // only up to `capacity`, and a full pool is waiting too).
            let done = self.done.recv().expect("reduce workers alive");
            self.busy -= 1;
            let Some(result) = done.result else {
                panic!("a reduce worker panicked");
            };
            let Some(chain) = chains.get_mut(done.chain) else {
                continue;
            };
            if chain.generation != done.generation {
                continue;
            }
            if let Some(pending) = chain
                .pending
                .iter_mut()
                .find(|pending| pending.index == done.index)
            {
                pending.result = Some(result);
            }
        }
    }
}

/// The coordinator's handle on a running pool ([`with_workers`]).
pub(in crate::reduce) struct Runner<'p> {
    pool: &'p mut Pool,
    attacker: u8,
    target: u32,
}

impl Runner<'_> {
    /// One greedy pass per `(order, state)`, all at once.  Returns each
    /// chain's state and whether the pass removed anything, in input order:
    /// exactly what [`super::greedy_pass`] leaves on each.
    pub(in crate::reduce) fn passes(
        &mut self,
        passes: Vec<(&[Candidate], ChainState)>,
    ) -> Vec<(ChainState, bool)> {
        // A settled chain's pass would take nothing (`ChainState::settled`):
        // it is not run, as the sequential `greedy_pass` returns at once.
        let mut out: Vec<Option<(ChainState, bool)>> = Vec::with_capacity(passes.len());
        let mut chains: Vec<Chain<'_>> = Vec::new();
        for (order, state) in passes {
            if state.skips_pass() {
                out.push(Some((state, false)));
            } else {
                out.push(None);
                chains.push(Chain::new(order, state, self.target));
            }
        }
        self.pool.run(&mut chains, self.attacker);
        let mut ran = chains.into_iter().map(|chain| {
            let mut state = chain.state;
            state.settled = !chain.changed;
            (state, chain.changed)
        });
        out.into_iter()
            .map(|slot| slot.unwrap_or_else(|| ran.next().expect("one result per pass")))
            .collect()
    }

    /// `while greedy_pass(state, candidates) {}` for every state, all at
    /// once; returned in input order.
    pub(in crate::reduce) fn sweep_to_fixpoint(
        &mut self,
        candidates: &[Candidate],
        states: Vec<ChainState>,
    ) -> Vec<ChainState> {
        let mut finished: Vec<Option<ChainState>> = vec![None; states.len()];
        let mut active: Vec<(usize, ChainState)> = states.into_iter().enumerate().collect();
        while !active.is_empty() {
            if super::stopped() {
                // The early stop: the chains are dropped by the caller.
                for (slot, state) in active {
                    finished[slot] = Some(state);
                }
                break;
            }
            let slots: Vec<usize> = active.iter().map(|(slot, _)| *slot).collect();
            let results = self.passes(
                active
                    .into_iter()
                    .map(|(_, state)| (candidates, state))
                    .collect(),
            );
            active = Vec::new();
            for (slot, (state, changed)) in slots.into_iter().zip(results) {
                if changed {
                    active.push((slot, state));
                } else {
                    finished[slot] = Some(state);
                }
            }
        }
        finished
            .into_iter()
            .map(|state| state.expect("every chain reaches its fixpoint"))
            .collect()
    }
}

/// Runs `body` with a pool of `options.jobs` workers (the caller's solver
/// is worker 0).  Every check is a pure function of its diagram and rule
/// (each solve starts from an empty table), so which worker takes a step
/// never matters.
pub(super) fn with_workers<R>(
    solver: &mut Solver,
    target: u32,
    options: &ReduceOptions,
    attacker: u8,
    body: impl FnOnce(&mut Runner<'_>) -> R,
) -> R {
    let capacity = options.jobs.max(1);
    // Every extra worker searches on a solver of its own with the
    // caller's table, so every candidate is judged alike.
    let table = solver.table_mode();
    let mut extra: Vec<Solver> = (1..capacity)
        .map(|_| {
            let mut worker = Solver::new();
            worker.set_table(table);
            worker
        })
        .collect();
    let (job_tx, job_rx) = channel::<StepJob>();
    let job_rx = Mutex::new(job_rx);
    let (done_tx, done_rx) = channel::<StepDone>();
    let abort = Arc::new(AtomicBool::new(false));
    // The workers' steps stop their sweeps on a budget-out as the
    // coordinator's would ([`super::stopped`]).
    let armed = super::STOP_ARMED.with(std::cell::Cell::get);
    // The workers' CPU time (microseconds), charged to this thread when
    // the pool closes ([`crate::record_clock::charge`]): the workers exit
    // before the record's cost is read.
    let worker_cpu = std::sync::atomic::AtomicU64::new(0);
    let charged = thread::scope(|scope| {
        for worker in std::iter::once(solver).chain(extra.iter_mut()) {
            let job_rx = &job_rx;
            let done_tx = done_tx.clone();
            let abort = Arc::clone(&abort);
            let worker_cpu = &worker_cpu;
            scope.spawn(move || {
                // Charged on every exit of the loop below.
                let _cpu = WorkerCpu(worker_cpu);
                super::STOP_ARMED.with(|cell| cell.set(armed));
                super::POOL_ABORT.with(|cell| *cell.borrow_mut() = Some(Arc::clone(&abort)));
                loop {
                    let job = {
                        let guard = job_rx.lock().expect("reduce job queue");
                        guard.recv()
                    };
                    // The coordinator dropped its sender: all work is done.
                    let Ok(job) = job else { break };
                    if abort.load(Ordering::Relaxed) {
                        // The record stopped: nobody reads this answer.
                        // (Until the coordinator commits the stopping
                        // step, a worker may still start one queued
                        // check: a budget-out counts only once committed
                        // in chain order, so no worker can stop early.)
                        continue;
                    }
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        evaluate_step(
                            worker,
                            &job.trial,
                            target,
                            options,
                            &job.rule,
                            &job.entry,
                            job.structural,
                        )
                    }))
                    .ok();
                    let failed = result.is_none();
                    let _ = done_tx.send(StepDone {
                        chain: job.chain,
                        generation: job.generation,
                        index: job.index,
                        result,
                    });
                    if failed {
                        break;
                    }
                }
            });
        }
        drop(done_tx);
        let mut pool = Pool {
            jobs: job_tx,
            done: done_rx,
            busy: 0,
            capacity,
            generation: 0,
            abort,
        };
        let out = body(&mut Runner {
            pool: &mut pool,
            attacker,
            target,
        });
        // Dropping the pool closes the job queue; the scope then waits for
        // any stale check still running.
        drop(pool);
        out
    });
    crate::record_clock::charge(worker_cpu.load(Ordering::Relaxed) as f64 * 1e-6);
    charged
}

/// Adds the CPU time of the worker thread it is dropped on to the pool's
/// total (measurement only, [`crate::record_clock`]).
struct WorkerCpu<'a>(&'a std::sync::atomic::AtomicU64);

impl Drop for WorkerCpu<'_> {
    fn drop(&mut self) {
        if let Some(seconds) = crate::cpu::thread_cpu_seconds() {
            self.0
                .fetch_add((seconds * 1e6).round() as u64, Ordering::Relaxed);
        }
    }
}
