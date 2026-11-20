//! The parallel prober (`--prober-threads` above 1): the shared tables, the
//! question each search publishes, the per-question budget, the helper
//! threads and their df-pn loop.

use super::*;

/// Nodes a prober of a parallel question takes from the shared budget at a
/// time ([`ClassicalProber::with_threads`]).
pub(super) const PAR_GRANT: u64 = 1024;

/// Nodes the main prober expands on its own before it wakes the helpers
/// (small questions never pay for them).
/// 1,024 until 2026-10-16; 128 since (the construction asks thousands of
/// questions of a few thousand nodes, and their serial prefix was 18 % of
/// the wall clock at P=16: `docs/experiments/par-dfpn/README.md` §8); 16
/// since 2026-10-27 (after the leaf search and the bitboards the prefix and
/// the questions below it were 14 % of the wall clock at P=16; 16 is 8 %
/// faster there and 4 % at P=8, 2-8 the same:
/// `docs/experiments/solver-scaling/README.md` §10).
pub(super) const PAR_JOIN_AFTER: u64 = 16;

/// What one visitor adds to a node's proof and disproof numbers as another
/// thread reads them.
pub(super) const PAR_VIRTUAL_UNIT: u32 = 1;

/// And per visitor, `1 / 2^PAR_VIRTUAL_SHIFT` of the number itself
/// (2026-10-28: the threads collide less under large numbers; 1.04-1.05x at
/// P=4-8 together with the later sweep, `docs/experiments/solver-scaling/
/// README.md` §15).
pub(super) const PAR_VIRTUAL_SHIFT: u32 = 5;

/// Descents that expanded nothing in a row before a node gives up its loop
/// (a sibling's numbers changing under it on every read).
pub(super) const PAR_STALL_LIMIT: u32 = 64;

/// Default size of the parallel prober's transposition table, in log2 slots
/// (32 bytes each, 512 MiB); the fact table has half as many.  24 since
/// 2026-10-26 (23 before): fewer duplicated expansions on 16 threads
/// (`docs/experiments/solver-scaling/README.md` §3).
pub const PAR_TABLE_BITS: u32 = 24;

/// Where the fact table packs a position's facts: `mate | refuted_below << 8
/// | pure << 16`, with [`FACT_WRITTEN`] (so the word is never zero).
#[inline]
pub(super) fn pack_fact_word(mate: u8, refuted_below: u8, pure: u8) -> u64 {
    pack_fact(mate, refuted_below, pure)
}

/// A packed transposition entry: `pn | dn << 32` (never zero: no node has
/// both numbers zero).
#[inline]
pub(super) fn pack_numbers(pn: u32, dn: u32) -> u64 {
    u64::from(pn) | u64::from(dn) << 32
}

/// The table's 16 bits of metadata: `pure` in bit 0, the class of the work
/// under the node ([`work_class`], at most 32) above.
#[inline]
pub(super) fn pack_meta(pure: bool, class: u32) -> u32 {
    u32::from(pure) | class.min(63) << 1
}

/// What the transposition slot of a node is worth when a bucket is full: a
/// settled node before an open one (as item 3's garbage collection keeps
/// them), then the class of the work under it.
#[inline]
pub(super) fn entry_priority(d: u64, meta: u32) -> u64 {
    let (pn, dn) = (d as u32, (d >> 32) as u32);
    let solved = pn == 0 || dn == 0;
    u64::from(meta >> 1) | u64::from(solved) << 8
}

/// The budget of one question of the parallel search, the question's own
/// (Codex review 2026-10-16, finding 1): a helper takes grants from and
/// gives them back to the budget of the question it is on, never another's.
/// The main prober **closes** it when it answers ([`QuestionBudget::close`],
/// one atomic `fetch_or` of [`BUDGET_CLOSED`]): no grant is taken after the
/// close (Codex r6: a helper descheduled between its finished check and
/// its `refill` would otherwise take one and expand a node after the
/// count), and the value the close saw is the count.
pub(super) struct QuestionBudget {
    /// Nodes still to be taken, with [`BUDGET_CLOSED`] once closed.
    pub(super) left: AtomicU64,
}

/// The bit of [`QuestionBudget::left`] that closes the budget.
pub(super) const BUDGET_CLOSED: u64 = 1 << 63;

impl QuestionBudget {
    /// The top bit is the close, so a budget of `2^63` or more is clamped
    /// to `2^63 - 1` nodes (Codex r7: it would otherwise read as closed and
    /// the question would answer `Unknown` without a search; the caller
    /// clamps the same way for its count).
    pub(super) fn new(budget: u64) -> Arc<Self> {
        Arc::new(QuestionBudget {
            left: AtomicU64::new(budget.min(BUDGET_CLOSED - 1)),
        })
    }

    /// Closes the budget and returns what was left at that moment (grants
    /// given back later are not counted: they were taken, so they are used).
    pub(super) fn close(&self) -> u64 {
        self.left.fetch_or(BUDGET_CLOSED, Ordering::AcqRel) & !BUDGET_CLOSED
    }

    /// Nodes left, open or closed (tests).
    #[cfg(test)]
    pub(super) fn left(&self) -> u64 {
        self.left.load(Ordering::Relaxed) & !BUDGET_CLOSED
    }
}

/// One question of the parallel search, as the helpers read it.
#[derive(Clone)]
pub(super) struct Question {
    pub(super) pos: Position,
    pub(super) node: Node,
    pub(super) attacker: u8,
    pub(super) epoch: u64,
    pub(super) generation: u64,
    pub(super) budget: Arc<QuestionBudget>,
}

/// The published question and the latest generation finished, under one
/// lock: a helper is admitted to a question only under it, and the main
/// prober finishes a question only under it, so a helper admitted to
/// generation `g` was counted in (`active`) before the main prober could
/// finish `g`, and a helper that finds `g` finished never runs it (whatever
/// it saw of the atomics before).
pub(super) struct Current {
    pub(super) question: Option<Question>,
    pub(super) done: u64,
}

/// What the probers of one parallel search share
/// ([`ClassicalProber::with_threads`]).
pub(crate) struct ParShared {
    /// The transposition table (keys as [`node_key`]; data [`pack_numbers`],
    /// metadata [`pack_meta`]).
    pub(super) tt: SeqTable,
    /// The fact table (keys as the interval table's; data [`pack_fact_word`],
    /// metadata the work class).
    pub(super) facts: SeqTable,
    /// The generation of the latest published question (0: none yet); the
    /// helpers' wake-up signal.
    pub(super) question: AtomicU64,
    /// The latest generation the main prober finished: the helpers' per-node
    /// stop signal (admission reads [`Current::done`] under the lock).
    pub(super) done: AtomicU64,
    /// Bumped by every [`ClassicalProber::reset`]: a helper seeing a new one
    /// empties its private memos.
    pub(super) epoch: AtomicU64,
    /// Helpers inside a question (counted before they read which).
    pub(super) active: AtomicUsize,
    /// Set once: the helpers leave their loops.
    pub(super) stop: AtomicBool,
    /// Set by a helper whose search panicked ([`helper_loop`]): the main
    /// prober's next answer panics too, instead of answering on fewer
    /// threads without a word.
    pub(super) failed: AtomicBool,
    /// The published question and the latest generation finished.
    pub(super) current: Mutex<Current>,
    /// The helpers' threads, to wake them.
    pub(super) helpers: Mutex<Vec<std::thread::Thread>>,
    /// `TSUME_PDFPN_TRACE`: one line to stderr per published question and
    /// helper join (measurement only).
    pub(super) trace: bool,
    /// `TSUME_PDFPN_PROF` (measurement only).
    pub(super) prof: ParProf,
}

impl ParShared {
    pub(super) fn new(table_bits: u32) -> Self {
        let shared = ParShared {
            tt: SeqTable::new(table_bits),
            facts: SeqTable::new(table_bits.saturating_sub(1).max(2)),
            question: AtomicU64::new(0),
            done: AtomicU64::new(0),
            epoch: AtomicU64::new(1),
            active: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            current: Mutex::new(Current {
                question: None,
                done: 0,
            }),
            helpers: Mutex::new(Vec::new()),
            trace: trace_on(),
            prof: ParProf::new(),
        };
        if shared.prof.on {
            shared.tt.set_counting(true);
            shared.facts.set_counting(true);
        }
        shared
    }

    /// Wakes every helper.
    pub(super) fn wake(&self) {
        let helpers = self
            .helpers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for thread in helpers.iter() {
            thread.unpark();
        }
    }

    /// Marks question `generation` finished: first for the helpers' per-node
    /// check, then under the admission lock (no helper is admitted to it
    /// from now on).  Idempotent.
    pub(super) fn finish(&self, generation: u64) {
        self.done.fetch_max(generation, Ordering::Release);
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        current.done = current.done.max(generation);
        if current
            .question
            .as_ref()
            .is_some_and(|q| q.generation <= generation)
        {
            current.question = None;
        }
    }
}

/// Counts a helper in [`ParShared::active`] while it lives, out when it is
/// dropped (also on a panic: the main prober's wait for `active == 0` must
/// not outlive the helper).
pub(super) struct ActiveGuard<'a>(&'a AtomicUsize);

impl<'a> ActiveGuard<'a> {
    pub(super) fn enter(active: &'a AtomicUsize) -> Self {
        active.fetch_add(1, Ordering::AcqRel);
        ActiveGuard(active)
    }
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Finishes the main prober's question when it unwinds out of the search
/// (a panic): the helpers would otherwise keep searching it until the budget
/// is gone.  Disarmed on the normal path, which finishes and then waits.
pub(super) struct FinishGuard<'a> {
    pub(super) par: &'a ParShared,
    pub(super) generation: u64,
    pub(super) armed: bool,
}

impl Drop for FinishGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.par.finish(self.generation);
        }
    }
}

/// Gives a helper's grant back to the shared budget when its search ends,
/// also by a panic: what the main prober counts as used is then what was
/// expanded.  (The visits the helper counted stay until their slots are
/// replaced or the next `reset`: nodes under it read as one visitor busier,
/// which costs work, never a verdict.)
pub(super) struct GrantGuard<'a> {
    pub(super) prober: &'a mut ClassicalProber,
    pub(super) budget: &'a QuestionBudget,
}

impl Drop for GrantGuard<'_> {
    fn drop(&mut self) {
        let budget = std::mem::take(&mut self.prober.budget);
        if budget > 0 {
            self.budget.left.fetch_add(budget, Ordering::Relaxed);
        }
    }
}

/// Tests only: the next helper that takes a grant panics
/// (`par_tests::a_helpers_panic_is_reported_and_its_grant_returned`).
#[cfg(test)]
pub(super) static PANIC_HELPER: AtomicBool = AtomicBool::new(false);

/// How a helper waits for the next question: [`HELPER_SPINS`] spins, then
/// [`HELPER_YIELDS`] yields of its core to any other thread that wants it,
/// then parked (woken by the main prober's publish, or every millisecond).
/// The yields are macOS only (2026-10-27): a solve asks a question every few
/// hundred microseconds, and there a parked helper woke so late that at P=8
/// the helpers were inside a question only 56 % of the time (yielding: 1.2-1.3
/// times faster at P=8-16, and 10 % less CPU, under load); on Windows the
/// park was not the loss, and the yields only cost CPU
/// (`docs/experiments/solver-scaling/README.md` §12).
pub(super) const HELPER_SPINS: u32 = 2_000;

pub(super) static HELPER_YIELDS: u32 = if cfg!(target_os = "macos") { 2_000 } else { 0 };

/// One helper thread of a parallel prober: waits for a question, searches it
/// from the root until the main prober finishes it, and waits for the next.
pub(super) fn helper_loop(shared: Arc<ParShared>, index: usize) {
    let mut prober = ClassicalProber::helper(Arc::clone(&shared), index);
    let mut seen = 0u64;
    loop {
        // Wait for a question newer than the last one seen (spin a little,
        // yield a while, then park; an unpark before the park is not lost).
        // With `TSUME_PDFPN_PROF`, the time of each phase is measured.
        let (mut spins, mut yields) = (0u32, 0u32);
        let wait_started = shared.prof.on.then(crate::clock::Instant::now);
        let mut phase_ns = [0u64; 3];
        let mut phase_at = wait_started;
        let mut phase = 0usize;
        let woke = loop {
            if shared.stop.load(Ordering::Acquire) {
                if shared.trace {
                    eprintln!("pdfpn helper {index}: stopped");
                }
                return;
            }
            let woke = shared.question.load(Ordering::Acquire);
            if woke != seen {
                break woke;
            }
            let next = if spins < HELPER_SPINS {
                spins += 1;
                std::hint::spin_loop();
                0
            } else if yields < HELPER_YIELDS {
                yields += 1;
                std::thread::yield_now();
                1
            } else {
                std::thread::park_timeout(std::time::Duration::from_millis(1));
                2
            };
            if next != phase {
                if let Some(at) = phase_at {
                    let now = crate::clock::Instant::now();
                    phase_ns[phase] += now.duration_since(at).as_nanos() as u64;
                    phase_at = Some(now);
                }
                phase = next;
            }
        };
        if let Some(at) = phase_at {
            phase_ns[phase] += at.elapsed().as_nanos() as u64;
            let prof = &shared.prof;
            prof.wait_spin_ns.fetch_add(phase_ns[0], Ordering::Relaxed);
            prof.wait_yield_ns.fetch_add(phase_ns[1], Ordering::Relaxed);
            prof.wait_park_ns.fetch_add(phase_ns[2], Ordering::Relaxed);
            if phase == 2 {
                prof.park_wakes.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Counted in before the admission, so the main prober's wait at the
        // end of a question covers every helper admitted to it (the count
        // went up before this helper took the lock that finishing takes, so
        // the main prober's reads of it after finishing see it); admitted
        // under that lock, so a finished question is never run
        // ([`Current`]).
        let active = ActiveGuard::enter(&shared.active);
        let admitted = {
            let current = shared
                .current
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            current
                .question
                .as_ref()
                .filter(|q| q.generation > current.done)
                .cloned()
        };
        // Seen: the question admitted to, else the one that woke this
        // helper -- never a newer one published meanwhile (the main prober
        // does not wait for the helpers, so it may have), which the next
        // wait must notice (Codex r4).
        seen = woke;
        if let Some(question) = admitted {
            seen = question.generation;
            let before = prober.stats.nodes;
            let stats_before = prober.stats;
            let started = crate::clock::Instant::now();
            // A panic in the search gives the grant back (`GrantGuard`),
            // leaves the question (`ActiveGuard`), marks the search failed
            // and ends this helper.
            let outcome = {
                let grant = GrantGuard {
                    prober: &mut prober,
                    budget: &question.budget,
                };
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    grant.prober.run_question(&question)
                }))
            };
            if outcome.is_err() {
                shared.failed.store(true, Ordering::Release);
                if shared.trace {
                    eprintln!(
                        "pdfpn helper {index}: panicked in question {}",
                        question.generation
                    );
                }
                return;
            }
            if shared.prof.on {
                let prof = &shared.prof;
                prof.helper_runs.fetch_add(1, Ordering::Relaxed);
                prof.helper_ns
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                prof.helper_nodes
                    .fetch_add(prober.stats.nodes - before, Ordering::Relaxed);
                prof.add_stats(&prober.stats, &stats_before);
            }
            if shared.trace {
                eprintln!(
                    "pdfpn helper {index}: question {} bound {} nodes {} in {:.3} s stalls {} exits {} drops {} creations {} sweeps {}",
                    question.generation,
                    question.node.bound,
                    prober.stats.nodes - before,
                    started.elapsed().as_secs_f64(),
                    prober.stats.stalls,
                    prober.stats.stall_exits,
                    prober.stats.drops,
                    prober.stats.creations,
                    prober.stats.sweeps,
                );
                prober.stats.stalls = 0;
                prober.stats.stall_exits = 0;
                prober.stats.drops = 0;
                prober.stats.sweeps = 0;
                prober.stats.creations = 0;
            }
        }
        drop(active);
    }
}

/// The main prober's bookkeeping of the question in progress
/// ([`ClassicalProber::with_threads`]).
pub(super) struct ParQuestion {
    pub(super) generation: u64,
    /// Whether the helpers were woken for it.
    pub(super) published: bool,
    /// The root, kept for the helpers.
    pub(super) root: Option<(Position, Node)>,
    /// `stats.nodes` when the question started.
    pub(super) own_start: u64,
    /// Measurement: when the question started and was published, and the
    /// main prober's nodes and stats then.
    pub(super) started: Option<crate::clock::Instant>,
    pub(super) published_at: Option<crate::clock::Instant>,
    pub(super) published_nodes: u64,
    pub(super) stats_start: CdfpnStats,
}

impl ClassicalProber {
    /// A helper of a parallel prober (`index >= 1`).
    pub(super) fn helper(shared: Arc<ParShared>, index: usize) -> Self {
        let mut prober = Self::new();
        prober.par = Some(shared);
        prober.par_index = index;
        prober
    }

    /// [`ClassicalProber::mate_within`] of the main prober of a parallel
    /// search (the module docs): the budget is shared with the helpers, which
    /// join once the main prober has expanded [`PAR_JOIN_AFTER`] nodes, and
    /// `stats.nodes` grows by every thread's expansions.
    pub(super) fn mate_within_par(
        &mut self,
        pos: &mut Position,
        node: Node,
        budget: u64,
    ) -> Outcome {
        let par = Arc::clone(self.par.as_ref().expect("a parallel prober"));
        debug_assert_eq!(self.par_index, 0, "only the main prober is asked");
        // A helper that panicked after the last answer was given (it was
        // still leaving) is reported here, before the next answer.
        assert!(
            !par.failed.load(Ordering::Acquire),
            "a helper of the parallel prober panicked (the question is void)"
        );
        self.par_q.generation += 1;
        self.par_q.published = false;
        self.par_q.root = Some((pos.clone(), node));
        self.par_q.own_start = self.stats.nodes;
        self.par_q.started = par.prof.on.then(crate::clock::Instant::now);
        self.par_q.published_at = None;
        self.par_q.stats_start = self.stats;
        // The top bit of the budget word is the close: a budget of 2^63 or
        // more (the CLI accepts any `u64`) is 2^63 - 1 nodes here, in the
        // count below as well (Codex r7, r8).
        let budget = budget.min(BUDGET_CLOSED - 1);
        let question_budget = QuestionBudget::new(budget);
        self.par_budget = Some(Arc::clone(&question_budget));
        self.budget = 0;
        let mut guard = FinishGuard {
            par: &par,
            generation: self.par_q.generation,
            armed: true,
        };
        let (pn, dn, _) = self.mid(pos, node, INF, INF);
        // Give back what is left of the last grant, stop the helpers and
        // wait for them to leave (they give back theirs before leaving).
        question_budget
            .left
            .fetch_add(self.budget, Ordering::Relaxed);
        self.budget = 0;
        self.par_budget = None;
        let waited = crate::clock::Instant::now();
        let own_nodes = self.stats.nodes - self.par_q.own_start;
        guard.armed = false;
        if self.par_q.published {
            // Finished; then wait for the helpers admitted (they leave at
            // their next node: spin a little, then give the core away).
            par.finish(self.par_q.generation);
            // A few microseconds for the helpers to leave (they do at their
            // next node), so that the count below has them nearly always;
            // the slow ones (descheduled) are not waited for, only in
            // `reset` (waiting here was 4-10 % of the wall clock at P=16:
            // `docs/experiments/par-dfpn/README.md` §9).  The nodes used are
            // then counted as the grants taken from the question's budget
            // and not given back when it is closed: every thread's
            // expansions, plus, for a helper still leaving, the rest of its
            // last grant (at most 1,024 nodes; never more than the budget in
            // all, so the budget's consumers stay on the safe side).
            let mut spins = 0u32;
            while par.active.load(Ordering::Acquire) > 0 && spins < 2_000 {
                spins += 1;
                std::hint::spin_loop();
            }
        }
        if let Some(started) = self.par_q.started {
            let prof = &par.prof;
            let add = |counter: &AtomicU64, value: u64| {
                counter.fetch_add(value, Ordering::Relaxed);
            };
            add(&prof.questions, 1);
            match self.par_q.published_at {
                None => {
                    add(&prof.small, 1);
                    add(&prof.small_nodes, own_nodes);
                    add(
                        &prof.small_ns,
                        waited.duration_since(started).as_nanos() as u64,
                    );
                }
                Some(published_at) => {
                    add(
                        &prof.pre_ns,
                        published_at.duration_since(started).as_nanos() as u64,
                    );
                    add(&prof.pre_nodes, self.par_q.published_nodes);
                    add(
                        &prof.par_ns,
                        waited.duration_since(published_at).as_nanos() as u64,
                    );
                    add(&prof.par_nodes, own_nodes - self.par_q.published_nodes);
                    let wait = waited.elapsed().as_nanos() as u64;
                    add(&prof.wait_ns, wait);
                    if wait > 1_000_000 {
                        add(&prof.wait_long, 1);
                    }
                    prof.wait_max_ns.fetch_max(wait, Ordering::Relaxed);
                }
            }
            prof.add_stats(&self.stats, &self.par_q.stats_start);
            if self.par_q.published {
                prof.print(&par, self.par_threads.len() + 1);
            }
        }
        // With the helpers gone, nothing holds a slot and the leaves go
        // through; with them still leaving, one try (the rest at the next
        // node).
        self.retry_leaves(false);
        self.par_q.root = None;
        let own = self.stats.nodes - self.par_q.own_start;
        // The grants taken and not given back when the budget is closed:
        // every thread's expansions (with `wait`, exactly), plus the rest
        // of the last grant of a helper still leaving (its return lands in
        // this closed budget, which nothing reads again: counted as used,
        // never lost).  No grant is taken after the close, so no node is
        // expanded uncounted.
        let used = budget.saturating_sub(question_budget.close());
        let helpers = used.saturating_sub(own);
        self.stats.nodes += helpers;
        self.stats.helper_nodes += helpers;
        #[cfg(test)]
        {
            self.par_last_budget = Some(question_budget);
        }
        assert!(
            !par.failed.load(Ordering::Acquire),
            "a helper of the parallel prober panicked (the question is void)"
        );
        if par.trace && self.par_q.published {
            eprintln!(
                "pdfpn main: question {} bound {} own {own} all {used} verdict {} waited {:.3} s stalls {} exits {} drops {} creations {} sweeps {} fill {}",
                self.par_q.generation,
                node.bound,
                if pn == 0 {
                    "mate"
                } else if dn == 0 {
                    "refuted"
                } else {
                    "unknown"
                },
                waited.elapsed().as_secs_f64(),
                self.stats.stalls,
                self.stats.stall_exits,
                self.stats.drops,
                self.stats.creations,
                self.stats.sweeps,
                par.tt.fill(),
            );
        }
        if pn == 0 {
            Outcome::Mate
        } else if dn == 0 {
            Outcome::Refuted
        } else {
            Outcome::Unknown
        }
    }

    /// Waits until no helper is inside a question (spin a little, then give
    /// the core away).
    pub(super) fn wait_helpers(par: &ParShared) {
        let mut spins = 0u32;
        while par.active.load(Ordering::Acquire) > 0 {
            if spins < 1_000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Wakes the helpers for the main prober's question in progress.
    pub(super) fn publish(&mut self) {
        let Some(par) = self.par.as_ref() else {
            return;
        };
        let Some((pos, node)) = self.par_q.root.as_ref() else {
            return;
        };
        self.par_q.published = true;
        self.stats.published += 1;
        if self.par_q.started.is_some() {
            self.par_q.published_at = Some(crate::clock::Instant::now());
            self.par_q.published_nodes = self.stats.nodes - self.par_q.own_start;
        }
        par.current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .question = Some(Question {
            pos: pos.clone(),
            node: *node,
            attacker: self.attacker,
            epoch: par.epoch.load(Ordering::Acquire),
            generation: self.par_q.generation,
            budget: Arc::clone(self.par_budget.as_ref().expect("a question in progress")),
        });
        par.question.store(self.par_q.generation, Ordering::Release);
        par.wake();
        // Tests only: with a panic injected, wait until a helper has taken
        // its grant (and panicked), so the test does not depend on a helper
        // waking within the few microseconds a small question lasts (on
        // macOS a helper yields its core while it waits, and under load it
        // may not be back in time).
        #[cfg(test)]
        {
            let mut rounds = 0u32;
            while PANIC_HELPER.load(Ordering::Acquire) && rounds < 1_000_000 {
                rounds += 1;
                std::thread::yield_now();
            }
        }
    }

    /// A helper's search of one published question, until the main prober
    /// finishes it ([`helper_loop`]).
    pub(super) fn run_question(&mut self, question: &Question) {
        if self.par_epoch != question.epoch || self.attacker != question.attacker {
            // A new `reset` of the main prober: the private memos go too,
            // and the leaves pending on the old epoch's entries.
            self.futility.clear();
            self.dominance.clear();
            self.par_leaves.clear();
            self.par_epoch = question.epoch;
            self.attacker = question.attacker;
        }
        self.par_q.generation = question.generation;
        self.par_budget = Some(Arc::clone(&question.budget));
        self.aborted = false;
        self.budget = 0;
        self.kids.clear();
        self.par_fresh.clear();
        self.futile_buf.clear();
        self.buf.buf.clear();
        let mut pos = question.pos.clone();
        self.mid(&mut pos, question.node, INF, INF);
        self.retry_leaves(false);
        // Given back to this question's own budget, whether or not the main
        // prober is on another question by now.
        question
            .budget
            .left
            .fetch_add(self.budget, Ordering::Relaxed);
        self.budget = 0;
        self.par_budget = None;
    }

    /// Whether the main prober has finished the question this helper is on
    /// (or was dropped: [`ParShared::stop`]).
    #[inline]
    pub(super) fn par_finished(&self) -> bool {
        match self.par.as_ref() {
            Some(par) if self.par_index != 0 => {
                par.done.load(Ordering::Relaxed) >= self.par_q.generation
                    || par.stop.load(Ordering::Relaxed)
            }
            _ => false,
        }
    }

    /// `v` as this thread reads it with `visitors` other threads under the
    /// node: a settled number stays, an open one grows per visitor by the
    /// virtual unit and by `v >> PAR_VIRTUAL_SHIFT` (never to `INF`).
    #[inline]
    pub(super) fn inflate(&self, v: u32, visitors: u8) -> u32 {
        if visitors == 0 || v == 0 || v == INF {
            return v;
        }
        let share = (u64::from(v) * u64::from(visitors)) >> PAR_VIRTUAL_SHIFT;
        v.saturating_add(PAR_VIRTUAL_UNIT.saturating_mul(u32::from(visitors)))
            .saturating_add(share.min(u64::from(u32::MAX)) as u32)
            .min(NEAR_INF)
    }

    /// Takes the next grant of a shared budget; `false` when there is no
    /// shared budget or it is spent.
    pub(super) fn refill(&mut self) -> bool {
        if self.par.is_some() {
            let Some(budget) = self.par_budget.as_ref() else {
                return false;
            };
            let mut left = budget.left.load(Ordering::Relaxed);
            loop {
                // Spent, or closed by the main prober's answer.
                if left == 0 || left & BUDGET_CLOSED != 0 {
                    return false;
                }
                let grant = left.min(PAR_GRANT);
                match budget.left.compare_exchange_weak(
                    left,
                    left - grant,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.budget = grant;
                        #[cfg(test)]
                        if self.par_index != 0 && PANIC_HELPER.swap(false, Ordering::AcqRel) {
                            panic!("injected: a helper panics with a fresh grant");
                        }
                        return true;
                    }
                    Err(now) => left = now,
                }
            }
        }
        false
    }

    /// Retries the visits this thread could not take back yet
    /// ([`ClassicalProber::par_leaves`]); with `all`, keeps at it for a
    /// while (a question's end), and then lets the rest go (a node that
    /// reads one visitor busier until its slot is replaced: cost, never a
    /// verdict).
    pub(super) fn retry_leaves(&mut self, all: bool) {
        let Some(par) = self.par.as_ref() else {
            return;
        };
        let mut rounds = 0u32;
        while !self.par_leaves.is_empty() {
            self.par_leaves
                .retain(|key| par.tt.leave(key) == Leave::Busy);
            if !all || rounds >= 1_000 {
                break;
            }
            rounds += 1;
            std::thread::yield_now();
        }
        // What is still busy stays on the list for the next node or
        // question (a count left up would keep its slot from every
        // replacement and sweep); `reset` (a new epoch) drops the list.
    }

    /// The df-pn loop on several threads.  The node's own numbers come from
    /// the children's real numbers; which child to search, and with what
    /// thresholds, from the numbers inflated by the other threads.
    pub(super) fn search_children_par(
        &mut self,
        pos: &mut Position,
        node: Node,
        kbase: usize,
        th_pn: u32,
        th_dn: u32,
    ) -> (u32, u32, bool) {
        let count = self.kids.len() - kbase;
        let or = node.kind.is_or();
        // A helper breaks ties among equal children from a different child
        // than the main prober (and the other helpers) does.
        let start = self.par_index % count;
        let mut stalls = 0u32;
        let fbase = self.par_fresh.len();
        self.par_fresh.resize(fbase + count, (0, 0, false));
        // The children are read once and then only the child just searched
        // again (2026-10-28): a sibling another thread changed is seen when
        // this node is searched next (as on one thread,
        // `search_children_cached`; a settled number is never stale, so
        // verdicts stay).  The leaf search is on here: `TSUME_CDFPN_LEAF=0`
        // with `--prober-threads` above 1 is refused (1.0.0).
        let vbase = self.par_values.len();
        for index in 0..count {
            let value = self.child_numbers(&self.kids[kbase + index], (0, 0, false));
            self.par_values.push(value);
        }
        let result = loop {
            let mut best = 0usize;
            let mut best_pair = (1u32, 1u32);
            let (mut min1, mut min2) = (INF, INF);
            let mut sum = 0u32;
            let (mut rmin1, mut rsum) = (INF, 0u32);
            let mut pure_any = false;
            let mut pure_all = true;
            for offset in 0..count {
                let index = if start == 0 {
                    offset
                } else {
                    (offset + start) % count
                };
                let ((rpn, rdn, pure), (pn, dn)) = self.par_values[vbase + index];
                let (lead, other) = if or { (pn, dn) } else { (dn, pn) };
                let (rlead, rother) = if or { (rpn, rdn) } else { (rdn, rpn) };
                sum = add(sum, other);
                rsum = add(rsum, rother);
                rmin1 = rmin1.min(rlead);
                if lead < min1 {
                    min2 = min1;
                    min1 = lead;
                    best = index;
                    best_pair = (pn, dn);
                } else if lead < min2 {
                    min2 = lead;
                }
                if or {
                    pure_any |= rpn == 0 && pure;
                } else {
                    pure_all &= rpn == 0 && pure;
                }
            }
            // The node's real numbers (stored, and returned to the caller).
            let (rpn, rdn) = if or { (rmin1, rsum) } else { (rsum, rmin1) };
            let pure = if rpn == 0 {
                if or {
                    pure_any
                } else {
                    pure_all
                }
            } else {
                false
            };
            // The inflated ones decide whether to go on and where.
            let (pn, dn) = if or { (min1, sum) } else { (sum, min1) };
            if self.aborted || pn >= th_pn || dn >= th_dn {
                break (rpn, rdn, pure && self.pure_node(node));
            }
            // Below a root (whose thresholds are infinite: it goes on until
            // settled or out of budget), a node whose children keep moving
            // under this thread hands back to its parent, which reads them
            // afresh.  Every move is another thread's expansion, so this
            // never loops without the budget going down.
            if stalls > PAR_STALL_LIMIT && (th_pn != INF || th_dn != INF) {
                self.stats.stall_exits += 1;
                break (rpn, rdn, pure && self.pure_node(node));
            }
            let child_th = if or {
                (
                    th_pn.min(bump(min2)),
                    if th_dn == INF {
                        INF
                    } else {
                        th_dn - dn + best_pair.1
                    },
                )
            } else {
                (
                    if th_pn == INF {
                        INF
                    } else {
                        th_pn - pn + best_pair.0
                    },
                    th_dn.min(bump(min2)),
                )
            };
            let child = self.kids[kbase + best];
            let before = self.stats.nodes;
            self.par_fresh[fbase + best] = self.descend(pos, &child, child_th);
            self.par_values[vbase + best] =
                self.child_numbers(&child, self.par_fresh[fbase + best]);
            // A descent that expanded nothing: the child's numbers moved
            // under us (another thread's).  Bounded, so no thread spins here.
            stalls = if self.stats.nodes == before {
                self.stats.stalls += 1;
                stalls + 1
            } else {
                0
            };
        };
        self.par_fresh.truncate(fbase);
        self.par_values.truncate(vbase);
        result
    }
}

#[cfg(test)]
mod par_tests {
    use super::*;
    use crate::sfen::parse_sfen;
    use crate::strict_material;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// A helper that panics gives its grant back and leaves the question,
    /// and the main prober reports the failure instead of answering -- with
    /// the answer when the helper had panicked by then, else (it was still
    /// leaving) at the next answer or `reset`.  With one helper, panicking
    /// before it expanded anything, the main prober's search is the serial
    /// search, so the nodes used are the serial count exactly (a leaked
    /// grant would add up to `PAR_GRANT`).
    #[test]
    fn a_budget_at_or_over_the_close_bit_is_clamped_and_still_open() {
        for budget in [BUDGET_CLOSED - 1, BUDGET_CLOSED, u64::MAX] {
            let q = QuestionBudget::new(budget);
            assert_eq!(q.left(), BUDGET_CLOSED - 1);
            assert_eq!(q.left.load(Ordering::Relaxed) & BUDGET_CLOSED, 0, "open");
            assert_eq!(q.close(), BUDGET_CLOSED - 1);
            assert_ne!(q.left.load(Ordering::Relaxed) & BUDGET_CLOSED, 0, "closed");
        }
        let q = QuestionBudget::new(7);
        assert_eq!(q.left(), 7);
        assert_eq!(q.close(), 7);
    }

    #[test]
    fn a_budget_over_the_close_bit_searches_and_counts_like_a_bounded_one() {
        // Codex r8: the count must use the clamped budget, or `u64::MAX`
        // would add 2^63 phantom helper nodes.
        let position = parse_sfen("k8/2G6/2R6/9/9/9/9/9/9 b - 1").expect("a position");
        let attacker = position.side;
        let root = strict_material::normalize(&position, attacker);
        let mut serial = ClassicalProber::new();
        // The parallel loop, which the main prober runs, has no 1+ε threshold.
        serial.no_epsilon = true;
        for budget in [BUDGET_CLOSED, u64::MAX] {
            serial.reset(attacker);
            let expected = serial.mate_within(&mut root.clone(), 3, budget);
            let serial_nodes = serial.stats.nodes;
            assert_ne!(expected, Outcome::Unknown);
            let mut parallel = ClassicalProber::with_threads(2, 20);
            parallel.reset(attacker);
            let outcome = parallel.mate_within(&mut root.clone(), 3, budget);
            let stats = parallel.stats;
            parallel.reset(attacker);
            assert_eq!(outcome, expected);
            assert_eq!(stats.nodes - stats.helper_nodes, serial_nodes);
            assert!(
                stats.helper_nodes <= 4 * PAR_GRANT,
                "{}",
                stats.helper_nodes
            );
            assert!(stats.nodes < BUDGET_CLOSED);
        }
    }

    #[test]
    fn a_helpers_panic_is_reported_and_its_grant_returned() {
        // Positions and bounds until one whose serial search is large
        // enough to publish (the refutations below a mate length, with a
        // full hand to drop from, are).
        let budget = 2_000_000;
        let mut serial = ClassicalProber::new();
        // The parallel loop, which the main prober runs, has no 1+ε threshold.
        serial.no_epsilon = true;
        let mut chosen = None;
        'search: for sfen in [
            "4+B1k2/2+R6/9/9/3R5/9/9/9/9 b b4g4s4n4l18p 1",
            "9/9/9/9/2+R6/9/6kB1/9/4+L3R b 2Sb4g2s4n3l18p 1",
            "k8/2G6/2R6/9/9/9/9/9/9 b - 1",
            "9/9/9/9/9/2+R6/4G1G2/9/4k1gS1 b GS 1",
        ] {
            let position = parse_sfen(sfen).expect("a position");
            let attacker = position.side;
            let root = strict_material::normalize(&position, attacker);
            for bound in (1..=9).step_by(2) {
                serial.reset(attacker);
                let outcome = serial.mate_within(&mut root.clone(), bound, budget);
                if outcome != Outcome::Unknown && serial.stats.nodes >= 4 * PAR_JOIN_AFTER {
                    chosen = Some((root, attacker, bound, serial.stats.nodes));
                    break 'search;
                }
            }
        }
        let (root, attacker, bound, serial_nodes) = chosen.expect("a bound that publishes");
        // A table the search fits in without evictions (which would change
        // the main prober's count).
        let mut parallel = ClassicalProber::with_threads(2, 24);
        parallel.reset(attacker);
        PANIC_HELPER.store(true, Ordering::Release);
        let result = catch_unwind(AssertUnwindSafe(|| {
            parallel.mate_within(&mut root.clone(), bound, budget)
        }));
        // The statistics of the question (a `reset` zeroes them).
        let stats = parallel.stats;
        if result.is_ok() {
            // The helper was still leaving when the answer was given: the
            // reset waits for it and reports.
            let later = catch_unwind(AssertUnwindSafe(|| parallel.reset(attacker)));
            assert!(
                later.is_err(),
                "the main prober reports the helper's panic at the latest at the reset"
            );
        }
        assert!(
            !PANIC_HELPER.load(Ordering::Acquire),
            "the helper took a grant"
        );
        let par = parallel.par.as_ref().expect("a parallel prober");
        assert!(par.failed.load(Ordering::Acquire));
        // The main prober does not wait for the helper (it may still be
        // leaving, its grant not yet given back): wait here.
        ClassicalProber::wait_helpers(par);
        assert!(parallel.par_q.published, "the question reached the helper");
        assert_eq!(
            stats.nodes - stats.helper_nodes,
            serial_nodes,
            "the main prober's own search is the serial search"
        );
        assert!(
            stats.helper_nodes <= PAR_GRANT,
            "at most the helper's one outstanding grant is counted: {stats:?}"
        );
        assert!(
            parallel.par_budget.is_none(),
            "the question's budget is let go with the answer"
        );
        // The helper's grant came back to the question's own budget, and
        // it expanded nothing there.
        let question = parallel
            .par_last_budget
            .as_ref()
            .expect("the last question's budget");
        assert_eq!(question.left(), budget - serial_nodes);
        assert_eq!(stats.published, 1);
        assert!(
            question.left.load(Ordering::Relaxed) & BUDGET_CLOSED != 0,
            "the budget was closed with the answer"
        );
    }
}
