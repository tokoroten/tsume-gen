//! Measurement of the prober (`TSUME_PDFPN_PROF`, `TSUME_PDFPN_TRACE`,
//! [`crate::diagnostics`]): nothing here changes a search.

use super::*;

/// Measurement of a parallel prober (`TSUME_PDFPN_PROF`): the time split of
/// its questions and the counters of every thread, summed; printed by the
/// main prober at every `reset` as one `pdfpn prof {json}` line on stderr
/// (cumulative since the prober was made).  Nothing here is read by a
/// search.
#[derive(Default)]
pub(super) struct ParProf {
    pub(super) on: bool,
    /// Questions asked; those never published (answered by the main prober
    /// alone), their nodes and wall nanoseconds.
    pub(super) questions: AtomicU64,
    pub(super) small: AtomicU64,
    pub(super) small_nodes: AtomicU64,
    pub(super) small_ns: AtomicU64,
    /// Published questions: the main prober's time and nodes before
    /// publishing, its time and nodes from publishing to its root's verdict,
    /// and its wait for the helpers after.
    pub(super) pre_ns: AtomicU64,
    pub(super) pre_nodes: AtomicU64,
    pub(super) par_ns: AtomicU64,
    pub(super) par_nodes: AtomicU64,
    pub(super) wait_ns: AtomicU64,
    /// Waits over a millisecond, and the longest.
    pub(super) wait_long: AtomicU64,
    pub(super) wait_max_ns: AtomicU64,
    /// Helpers: admissions, time inside questions and nodes.
    pub(super) helper_runs: AtomicU64,
    /// Helpers between questions: wall time spinning, yielding and parked
    /// (spinning and yielding hold a core), and waits that ended parked.
    pub(super) wait_spin_ns: AtomicU64,
    pub(super) wait_yield_ns: AtomicU64,
    pub(super) wait_park_ns: AtomicU64,
    pub(super) park_wakes: AtomicU64,
    pub(super) helper_ns: AtomicU64,
    pub(super) helper_nodes: AtomicU64,
    /// Counters of every thread ([`CdfpnStats`]), summed at question ends.
    pub(super) stalls: AtomicU64,
    pub(super) stall_exits: AtomicU64,
    pub(super) drops: AtomicU64,
    pub(super) creations: AtomicU64,
    pub(super) concurrent: AtomicU64,
    pub(super) late_stores: AtomicU64,
    pub(super) interval_hits: AtomicU64,
    pub(super) dominance_hits: AtomicU64,
    /// Wall nanoseconds of the prober's life until the last print.
    pub(super) born: Option<crate::clock::Instant>,
}

impl ParProf {
    pub(super) fn new() -> Self {
        let on = crate::diagnostics::pdfpn_prof();
        ParProf {
            on,
            born: on.then(crate::clock::Instant::now),
            ..Default::default()
        }
    }

    /// Adds a thread's counters since `before` (its stats at the question's
    /// start).
    pub(super) fn add_stats(&self, now: &CdfpnStats, before: &CdfpnStats) {
        if !self.on {
            return;
        }
        let add = |counter: &AtomicU64, value: u64| {
            counter.fetch_add(value, Ordering::Relaxed);
        };
        add(&self.stalls, now.stalls - before.stalls);
        add(&self.stall_exits, now.stall_exits - before.stall_exits);
        add(&self.drops, now.drops - before.drops);
        add(&self.creations, now.creations - before.creations);
        add(&self.concurrent, now.concurrent - before.concurrent);
        add(&self.late_stores, now.late_stores - before.late_stores);
        add(
            &self.interval_hits,
            now.interval_hits - before.interval_hits,
        );
        add(
            &self.dominance_hits,
            now.dominance_hits - before.dominance_hits,
        );
    }

    pub(super) fn print(&self, shared: &ParShared, threads: usize) {
        if !self.on {
            return;
        }
        let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let mut line = format!(
            "pdfpn prof {{\"threads\":{threads},\"lifeNs\":{},\"questions\":{},\"small\":{},\"smallNodes\":{},\"smallNs\":{},\"preNs\":{},\"preNodes\":{},\"parNs\":{},\"parNodes\":{},\"waitNs\":{},\"waitLong\":{},\"waitMaxNs\":{},\"waitSpinNs\":{},\"waitYieldNs\":{},\"waitParkNs\":{},\"parkWakes\":{},\"helperRuns\":{},\"helperNs\":{},\"helperNodes\":{},\"stalls\":{},\"stallExits\":{},\"drops\":{},\"creations\":{},\"concurrent\":{},\"lateStores\":{},\"intervalHits\":{},\"dominanceHits\":{}",
            self.born.map_or(0, |born| born.elapsed().as_nanos() as u64),
            get(&self.questions),
            get(&self.small),
            get(&self.small_nodes),
            get(&self.small_ns),
            get(&self.pre_ns),
            get(&self.pre_nodes),
            get(&self.par_ns),
            get(&self.par_nodes),
            get(&self.wait_ns),
            get(&self.wait_long),
            get(&self.wait_max_ns),
            get(&self.wait_spin_ns),
            get(&self.wait_yield_ns),
            get(&self.wait_park_ns),
            get(&self.park_wakes),
            get(&self.helper_runs),
            get(&self.helper_ns),
            get(&self.helper_nodes),
            get(&self.stalls),
            get(&self.stall_exits),
            get(&self.drops),
            get(&self.creations),
            get(&self.concurrent),
            get(&self.late_stores),
            get(&self.interval_hits),
            get(&self.dominance_hits),
        );
        for (name, table) in [("tt", &shared.tt), ("facts", &shared.facts)] {
            let (counts, sweeps, swept) = table.counters();
            line.push_str(&format!(",\"{name}\":{{"));
            for (key, value) in SeqTable::COUNTER_NAMES.iter().zip(counts.iter()) {
                line.push_str(&format!("\"{key}\":{value},"));
            }
            line.push_str(&format!(
                "\"sweeps\":{sweeps},\"swept\":{swept},\"fill\":{}}}",
                table.fill()
            ));
        }
        line.push('}');
        eprintln!("{line}");
    }
}

/// `TSUME_PDFPN_TRACE` (measurement only): one line to stderr per big
/// question of a prober, parallel or not.  Read once.
pub fn trace_on() -> bool {
    crate::diagnostics::pdfpn_trace()
}
