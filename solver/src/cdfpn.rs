//! Classical df-pn prober (solver v3, item 2; owner instruction 2026-10-09).
//!
//! # What it answers
//!
//! "Is `pos` mate within `b` rule-F plies?" under
//! [`crate::solver::UselessInterposition::Classical`]
//! -- exactly the question the exact solver's credit oracle
//! ([`crate::solver::Solver`]'s `oracle`) answers, by exactly the same rules
//! (`docs/rules.md` rule F, "The credit oracle"):
//!
//! * an ATTACK node mates within `b` when `b > 0` and some legal check's child
//!   mates within `b - 1`;
//! * a DEFENSE node with no legal reply mates when in check (at any bound); an
//!   all-futile F2 node mates at any bound; otherwise, at `b = 0` it does not;
//! * a defense node whose replies are all interposition drops on an
//!   interposable check (F2 applies but not every drop is futile) mates within
//!   `b` when every drop F2 does not prove futile has a child mating within
//!   `b - 1`;
//! * a **credit-line** defense node (interposition drops *and* other replies on
//!   an interposable check) mates within `b` (`b >= 2`; below, never) when every
//!   other reply's child mates within `b - 1` and every drop `r` satisfies
//!   `G_r = P(C_r, b - 1) OR (OR_f P(E_rf, q) AND W)`, `q` the largest even bound
//!   `<= b`, `E_rf` the defense node after the checker's capture form `f`, and
//!   `W = OR_j NOT P(A_j, q - 2)` over the other replies `j`;
//! * every other defense node mates within `b` when every reply's child mates
//!   within `b - 1`.
//!
//! With a sufficient budget its verdict is the credit oracle's, so it is the
//! exact solver's classical outcome at that bound; where the plain df-pn prober
//! ([`crate::dfpn`], F2 only) leaves a classical disproof inconclusive, this one
//! settles it.
//!
//! # How
//!
//! df-pn (Nagai 2002) over an AND/OR graph whose nodes are
//!
//! * the **real** nodes `(bound, position)`, attack (OR) and defense (AND), and
//! * the **virtual** nodes of a credit line, all anchored at the defense node
//!   `D`: `Gate(r, b)` (OR of `C_r` at `b - 1` and `Sub(r, q)`), `Sub(r, q)` (AND
//!   of `Ecap(r, q)` and `Witness(q)`), `Ecap(r, q)` (OR of the capture forms'
//!   `E` at `q`) and `Witness(q)` (OR of `NOT A_j` at `q - 2`).  A `NOT` edge
//!   swaps the child's numbers and thresholds.
//!
//! Every edge lowers the bound, except the credit edge into `E`, which keeps it
//! at most equal and spends a piece of the defender's hand, and the virtual
//! edges, which stay at `D`; so the graph is acyclic (the measure of
//! `docs/rules.md`, plus the order Gate > Sub > Ecap/Witness among the virtual
//! nodes of one `D`) and df-pn needs no cycle handling.
//!
//! The transposition table keys a real node by `(bound, position)` and a virtual
//! one by `(bound, D's position, kind, drop)`.  A second, bound-independent
//! table keeps per position the smallest bound proven mate and one more than
//! the largest proven not mate (sound because the verdict is monotone in the
//! bound, `docs/rules.md` rule F), and a pure mate (one that used no rule F)
//! also answers the same board with no less attacker hand (owner idea 4', the
//! credit oracle's dominance table).  All three tables have the fixed caps and
//! the deterministic garbage collection of item 3.
//!
//! Children are tried in USI order and the search is deterministic; the node
//! budget counts expanded nodes and is never time.
//!
//! # On several threads (`--prober-threads`, 2026-10-16)
//!
//! [`ClassicalProber::with_threads`] answers each question with `P` threads
//! at once -- a parallel df-pn with **virtual proof numbers** (Kishimoto &
//! Kotani 2003; Kaneko 2010, "Parallel depth first proof number search") over
//! one lock-free transposition table ([`crate::pardfpn::SeqTable`]):
//!
//! * the prober asked (the **main** prober) runs `MID` from the root exactly
//!   as on one thread, but on the shared tables; once it has expanded
//!   [`PAR_JOIN_AFTER`] nodes of a question it publishes the question (root, bound)
//!   and wakes the **helpers**, probers of their own on their own threads,
//!   which each run `MID` from the same root with thresholds `(INF, INF)`;
//! * every thread counts itself as a *visitor* of each node it is searching
//!   under (the table's visitor bits).  A thread reads a node's numbers
//!   *inflated* by [`PAR_VIRTUAL_UNIT`] per visitor other than itself -- a
//!   proven or disproven node is never inflated -- and chooses the most
//!   proving child and its thresholds from the inflated numbers, so threads
//!   spread over different children; but it computes and **stores** a
//!   node's own numbers from the children's real numbers, so the table holds
//!   what a one-thread search stores;
//! * the question ends when the main prober's root is settled (or the shared
//!   budget, taken 1,024 nodes at a time, is spent): it marks the question
//!   done, the helpers leave at their next node, and it waits for them.
//!
//! Exactness: a stored proof number `0` (or disproof number `0`) is computed
//! only from children whose stored numbers were `0` (or `INF`) at the time,
//! and from terminal verdicts, and such a verdict is a fact about the
//! position and the bound that never changes; inflation never turns a finite
//! number into `INF` nor any number into `0`; a lost table entry (the table is
//! fixed-size and lossy, and a write may be dropped under contention) is
//! searched again; so a `MATE` or `REFUTED` answer is the one-thread
//! prober's, whatever the threads did.  What depends on the threads is the
//! node count (`stats.nodes` counts every thread's expansions) and whether a
//! budget suffices.  `docs/experiments/par-dfpn/README.md`.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::keytable::KeyTable;
use crate::movegen::{MoveBuf, Safety};
use crate::pardfpn::{Leave, SeqTable, Visit};
use crate::position::{Move, Position};
use crate::solver::{
    board_key, capture_forms_of, dominated, record_dominance, sliding_check_line, table_key,
    CheckLine, Futility, KeyHasher, Outcome,
};

mod diagnostics;
mod horizon;
mod leaf;
mod parallel;
mod search;
mod serial;

pub use self::diagnostics::trace_on;
pub use self::horizon::DisproofKind;
pub use self::parallel::PAR_TABLE_BITS;
use self::{diagnostics::*, parallel::*};

/// Proof/disproof number standing for "impossible".
const INF: u32 = u32::MAX;
/// Largest value a *sum* of child numbers may reach.
const NEAR_INF: u32 = u32::MAX - 1;

/// The most entries the (bound-specific) df-pn table keeps (item 3).
pub const CDFPN_TABLE_CAP: usize = (1 << 22) / 8 * 7;
/// The most positions the bound-interval table keeps (item 3).
pub const CDFPN_INTERVAL_CAP: usize = (1 << 22) / 8 * 7;
/// The most boards the pure-mate dominance table keeps (item 3).
pub const CDFPN_DOMINANCE_CAP: usize = (1 << 17) / 8 * 7;
/// The largest bound the interval table keeps exactly.
const INTERVAL_BOUND_MAX: u32 = 253;

#[inline]
fn add(a: u32, b: u32) -> u32 {
    if a == INF || b == INF {
        INF
    } else {
        a.saturating_add(b).min(NEAR_INF)
    }
}

#[inline]
fn bump(value: u32) -> u32 {
    if value == INF {
        INF
    } else {
        value + 1
    }
}

#[inline]
const fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The kind of a node of the classical AND/OR graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Attack,
    Defense,
    Gate,
    Sub,
    Ecap,
    Witness,
}

impl Kind {
    #[inline]
    const fn is_or(self) -> bool {
        matches!(self, Kind::Attack | Kind::Gate | Kind::Ecap | Kind::Witness)
    }
    #[inline]
    const fn tag(self) -> u64 {
        match self {
            Kind::Attack => 1,
            Kind::Defense => 2,
            Kind::Gate => 3,
            Kind::Sub => 4,
            Kind::Ecap => 5,
            Kind::Witness => 6,
        }
    }
}

/// A node: its kind, its bound and, for the virtual nodes of a credit line,
/// the drop it is about (anchored at the defense node's position).
#[derive(Clone, Copy, Debug)]
struct Node {
    kind: Kind,
    bound: u32,
    drop: Move,
}

/// A child's numbers as the parent reads them: real `(pn, dn, pure)` and
/// inflated `(pn, dn)` ([`ClassicalProber::child_numbers`]).
type ChildValue = ((u32, u32, bool), (u32, u32));

/// One child of the node being expanded.
#[derive(Clone, Copy, Debug)]
struct Child {
    node: Node,
    /// Moves from the parent's anchor position to the child's: 0, 1 or 2.
    path: [Move; 2],
    len: u8,
    /// A `NOT` edge (`Witness`): the child's numbers are read swapped.
    negate: bool,
    key: [u64; 2],
}

/// The table key of a node anchored at a position with Zobrist key `hash`.
#[inline]
fn node_key(node: &Node, hash: u128) -> [u64; 2] {
    match node.kind {
        Kind::Attack | Kind::Defense => table_key(node.bound, hash),
        kind => {
            let code = u64::from(node.drop.to) | (u64::from(node.drop.piece) << 8);
            let salt = mix64((kind.tag() << 32) ^ code ^ 0x6364_6670_6e00_0000);
            let salt2 = mix64(salt ^ 0x5bd1_e995_1234_5678);
            let key = table_key(node.bound, hash);
            [key[0] ^ salt, key[1] ^ salt2]
        }
    }
}

/// What expanding a node gave.
enum Expanded {
    /// The node's value, without children.
    Terminal((u32, u32, bool)),
    /// Its children are on the stack; `impure` when any proof of the node relies
    /// on rule F by construction (an F2-futile reply skipped, a credit node).
    Children { impure: bool },
}

/// One table entry: the node's numbers, the work spent under it (item 3's
/// garbage collection keeps the expensive ones) and, for a proof, whether it
/// was pure.
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    pn: u32,
    dn: u32,
    work: u32,
    pure: bool,
}

/// Per position: the smallest bound proven mate (`u8::MAX` none), one more than
/// the largest bound proven not mate, the smallest bound proven mate by a pure
/// proof, and the work behind them.
#[derive(Clone, Copy, Debug, Default)]
struct Interval {
    mate: u8,
    refuted_below: u8,
    pure: u8,
    work: u32,
}

type Table = KeyTable<Entry>;
type IntervalTable = KeyTable<Interval>;
type DominanceTable = HashMap<[u64; 2], Vec<([u8; 7], u8)>, BuildHasherDefault<KeyHasher>>;

/// The power-of-two class of a work count.
#[inline]
fn work_class(work: u32) -> usize {
    (u32::BITS - work.leading_zeros()) as usize
}

/// Drops the entries of the lowest classes (as numbered by `class`) that
/// together hold at least half of the table.
fn collect_by<V: Copy + Default>(table: &mut KeyTable<V>, class: impl Fn(&V) -> usize) {
    let mut counts = [0usize; 80];
    for value in table.values() {
        counts[class(value)] += 1;
    }
    let target = table.len().div_ceil(2);
    let mut dropped = 0usize;
    let mut limit = 0usize;
    for (index, &count) in counts.iter().enumerate() {
        dropped += count;
        limit = index;
        if dropped >= target {
            break;
        }
    }
    table.retain(|value| class(value) > limit);
}

/// Marks a written slot's packed facts (a zero `v` is an empty slot).
const FACT_WRITTEN: u64 = 1 << 63;

/// `(mate, refuted_below, pure)` packed with [`FACT_WRITTEN`].
#[inline]
fn pack_fact(mate: u8, refuted_below: u8, pure: u8) -> u64 {
    FACT_WRITTEN | u64::from(mate) | u64::from(refuted_below) << 8 | u64::from(pure) << 16
}

#[inline]
fn unpack_fact(v: u64) -> (u8, u8, u8) {
    (v as u8, (v >> 8) as u8, (v >> 16) as u8)
}

/// Counters of one prober, for measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CdfpnStats {
    /// Nodes expanded (the budget's unit).
    pub nodes: u64,
    /// Answers taken from the interval or dominance table at a node's entry.
    pub interval_hits: u64,
    pub dominance_hits: u64,
    /// Garbage collections, all tables.
    pub collections: u64,
    /// On several threads: descents that expanded nothing (the child's
    /// numbers moved under this thread), and nodes whose loop gave up after
    /// [`PAR_STALL_LIMIT`] of them in a row.
    pub stalls: u64,
    pub stall_exits: u64,
    /// On several threads: writes the shared tables dropped (a full bucket
    /// whose every slot is being visited, or contention), and entries made for nodes
    /// at their expansion.
    pub drops: u64,
    pub creations: u64,
    /// On several threads: sweeps of the shared tables this thread ran.
    pub sweeps: u64,
    /// On several threads (measurement): expansions of a node another
    /// thread was already under, and stores of a settled value over an
    /// entry another thread had settled already (the work was done twice).
    pub concurrent: u64,
    pub late_stores: u64,
    /// On several threads: questions published to the helpers, and the
    /// nodes the helpers expanded that were counted into `nodes`
    /// (`nodes - helper_nodes` is this thread's own).
    pub published: u64,
    pub helper_nodes: u64,
}

/// Whether [`ClassicalProber::mid`] runs the leaf search (2026-10-26, on by
/// default).  Off only for `diff-reduce`'s budget records
/// (`TSUME_CDFPN_LEAF=0`): with it off the prober spends its budget as the
/// TypeScript one does, so the two reducers run out of budget alike.
static LEAF_SEARCH: AtomicBool = AtomicBool::new(true);

/// Turns the leaf search on or off for the whole process (tests only).
pub fn set_leaf_search(on: bool) {
    LEAF_SEARCH.store(on, Ordering::Relaxed);
}

/// Whether the leaf search is on ([`crate::stages::nodes_kind_prober`]).
pub fn leaf_search() -> bool {
    LEAF_SEARCH.load(Ordering::Relaxed)
}

/// Whether the df-pn loop keeps its children's numbers instead of reading
/// every child from the table again before each descent (one thread
/// 2026-10-27, the parallel loop 2026-10-28; owner: node counts may change,
/// the rating maps each solver version with its own table): always, except
/// with the leaf search off (`TSUME_CDFPN_LEAF=0`, tests only), which keeps
/// the TypeScript prober's search order.  `TSUME_CDFPN_REREAD=1`, which
/// turned it off for measurement, is no longer read (1.0.0).
pub fn child_cache() -> bool {
    leaf_search()
}

/// `--prober-threads` (and the table size, `PAR_TABLE_BITS` from the CLI) for the probers
/// [`ClassicalProber::configured`] makes from now on (the CLI sets them once,
/// before any search).
static PROBER_THREADS: AtomicUsize = AtomicUsize::new(1);
static PROBER_TABLE_BITS: AtomicUsize = AtomicUsize::new(PAR_TABLE_BITS as usize);

/// Sets what [`ClassicalProber::configured`] builds: `threads` threads per
/// question (`<= 1`: the one-thread prober) over a transposition table of
/// `2^table_bits` slots.
pub fn set_prober_threads(threads: usize, table_bits: u32) {
    PROBER_THREADS.store(threads.max(1), Ordering::Relaxed);
    PROBER_TABLE_BITS.store(table_bits.clamp(10, 34) as usize, Ordering::Relaxed);
}

/// [`set_prober_threads`]'s threads.
pub fn prober_threads() -> usize {
    PROBER_THREADS.load(Ordering::Relaxed)
}

/// The classical df-pn prober.  Its tables persist between questions about the
/// same search (every verdict is exact, so a later question may reuse them)
/// until [`ClassicalProber::reset`].
pub struct ClassicalProber {
    buf: MoveBuf,
    futility: Futility,
    futile_buf: Vec<Option<u32>>,
    kids: Vec<Child>,
    /// Reused per expansion: the moves of the node being expanded and, at a
    /// defense node, which of them rule F2 proves futile.
    moves: Vec<Move>,
    futile: Vec<bool>,
    /// Reused by the leaf search ([`ClassicalProber::leaf_attack1`]): the
    /// checks, and the defense node's moves and futility marks.
    leaf_checks: Vec<Move>,
    leaf_moves: Vec<Move>,
    leaf_futile: Vec<bool>,
    /// The children's numbers of the one-thread loops on the stack
    /// ([`ClassicalProber::search_children_cached`]).
    child_values: Vec<(u32, u32, bool)>,
    table: Table,
    intervals: IntervalTable,
    dominance: DominanceTable,
    attacker: u8,
    /// Nodes left in the current question's budget.
    budget: u64,
    aborted: bool,
    stats: CdfpnStats,
    caps: [usize; 3],
    /// The parallel search's shared state ([`ClassicalProber::with_threads`]);
    /// `None` for a prober on its own tables.
    par: Option<Arc<ParShared>>,
    /// This prober's place in the parallel search: 0 the main prober (the one
    /// asked), `i >= 1` the helper on thread `i`.
    par_index: usize,
    /// The main prober's question in progress.
    par_q: ParQuestion,
    /// The helper threads (the main prober's; joined when it is dropped).
    par_threads: Vec<std::thread::JoinHandle<()>>,
    /// The epoch of the main prober's `reset` this helper last saw.
    par_epoch: u64,
    /// The budget of the question this thread is on (the main prober's
    /// own, or the one a helper was admitted to); `None` between questions.
    par_budget: Option<Arc<QuestionBudget>>,
    /// Tests only: the budget of the main prober's last question.
    #[cfg(test)]
    par_last_budget: Option<Arc<QuestionBudget>>,
    /// Visits of this thread not yet taken back (the slot was busy when it
    /// left the node); retried at the next node and at the question's end.
    par_leaves: Vec<[u64; 2]>,
    /// The values the children of the expansions in progress last returned
    /// (a stack parallel to `kids`; `(0, 0, false)` none yet): what a parent
    /// reads for a child whose entry the shared table does not hold (its
    /// store dropped on a full bucket of visited slots, or evicted), so a dropped store never
    /// sends the parent back down the same child for the same answer.
    par_fresh: Vec<(u32, u32, bool)>,
    /// The parallel loop's children's numbers, real and inflated (a stack
    /// parallel to `kids`).
    par_values: Vec<ChildValue>,
    /// Tests only: this prober's one-thread loop never uses the 1+ε threshold
    /// (as the parallel loop, which the main prober of a parallel search
    /// runs: the tests that compare the two counts).
    #[cfg(test)]
    no_epsilon: bool,
}

impl Default for ClassicalProber {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ClassicalProber {
    /// The main prober tells its helpers to stop and lets them go (they
    /// leave within a millisecond and free the shared tables with the last
    /// reference).  It never joins them: a prober lives in a thread-local
    /// (`crate::reduce`), whose destructor runs while the thread is already
    /// exiting, and on Windows a thread exiting cannot wait for another
    /// thread to exit (the loader lock; measured as a hang, 2026-10-16).
    fn drop(&mut self) {
        if let Some(par) = self.par.as_ref() {
            if self.par_index == 0 {
                par.stop.store(true, Ordering::Release);
                par.wake();
                self.par_threads.clear();
            }
        }
    }
}

impl ClassicalProber {
    pub fn new() -> Self {
        Self {
            buf: MoveBuf::new(),
            futility: Futility::new(),
            futile_buf: Vec::new(),
            kids: Vec::new(),
            moves: Vec::new(),
            futile: Vec::new(),
            leaf_checks: Vec::new(),
            leaf_moves: Vec::new(),
            leaf_futile: Vec::new(),
            child_values: Vec::new(),
            table: Table::default(),
            intervals: IntervalTable::default(),
            dominance: DominanceTable::default(),
            attacker: crate::piece::BLACK,
            budget: 0,
            aborted: false,
            stats: CdfpnStats::default(),
            caps: [CDFPN_TABLE_CAP, CDFPN_INTERVAL_CAP, CDFPN_DOMINANCE_CAP],
            par: None,
            par_index: 0,
            par_q: ParQuestion {
                generation: 0,
                published: false,
                root: None,
                own_start: 0,
                started: None,
                published_at: None,
                published_nodes: 0,
                stats_start: CdfpnStats::default(),
            },
            par_threads: Vec::new(),
            par_epoch: 0,
            par_budget: None,
            #[cfg(test)]
            par_last_budget: None,
            par_leaves: Vec::new(),
            par_fresh: Vec::new(),
            par_values: Vec::new(),
            #[cfg(test)]
            no_epsilon: false,
        }
    }

    /// A prober that answers every question on `threads` threads (see the
    /// module docs): this one, plus `threads - 1` helper threads of its own,
    /// over a transposition table of `2^table_bits` slots of 32 bytes and a
    /// fact table of half as many.  `threads <= 1` is [`ClassicalProber::new`].
    /// Verdicts are the one-thread prober's; node counts are every thread's.
    pub fn with_threads(threads: usize, table_bits: u32) -> Self {
        let threads = threads.max(1);
        if threads == 1 {
            return Self::new();
        }
        let shared = Arc::new(ParShared::new(table_bits));
        let mut prober = Self::new();
        prober.par = Some(Arc::clone(&shared));
        prober.par_index = 0;
        let mut handles = Vec::with_capacity(threads - 1);
        let mut threads_of = Vec::with_capacity(threads - 1);
        for index in 1..threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("tsume-prober-{index}"))
                .spawn(move || helper_loop(shared, index))
                .expect("spawning a prober helper thread");
            threads_of.push(handle.thread().clone());
            handles.push(handle);
        }
        *shared
            .helpers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = threads_of;
        prober.par_threads = handles;
        prober
    }

    /// A prober as [`set_prober_threads`] configured: the one-thread prober
    /// unless the CLI asked for `--prober-threads`.
    pub fn configured() -> Self {
        Self::with_threads(
            PROBER_THREADS.load(Ordering::Relaxed),
            PROBER_TABLE_BITS.load(Ordering::Relaxed) as u32,
        )
    }

    /// Threads this prober answers a question with.
    pub fn threads(&self) -> usize {
        self.par_threads.len() + 1
    }

    /// CPU seconds the helper threads of this (main) prober consumed since
    /// they started (`Some(0.0)` for a prober without helpers), or `None`
    /// where a thread's clock cannot be read through its handle
    /// ([`crate::cpu::joined_thread_cpu_seconds`]: macOS).  Measurement only,
    /// for the catalog records' `cpuMs` ([`crate::record_clock`]).
    pub fn helper_cpu_seconds(&self) -> Option<f64> {
        self.par_threads
            .iter()
            .map(crate::cpu::joined_thread_cpu_seconds)
            .sum()
    }

    /// Bytes of the shared tables of a parallel prober (0 for one thread).
    pub fn shared_table_bytes(&self) -> usize {
        self.par
            .as_ref()
            .map_or(0, |par| par.tt.bytes() + par.facts.bytes())
    }

    /// Lowers the item-3 caps (tests only).
    #[doc(hidden)]
    pub fn set_caps_for_tests(&mut self, table: usize, intervals: usize, dominance: usize) {
        self.caps = [table.max(1), intervals.max(1), dominance.max(1)];
    }

    /// Starts a new search for `attacker`: every table emptied, counters zeroed.
    pub fn reset(&mut self, attacker: u8) {
        self.table.clear();
        self.intervals.clear();
        self.dominance.clear();
        self.futility.clear();
        self.futile_buf.clear();
        self.kids.clear();
        self.buf.buf.clear();
        self.attacker = attacker;
        self.aborted = false;
        self.stats = CdfpnStats::default();
        self.par_leaves.clear();
        if let Some(par) = self.par.as_ref().map(Arc::clone) {
            debug_assert_eq!(self.par_index, 0, "only the main prober is reset");
            // The helpers must be out of every question before the tables
            // are cleared (a new epoch, not a write): with `wait` the main
            // prober waited at the end of each question; otherwise here.
            par.finish(self.par_q.generation);
            Self::wait_helpers(&par);
            assert!(
                !par.failed.load(Ordering::Acquire),
                "a helper of the parallel prober panicked (the question is void)"
            );
            // Their pending leaves and this thread's go through now (no
            // slot is held).
            self.retry_leaves(true);
            self.par_leaves.clear();
            par.tt.clear();
            par.facts.clear();
            par.epoch.fetch_add(1, Ordering::AcqRel);
            par.prof.print(&par, self.par_threads.len() + 1);
        }
    }

    pub fn stats(&self) -> CdfpnStats {
        self.stats
    }

    /// Whether `pos` (already normalised, side to move either colour; the
    /// attacker is the one [`ClassicalProber::reset`] set) is mate within
    /// `bound` rule-F plies: `MATE`, `REFUTED`, or `UNKNOWN` when `budget` more
    /// expanded nodes did not settle it.  The tables carry over between calls.
    pub fn mate_within(&mut self, pos: &mut Position, bound: u32, budget: u64) -> Outcome {
        self.budget = budget;
        self.aborted = false;
        let node = Node {
            kind: if pos.side == self.attacker {
                Kind::Attack
            } else {
                Kind::Defense
            },
            bound,
            drop: Move::drop(0, 0),
        };
        if self.par.is_some() {
            return self.mate_within_par(pos, node, budget);
        }
        let before = self.stats.nodes;
        self.par_q.generation += 1;
        let (pn, dn, _) = self.mid(pos, node, INF, INF);
        if self.stats.nodes - before >= PAR_JOIN_AFTER && trace_on() {
            eprintln!(
                "pdfpn serial: question {} bound {} nodes {} verdict {}",
                self.par_q.generation,
                bound,
                self.stats.nodes - before,
                if pn == 0 {
                    "mate"
                } else if dn == 0 {
                    "refuted"
                } else {
                    "unknown"
                }
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

    /// Rule G clause (d) (`PIPELINE_VERSION` 0.16.0): whether the defense
    /// node `pos` (defender to move, already normalised) is still mate within
    /// `bound` rule-F plies **against the one reply `reply`** -- `MATE` when
    /// the reply does not refute (rule F2 proves it futile, or the child the
    /// classical AND node has for it is proven: a real attack node at
    /// `bound - 1`, or the credit line's `Gate(reply, bound)`), `REFUTED`
    /// when it refutes (that child is disproven), `UNKNOWN` when `budget`
    /// more expanded nodes did not settle it.
    ///
    /// The node is expanded exactly as [`ClassicalProber::mate_within`]
    /// expands it, so `pos` is mate within `bound` iff every legal reply
    /// answers `MATE` here: the replies answering `REFUTED` are the set of
    /// replies that refute the check within the bound, a fact about the
    /// position and the bound alone.  Only a prober on its own tables (not
    /// [`ClassicalProber::with_threads`]).
    pub fn reply_holds(
        &mut self,
        pos: &mut Position,
        bound: u32,
        reply: Move,
        budget: u64,
    ) -> Outcome {
        assert!(self.par.is_none(), "reply_holds runs on one thread");
        debug_assert_ne!(pos.side, self.attacker, "a defense node");
        self.budget = budget;
        self.aborted = false;
        let kbase = self.kids.len();
        let child = match self.expand_defense(pos, bound) {
            Expanded::Terminal((pn, _, _)) => {
                self.kids.truncate(kbase);
                if pn == 0 {
                    // No reply, or every reply futile (rule F2): nothing refutes.
                    return Outcome::Mate;
                }
                // Not mate without children: at bound 0 (or on a credit line
                // below bound 2) every reply refutes except a drop rule F2
                // proves futile, which is no resistance at any bound.
                return if self.futile_reply(pos, reply) {
                    Outcome::Mate
                } else {
                    Outcome::Refuted
                };
            }
            Expanded::Children { .. } => {
                self.kids[kbase..]
                    .iter()
                    .copied()
                    .find(|kid| match kid.node.kind {
                        Kind::Attack => kid.len == 1 && kid.path[0] == reply,
                        Kind::Gate => kid.node.drop == reply,
                        _ => false,
                    })
            }
        };
        self.kids.truncate(kbase);
        // A legal reply with no child is one rule F2 skipped as futile.
        let Some(child) = child else {
            return Outcome::Mate;
        };
        let (pn, dn, _) = if child.len == 1 {
            let undo = pos.make(reply);
            let numbers = self.mid(pos, child.node, INF, INF);
            pos.unmake(reply, undo);
            numbers
        } else {
            self.mid(pos, child.node, INF, INF)
        };
        if pn == 0 {
            Outcome::Mate
        } else if dn == 0 {
            Outcome::Refuted
        } else {
            Outcome::Unknown
        }
    }

    /// Whether rule F2 proves the reply `reply` of the defense node `pos`
    /// futile (the classical reading's per-drop verdict, as
    /// [`ClassicalProber::expand_defense`] reads it).
    fn futile_reply(&mut self, pos: &mut Position, reply: Move) -> bool {
        if !reply.is_drop() {
            return false;
        }
        let safety = Safety::new(pos);
        let range = self.buf.gen_legal_with(pos, true, &safety);
        Move::sort_usi(&mut self.buf.buf[range.clone()]);
        let index = self.buf.buf[range.clone()].iter().position(|&m| m == reply);
        let fbase = self.futile_buf.len();
        let classical = self.futility.classify_on(
            &mut self.buf,
            self.attacker,
            pos,
            range.clone(),
            &mut self.futile_buf,
            safety.check_line(),
        );
        let futile =
            classical && index.is_some_and(|index| self.futile_buf[fbase + index].is_some());
        self.futile_buf.truncate(fbase);
        self.buf.buf.truncate(range.start);
        futile
    }

    /// Entries of the transposition and interval tables (profiling; a
    /// parallel prober reports its shared tables' occupied slots).
    pub fn table_sizes(&self) -> [usize; 2] {
        match self.par.as_ref() {
            None => [self.table.len(), self.intervals.len()],
            Some(par) => [par.tt.occupied(), par.facts.occupied()],
        }
    }
}

/// What [`entry_probe`] found for one diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryProbe {
    /// `MATE` (mate within the bound, `mate_ply` its rule-F length),
    /// `REFUTED` (no mate within the bound, a proof) or `UNKNOWN` (the node
    /// budget ran out before the verdict or the length was settled).
    pub outcome: Outcome,
    /// The rule-F mate length, only with `MATE`.
    pub mate_ply: Option<u32>,
    /// Prober nodes spent on every question of this diagram.
    pub nodes: u64,
    /// The budget ran out (`UNKNOWN` only).
    pub budget_exhausted: bool,
}

/// The RL entry (owner decision 2026-10-10, `docs/experiments/ply13/README.md`
/// §7 and §12): "is `position` mate within `max_ply` rule-F plies, and how
/// long is the shortest mate", answered by the classical prober alone.
///
/// The root is normalised exactly as the publishing gate's fast path
/// normalises it (`fast_validate` in [`crate::reduce`]): the side to move is
/// the attacker and [`crate::strict_material::normalize`] hands the defender
/// every remaining piece.  The prober is reset for that attacker, so the
/// answer depends on the diagram alone (never on what the worker solved
/// before), and every question shares the one `budget`:
///
/// 1. "mate within `max_ply`?" -- `REFUTED` and `UNKNOWN` end here;
/// 2. on `MATE`, "mate within `b`?" for `b` = 1, 3, ... below `max_ply`, in
///    that order; the first `MATE` is the length, and none means `max_ply`
///    itself.  The verdict is monotone in the bound, so this is the shortest
///    mate.  A budget cut while climbing is `UNKNOWN`, never a guessed length.
///
/// With a sufficient budget the verdict and the length are the exact
/// solver's under rule F classical at that bound (the prober's contract), so
/// the publishing gate at `--max-ply L` finds the same mate.
pub fn entry_probe(
    prober: &mut ClassicalProber,
    position: &Position,
    max_ply: u32,
    budget: u64,
) -> EntryProbe {
    let attacker = position.side;
    let root = crate::strict_material::normalize(position, attacker);
    prober.reset(attacker);
    let ask = |prober: &mut ClassicalProber, bound: u32| {
        let left = budget.saturating_sub(prober.stats().nodes);
        let mut work = root.clone();
        prober.mate_within(&mut work, bound, left)
    };
    let settled = |prober: &ClassicalProber, outcome: Outcome, mate_ply: Option<u32>| EntryProbe {
        outcome,
        mate_ply,
        nodes: prober.stats().nodes,
        budget_exhausted: outcome == Outcome::Unknown,
    };
    match ask(prober, max_ply) {
        Outcome::Mate => {}
        other => return settled(prober, other, None),
    }
    // The attacker is the side to move, so every mate length is odd.
    let mut bound = 1;
    while bound < max_ply {
        match ask(prober, bound) {
            Outcome::Mate => return settled(prober, Outcome::Mate, Some(bound)),
            Outcome::Refuted => bound += 2,
            Outcome::Unknown => return settled(prober, Outcome::Unknown, None),
        }
    }
    settled(prober, Outcome::Mate, Some(max_ply))
}

/// The kind of the refutation [`entry_probe`] just answered (`probe --engine
/// cdfpn`'s `disproofKind`, owner request 2026-11-16): call it right after
/// `entry_probe(prober, position, max_ply, _)` returned `REFUTED`, on the same
/// prober (its tables still hold that question).  Measurement only: it
/// reads the tables and changes no verdict, length or node count
/// ([`ClassicalProber::disproof_kind`]).
pub fn entry_disproof_kind(
    prober: &mut ClassicalProber,
    position: &Position,
    max_ply: u32,
) -> DisproofKind {
    let root = crate::strict_material::normalize(position, position.side);
    prober.disproof_kind(&root, max_ply)
}
