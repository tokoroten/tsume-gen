//! Depth-limited df-pn mate **prober** (`docs/compute-plan.md` C9).
//!
//! # What this is, and what it is not
//!
//! [`crate::solver::Solver`] is and stays the verdict authority: every emitted
//! puzzle's outcome, `stats.nodes`, solution graph and rule A-G judgement come
//! from the bounded exact AND/OR search, and the TS<->Rust parity contract
//! (`scripts/diff-solve.ts`, `diff-reduce.ts`, `diff-movegen.ts`) is a statement
//! about *that* engine.  Nothing in this module is ever consulted for a published
//! verdict, and nothing here has a TypeScript counterpart -- it is a Rust-only
//! tooling stage like `extend`.
//!
//! This module answers exactly one question, fast:
//!
//! > is there a forced mate within `N` plies from this position under the strict
//! > tsume convention, and what is one line?
//!
//! `distill` uses it to make its per-step keep/discard decisions and to find the
//! source's mate length on diagrams where the exact solver runs out of budget;
//! `mine` uses it to detect candidates on the same kind of dense board.  Both
//! still hand every *recorded* diagram to the exact solver.
//!
//! # Semantics, and why the two engines cannot disagree
//!
//! The node set is the exact solver's node set, ply for ply:
//!
//! * an ATTACK node (attacker to move) expands **only legal checking moves**; with
//!   `remaining == 0`, or with no checking move, it is **disproven**;
//! * a DEFENSE node expands **every legal reply** (drops from the strict hand
//!   included, [`crate::strict_material`]); no legal reply while in check is
//!   **proven** (mate), no legal reply while not in check is a stalemate and is
//!   **disproven**, and a reply with `remaining == 0` is **disproven**.
//!
//! So, with the same attacker, the same strict-material normalisation and a
//! budget large enough for both, "proven at bound `N`" is precisely
//! `Solver::solve(.., max_ply = N).outcome == MATE` and "disproven at bound `N`"
//! is precisely `REFUTED` -- **in the rule-F mode the prober runs in**
//! ([`ProbeOptions::useless`], below).  Only the traversal order differs:
//! best-first by proof/disproof number instead of depth-first in USI order.
//! Under a finite budget the two can still differ in which one gives up first.
//! `tests/dfpn.rs` asserts the agreement on every fixture the exact solver has
//! plus 300 mined positions, in both modes.
//!
//! ## Rule F
//!
//! Rule F (無駄合, `docs/rules.md`) has two parts, and they touch this question
//! differently:
//!
//! * **F1** (`Prune`) changes a mate's *length*, never its *existence* at a bound:
//!   a mating defence node still needs every legal reply to mate.  The prober never
//!   models it; under `Keep` and `Prune` its verdict at a bound is the exact
//!   solver's.
//! * **F2** (`Classical`, the composition reading since `PIPELINE_VERSION` 0.10.0)
//!   *does* change existence at a bound: a defence node whose every reply is a
//!   futile interposition drop is mate at **any** remaining bound, 0 included, and
//!   its futile drops are never searched.  A prober built with
//!   `.useless(UselessInterposition::Classical)` applies exactly that, through the
//!   very same [`crate::solver::Futility`] test the exact solver uses: at a
//!   defence node the futile drops are classified *before* the bound is consulted,
//!   a node whose replies are all futile is proven, and otherwise only the
//!   non-futile replies become children.  Under `PIPELINE_VERSION` 0.10.0 its
//!   verdict at a bound was the exact solver's `Classical` verdict at that bound.
//!   Since 0.11.0 the exact `Classical` search also counts its bound in rule-F
//!   plies -- F1's 無駄合 and their captures consume none of it, through the
//!   credit search -- which this prober does not reproduce (stage 1 of
//!   `docs/experiments/rule-f-bound/README.md` §4.4).  Its proofs stay sound for the
//!   classical gate (a mate it proves within a bound is a classical mate within
//!   it), its disproofs are not conclusive, and neither are a `Keep` prober's
//!   ([`crate::exact::dfpn_disproof_is_conclusive`]).
//!
//! **Lengths.**  [`Prober::minimal_mate_ply`] reports the smallest *bound* that
//! proves.  Under `Keep` that is the real-ply minimum -- the `keep` length, which
//! is what [`crate::exact::minimum_mate_validate`] reports with
//! [`crate::exact::TsumeRules::off`].  Under `Classical` it is an upper bound on
//! the rule-F mate distance, which since `PIPELINE_VERSION` 0.11.0 is the bound
//! [`crate::exact::minimum_mate_validate_with_rules`] stops at: F1 can make the
//! distance shorter than the smallest bound this prober proves (a mate in M whose
//! 無駄合 need a real-ply bound P > M).  A caller that needs the rule-F distance
//! asks the exact solver at the bound this prober found.
//! Rules A, B, C and G are none of this module's business.
//!
//! # Algorithm
//!
//! df-pn (Nagai 2002) in its multiple-iterative-deepening form: one recursive
//! `MID` routine driven by a `(thpn, thdn)` threshold pair, with the child
//! thresholds
//!
//! ```text
//! OR  node: thpn_child = min(thpn, pn2 + 1)          thdn_child = thdn - dn + dn_best
//! AND node: thpn_child = thpn - pn + pn_best         thdn_child = min(thdn, dn2 + 1)
//! ```
//!
//! where `pn2`/`dn2` is the second-smallest child number.  The `+ 1` is the plain
//! increment, **not** the ε-trick (`min(thpn, ceil(pn2 * (1 + eps)))`): the ε-trick
//! trades a little re-expansion for a little over-search, and with a hard node
//! budget the plain increment is the variant whose behaviour is easiest to reason
//! about.  Unvisited nodes start at `(1, 1)`.
//!
//! ## Transposition table and GHI
//!
//! The table is keyed by the tuple `(remaining, 128-bit Zobrist key)` -- the very
//! key [`crate::solver`] uses, and for the same reason.  A node's value then
//! answers only "mate within `remaining` plies from here", which is a function of
//! the position and the bound alone (the rule-F mode is fixed per call), so it
//! cannot be invalidated by the path that reached it: that is the
//! graph-history-interaction problem, and including `remaining` in the key is the
//! cure `docs/compute-plan.md` C9 requires.  Because `remaining` strictly
//! decreases along every edge, a repetition in the position graph maps to
//! *distinct* keys at distinct depths, the recursion is bounded by the depth
//! limit, and no path-based cycle check is needed.  (Consequence, stated plainly:
//! a mate that only exists by forcing a repetition is not something this prober
//! -- or the exact solver -- can express, and that is the frozen contract of
//! `docs/rules.md` §"Bounded exact mate search".)  As in the exact solver, two
//! positions with the same 128-bit Zobrist key are not told apart.
//!
//! The table and the F2 futility memos are cleared at the start of every public
//! entry point, so one call's node count never depends on what the same
//! [`Prober`] was asked before.  That is what lets `distill` add prober nodes to
//! its per-source budget and stay deterministic per record.  The F2 test is not a
//! search: like in the exact solver it creates no node and costs no budget.
//!
//! ## Budget
//!
//! Node budget only, never time (spec §35).  When it is spent the search is
//! abandoned and the result is [`Verdict::Unknown`]; an `Unknown` is never reported
//! as a mate, exactly as in the exact solver.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::movegen::{checking_moves_sorted, legal_moves_sorted, MoveBuf};
use crate::position::{Position, Usi};
use crate::solver::{Futility, KeyHasher, SearchRules, UselessInterposition};
use crate::strict_material;

/// Proof/disproof number standing for "impossible".
pub const INF: u32 = u32::MAX;
/// Largest value a *sum* of child numbers may reach, so a saturated sum can never
/// be mistaken for [`INF`] (which is a proof, not an estimate).
const NEAR_INF: u32 = u32::MAX - 1;

/// Highest ply bound a probe accepts.
pub const MAX_PROBE_PLY: u32 = 255;

/// The most entries the prober's table keeps (solver v3, item 3): 7/8 of 2^22,
/// the largest count a 2^22-bucket table holds without growing (about 140 MB).
/// Past it `Prober::collect` drops the cheaper half.  Dropping an entry loses
/// work but never a verdict, and which entries go depends only on the entries,
/// so the search stays deterministic.  (It replaces the clear-between-bounds of
/// a table past 4,000,000 entries.)
pub const DFPN_TABLE_CAP: usize = (1 << 22) / 8 * 7;

/// What a probe concluded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// A forced mate within the bound exists (`Solver` would report `MATE`).
    Proven,
    /// No forced mate within the bound (`Solver` would report `REFUTED`).
    Disproven,
    /// The node budget was spent first.  Never a mate.
    Unknown,
}

impl Verdict {
    pub const fn as_str(self) -> &'static str {
        match self {
            Verdict::Proven => "PROVEN",
            Verdict::Disproven => "DISPROVEN",
            Verdict::Unknown => "UNKNOWN",
        }
    }
}

/// Which engine answers a tooling stage's existence/length questions.
///
/// The default is [`ProbeEngine::Exact`], which is the behaviour every stage had
/// before this module existed, byte for byte.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ProbeEngine {
    #[default]
    Exact,
    Dfpn,
}

impl ProbeEngine {
    pub const fn as_str(self) -> &'static str {
        match self {
            ProbeEngine::Exact => "exact",
            ProbeEngine::Dfpn => "dfpn",
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "exact" => Some(ProbeEngine::Exact),
            "dfpn" => Some(ProbeEngine::Dfpn),
            _ => None,
        }
    }
    pub const fn is_dfpn(self) -> bool {
        matches!(self, ProbeEngine::Dfpn)
    }
}

/// Probe configuration: the exact solver's [`SearchRules`] (attacker, strict
/// material, rule F), a ply bound and a node budget.
///
/// Of rule F only [`UselessInterposition::Classical`] changes anything: it
/// applies F2 (see the module docs).  `Keep` (the default) and `Prune` probe the
/// bare contract, which is every earlier probe, bit for bit.
#[derive(Clone, Copy, Debug)]
pub struct ProbeOptions {
    pub rules: SearchRules,
    /// Ply bound.  A node at this depth whose defender is not mated is disproven.
    pub max_ply: u32,
    /// Hard node budget for the whole call (every bound of
    /// [`Prober::minimal_mate_ply`] and its PV extraction together).
    pub node_budget: u64,
}

impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            rules: SearchRules::default(),
            max_ply: crate::exact::DEFAULT_MAX_PLY,
            node_budget: 1_000_000,
        }
    }
}

impl ProbeOptions {
    pub fn max_ply(mut self, max_ply: u32) -> Self {
        self.max_ply = max_ply;
        self
    }
    pub fn node_budget(mut self, node_budget: u64) -> Self {
        self.node_budget = node_budget;
        self
    }
    pub fn attacker(mut self, attacker: u8) -> Self {
        self.rules.attacker = Some(attacker);
        self
    }
    pub fn strict_material(mut self, strict: bool) -> Self {
        self.rules.strict_material = strict;
        self
    }
    pub fn useless(mut self, mode: UselessInterposition) -> Self {
        self.rules.useless = mode;
        self
    }
}

/// One fixed-bound probe.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub verdict: Verdict,
    /// The bound the question was asked at.
    pub max_ply: u32,
    /// df-pn nodes expanded.  **Not** comparable with `SolveStats::nodes`: a
    /// different traversal of the same tree, and never published metadata.
    pub nodes: u64,
    pub budget_exhausted: bool,
}

/// [`Prober::minimal_mate_ply`]: the smallest bound `N` that proves a mate.
#[derive(Clone, Debug)]
pub struct MinimalMate {
    pub verdict: Verdict,
    /// `Some(N)` exactly when `verdict == Proven`.  Under `Keep` this is the
    /// **real-ply** minimum (rule F not applied); under `Classical` the smallest
    /// proving bound, which can exceed the rule-F distance -- see the module docs.
    pub mate_ply: Option<u32>,
    /// One shortest line, `mate_ply` plies long: at every attack node the
    /// lexicographically smallest correct move, at every defence node the reply
    /// that resists longest with the same tie-break -- [`crate::exact::usi_first_line`]
    /// read with rule F off.  Under `Classical` futile drops are never a defence
    /// node's choice.  Since 2026-09-19 this is no longer the designated main line
    /// of `docs/rules.md`, which prefers replies that are not 変同駒余り; the
    /// prober has no graph to judge that on, and its `pv` is a measurement column
    /// only.  Empty when the root is not proven, or truncated when the budget ran
    /// out during extraction.
    pub pv: Vec<Usi>,
    pub nodes: u64,
    pub budget_exhausted: bool,
}

/// Saturating sum that never reaches [`INF`] by accident.
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

/// `(remaining, 128-bit Zobrist key)` folded into 128 bits, as the exact
/// solver's table does ([`crate::solver::table_key`]).
type Key = [u64; 2];

/// One table entry: the node's proof and disproof numbers, and the df-pn nodes
/// spent under it (saturating), which is what the garbage collection keeps.
#[derive(Clone, Copy, Debug)]
struct Entry {
    pn: u32,
    dn: u32,
    work: u32,
}

type Table = HashMap<Key, Entry, BuildHasherDefault<KeyHasher>>;

/// A reusable df-pn prober.  Keeping one instance across many positions reuses the
/// move stack and table allocation; the table itself is cleared at every entry
/// point, so results and node counts never depend on what came before.
pub struct Prober {
    buf: MoveBuf,
    table: Table,
    /// Child position hashes, one frame per node on the search stack (the
    /// `prune_buf` pattern of [`crate::solver::Solver`]).
    kids: Vec<u128>,
    /// F2 verdicts of the replies of the defence node being expanded.
    futile_buf: Vec<Option<u32>>,
    /// The F2 test, shared with the exact solver; memos cleared per entry point.
    futility: Futility,
    attacker: u8,
    useless: UselessInterposition,
    nodes: u64,
    budget: u64,
    aborted: bool,
    /// Garbage collections of the table (measurement only).
    collections: u64,
    /// [`DFPN_TABLE_CAP`] (lowered only by tests).
    table_cap: usize,
}

impl Default for Prober {
    fn default() -> Self {
        Self::new()
    }
}

impl Prober {
    pub fn new() -> Self {
        Self {
            buf: MoveBuf::new(),
            table: Table::default(),
            kids: Vec::new(),
            futile_buf: Vec::new(),
            futility: Futility::new(),
            attacker: crate::piece::BLACK,
            useless: UselessInterposition::Keep,
            nodes: 0,
            budget: 0,
            aborted: false,
            collections: 0,
            table_cap: DFPN_TABLE_CAP,
        }
    }

    /// Lowers [`DFPN_TABLE_CAP`] so a test can make the garbage collection run
    /// on a small probe.
    #[doc(hidden)]
    pub fn set_table_cap_for_tests(&mut self, cap: usize) {
        self.table_cap = cap.max(1);
    }

    /// Garbage collections of the table since the prober was built.
    pub fn collections(&self) -> u64 {
        self.collections
    }

    /// Stores a node's numbers, adding the `work` it cost; a new key first
    /// makes room when the table is full ([`DFPN_TABLE_CAP`]).
    fn store(&mut self, key: Key, (pn, dn): (u32, u32), work: u64) {
        if self.table.len() >= self.table_cap && !self.table.contains_key(&key) {
            self.collect();
        }
        let entry = self.table.entry(key).or_insert(Entry { pn, dn, work: 0 });
        entry.pn = pn;
        entry.dn = dn;
        entry.work = entry
            .work
            .saturating_add(work.min(u64::from(u32::MAX)) as u32);
    }

    /// The table's deterministic garbage collection: drops every entry in the
    /// lowest classes -- unsolved before solved, then by the power-of-two class
    /// of its work -- the fewest classes that hold at least half of the
    /// entries.  The choice depends on the entries alone, never on where they
    /// sit in the table.
    fn collect(&mut self) {
        let class = |entry: &Entry| -> usize {
            let solved = entry.pn == 0 || entry.dn == 0;
            (u32::BITS - entry.work.leading_zeros()) as usize + if solved { 33 } else { 0 }
        };
        let mut classes = [0usize; 66];
        for entry in self.table.values() {
            classes[class(entry)] += 1;
        }
        let target = self.table.len().div_ceil(2);
        let mut dropped = 0usize;
        let mut limit = 0usize;
        for (index, &count) in classes.iter().enumerate() {
            dropped += count;
            limit = index;
            if dropped >= target {
                break;
            }
        }
        self.table.retain(|_, entry| class(entry) > limit);
        self.collections += 1;
    }
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    fn begin(&mut self, attacker: u8, options: &ProbeOptions) {
        self.table.clear();
        self.kids.clear();
        self.futile_buf.clear();
        self.futility.clear();
        self.buf.buf.clear();
        self.attacker = attacker;
        self.useless = options.rules.useless;
        self.nodes = 0;
        self.budget = options.node_budget;
        self.aborted = false;
    }

    #[inline]
    fn classical(&self) -> bool {
        self.useless == UselessInterposition::Classical
    }

    fn root_of(&self, position: &Position, options: &ProbeOptions) -> (Position, u8) {
        let attacker = options.rules.attacker.unwrap_or(position.side);
        let root = if options.rules.strict_material {
            strict_material::normalize(position, attacker)
        } else {
            position.clone()
        };
        (root, attacker)
    }

    /// Is there a forced mate within `options.max_ply` plies?
    pub fn prove(&mut self, position: &Position, options: &ProbeOptions) -> ProbeResult {
        let _prof = crate::prof::engine(crate::prof::Engine::Dfpn);
        assert!(
            options.max_ply <= MAX_PROBE_PLY,
            "df-pn ply bound out of range"
        );
        let (root, attacker) = self.root_of(position, options);
        self.begin(attacker, options);
        let mut work = root;
        let verdict = match self.resolve(&mut work, options.max_ply) {
            Some(true) => Verdict::Proven,
            Some(false) => Verdict::Disproven,
            None => Verdict::Unknown,
        };
        ProbeResult {
            verdict,
            max_ply: options.max_ply,
            nodes: self.nodes,
            budget_exhausted: self.aborted,
        }
    }

    /// The smallest bound `N` that proves a mate, by probing `N = 1, 3, 5, …` from
    /// an attacker-to-move root, or `N = 0, 2, 4, …` when the defender is to move.
    ///
    /// A mate measured from an attacker-to-move root always falls on an attacker
    /// move, so only odd bounds can be minimal there (and only even ones from a
    /// defender-to-move root).  The first bound that proves is the minimum,
    /// because every smaller bound of that parity has already been disproven --
    /// the same argument [`crate::exact::minimum_mate_validate`] makes, and the
    /// reason this is a drop-in replacement for its length half.  See the module
    /// docs for what the number means under `Classical`.
    pub fn minimal_mate_ply(&mut self, position: &Position, options: &ProbeOptions) -> MinimalMate {
        let _prof = crate::prof::engine(crate::prof::Engine::Dfpn);
        assert!(
            options.max_ply <= MAX_PROBE_PLY,
            "df-pn ply bound out of range"
        );
        let (root, attacker) = self.root_of(position, options);
        self.begin(attacker, options);
        let mut work = root.clone();
        let mut verdict = Verdict::Disproven;
        let mut mate_ply = None;
        let mut bound = if root.side == attacker { 1 } else { 0 };
        while bound <= options.max_ply {
            match self.resolve(&mut work, bound) {
                Some(true) => {
                    verdict = Verdict::Proven;
                    mate_ply = Some(bound);
                    break;
                }
                Some(false) => {}
                None => {
                    verdict = Verdict::Unknown;
                    break;
                }
            }
            bound += 2;
        }
        let pv = match mate_ply {
            Some(ply) => self.extract_pv(&root, ply),
            None => Vec::new(),
        };
        MinimalMate {
            verdict,
            mate_ply,
            pv,
            nodes: self.nodes,
            budget_exhausted: self.aborted,
        }
    }

    /// One complete df-pn search at `bound`: `Some(true)` proven, `Some(false)`
    /// disproven, `None` out of budget.
    fn resolve(&mut self, pos: &mut Position, bound: u32) -> Option<bool> {
        if self.aborted {
            return None;
        }
        let (pn, dn) = self.mid(pos, bound, INF, INF);
        if pn == 0 {
            Some(true)
        } else if dn == 0 {
            Some(false)
        } else {
            None
        }
    }

    /// df-pn `MID`.  Returns the node's `(pn, dn)` once one of them reaches its
    /// threshold, and leaves that pair in the table.
    fn mid(&mut self, pos: &mut Position, remaining: u32, th_pn: u32, th_dn: u32) -> (u32, u32) {
        let key: Key = crate::solver::table_key(remaining, pos.hash());
        let entry = self
            .table
            .get(&key)
            .map_or((1, 1), |entry| (entry.pn, entry.dn));
        if entry.0 >= th_pn || entry.1 >= th_dn {
            return entry;
        }
        if self.aborted {
            return entry;
        }
        if self.nodes >= self.budget {
            self.aborted = true;
            return entry;
        }
        let started = self.nodes;
        self.nodes += 1;

        let attack = pos.side == self.attacker;
        if attack && remaining == 0 {
            // No attacker ply left: `Solver` reports REFUTED here too.
            let result = (INF, 0);
            self.store(key, result, self.nodes - started);
            return result;
        }
        if !attack && remaining == 0 && self.buf.has_legal_board_move(pos) {
            // A defense node at the bound with a legal board move: that reply is
            // no interposition drop, so F2 cannot make the node mate, and the
            // verdict below is (INF, 0) without the reply list.
            let result = (INF, 0);
            self.store(key, result, self.nodes - started);
            return result;
        }

        let base = self.buf.buf.len();
        let kbase = self.kids.len();
        let mut count = self.expand(pos, attack);
        let mut terminal = None;
        if count == 0 {
            terminal = Some(if attack {
                // No legal checking move.
                (INF, 0)
            } else if pos.is_in_check(pos.side) {
                // Mate: the frozen terminal of `docs/rules.md`.
                (0, INF)
            } else {
                // Stalemate is not mate; the exact solver calls it REFUTED.
                (INF, 0)
            });
        } else if !attack && self.classical() {
            // F2, decided before the bound like in the exact solver: a futile
            // drop is not resistance, and a check answered only by futile drops
            // is mate at any remaining bound.
            match self.drop_futile_replies(pos, base, kbase, count) {
                Some(left) => count = left,
                None => terminal = Some((0, INF)),
            }
        }
        if terminal.is_none() && remaining == 0 {
            // A defense node with a (non-futile) legal answer and no attacker ply
            // left.
            terminal = Some((INF, 0));
        }
        if let Some(result) = terminal {
            self.buf.buf.truncate(base);
            self.kids.truncate(kbase);
            self.store(key, result, self.nodes - started);
            return result;
        }

        let child_remaining = remaining - 1;
        let result = loop {
            // Recompute this node's numbers from the table entries of its children.
            let mut best = 0usize;
            let mut best_pair = (1u32, 1u32);
            let (pn, dn);
            if attack {
                // OR: pn = min child pn, dn = sum child dn.
                let (mut min1, mut min2) = (INF, INF);
                let mut sum = 0u32;
                for index in 0..count {
                    let pair = self.child(kbase + index, child_remaining);
                    sum = add(sum, pair.1);
                    if pair.0 < min1 {
                        min2 = min1;
                        min1 = pair.0;
                        best = index;
                        best_pair = pair;
                    } else if pair.0 < min2 {
                        min2 = pair.0;
                    }
                }
                pn = min1;
                dn = sum;
                if self.aborted || pn >= th_pn || dn >= th_dn {
                    break (pn, dn);
                }
                let child_th = (
                    th_pn.min(bump(min2)),
                    if th_dn == INF {
                        INF
                    } else {
                        th_dn - dn + best_pair.1
                    },
                );
                self.descend(pos, base + best, child_remaining, child_th);
            } else {
                // AND: pn = sum child pn, dn = min child dn.
                let (mut min1, mut min2) = (INF, INF);
                let mut sum = 0u32;
                for index in 0..count {
                    let pair = self.child(kbase + index, child_remaining);
                    sum = add(sum, pair.0);
                    if pair.1 < min1 {
                        min2 = min1;
                        min1 = pair.1;
                        best = index;
                        best_pair = pair;
                    } else if pair.1 < min2 {
                        min2 = pair.1;
                    }
                }
                pn = sum;
                dn = min1;
                if self.aborted || pn >= th_pn || dn >= th_dn {
                    break (pn, dn);
                }
                let child_th = (
                    if th_pn == INF {
                        INF
                    } else {
                        th_pn - pn + best_pair.0
                    },
                    th_dn.min(bump(min2)),
                );
                self.descend(pos, base + best, child_remaining, child_th);
            }
        };

        self.buf.buf.truncate(base);
        self.kids.truncate(kbase);
        if !self.aborted {
            self.store(key, result, self.nodes - started);
        }
        result
    }

    /// F2 at the defence node `pos`, whose `count` replies sit at `base` on the
    /// move stack and `kbase` on [`Prober::kids`]: removes the futile drops from
    /// both, in place and in order, and returns how many replies are left --
    /// `None` when every reply was futile, i.e. the node is mate.  A node that is
    /// not all interposition drops is left untouched ([`Futility::classify`]).
    fn drop_futile_replies(
        &mut self,
        pos: &mut Position,
        base: usize,
        kbase: usize,
        count: usize,
    ) -> Option<usize> {
        let fbase = self.futile_buf.len();
        if !self.futility.classify(
            &mut self.buf,
            self.attacker,
            pos,
            base..base + count,
            &mut self.futile_buf,
        ) {
            return Some(count);
        }
        let mut write = 0usize;
        for read in 0..count {
            if self.futile_buf[fbase + read].is_none() {
                self.buf.buf[base + write] = self.buf.buf[base + read];
                self.kids[kbase + write] = self.kids[kbase + read];
                write += 1;
            }
        }
        self.futile_buf.truncate(fbase);
        self.buf.buf.truncate(base + write);
        self.kids.truncate(kbase + write);
        (write > 0).then_some(write)
    }

    /// Plays the move at `slot`, searches the child, takes the move back.
    fn descend(&mut self, pos: &mut Position, slot: usize, remaining: u32, th: (u32, u32)) {
        let m = self.buf.buf[slot];
        let undo = pos.make(m);
        self.mid(pos, remaining, th.0, th.1);
        pos.unmake(m, undo);
    }

    /// Appends this node's moves to the move stack and their child hashes to
    /// [`Prober::kids`], returning how many there are.
    ///
    /// Attack nodes keep only checking moves, which is the frozen contract of
    /// `docs/rules.md`; defense nodes keep every legal reply.  The order is
    /// [`crate::movegen`]'s generation order rather than USI order, a bishop's,
    /// rook's or pawn's promotion before its twin at an attack node (solver
    /// 1.7.0) -- df-pn picks its own child anyway, so the order changes only
    /// node counts, never the verdict.
    fn expand(&mut self, pos: &mut Position, attack: bool) -> usize {
        // Checks come in the same relative order `gen_legal` gives them.
        let range = if attack {
            let range = self.buf.gen_checks(pos, true);
            // A bishop's, rook's or pawn's promotion before its twin (solver
            // 1.7.0, [`Move::promotion_first`]).
            crate::position::Move::promotion_first(&mut self.buf.buf[range.clone()]);
            range
        } else {
            self.buf.gen_legal(pos, true)
        };
        for read in range.clone() {
            let m = self.buf.buf[read];
            let undo = pos.make(m);
            self.kids.push(pos.hash());
            pos.unmake(m, undo);
        }
        range.len()
    }

    #[inline]
    fn child(&self, slot: usize, remaining: u32) -> (u32, u32) {
        self.table
            .get(&crate::solver::table_key(remaining, self.kids[slot]))
            .map_or((1, 1), |entry| (entry.pn, entry.dn))
    }

    /// The F2-futile flags of `moves`, the legal replies of the defence node
    /// `pos`; all `false` unless the prober runs `Classical` and F2 applies.
    fn futile_flags(&mut self, pos: &mut Position, moves: &[crate::position::Move]) -> Vec<bool> {
        let mut flags = vec![false; moves.len()];
        if !self.classical() || moves.is_empty() {
            return flags;
        }
        let base = self.buf.buf.len();
        self.buf.buf.extend_from_slice(moves);
        let fbase = self.futile_buf.len();
        if self.futility.classify(
            &mut self.buf,
            self.attacker,
            pos,
            base..base + moves.len(),
            &mut self.futile_buf,
        ) {
            for (flag, verdict) in flags.iter_mut().zip(&self.futile_buf[fbase..]) {
                *flag = verdict.is_some();
            }
        }
        self.futile_buf.truncate(fbase);
        self.buf.buf.truncate(base);
        flags
    }

    /// Reads one shortest line out of the prober, given the proven mate length.
    ///
    /// At a node whose mate distance is `d`, a move is on the line exactly when its
    /// child has distance `d - 1`, i.e. the child is proven at bound `d - 1` and
    /// **disproven** at bound `d - 3`.  At an attack node that is the definition of
    /// a *correct* move, at a defence node the definition of the *longest*
    /// resistance; the first such move in USI order is taken, which is the same
    /// tie-break [`crate::exact::usi_first_line`] uses (not the 2026-09-19
    /// designated main line; see [`MinimalMate::pv`]).  Rule F1 is not applied,
    /// so on a diagram with 無駄合 branches this is the `keep`-mode main line;
    /// under `Classical` a futile drop is never a defence node's choice.  When the
    /// budget runs out in the middle -- either bound unresolved -- the line stops
    /// there rather than guess.
    fn extract_pv(&mut self, root: &Position, mate_ply: u32) -> Vec<Usi> {
        let mut pv = Vec::new();
        let mut pos = root.clone();
        let mut distance = mate_ply;
        'line: while distance > 0 {
            let attack = pos.side == self.attacker;
            let moves = if attack {
                checking_moves_sorted(&pos)
            } else {
                legal_moves_sorted(&pos)
            };
            let futile = if attack {
                vec![false; moves.len()]
            } else {
                self.futile_flags(&mut pos, &moves)
            };
            let mut chosen = None;
            for (&m, &futile) in moves.iter().zip(futile.iter()) {
                if futile {
                    continue;
                }
                let undo = pos.make(m);
                let within = self.resolve(&mut pos, distance - 1);
                let exact_length = match within {
                    Some(true) if distance < 3 => Some(true),
                    Some(true) => self.resolve(&mut pos, distance - 3).map(|shorter| !shorter),
                    Some(false) => Some(false),
                    None => None,
                };
                pos.unmake(m, undo);
                match exact_length {
                    Some(true) => {
                        chosen = Some(m);
                        break;
                    }
                    Some(false) => {}
                    // Out of budget: a move whose length is unknown is not put
                    // on the line.
                    None => break 'line,
                }
            }
            let Some(m) = chosen else {
                // Only reachable when the budget ran out mid-extraction.
                break;
            };
            pv.push(m.to_usi());
            pos.make(m);
            distance -= 1;
        }
        pv
    }
}

/// One fixed-bound probe with a fresh prober.
pub fn prove(position: &Position, max_ply: u32, node_budget: u64) -> ProbeResult {
    Prober::new().prove(
        position,
        &ProbeOptions::default()
            .max_ply(max_ply)
            .node_budget(node_budget),
    )
}

/// The smallest `N` with a forced mate in `N`, with a fresh prober.
pub fn minimal_mate_ply(position: &Position, max_ply: u32, node_budget: u64) -> MinimalMate {
    Prober::new().minimal_mate_ply(
        position,
        &ProbeOptions::default()
            .max_ply(max_ply)
            .node_budget(node_budget),
    )
}
