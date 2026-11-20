//! `extend`: the model-free reverse method (逆算法, owner policy "B").
//!
//! One source puzzle of mate length `k` goes in; every 2-ply predecessor
//! [`crate::unmove::grouped_predecessors`] finds comes out, each one put through
//! the ordinary publishing gate — `exact_validate_with_rules(TsumeRules::puzzle())`
//! at `max_ply = k + 2` with `target_mate_ply = k + 2` — so an emitted record is
//! either a strict-valid `k + 2` puzzle or a rejection that names why.
//!
//! This is step 2 of `docs/research/learned-reverse-method.md` §9: the baseline a
//! learned proposer has to beat.  No model orders the candidates here; every one of
//! them is solved.
//!
//! Records are reproducible from `(parent sourceId, check, reply)`: the parent's
//! whole `provenance.extra` map is carried over untouched, `parentSourceId` is set
//! in it, and the transformation string `extend:unmove-2ply:<check>,<reply>` names
//! the exact retraction.
//!
//! Origin provenance (`docs/legal-data.md` §3) travels the same way it does through
//! `distill`: a candidate is a derivation with one input, so it *copies* its
//! parent's `originClass`, `origins` and `rootSourceId` unchanged and appends one
//! typed `provenance.lineage` entry naming the parent, this tool and
//! [`EXTEND_VERSION`].  A puzzle reverse-extended from a YaneuraOu position
//! therefore still states `"origins": ["yaneuraou-2020-12-25"]`, which is the
//! condition the owner's 2026-09-13 decision attaches to using that data.  A parent
//! that states no origin produces no candidate at all: the set is never defaulted
//! to `own`.
//!
//! # Rule D: every accepted candidate is reduced
//!
//! The publishing gate checks rules A–G, but rule D (a published puzzle carries no
//! piece it can do without) is enforced by the `reduce` stage, not by the gate, and
//! a 2-ply retraction routinely adds material the mate does not need: an uncapture
//! puts a piece back on the board, an un-drop hands one back.  Until 2026-09-19
//! `extend` stopped at the gate, so its output — most of the 9- and 11-ply stock —
//! was never reduced, while `distill` output always was.
//!
//! So an accepted candidate now goes through [`crate::stages::reduce_stage`], the
//! very stage the pipeline runs, and then through the gate again, because a
//! diagram that lost pieces needs its own solution.  The reducer is given the
//! gate's own bound and budget (`max_ply = k + 2`, `node_budget`), so its entry
//! check is the verdict the gate just reached and cannot refuse the candidate for
//! want of search; the reducer keeps the target length, so the result is still a
//! `k + 2` puzzle.  Its seed derives from the candidate's `sourceId`, exactly as
//! in the stage, so the output stays reproducible and identical under `--order`
//! and `--probe dfpn`.  `ExtendOptions::reduce = false` (library only since 2026-10-22) writes the
//! pre-2026-09-19 bytes.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

use crate::dfpn::{ProbeEngine, ProbeOptions, Prober, Verdict};
use crate::exact::{
    dfpn_disproof_is_conclusive, dfpn_useless_mode, measure_main_line_yozume, TsumeRules,
};
use crate::jsonl::{PipelineRecord, Rejection};
use crate::position::{Move, Position, NO_SQUARE};
use crate::reduce::ReduceOptions;
use crate::sfen::parse_sfen;
use crate::solver::{Outcome, SolveOptions, Solver};
use crate::stages::{
    self, record_position, solve_stage_reported, validate_stage, ReduceAcceptedOutcome,
    SolveStageOptions, SolveStageReport, MISSING_ORIGIN_REASON,
};
use crate::unmove::{grouped_predecessors, PredecessorCounts, UnCheckKind, UnReplyKind};

/// Version stamped into `provenance.lineage[].version`, as `DISTILL_VERSION` is
/// for a distilled variant.
pub const EXTEND_VERSION: &str = "extend-unmove-2ply-v1";

/// `provenance.lineage[].tool` for a reverse-extended candidate.
pub const EXTEND_TOOL: &str = "extend";

/// Reasons that the reducer/distiller is *expected* to repair, so a candidate
/// rejected for nothing else is a "repairable" one (§9 step 4 of the design note).
const REPAIRABLE_REASONS: [&str; 3] =
    ["koma_amari", "inert_attacker_piece", "inert_defender_piece"];

/// Which records `extend` writes to stdout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emit {
    /// Only strict-valid `k + 2` puzzles.
    Accepted,
    /// Every solved candidate, accepted or rejected.
    All,
    /// Enumerate only: one lean row per candidate, **no search at all**.  This is
    /// the ranker's input (`tools/reverse-ranker/score.py`); nothing here decides
    /// acceptance, so no row is a puzzle and none carries a `solution`.
    Candidates,
}

impl Emit {
    /// Parses the `--emit` flag.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "accepted" => Some(Emit::Accepted),
            "all" => Some(Emit::All),
            "candidates" => Some(Emit::Candidates),
            _ => None,
        }
    }

    /// The flag spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Emit::Accepted => "accepted",
            Emit::All => "all",
            Emit::Candidates => "candidates",
        }
    }
}

/// A scored verification order, read from the `--order` JSONL file.
///
/// One row per scored candidate: `{"parentSourceId": …, "candidateIndex": i,
/// "score": s}`.  The file only *reorders* what the enumerator already produced —
/// it can never introduce a candidate, and it never decides acceptance.  That
/// stays with `exact_validate_with_rules`, so rules A–G hold whatever the model
/// says (`docs/rules.md`).
#[derive(Clone, Debug, Default)]
pub struct ExtendOrder {
    /// Free-form model id written to `provenance.extra.ranker.id`, e.g. the
    /// sha256 of the LightGBM `model.txt`.
    pub id: String,
    /// `parentSourceId` -> `candidateIndex` -> score.
    scores: BTreeMap<String, BTreeMap<usize, f64>>,
    /// Rows read, including ones naming a parent this run never sees.
    pub rows: usize,
}

impl ExtendOrder {
    /// Parses the whole `--order` file.  Blank lines are skipped; a row that does
    /// not carry the three fields is an error, because silently dropping it would
    /// change the verification order without saying so.
    pub fn parse(text: &str, id: &str) -> Result<Self> {
        let mut out = ExtendOrder {
            id: id.to_string(),
            ..ExtendOrder::default()
        };
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(line)
                .with_context(|| format!("--order line {}", number + 1))?;
            let parent = row
                .get("parentSourceId")
                .and_then(Value::as_str)
                .with_context(|| format!("--order line {}: parentSourceId", number + 1))?;
            let index = row
                .get("candidateIndex")
                .and_then(Value::as_u64)
                .with_context(|| format!("--order line {}: candidateIndex", number + 1))?;
            let score = row
                .get("score")
                .and_then(Value::as_f64)
                .with_context(|| format!("--order line {}: score", number + 1))?;
            out.scores
                .entry(parent.to_string())
                .or_default()
                .insert(index as usize, score);
            out.rows += 1;
        }
        Ok(out)
    }

    /// Reads the `--order` file from disk.
    pub fn read(path: &str, id: &str) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading --order {path}"))?;
        Self::parse(&text, id)
    }

    /// The scores this file states for one parent, if any.
    pub fn for_parent(&self, parent_source_id: &str) -> Option<&BTreeMap<usize, f64>> {
        self.scores.get(parent_source_id)
    }

    /// Parents named by the file.
    pub fn parents(&self) -> usize {
        self.scores.len()
    }
}

/// Options of one `extend` run.
#[derive(Clone, Copy, Debug)]
pub struct ExtendOptions {
    /// Node budget of one candidate's publishing gate.
    pub node_budget: u64,
    /// Ply bound used to measure `k` when the input record does not carry it.
    pub source_max_ply: u32,
    pub emit: Emit,
    /// Candidates *verified* per source; `0` verifies every one of them.  It cuts
    /// the verification order, so with an `--order` file it keeps the K
    /// best-scored candidates.  (The old sampling knobs, `max_candidates` with
    /// its `seed` and `listed_only`, went on 2026-10-27: the CLI always ran them
    /// at `0` / `1` / `false`, and the record keeps `"seed":"1"` and
    /// `"maxCandidates":0`.)
    pub max_verify: usize,
    /// Stop a source once this many candidates have been accepted; `0` never
    /// stops early.
    pub stop_after_accepted: usize,
    /// Which engine answers the per-candidate "is there a forced mate of
    /// exactly `k + 2` plies here?" question *before* the exact gate runs.
    ///
    /// [`ProbeEngine::Exact`] is the default and reproduces every earlier run:
    /// every candidate goes straight to `exact_validate_with_rules`.
    /// [`ProbeEngine::Dfpn`] puts [`crate::dfpn`] in front of it as a pure
    /// pre-filter -- it can only *reject*, never accept, and the exact solver
    /// still produces every emitted record, its graph, its `stats.nodes` and its
    /// rule A-G verdict.  An accepted record is byte-identical either way, which
    /// is asserted in `tests/unmove.rs`.
    pub probe: ProbeEngine,
    /// df-pn node budget of one candidate probe.  Ignored with
    /// [`ProbeEngine::Exact`].  df-pn nodes are a different unit from the exact
    /// nodes `node_budget` bounds, and are counted separately.
    pub dfpn_node_budget: u64,
    /// Put every accepted candidate through the `reduce` stage (rule D) and the
    /// gate again before it is written; see "Rule D" in the module docs.  On by
    /// default; `false` reproduces the unreduced output of earlier runs.
    pub reduce: bool,
}

/// `rejection.reasons` of a parent `extend` refuses because its own main line
/// carries yozume (owner decision 2026-10-04, `docs/TODO.md` P8-016 rule 1:
/// retrograde extension picks only yozume-free parents).  Opt-in under
/// `--strict-rule-b` until `PIPELINE_VERSION` 0.12.0, always on since: rule B
/// holds at every length, and a child keeps its parent's main-line yozume
/// whenever the parent stays in the child's main-line region.
pub const PARENT_HAS_YOZUME: &str = "parent_has_yozume";
/// ... or because whether it does could not be decided inside the node budget.
pub const PARENT_YOZUME_UNKNOWN: &str = "parent_yozume_unknown";

impl Default for ExtendOptions {
    fn default() -> Self {
        Self {
            node_budget: 250_000,
            source_max_ply: 11,
            emit: Emit::Accepted,
            max_verify: 0,
            stop_after_accepted: 0,
            probe: ProbeEngine::Exact,
            dfpn_node_budget: 1_000_000,
            reduce: true,
        }
    }
}

impl ExtendOptions {
    /// The publishing gate's rules: [`TsumeRules::puzzle`] (rule B at every length
    /// since `PIPELINE_VERSION` 0.12.0, so a child with yozume is rejected with
    /// `first_move_not_unique` / `yozume_in_main_line`, and the rule D reduction,
    /// run under the same rules, never creates yozume).
    pub fn gate_rules(&self) -> TsumeRules {
        TsumeRules::puzzle()
    }

    /// The option block recorded on every emitted record.
    ///
    /// The ranking knobs (`max_verify`, `stop_after_accepted`) are
    /// deliberately **not** here.  They say how much of the enumeration this run
    /// paid for, not how the candidate was derived, and a record has to stay
    /// byte-identical whether it was reached through the ranked order or the
    /// exhaustive one — that identity is what makes the ranker a pure ordering and
    /// is asserted in `tests/unmove.rs`.  `run_json` carries them to `--stats-out`.
    ///
    /// `probe` and `dfpn_node_budget` are out for the same reason and a stronger
    /// one: the prober never decides an acceptance, so no record may state which
    /// engine screened it -- that identity is what keeps a `--probe dfpn` run's
    /// accepted output byte-identical to the exhaustive one.  They go to
    /// `--stats-out` only.
    ///
    /// `reduce` *is* here, because it changes the puzzle a record describes.  It
    /// is written only when on, so a record without it keeps the exact bytes of
    /// the runs made before the reducer was added.
    pub fn to_json(&self) -> Value {
        let mut block = json!({
            "version": EXTEND_VERSION,
            // Fixed since the sampling knobs went (2026-10-27).
            "seed": "1",
            "maxCandidates": 0,
            "nodeBudget": self.node_budget,
            "sourceMaxPly": self.source_max_ply,
            "emit": self.emit.as_str(),
        });
        if self.reduce {
            if let Value::Object(map) = &mut block {
                map.insert("reduce".to_string(), json!(true));
            }
        }
        block
    }

    /// [`ExtendOptions::to_json`] plus the per-source verification caps, for
    /// `--stats-out` only.
    pub fn run_json(&self) -> Value {
        let mut block = self.to_json();
        if let Value::Object(map) = &mut block {
            map.insert("maxVerify".to_string(), json!(self.max_verify));
            map.insert(
                "stopAfterAccepted".to_string(),
                json!(self.stop_after_accepted),
            );
            map.insert("listedOnly".to_string(), json!(false));
            map.insert("probe".to_string(), json!(self.probe.as_str()));
            map.insert("dfpnNodeBudget".to_string(), json!(self.dfpn_node_budget));
        }
        block
    }
}

/// Counters of one `extend` run.  Every field is a sum or a set, so merging across
/// workers is order-independent and the aggregate is reproducible.
#[derive(Clone, Debug, Default)]
pub struct ExtendStats {
    /// Input records read.
    pub sources: usize,
    /// Inputs that carried a rejection, stated no origin, or whose mate length
    /// could not be measured.
    pub sources_skipped: usize,
    /// Sources that yielded at least one predecessor.
    pub sources_with_candidates: usize,
    /// Distinct predecessor positions enumerated.
    pub candidates: usize,
    /// Predecessors actually put through the gate.
    pub solved: usize,
    /// Enumerated candidates never verified because `max_verify` or
    /// `stop_after_accepted` cut the source short.  The verifier time these
    /// represent is exactly what the ranker saves.
    pub skipped_by_cap: usize,
    /// Always 0 since `listed_only` went (2026-10-27); kept so `--stats-out`
    /// keeps its keys.
    pub skipped_unlisted: usize,
    /// Sources cut short by `max_verify`.
    pub sources_hit_max_verify: usize,
    /// Sources cut short by `stop_after_accepted`.
    pub sources_hit_stop_after_accepted: usize,
    /// Sources for which the order file scored at least one candidate.
    pub sources_ranked: usize,
    /// Strict-valid `k + 2` puzzles.
    pub accepted: usize,
    /// Gate search nodes over every solved candidate.
    pub nodes: u64,
    /// Gate search nodes over the accepted ones only.
    pub accepted_nodes: u64,
    /// `(check, reply)` pairs enumerated, by kind.
    pub pairs: PredecessorCounts,
    /// Distinct candidates by the kind of their first `(check, reply)` pair.
    pub candidate_kinds: PredecessorCounts,
    /// Accepted candidates by the same key.
    pub accepted_kinds: PredecessorCounts,
    /// Rejection reasons, counted once per candidate that reported them.
    pub reasons: BTreeMap<String, usize>,
    /// The whole sorted reason list, `+`-joined: which *combinations* occur.
    pub reason_sets: BTreeMap<String, usize>,
    /// Candidates rejected for nothing but 駒余り and rule G — the population the
    /// repair measurement in §9 step 4 runs `distill` over.
    pub repairable: usize,
    /// One entry per source, for mean/median/max.
    pub candidates_per_source: Vec<usize>,
    /// One entry per source with at least one candidate.
    pub accepted_per_source: Vec<usize>,
    /// One entry per source with at least one candidate: candidates verified.
    pub verified_per_source: Vec<usize>,
    /// One entry per source with at least one candidate: candidates enumerated
    /// but not verified (`skipped_by_cap` + `skipped_unlisted` for that source).
    pub skipped_per_source: Vec<usize>,
    /// Canonical hashes of the accepted puzzles: the distinct-puzzle count.
    pub accepted_hashes: BTreeSet<String>,
    /// Candidates the df-pn prober was asked about (`--probe dfpn` only).
    pub dfpn_probes: usize,
    /// Candidates the prober settled alone, so the exact gate never ran.
    pub dfpn_rejected: usize,
    /// ... of those: no forced mate inside `k + 2` at all.  Whenever the exact
    /// gate resolves such a candidate it writes the identical `rejection.reasons`.
    pub dfpn_rejected_no_mate: usize,
    /// ... of those: a forced mate already inside `k`, so the minimum is not
    /// `k + 2`.  `target_mate_ply_changed` is what the exact gate reports too,
    /// possibly beside further reasons this path never looks for.
    pub dfpn_rejected_shorter: usize,
    /// Candidates the prober handed on to the exact gate.
    pub dfpn_passed: usize,
    /// Probes that ran out of df-pn budget and therefore decided nothing.
    pub dfpn_budget_exhausted: usize,
    /// df-pn nodes.  **Not** comparable with `nodes`: a different traversal of
    /// the same tree, and never published metadata.
    pub dfpn_nodes: u64,
    /// Accepted candidates the reducer ran on (`reduce` on).
    pub reduced: usize,
    /// ... of those: at least one piece came off, so the gate ran again.
    pub reduced_changed: usize,
    /// Pieces the reducer removed over all of them (board pieces and attacker
    /// hand units, as `reduction.removed` lists them).
    pub reduced_pieces: usize,
    /// Accepted candidates written *unreduced*, because the reducer refused them at
    /// entry or the reduced diagram failed the gate.  Neither should happen — the
    /// reducer's entry check is the gate's own verdict — so a non-zero count is a
    /// finding, not a routine outcome.
    pub reduce_fallback: usize,
    /// Exact nodes of the second gate run on a reduced diagram.  Kept apart from
    /// `nodes`, which stays the cost of choosing *which* candidates are puzzles.
    pub reduce_resolve_nodes: u64,
    /// Source mate lengths seen, `k` -> sources.
    pub source_mate_ply: BTreeMap<u32, usize>,
    /// Sources skipped because their own main line carries yozume
    /// ([`PARENT_HAS_YOZUME`]).  Also counted in `sources_skipped`.
    pub sources_parent_yozume: usize,
    /// Sources skipped because the yozume measurement ran out of budget
    /// ([`PARENT_YOZUME_UNKNOWN`]).  Also counted in `sources_skipped`.
    pub sources_parent_yozume_unknown: usize,
    /// Search nodes of those parent measurements.
    pub parent_yozume_nodes: u64,
}

fn bump(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

impl ExtendStats {
    /// Adds `other` into `self`.
    pub fn merge(&mut self, other: &ExtendStats) {
        self.sources += other.sources;
        self.sources_skipped += other.sources_skipped;
        self.sources_with_candidates += other.sources_with_candidates;
        self.candidates += other.candidates;
        self.solved += other.solved;
        self.skipped_by_cap += other.skipped_by_cap;
        self.skipped_unlisted += other.skipped_unlisted;
        self.sources_hit_max_verify += other.sources_hit_max_verify;
        self.sources_hit_stop_after_accepted += other.sources_hit_stop_after_accepted;
        self.sources_ranked += other.sources_ranked;
        self.accepted += other.accepted;
        self.nodes += other.nodes;
        self.accepted_nodes += other.accepted_nodes;
        self.pairs.merge(&other.pairs);
        self.candidate_kinds.merge(&other.candidate_kinds);
        self.accepted_kinds.merge(&other.accepted_kinds);
        self.repairable += other.repairable;
        self.dfpn_probes += other.dfpn_probes;
        self.dfpn_rejected += other.dfpn_rejected;
        self.dfpn_rejected_no_mate += other.dfpn_rejected_no_mate;
        self.dfpn_rejected_shorter += other.dfpn_rejected_shorter;
        self.dfpn_passed += other.dfpn_passed;
        self.dfpn_budget_exhausted += other.dfpn_budget_exhausted;
        self.dfpn_nodes += other.dfpn_nodes;
        self.reduced += other.reduced;
        self.reduced_changed += other.reduced_changed;
        self.reduced_pieces += other.reduced_pieces;
        self.reduce_fallback += other.reduce_fallback;
        self.reduce_resolve_nodes += other.reduce_resolve_nodes;
        self.sources_parent_yozume += other.sources_parent_yozume;
        self.sources_parent_yozume_unknown += other.sources_parent_yozume_unknown;
        self.parent_yozume_nodes += other.parent_yozume_nodes;
        for (key, value) in other.reasons.iter() {
            *self.reasons.entry(key.clone()).or_insert(0) += value;
        }
        for (key, value) in other.reason_sets.iter() {
            *self.reason_sets.entry(key.clone()).or_insert(0) += value;
        }
        self.candidates_per_source
            .extend_from_slice(&other.candidates_per_source);
        self.accepted_per_source
            .extend_from_slice(&other.accepted_per_source);
        self.verified_per_source
            .extend_from_slice(&other.verified_per_source);
        self.skipped_per_source
            .extend_from_slice(&other.skipped_per_source);
        self.accepted_hashes
            .extend(other.accepted_hashes.iter().cloned());
        for (key, value) in other.source_mate_ply.iter() {
            *self.source_mate_ply.entry(*key).or_insert(0) += value;
        }
    }

    /// Mean, median and maximum of a per-source series.
    fn spread(values: &[usize]) -> Value {
        if values.is_empty() {
            return json!({ "count": 0, "mean": 0.0, "median": 0, "max": 0, "total": 0 });
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let total: usize = sorted.iter().sum();
        let median = sorted[sorted.len() / 2];
        json!({
            "count": sorted.len(),
            "mean": total as f64 / sorted.len() as f64,
            "median": median,
            "max": *sorted.last().unwrap_or(&0),
            "total": total,
        })
    }

    fn kinds_json(counts: &PredecessorCounts) -> Value {
        let mut check = Map::new();
        for kind in crate::unmove::UN_CHECK_KINDS {
            check.insert(kind.as_str().to_string(), json!(counts.check[kind.index()]));
        }
        let mut reply = Map::new();
        for kind in crate::unmove::UN_REPLY_KINDS {
            reply.insert(kind.as_str().to_string(), json!(counts.reply[kind.index()]));
        }
        json!({ "total": counts.pairs, "check": check, "reply": reply })
    }

    /// The `--stats-out` payload.
    pub fn to_json(&self) -> Value {
        let mate_ply: Map<String, Value> = self
            .source_mate_ply
            .iter()
            .map(|(key, value)| (key.to_string(), json!(value)))
            .collect();
        let mut payload = json!({
            "sources": self.sources,
            "sourcesSkipped": self.sources_skipped,
            "sourcesWithCandidates": self.sources_with_candidates,
            "sourceMatePly": mate_ply,
            "candidates": self.candidates,
            "solved": self.solved,
            "verified": self.solved,
            "skippedByCap": self.skipped_by_cap,
            "skippedUnlisted": self.skipped_unlisted,
            "sourcesRanked": self.sources_ranked,
            "sourcesHitMaxVerify": self.sources_hit_max_verify,
            "sourcesHitStopAfterAccepted": self.sources_hit_stop_after_accepted,
            "accepted": self.accepted,
            "acceptanceRate": if self.solved == 0 { 0.0 } else { self.accepted as f64 / self.solved as f64 },
            "distinctAcceptedHashes": self.accepted_hashes.len(),
            "repairableRejections": self.repairable,
            "nodes": self.nodes,
            "acceptedNodes": self.accepted_nodes,
            "dfpnProbes": self.dfpn_probes,
            "dfpnRejected": self.dfpn_rejected,
            "dfpnRejectedNoMate": self.dfpn_rejected_no_mate,
            "dfpnRejectedShorter": self.dfpn_rejected_shorter,
            "dfpnPassed": self.dfpn_passed,
            "dfpnBudgetExhausted": self.dfpn_budget_exhausted,
            "dfpnNodes": self.dfpn_nodes,
            "reduced": self.reduced,
            "reducedChanged": self.reduced_changed,
            "reducedPieces": self.reduced_pieces,
            "reduceFallback": self.reduce_fallback,
            "reduceResolveNodes": self.reduce_resolve_nodes,
            "nodesPerCandidate": if self.solved == 0 { 0.0 } else { self.nodes as f64 / self.solved as f64 },
            "nodesPerAccepted": if self.accepted == 0 { 0.0 } else { self.nodes as f64 / self.accepted as f64 },
            "candidatesPerSource": Self::spread(&self.candidates_per_source),
            "verifiedPerSource": Self::spread(&self.verified_per_source),
            "skippedPerSource": Self::spread(&self.skipped_per_source),
            "acceptedPerSource": Self::spread(&self.accepted_per_source),
            "pairKinds": Self::kinds_json(&self.pairs),
            "candidateKinds": Self::kinds_json(&self.candidate_kinds),
            "acceptedKinds": Self::kinds_json(&self.accepted_kinds),
            "rejectionReasons": self.reasons.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<String, Value>>(),
            "rejectionReasonSets": self.reason_sets.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<String, Value>>(),
        });
        // The parent-yozume counters, only when that path measured anything (a
        // run whose every source was skipped earlier, e.g. `missing_origin`, keeps
        // the keys it had before 0.12.0).
        if self.parent_yozume_nodes > 0
            || self.sources_parent_yozume > 0
            || self.sources_parent_yozume_unknown > 0
        {
            if let Value::Object(map) = &mut payload {
                map.insert(
                    "sourcesParentYozume".to_string(),
                    json!(self.sources_parent_yozume),
                );
                map.insert(
                    "sourcesParentYozumeUnknown".to_string(),
                    json!(self.sources_parent_yozume_unknown),
                );
                map.insert(
                    "parentYozumeNodes".to_string(),
                    json!(self.parent_yozume_nodes),
                );
            }
        }
        payload
    }
}

/// What one source produced.
pub struct ExtendOutput {
    /// Pipeline records, under [`Emit::Accepted`] and [`Emit::All`].
    pub records: Vec<PipelineRecord>,
    /// Lean candidate rows, under [`Emit::Candidates`] only.  One per enumerated
    /// candidate, in enumeration order, carrying exactly what
    /// `tools/reverse-ranker/rrlib.py` reads.
    pub candidates: Vec<Value>,
    /// The candidates the gate judged and `--emit accepted` does not write
    /// (rejected ones), kept only while [`crate::catalog_record::capture_on`]
    /// (`--emit-record` writes a record of every judged diagram), each with the
    /// number of `records` judged before it, so the two merge back into the
    /// judgement order.  Empty otherwise, and under [`Emit::All`], which writes
    /// them in `records`.
    pub unwritten: Vec<(usize, PipelineRecord)>,
    pub stats: ExtendStats,
}

/// Extends one source record by two plies.
///
/// A record that is not accepted, or whose mate length cannot be measured inside
/// `source_max_ply`, produces nothing and is counted as skipped: there is no
/// verified puzzle to retract from.  A record that states no origin is skipped for
/// a different reason — there is a puzzle, but nothing derived from it could be
/// published (`docs/legal-data.md` §3) — and under `--emit all` it produces one
/// `missing_origin` rejection so the run says why.
pub fn extend_record(
    solver: &mut Solver,
    record: &PipelineRecord,
    options: &ExtendOptions,
) -> Result<ExtendOutput> {
    extend_record_ordered(solver, record, options, None)
}

/// [`extend_record`] with a scored verification order.
///
/// `order` never changes *which* candidates exist, only the sequence they are put
/// through the gate in, and — together with `ExtendOptions::max_verify` and
/// `ExtendOptions::stop_after_accepted` — how many of them are reached at all.
/// The verdict stays `exact_validate_with_rules`, so an accepted record is
/// byte-identical to the one the exhaustive run writes except for the
/// `provenance.extra.ranker` block naming the model, its score and the rank.
///
/// That block is **not** a `provenance.lineage` entry: the ranker did not derive
/// the puzzle, it only chose when to look at it, and the same puzzle comes out of
/// the unranked run unchanged.  Lineage records derivations (`docs/legal-data.md`
/// §3); an ordering heuristic is not one.
pub fn extend_record_ordered(
    solver: &mut Solver,
    record: &PipelineRecord,
    options: &ExtendOptions,
    order: Option<&ExtendOrder>,
) -> Result<ExtendOutput> {
    extend_record_probed(solver, &mut Prober::new(), record, options, order)
}

/// [`extend_record_ordered`] with a caller-owned df-pn prober, so one worker
/// reuses a single prober across records instead of rebuilding it per source.
/// The prober is idle unless `options.probe` is [`ProbeEngine::Dfpn`]; its
/// transposition table is cleared at every probe, so results never depend on
/// which records a worker happened to see before.
pub fn extend_record_probed(
    solver: &mut Solver,
    prober: &mut Prober,
    record: &PipelineRecord,
    options: &ExtendOptions,
    order: Option<&ExtendOrder>,
) -> Result<ExtendOutput> {
    let mut stats = ExtendStats {
        sources: 1,
        ..ExtendStats::default()
    };
    if !record.is_accepted() {
        stats.sources_skipped = 1;
        return Ok(ExtendOutput {
            records: Vec::new(),
            candidates: Vec::new(),
            unwritten: Vec::new(),
            stats,
        });
    }
    // `docs/legal-data.md` §3: every candidate inherits its source's origin set, so
    // a source that states no origin can produce no publishable candidate.  The
    // result is rejected, never defaulted to `own`; the reason is the string
    // `validate` and `distill` write for the same defect.  The check runs before the
    // length probe, so an unattributed source costs no search at all.  Under
    // `--emit accepted` nothing is written, because a rejection is not a puzzle; the
    // per-candidate `validate_stage` below is the same refusal one level down, and
    // is what catches an origin field lost further along a chain.
    if !record.provenance.is_root_complete() {
        stats.sources_skipped = 1;
        return Ok(ExtendOutput {
            records: match options.emit {
                Emit::All => vec![refused_source_record(record, MISSING_ORIGIN_REASON)],
                // A rejection is not a puzzle, and `--emit candidates` writes no
                // records at all: an unattributed source simply enumerates nothing.
                Emit::Accepted | Emit::Candidates => Vec::new(),
            },
            candidates: Vec::new(),
            unwritten: Vec::new(),
            stats,
        });
    }
    let position = record_position(record)?;
    let Some(source_ply) = source_mate_ply(solver, record, &position, options) else {
        stats.sources_skipped = 1;
        return Ok(ExtendOutput {
            records: Vec::new(),
            candidates: Vec::new(),
            unwritten: Vec::new(),
            stats,
        });
    };
    // P8-016 rule 1: only a yozume-free parent is extended (opt-in under
    // `--strict-rule-b` until 0.12.0, always since).  Measured at the parent's own
    // length and the gate's budget; a parent that cannot be measured is skipped
    // too (fail safe).  Under `--emit all` the refusal is written, as
    // `missing_origin` is.
    let (yozume, nodes) = measure_main_line_yozume(
        solver,
        &position,
        Some(source_ply),
        source_ply,
        options.node_budget,
    );
    stats.parent_yozume_nodes += nodes;
    let refusal = match yozume {
        Some(false) => None,
        Some(true) => {
            stats.sources_parent_yozume = 1;
            Some(PARENT_HAS_YOZUME)
        }
        None => {
            stats.sources_parent_yozume_unknown = 1;
            Some(PARENT_YOZUME_UNKNOWN)
        }
    };
    if let Some(reason) = refusal {
        stats.sources_skipped = 1;
        return Ok(ExtendOutput {
            records: match options.emit {
                Emit::All => vec![refused_source_record(record, reason)],
                Emit::Accepted | Emit::Candidates => Vec::new(),
            },
            candidates: Vec::new(),
            unwritten: Vec::new(),
            stats,
        });
    }
    *stats.source_mate_ply.entry(source_ply).or_insert(0) += 1;
    let target = source_ply + 2;

    let groups = grouped_predecessors(&position);
    stats.candidates = groups.len();
    stats.candidates_per_source.push(groups.len());
    if groups.is_empty() {
        return Ok(ExtendOutput {
            records: Vec::new(),
            candidates: Vec::new(),
            unwritten: Vec::new(),
            stats,
        });
    }
    stats.sources_with_candidates = 1;
    for group in groups.iter() {
        for &(_, _, kind) in group.moves.iter() {
            stats.pairs.check[kind.0.index()] += 1;
            stats.pairs.reply[kind.1.index()] += 1;
            stats.pairs.pairs += 1;
        }
        let (_, _, primary) = group.moves[0];
        stats.candidate_kinds.check[primary.0.index()] += 1;
        stats.candidate_kinds.reply[primary.1.index()] += 1;
        stats.candidate_kinds.pairs += 1;
    }

    // Every candidate, in enumeration order (the sampling of `max_candidates`
    // went on 2026-10-27).
    let chosen: Vec<usize> = (0..groups.len()).collect();

    let parent_source_id = record.source_id.clone();
    let parent_board_pieces = position.board_piece_count();

    // The verification order.  Without `--order` it is enumeration order, exactly
    // as every earlier run had it.  With one, the candidates the file scores come
    // first in descending score — ties broken by `candidateIndex`, so the order is
    // total and the same file always produces the same sequence — and the ones it
    // does not score follow in enumeration order.
    let scores = order.and_then(|order| order.for_parent(&parent_source_id));
    if scores.is_some() {
        stats.sources_ranked = 1;
    }
    let mut listed: Vec<(usize, f64)> = Vec::new();
    let mut unlisted: Vec<usize> = Vec::new();
    for &index in chosen.iter() {
        match scores.and_then(|scores| scores.get(&index)) {
            Some(&score) => listed.push((index, score)),
            None => unlisted.push(index),
        }
    }
    // `total_cmp` rather than `partial_cmp`: a NaN in the file must not make the
    // sort order depend on the input permutation.
    listed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut sequence: Vec<(usize, Option<f64>)> = listed
        .iter()
        .map(|&(index, score)| (index, Some(score)))
        .collect();
    let mut skipped_here = 0usize;
    sequence.extend(unlisted.iter().map(|&index| (index, None)));

    let mut records = Vec::with_capacity(sequence.len());
    let mut unwritten = Vec::new();
    let mut candidates = Vec::new();
    let mut accepted_here = 0usize;
    let mut verified_here = 0usize;
    for (rank, &(index, score)) in sequence.iter().enumerate() {
        // Both caps are per source and both are checked before the gate runs, so
        // the candidates they cut cost no search at all.  That is the saving.
        if options.emit != Emit::Candidates {
            if options.max_verify != 0 && verified_here >= options.max_verify {
                stats.sources_hit_max_verify = 1;
                skipped_here += sequence.len() - rank;
                stats.skipped_by_cap += sequence.len() - rank;
                break;
            }
            if options.stop_after_accepted != 0 && accepted_here >= options.stop_after_accepted {
                stats.sources_hit_stop_after_accepted = 1;
                skipped_here += sequence.len() - rank;
                stats.skipped_by_cap += sequence.len() - rank;
                break;
            }
        }
        let group = &groups[index];
        let (check, reply, kind) = group.moves[0];
        // `docs/legal-data.md` §3, exactly as `distill` does for a variant: the
        // copy carries the parent's `originClass`, `origins` and `rootSourceId`
        // — a derivation with one input copies the set — and this step
        // appends itself to the typed lineage.
        let mut out = record.derived(
            format!("{parent_source_id}-x{index}"),
            group.sfen.clone(),
            target,
            EXTEND_TOOL,
            EXTEND_VERSION,
        );
        out.provenance.discovered_mate_ply = None;
        out.provenance.transformations.push(format!(
            "extend:unmove-2ply:{},{}",
            check.to_usi(),
            reply.to_usi()
        ));
        // The parent id also stays in `provenance.extra`, where the ranker's
        // training rows read it.  There is no second `lineage` array beside it: the
        // typed field above carries the same `{sourceId, tool, version}` step.
        out.provenance.extra.insert(
            "parentSourceId".to_string(),
            Value::String(parent_source_id.clone()),
        );
        out.provenance.extra.insert(
            "extend".to_string(),
            extend_provenance(
                options,
                &parent_source_id,
                source_ply,
                check,
                reply,
                kind,
                index,
            ),
        );
        // The one thing a ranked run adds to a record: which model put this
        // candidate where in the queue.  Not a lineage entry — see the doc comment
        // on `extend_record_ordered`.  `score` is null for a candidate the order
        // file did not name; `rank` is 1-based in the verification order.
        if let Some(order) = order {
            out.provenance.extra.insert(
                "ranker".to_string(),
                json!({ "id": order.id, "score": score, "rank": rank + 1 }),
            );
        }

        validate_stage(&mut out);

        // `--emit candidates`: the enumeration is the product.  The row is built
        // from the same `out` the solving path uses, so `normalizedSfen` and
        // `canonicalHash` are the ones the full run writes, by construction.
        if options.emit == Emit::Candidates {
            candidates.push(json!({
                "sourceId": out.source_id,
                "parentSourceId": parent_source_id,
                "sourceSfen": out.source_sfen,
                "normalizedSfen": out.normalized_sfen,
                "canonicalHash": out.canonical_hash,
                "extend": candidate_block(
                    &parent_source_id,
                    source_ply,
                    parent_board_pieces,
                    groups.len(),
                    group,
                    index,
                ),
            }));
            continue;
        }
        // The df-pn pre-filter, when it is on.  It can only *reject*, and only
        // on the two verdicts the exact gate would have reached itself given
        // enough budget: no forced mate inside the target at all, or a mate that
        // is already shorter than it.  Everything it passes goes through
        // `solve_stage_reported` untouched, which is why an accepted record is
        // byte-identical under either engine.
        let report = match probe_gate(prober, &mut out, options, target, &mut stats) {
            Some(report) => report,
            None => solve_stage_reported(
                &mut out,
                &SolveStageOptions {
                    max_ply: target,
                    node_budget: options.node_budget,
                    rules: options.gate_rules(),
                },
                solver,
            )?,
        };
        observe(&mut stats, &report);
        // `solve` keeps the solution of a rejection by rule B alone
        // (`stages::rule_b_only`); `--emit all` writes a rejected candidate as
        // before, without one (its `extra.extend` block has the length and reasons).
        if !report.accepted {
            out.solution = None;
        }
        let mut reduce_fallback = None;
        if report.accepted {
            // Rule D before the record is counted, so `accepted_hashes` holds the
            // puzzle that is actually written.
            if options.reduce {
                reduce_fallback = reduce_accepted(&mut out, target, options, solver, &mut stats)?;
            }
            accepted_here += 1;
            stats.accepted_kinds.check[kind.0.index()] += 1;
            stats.accepted_kinds.reply[kind.1.index()] += 1;
            stats.accepted_kinds.pairs += 1;
            if let Some(hash) = out.canonical_hash.clone() {
                stats.accepted_hashes.insert(hash);
            }
        }
        let mut block = candidate_block(
            &parent_source_id,
            source_ply,
            parent_board_pieces,
            groups.len(),
            group,
            index,
        );
        if let Value::Object(map) = &mut block {
            map.insert("accepted".to_string(), json!(report.accepted));
            map.insert("matePly".to_string(), json!(report.mate_ply));
            map.insert("nodes".to_string(), json!(report.nodes));
            map.insert("reasons".to_string(), json!(report.reasons));
            // Only on the rare record written unreduced: why the reducer's result
            // was not used.  A reduced record says so in its own `reduction` field.
            if let Some(reasons) = reduce_fallback {
                map.insert("reduceFallback".to_string(), json!(reasons));
            }
        }
        out.extra.insert("extend".to_string(), block);
        if options.emit == Emit::All || report.accepted {
            records.push(out);
        } else if crate::catalog_record::capture_on() {
            unwritten.push((records.len(), out));
        }
        verified_here += 1;
    }
    stats.accepted_per_source.push(accepted_here);
    stats.verified_per_source.push(verified_here);
    stats.skipped_per_source.push(skipped_here);
    Ok(ExtendOutput {
        records,
        candidates,
        unwritten,
        stats,
    })
}

/// The `extra.extend` block every candidate carries, verdict aside.
///
/// `--emit candidates` writes exactly this; the solving path appends `accepted`,
/// `matePly`, `nodes` and `reasons` to it.  One builder, so the two shapes cannot
/// drift and `tools/reverse-ranker/rrlib.py` reads one set of names.
fn candidate_block(
    parent_source_id: &str,
    source_ply: u32,
    parent_board_pieces: usize,
    source_candidates: usize,
    group: &crate::unmove::PredecessorGroup,
    index: usize,
) -> Value {
    let (check, reply, kind) = group.moves[0];
    json!({
        "version": EXTEND_VERSION,
        "parentSourceId": parent_source_id,
        "candidateIndex": index,
        "sourceCandidates": source_candidates,
        "sourceMatePly": source_ply,
        "targetMatePly": source_ply + 2,
        "parentBoardPieces": parent_board_pieces,
        "check": check.to_usi().to_string(),
        "reply": reply.to_usi().to_string(),
        "unCheckKind": kind.0.as_str(),
        "unReplyKind": kind.1.as_str(),
        "pairs": group
            .moves
            .iter()
            .map(|(c, r, k)| json!({
                "check": c.to_usi().to_string(),
                "reply": r.to_usi().to_string(),
                "unCheckKind": k.0.as_str(),
                "unReplyKind": k.1.as_str(),
            }))
            .collect::<Vec<Value>>(),
    })
}

/// Folds one gate verdict into the counters.
fn observe(stats: &mut ExtendStats, report: &SolveStageReport) {
    stats.solved += 1;
    stats.nodes += report.nodes;
    if report.accepted {
        stats.accepted += 1;
        stats.accepted_nodes += report.nodes;
        return;
    }
    let mut sorted = report.reasons.clone();
    sorted.sort();
    sorted.dedup();
    for reason in sorted.iter() {
        bump(&mut stats.reasons, reason);
    }
    if !sorted.is_empty() {
        bump(&mut stats.reason_sets, &sorted.join("+"));
        if sorted
            .iter()
            .all(|reason| REPAIRABLE_REASONS.contains(&reason.as_str()))
        {
            stats.repairable += 1;
        }
    }
}

/// Rule D on one accepted candidate: [`crate::stages::reduce_accepted`] under
/// the gate's own bound, budget and rules.
///
/// `out` arrives accepted, carrying the gate's solution, and leaves as the record
/// to write: unchanged but for `reduction` when nothing came off, re-solved when
/// pieces did.  On a **fallback** — the reducer refused the candidate, or the
/// reduced diagram failed the gate — the record is written exactly as the gate
/// left it and the reasons are returned for `extra.extend.reduceFallback`.
/// Neither should occur: the reducer's entry check is the gate's own verdict, and
/// every step it takes passes the same rules.  So it is counted apart, as a
/// finding.
fn reduce_accepted(
    out: &mut PipelineRecord,
    target: u32,
    options: &ExtendOptions,
    solver: &mut Solver,
    stats: &mut ExtendStats,
) -> Result<Option<Vec<String>>> {
    let report = stages::reduce_accepted(
        out,
        // Rule D's goal is no useless piece, not the fewest pieces (owner
        // 2026-09-19): one greedy attempt plus the fixpoint sweep guarantees it.
        ReduceOptions {
            attempts: 1,
            max_ply: target,
            node_budget: options.node_budget,
            until_fixpoint: true,
            // P8-016 rule 4: the child was accepted under the gate's rules, so it
            // is yozume-free and stays so.
            rules: options.gate_rules(),
            ..ReduceOptions::default()
        },
        // Rule D v5: the main-line-preserving reduction alone.
        crate::reduce::TwoKindsOptions::default(),
        &SolveStageOptions {
            max_ply: target,
            node_budget: options.node_budget,
            rules: options.gate_rules(),
        },
        solver,
    )?;
    stats.reduce_resolve_nodes += report.resolve_nodes;
    match report.outcome {
        ReduceAcceptedOutcome::Unchanged => stats.reduced += 1,
        ReduceAcceptedOutcome::Reduced { removed } => {
            stats.reduced += 1;
            stats.reduced_changed += 1;
            stats.reduced_pieces += removed;
        }
        ReduceAcceptedOutcome::Fallback { reasons } => {
            stats.reduce_fallback += 1;
            return Ok(Some(reasons));
        }
    }
    Ok(None)
}

/// The df-pn pre-filter in front of the publishing gate (`--probe dfpn`).
///
/// `Some(report)` means the prober alone settled that this candidate is not a mate
/// of exactly `target` plies: the record is written as a `solve`-stage rejection
/// and the exact solver never runs.  `None` means "ask the exact solver", which is
/// what happens for every candidate under `--probe exact`, for every candidate the
/// prober proves at the target bound and cannot prove two plies shorter, and for
/// every probe that ran out of df-pn budget.
///
/// Why this cannot change a verdict.  Rule F1 (`docs/rules.md`) is a statement
/// about a mate's *length*, never about its *existence* at a bound -- a defence
/// node's outcome still requires EVERY legal reply, useless ones included, to be
/// MATE -- and the prober applies F2 exactly as the exact solver does, so the
/// prober's proven/disproven verdict at a bound is the exact solver's
/// `MATE`/`REFUTED` at the same bound in the gate's rule-F mode ([`crate::dfpn`],
/// `tests/dfpn.rs`).  The shorter-mate test is sound for that reason plus one more:
/// the rule-F-pruned length is never longer than the real-ply length the prober
/// measures, so a real-ply mate inside `target - 2` is a pruned mate inside
/// `target - 2` as well, and `target_mate_ply_changed` holds either way.
///
/// Since `PIPELINE_VERSION` 0.10.0 the gate reads rule F classically, and that
/// reading *does* change existence at a bound (an all-futile check is mate even
/// when its capture chain runs past the bound).  The prober reads F2 the same way
/// ([`crate::exact::dfpn_useless_mode`], F2 in [`crate::dfpn`]), but since 0.11.0
/// the gate's bound also counts rule-F plies (F1's credit search), which the
/// prober does not reproduce: its proofs stay sound, its disproofs are not
/// conclusive ([`crate::exact::dfpn_disproof_is_conclusive`]) and are handed to
/// the exact gate here.
fn probe_gate(
    prober: &mut Prober,
    record: &mut PipelineRecord,
    options: &ExtendOptions,
    target: u32,
    stats: &mut ExtendStats,
) -> Option<SolveStageReport> {
    if !options.probe.is_dfpn() {
        return None;
    }
    // The two guards `solve_stage_reported` opens with: a candidate that already
    // carries a rejection, or that never normalised, is searched by neither engine
    // and must not be counted as probed either.
    if !record.is_accepted() {
        return None;
    }
    let position = parse_sfen(record.normalized_sfen.as_deref()?).ok()?;
    // The gate is `TsumeRules::puzzle()` (strict rule B or not, the rule-F reading
    // is the same), and the prober reads rule F the way it does.
    let gate_rules = TsumeRules::puzzle();
    let mode = dfpn_useless_mode(gate_rules);
    let probe = ProbeOptions::default()
        .max_ply(target)
        .node_budget(options.dfpn_node_budget)
        .useless(mode);
    stats.dfpn_probes += 1;
    let within = prober.prove(&position, &probe);
    stats.dfpn_nodes += within.nodes;
    match within.verdict {
        // Rule F's classical reading changes a mate's *existence* at a bound: F2
        // (an all-futile check is mate) since `PIPELINE_VERSION` 0.10.0, and the
        // bound counted in rule-F plies (F1's credit search) since 0.11.0.  The
        // prober applies F2 but not the credit search, so under the classical
        // rules its disproof is handed to the exact gate here.
        Verdict::Disproven if !dfpn_disproof_is_conclusive(gate_rules) => {
            stats.dfpn_passed += 1;
            return None;
        }
        Verdict::Disproven => {
            stats.dfpn_rejected += 1;
            stats.dfpn_rejected_no_mate += 1;
            return Some(probe_rejection(record, no_mate_reasons(&position, record)));
        }
        Verdict::Unknown => {
            stats.dfpn_budget_exhausted += 1;
            stats.dfpn_passed += 1;
            return None;
        }
        Verdict::Proven => {}
    }
    // Minimal length.  A mate proven two plies shorter settles that the minimum is
    // not `target`; anything else is the exact solver's business.
    if target >= 3 {
        let shorter = prober.prove(&position, &probe.max_ply(target - 2));
        stats.dfpn_nodes += shorter.nodes;
        if shorter.budget_exhausted {
            stats.dfpn_budget_exhausted += 1;
        }
        if shorter.verdict == Verdict::Proven {
            stats.dfpn_rejected += 1;
            stats.dfpn_rejected_shorter += 1;
            return Some(probe_rejection(
                record,
                vec!["target_mate_ply_changed".to_string()],
            ));
        }
    }
    stats.dfpn_passed += 1;
    None
}

/// Writes a `solve`-stage rejection exactly as [`solve_stage_reported`] does and
/// reports it with a **zero** exact-node cost, because no exact search ran.  What
/// the probe did cost is in `ExtendStats::dfpn_nodes`, which is a different unit.
fn probe_rejection(record: &mut PipelineRecord, reasons: Vec<String>) -> SolveStageReport {
    record.stage = "solved".to_string();
    record.rejection = Some(Rejection {
        stage: "solve".to_string(),
        reasons: reasons.clone(),
        // Rule G reads a solution tree and this branch has none, so no square is
        // named -- the same `None` the exact gate writes when rule G never fired.
        inert_pieces: None,
    });
    SolveStageReport {
        solved: true,
        accepted: false,
        reasons,
        mate_ply: None,
        nodes: 0,
        main_line: Vec::new(),
        leftover_hand: 0,
    }
}

/// `rejection.reasons` of a candidate with no forced mate inside the target bound,
/// in the order `exact_validate_with_rules` produces them.
///
/// Every entry follows from "the search finished and proved REFUTED at
/// `max_ply = target`" alone: there is no proven distance, so the target changed and
/// the minimum-length move set is empty; the root outcome is not MATE, so defence
/// coverage is not verified; and rules B, C and G are not evaluated at all because
/// they read a solution tree.  Rule A is a property of the diagram and is checked
/// here.  `incomplete_response_graph` is deliberately absent: it tracks
/// `stats.budgetExhausted`, and this branch is only reached on a *finished* probe.
fn no_mate_reasons(position: &Position, record: &PipelineRecord) -> Vec<String> {
    let mut reasons = vec!["no_forced_mate_within_bound".to_string()];
    if record.target_mate_ply.is_some() {
        reasons.push("target_mate_ply_changed".to_string());
    }
    reasons.push("first_move_not_unique".to_string());
    reasons.push("defense_coverage_incomplete".to_string());
    let rules = TsumeRules::puzzle().allowing_attacker_king(record.allows_attacker_king());
    // The attacker is the side to move at the root, exactly as `SolveOptions`
    // defaults it, and `strict_material::normalize` moves no king.
    if rules.composition_rules
        && !rules.allow_attacker_king
        && position.king_square(position.side) != NO_SQUARE
    {
        reasons.push("attacker_king_present".to_string());
    }
    reasons
}

/// The `provenance.extra.extend` block: everything needed to replay the retraction.
#[allow(clippy::too_many_arguments)]
fn extend_provenance(
    options: &ExtendOptions,
    parent_source_id: &str,
    source_ply: u32,
    check: Move,
    reply: Move,
    kind: (UnCheckKind, UnReplyKind),
    index: usize,
) -> Value {
    let mut block = options.to_json();
    if let Value::Object(map) = &mut block {
        map.insert("parentSourceId".to_string(), json!(parent_source_id));
        map.insert("sourceMatePly".to_string(), json!(source_ply));
        map.insert("targetMatePly".to_string(), json!(source_ply + 2));
        map.insert("candidateIndex".to_string(), json!(index));
        map.insert("check".to_string(), json!(check.to_usi().to_string()));
        map.insert("reply".to_string(), json!(reply.to_usi().to_string()));
        map.insert("unCheckKind".to_string(), json!(kind.0.as_str()));
        map.insert("unReplyKind".to_string(), json!(kind.1.as_str()));
    }
    block
}

/// The one record a source that states no origin produces.
///
/// `distill` answers the same defect the same way — one rejection record, no
/// derived positions — and both name the string `validate` writes for it, so an
/// audit sees one reason for one cause wherever it is raised.
///
/// `extend` refuses a yozume parent (P8-016) the same way, with
/// [`PARENT_HAS_YOZUME`] or [`PARENT_YOZUME_UNKNOWN`].
fn refused_source_record(record: &PipelineRecord, reason: &str) -> PipelineRecord {
    let mut out = record.refused("extend", vec![reason.to_string()]);
    // No `parentSourceId` here: this record *is* the source, not something derived
    // from it, and the ranker keys its training rows on real candidates only.
    out.extra.insert(
        "extend".to_string(),
        json!({ "version": EXTEND_VERSION, "rejected": true }),
    );
    out
}

/// `k` for the source: the published mate length when the record carries one, the
/// declared target next, and otherwise a measured one.
///
/// The measurement deepens from one ply upward rather than solving once at
/// `source_max_ply`, because the strict convention puts the whole remaining set in
/// the defender's hand: a bare 1-ply diagram searched at bound 11 exhausts a
/// 250 000-node budget and reports UNKNOWN.  The first bound that proves a mate
/// gives the minimum length, which is what the solver returns.
fn source_mate_ply(
    solver: &mut Solver,
    record: &PipelineRecord,
    position: &crate::position::Position,
    options: &ExtendOptions,
) -> Option<u32> {
    if let Some(solution) = record.solution.as_ref() {
        return Some(solution.mate_ply);
    }
    if let Some(target) = record.target_mate_ply {
        return Some(target);
    }
    let mut bound = 1;
    while bound <= options.source_max_ply {
        let result = solver.solve(
            position,
            &SolveOptions::default()
                .max_ply(bound)
                .node_budget(options.node_budget)
                .strict_material(true)
                .build_graph(false)
                .useless_interposition(TsumeRules::puzzle().useless_mode()),
        );
        if result.outcome == Outcome::Mate && result.distance.is_some() {
            return result.distance;
        }
        bound += 2;
    }
    None
}

/// Builds the imported-stage record `extend` uses for a bare SFEN input line:
/// [`PipelineRecord::imported_from_sfen`] with source ids `extend-…` and the
/// transformation `extend:sfen-input`, exactly as `distill` requires an origin.
pub fn imported_record_from_sfen(
    position: &crate::position::Position,
    ordinal: usize,
    origin: &str,
    origin_class: &str,
) -> PipelineRecord {
    PipelineRecord::imported_from_sfen(
        position,
        ordinal,
        origin,
        origin_class,
        "extend",
        EXTEND_VERSION,
    )
}
