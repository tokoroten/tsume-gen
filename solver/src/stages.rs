//! JSONL pipeline stages: `validate`, `reduce`, `solve`.
//!
//! Each function mutates a [`PipelineRecord`] exactly as the corresponding
//! TypeScript stage in `tools/puzzle-pipeline/src/stages.ts` does, so a Rust
//! stage can replace a TypeScript stage anywhere in the pipe.

use anyhow::Result;
use indexmap::IndexMap;

use crate::canonical::canonicalize;
use crate::exact::{exact_validate_with_rules, prune_solver_graph, PrunedGraph, TsumeRules};
use crate::features;
use crate::jsonl::{
    GraphMoveJson, GraphNodeJson, PipelineRecord, Rejection, Solution, SolveStatsJson,
    SolverGraphJson, Validation, PIPELINE_VERSION,
};
use crate::piece::BLACK;
use crate::position::Position;
use crate::reduce::{self, ReduceOptions, ReduceSetResult, Reduced, TwoKindsOptions};
use crate::sfen::{legacy_sfen_key_of, parse_sfen, to_sfen};
use crate::solver::{non_futile_node_count, SolveStats, Solver};
use crate::strict_material;
use crate::util::{seed_from_hex_prefix, sha256_text};
use crate::validate::{validate_position, Issue, ValidateOptions};

/// The one rejection reason origin provenance has (`docs/legal-data.md` §3).
/// `tools/puzzle-pipeline/src/util.ts` `MISSING_ORIGIN_REASON` spells it
/// identically, and both implementations reject with it *alone*.
pub const MISSING_ORIGIN_REASON: &str = "missing_origin";

/// The `validation.issues` message that accompanies [`MISSING_ORIGIN_REASON`];
/// byte-identical to the TypeScript `MISSING_ORIGIN_MESSAGE`.
pub const MISSING_ORIGIN_MESSAGE: &str =
    "provenance must carry originClass, a non-empty origins and rootSourceId (solver/README.md, Provenance)";

/// Rule B's two reason codes (`docs/rules.md`): the root and the deeper attack
/// nodes of the main-line region `R`.  Same spelling as `RULE_B_REASONS` in
/// `tools/puzzle-pipeline/src/stages.ts`.
pub const RULE_B_REASONS: [&str; 2] = ["first_move_not_unique", "yozume_in_main_line"];

/// A `solve` rejection whose every reason is rule B's: the diagram is a proven
/// mate within the bound, of the target length, of proven minimum length, with
/// complete defence coverage, that passes rules A, C and G -- it has yozume and
/// nothing else is wrong with it.  `solve` keeps the solution of such a record
/// (see [`solve_stage_reported`]); every other rejection is unchanged.
pub fn rule_b_only(reasons: &[String]) -> bool {
    !reasons.is_empty()
        && reasons
            .iter()
            .all(|reason| RULE_B_REASONS.contains(&reason.as_str()))
}

/// Strict-tsume normalisation, structural validation and the canonical hash.
pub fn validate_stage(record: &mut PipelineRecord) {
    // A (re)validated record is a diagram no gate has judged yet.
    record.gate = None;
    if record.rejection.is_some() {
        return;
    }
    // `docs/legal-data.md` §3: a row whose origin is not stated is rejected at the
    // stage boundary, never repaired with a default.  The check runs before the
    // SFEN is even parsed, so `missing_origin` is always the sole reason and both
    // implementations agree on the string and its position without further rules.
    if !record.provenance.is_root_complete() {
        record.stage = "validated".to_string();
        record.validation = Some(Validation {
            valid: false,
            issues: vec![Issue {
                code: MISSING_ORIGIN_REASON.to_string(),
                message: MISSING_ORIGIN_MESSAGE.to_string(),
            }],
            strict_defender_material: true,
        });
        record.rejection = Some(Rejection {
            stage: "validate".to_string(),
            reasons: vec![MISSING_ORIGIN_REASON.to_string()],
            inert_pieces: None,
        });
        return;
    }
    let parsed = parse_sfen(&record.source_sfen);
    record.stage = "validated".to_string();
    let source = match parsed {
        Ok(source) => source,
        Err(error) => {
            record.validation = Some(Validation {
                valid: false,
                issues: vec![Issue {
                    code: "parse".to_string(),
                    message: error.0,
                }],
                strict_defender_material: true,
            });
            record.rejection = Some(Rejection {
                stage: "validate".to_string(),
                reasons: vec!["parse".to_string()],
                inert_pieces: None,
            });
            return;
        }
    };
    let normalized = crate::canonical::normalize_attacker(&source);
    let strict = strict_material::normalize(&normalized, BLACK);
    let issues = validate_position(&strict, ValidateOptions::pipeline());
    let canonical = canonicalize(&strict);
    let valid = issues.is_empty();
    record.normalized_sfen = Some(to_sfen(&strict));
    record.canonical_hash = Some(canonical.hash);
    record.canonical_mirrored = Some(canonical.mirrored);
    record.add_transformations(&[
        "normalize:attacker-black",
        "normalize:strict-defender-remainder-hand",
        "validate:strict-tsume-structure",
    ]);
    if !valid {
        record.rejection = Some(Rejection {
            stage: "validate".to_string(),
            reasons: issues.iter().map(|issue| issue.code.clone()).collect(),
            inert_pieces: None,
        });
    }
    record.validation = Some(Validation {
        valid,
        issues,
        strict_defender_material: true,
    });
}

/// Options for [`reduce_stage`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReduceStageOptions {
    pub reduce: ReduceOptions,
    /// Overrides the seed derived from `sourceId`.
    pub seed: Option<u64>,
    /// Rule D v5: what the stage emits besides the main-line-preserving
    /// reduction (`--length-k`, `--no-rule-b-entry`; K_y is fixed).  The
    /// default (`length_k` 0, rule B entry on) emits one record per input:
    /// the main-line-preserving reduction, or for an input the rule B entry
    /// let in its best length-preserving one or `yozume_not_reducible`.
    pub kinds: TwoKindsOptions,
}

/// The lineage `tool` of a length-preserving reduction's record.
pub const REDUCE_TOOL: &str = "reduce";

/// Rule D v5 (`PIPELINE_VERSION` 0.13.0, `docs/rules.md` "Reduction v5"):
/// seeded multi-order greedy piece removal, in two kinds.
///
/// `record` becomes the **main-line-preserving** reduction (`reduction.kind =
/// "mainline"`, `rank` 0; the input itself when nothing could come off), or its
/// rejection.  The **length-preserving** reductions (`options.kinds.length_k`
/// of them at most) are returned as new records, rank order: each is a copy of
/// the input with `sourceId` `<sourceId>-L<rank>`, one `lineage` entry
/// (`tool: "reduce"`) naming the input, and `reduction.kind = "length"`.
///
/// An input the rule B entry let in (`options.kinds.rule_b_entry`, on by
/// default) has no main-line-preserving reduction: `record` becomes its best
/// length-preserving reduction (`rank` 1) and the others (rank 2 on) are
/// returned as above, all marked `reduction.entry = "rule-b-relaxed"`; when no
/// step removes its yozume it is rejected `yozume_not_reducible`.
pub fn reduce_stage(
    record: &mut PipelineRecord,
    options: &ReduceStageOptions,
    solver: &mut Solver,
) -> Result<Vec<PipelineRecord>> {
    let started = crate::posprof::start();
    let meter = crate::catalog_record::capture_on().then(crate::record_clock::Meter::start);
    let mut out = reduce_stage_inner(record, options, solver);
    // Closed before the profile line is printed (`TSUME_PROFILE`), which is
    // not rule D's cost.
    let cost = meter.map(|meter| meter.finish());
    crate::posprof::flush(&record.source_id, started);
    // `--emit-record`: what rule D on this input took, on the record and on
    // every length-preserving reduction it returned.
    if let Some(cost) = cost {
        if let Ok(children) = out.as_mut() {
            for child in children.iter_mut() {
                child.reduce_cost = Some(cost.clone());
            }
        }
        record.reduce_cost = Some(cost);
    }
    out
}

/// [`reduce_stage`] without the profile line.
fn reduce_stage_inner(
    record: &mut PipelineRecord,
    options: &ReduceStageOptions,
    solver: &mut Solver,
) -> Result<Vec<PipelineRecord>> {
    if !record.is_accepted() {
        return Ok(Vec::new());
    }
    let Some(normalized_sfen) = record.normalized_sfen.clone() else {
        return Ok(Vec::new());
    };
    let original = parse_sfen(&normalized_sfen)?;
    // `--keep-squares` naming this record's attacker king: rule A deletes the
    // king first, so the keep cannot be honoured, and the run stops rather
    // than quietly dropping it (Codex review 2026-11-22).
    if let Some(square) = options
        .reduce
        .keep
        .kept_attacker_king(&original, original.side)
    {
        anyhow::bail!(
            "reduce {}: --keep-squares names square {square}, the attacker king's; rule A deletes it, so it cannot be kept",
            record.source_id
        );
    }
    let base_seed = options
        .seed
        .unwrap_or_else(|| seed_from_hex_prefix(&sha256_text(&record.source_id)));
    record.stage = "reduced".to_string();
    // Rule A is a per-record property: the reducer must know whether it is allowed
    // to leave the attacker's king on the board.
    let reduce_options = ReduceOptions {
        rules: options
            .reduce
            .rules
            .allowing_attacker_king(record.allows_attacker_king()),
        ..options.reduce
    };
    let set = match reduce::reduce_two_kinds_seeded(
        solver,
        &original,
        record.target_mate_ply,
        &reduce_options,
        &options.kinds,
        base_seed,
    ) {
        ReduceSetResult::Rejected { reasons } => {
            record.rejection = Some(Rejection {
                stage: "reduce".to_string(),
                reasons,
                inert_pieces: None,
            });
            return Ok(Vec::new());
        }
        ReduceSetResult::Reduced(set) => *set,
    };
    let stamp = ReductionStamp {
        seed: base_seed,
        attempts: options.reduce.attempts,
        source_canonical_hash: canonicalize(&original).hash,
        entry: set.entry.label(),
        decided_budget: set.decided_budget,
    };
    let parent = record.clone();
    let mut reductions = set.length.into_iter();
    let main_outs = &set.main_line_budget_outs;
    let length_outs = &set.length_budget_outs;
    match set.main_line {
        Some(main) => stamp.apply(record, main, "mainline", 0, set.paths_capped, main_outs),
        // The rule B entry: the record is its best length-preserving reduction
        // (`reduce_two_kinds` rejects an input with none).
        None => match reductions.next() {
            Some(first) => stamp.apply(record, first, "length", 1, false, length_outs),
            None => {
                record.rejection = Some(Rejection {
                    stage: "reduce".to_string(),
                    reasons: vec![match set.entry {
                        reduce::EntryKind::RuleCg => reduce::CG_NOT_REPAIRABLE,
                        _ => reduce::YOZUME_NOT_REDUCIBLE,
                    }
                    .to_string()],
                    inert_pieces: None,
                });
            }
        },
    }
    let first_rank = if stamp.entry.is_some() { 2 } else { 1 };
    let mut length = Vec::new();
    for (index, reduced) in reductions.enumerate() {
        let rank = index as u32 + first_rank;
        let mut child = parent.clone();
        child.source_id = format!("{}-L{rank}", parent.source_id);
        child
            .provenance
            .append_lineage(&parent.source_id, REDUCE_TOOL, PIPELINE_VERSION);
        stamp.apply(&mut child, reduced, "length", rank, false, length_outs);
        length.push(child);
    }
    Ok(length)
}

/// What every record one [`reduce_stage`] call writes has in common.
struct ReductionStamp {
    seed: u64,
    attempts: u32,
    source_canonical_hash: String,
    /// `reduction.entry` of a relaxed entry (`rule-b-relaxed`, or
    /// `rule-cg-relaxed` for the opt-in C/G entry), `None` for the strict one.
    entry: Option<&'static str>,
    /// [`reduce::ReducedSet::decided_budget`] (never serialised).
    decided_budget: u64,
}

impl ReductionStamp {
    /// Writes one reduction into `record`, as the TypeScript `reduceStage` does.
    fn apply(
        &self,
        record: &mut PipelineRecord,
        reduced: Reduced,
        kind: &str,
        rank: u32,
        pi_capped: bool,
        budget_outs: &reduce::BudgetOutTally,
    ) {
        let canonical = canonicalize(&reduced.position);
        record.normalized_sfen = Some(to_sfen(&reduced.position));
        record.target_mate_ply = Some(reduced.target_mate_ply);
        record.canonical_hash = Some(canonical.hash);
        record.canonical_mirrored = Some(canonical.mirrored);
        record.reduction = Some(crate::jsonl::Reduction {
            seed: self.seed.to_string(),
            attempts: self.attempts,
            original_board_pieces: reduced.original_board_pieces,
            final_board_pieces: reduced.final_board_pieces,
            removed: reduced.removed,
            target_mate_ply: reduced.target_mate_ply,
            kind: Some(kind.to_string()),
            rank: Some(rank),
            source_canonical_hash: Some(self.source_canonical_hash.clone()),
            pi_capped,
            entry: self.entry.map(str::to_string),
            minimality: Some(budget_outs.minimality().to_string()),
            budget_out_checks: Some(budget_outs.checks),
            // A record the early stop ended (`ReducedSet::stopped`) names no
            // stage: which check stopped it may depend on `--jobs`.
            budget_out_stages: (!budget_outs.stages.is_empty()).then(|| {
                budget_outs
                    .stages
                    .iter()
                    .map(|(stage, count)| (stage.to_string(), *count))
                    .collect()
            }),
            decided_budget: Some(self.decided_budget),
        });
        record.provenance.discovered_mate_ply = Some(reduced.target_mate_ply);
        if let Some(entry) = self.entry {
            record.add_transformations(&[&format!("reduce:entry-{entry}")]);
        }
        // Rule D v5: both kinds run to their fixpoint (a record the early stop
        // ended, `ReducedSet::stopped`, keeps this entry as every `"unverified"`
        // record since 2026-10-22 does: its `minimality` says the fixpoint is not
        // proven, and readers go by that).  Until 0.12.0 the
        // entry was `reduce:greedy-multiseed-<attempts>[-fixpoint]-v3` (with a
        // `-strict-b` infix for `--strict-rule-b`, P8-016); records written
        // before keep it.
        record.provenance.transformations.push(format!(
            "reduce:{kind}-multiseed-{}-fixpoint-v5",
            self.attempts
        ));
        record.add_transformations(&["reduce:removed-pieces-assigned-to-defender-hand"]);
    }
}

/// The two transformations [`solve_stage`] appends to an accepted record.
pub const SOLVE_TRANSFORMATIONS: [&str; 2] = [
    "solve:iterative-minimum-proof",
    "solve:complete-and-or-defense-graph",
];

/// What [`reduce_accepted`] did to one accepted, solved record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReduceAcceptedOutcome {
    /// Nothing could come off.  The record is the gate's, plus `reduction`.
    Unchanged,
    /// `removed` pieces came off and the smaller diagram passed the same solve
    /// gate again; `solution` is the re-solve's.
    Reduced { removed: usize },
    /// The reducer or the re-solve rejected it (in practice a node budget), so the
    /// record is restored exactly as it arrived.
    Fallback { reasons: Vec<String> },
}

/// [`reduce_accepted`]'s verdict and the nodes the re-solve cost.
#[derive(Clone, Debug)]
pub struct ReduceAcceptedReport {
    pub outcome: ReduceAcceptedOutcome,
    /// Gate nodes of the re-solves (the record's and every length-preserving
    /// reduction's); `0` when no re-solve ran.
    pub resolve_nodes: u64,
    /// The length-preserving reductions (rule D v5) the caller asked for
    /// (`kinds.length_k`), each solved by the same gate; only those it
    /// accepts, rank order.
    pub length: Vec<PipelineRecord>,
}

/// Rule D after a generator's own gate: runs the standard [`reduce_stage`]
/// (strict entry, rule D v5) on a record that `solve` has already accepted, and
/// re-solves it when pieces came off.
///
/// Generators (`extend`, `distill`) accept a candidate on the publishing gate
/// alone, which says nothing about minimality: a guard the new first move no
/// longer needs, or a piece a variant search never tried to drop, survives.
/// This is the one place that makes such a record as small as `reduce` would.
///
/// `reduce.max_ply` and `solve.max_ply` should both be the bound the generator
/// solved under, so the reducer and the re-solve measure the length the gate
/// did.  The target length is kept (`reduce` never shortens a mate), and so is
/// a main line (since 0.14.0 any path of the main-line region, not necessarily
/// the displayed one): the record becomes its main-line-preserving reduction
/// (rule D v5).  With `kinds.length_k` above 0 the length-preserving
/// reductions come back solved in [`ReduceAcceptedReport::length`].
///
/// In the two reduced outcomes `provenance.transformations` ends `reduce:…,
/// reduce:…, solve:…, solve:…` — the pipeline's order — and never carries the
/// `solve:` pair twice.
pub fn reduce_accepted(
    record: &mut PipelineRecord,
    reduce: ReduceOptions,
    kinds: TwoKindsOptions,
    solve: &SolveStageOptions,
    solver: &mut Solver,
) -> Result<ReduceAcceptedReport> {
    let meter = crate::catalog_record::capture_on().then(crate::record_clock::Meter::start);
    let mut report = reduce_accepted_measured(record, reduce, kinds, solve, solver)?;
    // `--emit-record`: rule D and the gate's re-solve, on the record and its
    // length-preserving reductions (a fallback record carries no reduction,
    // so its cost is never read).
    if let Some(meter) = meter {
        let cost = meter.finish();
        for child in report.length.iter_mut() {
            child.reduce_cost = Some(cost.clone());
        }
        record.reduce_cost = Some(cost);
    }
    Ok(report)
}

/// [`reduce_accepted`] before its cost is put on the records.
fn reduce_accepted_measured(
    record: &mut PipelineRecord,
    reduce: ReduceOptions,
    kinds: TwoKindsOptions,
    solve: &SolveStageOptions,
    solver: &mut Solver,
) -> Result<ReduceAcceptedReport> {
    let before = record.clone();
    let fallback = |record: &mut PipelineRecord, reasons, resolve_nodes, length| {
        *record = before.clone();
        Ok(ReduceAcceptedReport {
            outcome: ReduceAcceptedOutcome::Fallback { reasons },
            resolve_nodes,
            length,
        })
    };
    // The gate appended its `solve:` pair last.  The history before it is the
    // parent's, inherited through the clone, so only this tail is moved.
    let had_solve_tail = {
        let transformations = &mut record.provenance.transformations;
        let tail = transformations
            .len()
            .saturating_sub(SOLVE_TRANSFORMATIONS.len());
        let matches = transformations[tail..]
            .iter()
            .map(String::as_str)
            .eq(SOLVE_TRANSFORMATIONS.iter().copied());
        if matches {
            transformations.truncate(tail);
        }
        matches
    };

    let stage = ReduceStageOptions {
        reduce,
        seed: None,
        kinds,
    };
    let children = reduce_stage(record, &stage, solver)?;
    // The length-preserving reductions pass the same gate as the record.
    let mut length = Vec::with_capacity(children.len());
    let mut length_nodes = 0u64;
    for mut child in children {
        child.solution = None;
        let report = solve_stage_reported(&mut child, solve, solver)?;
        length_nodes += report.nodes;
        if report.accepted {
            length.push(child);
        }
    }
    if let Some(rejection) = record.rejection.take() {
        return fallback(record, rejection.reasons, length_nodes, length);
    }
    let removed = record
        .reduction
        .as_ref()
        .map_or(0, |reduction| reduction.removed.len());
    // Compared as positions, not text: a record written before 2026-10-04 spells its
    // hands P L N S G B R, and `reduce_stage` rewrites it in `SFEN_HAND_ORDER`.
    let unchanged_position = match (&record.normalized_sfen, &before.normalized_sfen) {
        (Some(after), Some(earlier)) => legacy_sfen_key_of(after) == legacy_sfen_key_of(earlier),
        (after, earlier) => after == earlier,
    };
    if removed == 0 && unchanged_position {
        record.stage = before.stage.clone();
        if had_solve_tail {
            record.add_transformations(&SOLVE_TRANSFORMATIONS);
        }
        return Ok(ReduceAcceptedReport {
            outcome: ReduceAcceptedOutcome::Unchanged,
            resolve_nodes: length_nodes,
            length,
        });
    }

    record.solution = None;
    let report = solve_stage_reported(record, solve, solver)?;
    if !report.accepted {
        return fallback(record, report.reasons, report.nodes + length_nodes, length);
    }
    // `--emit-record`: the diagram the gate accepted before rule D took pieces
    // off it is a sound but not minimal one, written as its own record.
    // Its record keeps no graph (a negative), so the copy drops it.
    if let Some(facts) = record.gate.as_mut() {
        let mut source = before;
        source.solution = None;
        facts.reduced_from = Some(Box::new(source));
    }
    Ok(ReduceAcceptedReport {
        outcome: ReduceAcceptedOutcome::Reduced { removed },
        resolve_nodes: report.nodes + length_nodes,
        length,
    })
}

/// Options for [`solve_stage`].
#[derive(Clone, Copy, Debug)]
pub struct SolveStageOptions {
    pub max_ply: u32,
    pub node_budget: u64,
    /// The publishing gate.  `solve` is the *last* stage, so this is the full
    /// rule set; only rule A is overridden per record.
    pub rules: TsumeRules,
}

impl Default for SolveStageOptions {
    fn default() -> Self {
        Self {
            max_ply: 11,
            node_budget: 500_000,
            rules: TsumeRules::puzzle(),
        }
    }
}

/// What one [`solve_stage_reported`] call decided and what it cost.
///
/// The stage itself only writes `solution` or `rejection` (both for a rejection
/// by rule B alone, [`rule_b_only`]); callers that budget
/// their own search — `extend`, which solves one candidate per enumerated un-move —
/// need the verdict and the node count without re-reading the record.
#[derive(Clone, Debug, Default)]
pub struct SolveStageReport {
    /// `false` when the record arrived already rejected (or unnormalised) and no
    /// search ran at all.
    pub solved: bool,
    /// The publishing verdict: the record carries a `solution` and no
    /// `rejection` (a rejection by rule B alone keeps its solution too,
    /// [`rule_b_only`]).
    pub accepted: bool,
    /// `rejection.reasons`, empty on acceptance.
    pub reasons: Vec<String>,
    /// Proven minimum mate length, when there was one.
    pub mate_ply: Option<u32>,
    /// Search nodes of the gate: the published `stats.nodes` plus the extra
    /// main-line solve `exact_validate_with_rules` charges separately.
    ///
    /// The minimality re-solve at one ply shorter is *not* included, because its
    /// nodes are deliberately kept out of both counters so `stats.nodes` stays
    /// bit-identical to the TypeScript stage.
    pub nodes: u64,
    /// The designated main line ([`crate::exact::designated_main_line`]), USI,
    /// attacker and defender moves alternating; empty when rules B and C were
    /// never evaluated.
    ///
    /// It rides on the *report*, never on the record: `Solution`'s key order is
    /// part of the TS<->Rust byte-identity contract.  It is here because the
    /// published graph is the pruned runtime graph, from which the `Clean`-first
    /// walk cannot be reconstructed -- a caller that derives its own line from
    /// that graph gets a different, possibly koma-amari line
    /// (`docs/web/generate/DESIGN.md` §12).
    pub main_line: Vec<String>,
    /// Pieces left in the attacker's hand at the end of `main_line`.
    pub leftover_hand: usize,
}

/// Solver v3 item 1 in the publishing gate
/// ([`crate::reduce::constructed_validation`]).  **On by default** (owner
/// decision 2026-10-09 23:00 JST): the verdict, the main line and the pruned
/// graph's shape are the full-width gate's, while the published node counts,
/// features and graph ids become the construction's, and the record says so
/// with `stats.nodesKind: "prober"` (the rating reads it,
/// `DIFFICULTY_MODEL_VERSION` 4).  The CLI turns it off for
/// `TSUME_SOLVE_CONSTRUCT=0`, as the TypeScript `solveStage` does.  When the
/// prober or the construction runs out of budget, the gate's answer is a
/// budget-out (`solver_budget_exhausted`, `incomplete_response_graph`: status
/// `unknown`, `--emit-unresolved`); since solver 1.4.0 the one-thread
/// full-width gate is no longer run again on the same budget (owner
/// 2026-11-05, in line with the owner's rule of 2026-10-23: no re-run on one thread).
/// With `TSUME_SOLVE_CONSTRUCT=0` the full-width gate is the gate, as before.
static SOLVE_CONSTRUCT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether [`SOLVE_CONSTRUCT`] is on: the publishing gate, and the measurements
/// that read the same region ([`crate::exact::measure_main_line_yozume`]), are
/// decided by solver v3's construction where its budget allows.
pub fn solve_construct_on() -> bool {
    SOLVE_CONSTRUCT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turns [`SOLVE_CONSTRUCT`] on or off; set it before any solve starts.
pub fn set_solve_construct(on: bool) {
    SOLVE_CONSTRUCT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Exact bounded solve plus uniqueness/coverage gates, features and the pruned
/// runtime graph.
pub fn solve_stage(
    record: &mut PipelineRecord,
    options: &SolveStageOptions,
    solver: &mut Solver,
) -> Result<()> {
    solve_stage_reported(record, options, solver).map(|_| ())
}

/// [`solve_stage`], also reporting the verdict and the node cost.  The record it
/// writes is identical either way.
///
/// While [`crate::catalog_record::capture_on`] (`--emit-record`), the verdict,
/// the designated main line and the yozume location are also kept beside the
/// record ([`PipelineRecord::gate`], never serialised) for its catalog record.
pub fn solve_stage_reported(
    record: &mut PipelineRecord,
    options: &SolveStageOptions,
    solver: &mut Solver,
) -> Result<SolveStageReport> {
    let started = crate::posprof::start();
    let flush = |record: &PipelineRecord| crate::posprof::flush(&record.source_id, started);
    if !crate::catalog_record::capture_on() {
        let report = solve_stage_gate(record, options, solver);
        flush(record);
        return report;
    }
    record.gate = None;
    let meter = crate::record_clock::Meter::start();
    let report = solve_stage_gate(record, options, solver);
    // Closed before the profile line is printed (`TSUME_PROFILE`).
    let cost = meter.finish();
    flush(record);
    let report = report?;
    if report.solved {
        crate::catalog_record::capture(record, &report, options, cost);
    }
    Ok(report)
}

/// The publishing gate of [`solve_stage_reported`].
fn solve_stage_gate(
    record: &mut PipelineRecord,
    options: &SolveStageOptions,
    solver: &mut Solver,
) -> Result<SolveStageReport> {
    let mut report = SolveStageReport::default();
    if !record.is_accepted() {
        return Ok(report);
    }
    let Some(normalized_sfen) = record.normalized_sfen.clone() else {
        return Ok(report);
    };
    let position = parse_sfen(&normalized_sfen)?;
    let rules = options
        .rules
        .allowing_attacker_king(record.allows_attacker_king());
    let constructed = if SOLVE_CONSTRUCT.load(std::sync::atomic::Ordering::Relaxed) {
        crate::reduce::constructed_gate(
            &position,
            record.target_mate_ply,
            options.max_ply,
            options.node_budget,
            rules,
        )
    } else {
        None
    };
    // `stats.nodesKind: "prober"` marks a solution the construction wrote.  A
    // fast-path budget-out (`budget_out`) writes no solution.
    let prober = constructed.as_ref().is_some_and(|gate| !gate.budget_out);
    let (exact, gate) = match constructed {
        // Decided by the construction, or out of budget on the fast path: in
        // neither case is the full-width gate run (solver 1.4.0).
        Some(constructed) => {
            let crate::reduce::ConstructedGate {
                validation,
                prefilter_nodes,
                budget_out: _,
            } = constructed;
            (validation, Some(prefilter_nodes))
        }
        // The construction off (`TSUME_SOLVE_CONSTRUCT=0`), rule F other than
        // classical, or the test-only `TSUME_FAST_EXACT_FALLBACK=1`.
        None => {
            // The browser gate: the prober's tables are dead weight from here
            // on (`reduce::release_prober`), and wasm32's memory never shrinks.
            #[cfg(target_arch = "wasm32")]
            crate::reduce::release_prober();
            let started = crate::clock::Instant::now();
            crate::reduce::count_full_width_gate();
            let exact = exact_validate_with_rules(
                solver,
                &position,
                record.target_mate_ply,
                options.max_ply,
                options.node_budget,
                true,
                rules,
            );
            crate::reduce::charge_phase(3, started);
            crate::posprof::charge_since(crate::posprof::K::ExactNs, started);
            crate::posprof::add(
                crate::posprof::K::ExactNodes,
                exact.result.stats.nodes + exact.extra_nodes,
            );
            (exact, None)
        }
    };
    // `stats.nodesProof`: the prefilter already proved "mate within
    // `max_ply`" with a fresh prober on this budget, the question
    // `proof_nodes` asks when the mate length is `max_ply` (same count).
    let proof_for = |distance: u32| {
        crate::reduce::proof_nodes_after_gate(
            &position,
            gate.flatten(),
            options.max_ply,
            distance,
            options.node_budget,
        )
    };
    // Rule G's count, when the gate analysed this very graph for this attacker
    // (finding 9: no second walk in the features).
    let inert = (rules.composition_rules
        && !rules.allow_inert_pieces
        && exact.main_line.is_some()
        && exact.result.graph.is_some()
        && exact.result.attacker == position.side)
        .then_some(exact.inert_pieces.len() as u64);
    record.stage = "solved".to_string();
    report.solved = true;
    report.nodes = exact.result.stats.nodes + exact.extra_nodes;
    report.mate_ply = exact.result.distance;
    if let Some(line) = exact.main_line.as_ref() {
        report.main_line = line.moves.iter().map(|usi| usi.to_string()).collect();
        report.leftover_hand = line.leftover_hand;
    }
    let (Some(distance), Some(first_move), true) =
        (exact.result.distance, exact.first_move, exact.accepted)
    else {
        report.reasons = exact.reasons.clone();
        match (exact.result.distance, exact.first_move) {
            // Rejected by rule B alone (owner 2026-10-09, "the solver should say
            // there is yozume"): the solution is written as on acceptance, so a
            // reader gets the mate length, the main-line region's graph and its
            // counters from the record itself.  `stats.nodesNonFutile` stays
            // accepted-only (owner decision 2026-09-25), and the verdict is the
            // `rejection` below, unchanged.  Same in the TypeScript `solveStage`.
            (Some(distance), Some(first_move)) if rule_b_only(&exact.reasons) => {
                let proof = prober.then(|| proof_for(distance));
                record.solution = Some(Box::new(published_solution(
                    &position, &exact, distance, first_move, false, proof, inert,
                )));
            }
            // A failed re-solve keeps an earlier `solution` as it always has, minus
            // `stats.nodesNonFutile`, which only an accepted record carries (owner
            // decision 2026-09-25); the TypeScript `solveStage` does the same.
            _ => {
                if let Some(solution) = record.solution.as_mut() {
                    solution.stats.nodes_non_futile = None;
                }
            }
        }
        record.rejection = Some(Rejection {
            stage: "solve".to_string(),
            reasons: exact.reasons,
            // Rule G names the squares it objects to, so a rejected record can be
            // read without re-solving it.
            inert_pieces: if exact.inert_pieces.is_empty() {
                None
            } else {
                Some(exact.inert_pieces)
            },
        });
        return Ok(report);
    };
    report.accepted = true;
    let canonical = canonicalize(&position);
    record.canonical_hash = Some(canonical.hash);
    record.canonical_mirrored = Some(canonical.mirrored);
    let proof = prober.then(|| proof_for(distance));
    record.solution = Some(Box::new(published_solution(
        &position, &exact, distance, first_move, true, proof, inert,
    )));
    record.provenance.discovered_mate_ply = Some(distance);
    record.add_transformations(&SOLVE_TRANSFORMATIONS);
    Ok(report)
}

/// The `solution` block `solve` publishes for a proven mate: on acceptance, and
/// on a rejection by rule B alone ([`rule_b_only`]).  `non_futile` writes
/// `stats.nodesNonFutile`, which only an accepted record carries.  `prober` is
/// `Some` when the construction solved it: `stats.nodesProof` (its inner value,
/// [`crate::reduce::proof_nodes`]) and `stats.nodesKind: "prober"`.  `inert`
/// is rule G's piece count when the gate analysed this graph already.
#[allow(clippy::too_many_arguments)]
fn published_solution(
    position: &Position,
    exact: &crate::exact::ExactValidation,
    distance: u32,
    first_move: crate::position::Usi,
    non_futile: bool,
    prober: Option<Option<u64>>,
    inert: Option<u64>,
) -> Solution {
    let features = features::extract_with_inert(position, &exact.result, inert);
    let graph = exact
        .result
        .graph
        .as_ref()
        .expect("solve stage requires a graph");
    let pruned = prune_solver_graph(graph);
    Solution {
        mate_ply: distance,
        unique_first_move: exact.unique_first_move,
        first_move: first_move.to_string(),
        // Rule B (docs/rules.md): `yozumeAllowed` said whether the length
        // exempted this puzzle (0.7.0-0.11.0, 9 plies and longer).  Since 0.12.0
        // rule B applies at every length, so it is always `false`; the field
        // stays for format compatibility.  The two counters say how wide the
        // main-line region is.  Key order is part of the TS<->Rust byte-identity
        // contract -- `tools/puzzle-pipeline/src/stages.ts` `solveStage` builds
        // these three in the same place.
        yozume_allowed: false,
        attacker_branch_nodes: exact.attacker_branch_nodes as u64,
        max_attacker_branching: exact.max_attacker_branching as u64,
        captured_promo_alt_nodes: exact
            .main_line
            .as_ref()
            .map_or(0, |line| line.captured_promo_alt_nodes as u64),
        // Only the runtime-relevant subgraph is persisted; features above were
        // extracted from the complete search graph.
        graph: pruned_graph_to_json(&pruned),
        stats: SolveStatsJson {
            nodes_non_futile: non_futile.then(|| non_futile_node_count(graph, distance)),
            nodes_proof: prober.flatten(),
            nodes_kind: prober.map(|_| nodes_kind_prober().to_string()),
            ..stats_to_json(&exact.result.stats)
        },
        defense_coverage_verified: exact.defense_coverage_verified,
        features,
        difficulty: None,
        quality: None,
    }
}

/// Converts solver stats to the published shape.
pub fn stats_to_json(stats: &SolveStats) -> SolveStatsJson {
    SolveStatsJson {
        nodes: stats.nodes,
        cache_hits: stats.cache_hits,
        budget_exhausted: stats.budget_exhausted,
        max_ply: stats.max_ply,
        exhaustive_within_bound: stats.exhaustive_within_bound,
        nodes_non_futile: None,
        nodes_proof: None,
        nodes_kind: None,
    }
}

/// `stats.nodesKind` of a solution the construction wrote: `stats.nodes` is the
/// classical df-pn prober's node count, `stats.cacheHits` the construction's
/// transpositions, and `stats.nodesNonFutile` and the features are counted on
/// the constructed graph.  Same spelling as `NODES_KIND_PROBER` in
/// `tools/puzzle-pipeline/src/stages.ts`.  Only with the prober's leaf search
/// off (`TSUME_CDFPN_LEAF=0`) since 2026-10-27: otherwise
/// [`NODES_KIND_PROBER_LEAF3`].
pub const NODES_KIND_PROBER: &str = "prober";

/// `stats.nodesKind` of a solution the construction wrote with the prober's
/// leaf search on and its loop reading every child from the table before each
/// descent (2026-10-26 to 10-27): its node counts are smaller than
/// [`NODES_KIND_PROBER`]'s, so the rating maps them with their own table
/// (owner 2026-10-27, `DIFFICULTY_MODEL_VERSION` 6).  No longer written
/// (1.0.0; `TSUME_CDFPN_REREAD=1` is gone); kept for the records that carry it.
pub const NODES_KIND_PROBER_LEAF3: &str = "prober-leaf3";

/// `stats.nodesKind` of a solution whose one-thread prober kept its
/// children's numbers ([`crate::cdfpn::child_cache`], 2026-10-27 to 10-28;
/// owner: node counts may change, the rating maps each kind with its own
/// table).  No longer written on one thread (1.0.0; `TSUME_CDFPN_EPS=0` is
/// gone); kept for the records that carry it.  A parallel prober still writes
/// it with [`NODES_KIND_PAR_SUFFIX`].
pub const NODES_KIND_PROBER_V2: &str = "prober-v2";

/// `stats.nodesKind` of a solution whose one-thread prober, besides keeping
/// its children's numbers, searched a child until it passed the second-best
/// sibling by a quarter (the 1+ε threshold, since 2026-10-28; node counts
/// 0.84-1.00 of [`NODES_KIND_PROBER_V2`]'s by mate length,
/// `docs/experiments/solver-scaling/README.md` §15.8).  A parallel prober does
/// not use it (its kind stays `prober-v2-par`).
pub const NODES_KIND_PROBER_V3: &str = "prober-v3";

/// The suffix of the `stats.nodesKind` a parallel prober writes
/// (`--prober-threads` above 1, 2026-10-27): its counts depend on the threads,
/// so they are told apart from the one-thread kind they extend.
pub const NODES_KIND_PAR_SUFFIX: &str = "-par";

/// The `stats.nodesKind` this process's prober writes: [`NODES_KIND_PROBER`]
/// with the leaf search off (`TSUME_CDFPN_LEAF=0`, tests only), else
/// [`NODES_KIND_PROBER_V3`]; a parallel prober (which never uses the 1+ε
/// threshold) writes `prober-v2-par` ([`NODES_KIND_PAR_SUFFIX`]).  The CLI
/// refuses the leaf search off on several threads (1.0.0), so it never writes
/// `prober-par`; the library still names it.
pub fn nodes_kind_prober() -> &'static str {
    nodes_kind_for(
        crate::cdfpn::leaf_search(),
        crate::cdfpn::prober_threads() > 1,
    )
}

/// [`nodes_kind_prober`] for the given settings.
pub fn nodes_kind_for(leaf: bool, parallel: bool) -> &'static str {
    match (leaf, parallel) {
        (false, false) => NODES_KIND_PROBER,
        (false, true) => "prober-par",
        (true, false) => NODES_KIND_PROBER_V3,
        (true, true) => "prober-v2-par",
    }
}

/// Converts a pruned graph to the published `SolverGraph` shape.
pub fn pruned_graph_to_json(graph: &PrunedGraph) -> SolverGraphJson {
    let mut nodes = IndexMap::with_capacity(graph.nodes.len());
    for node in graph.nodes.iter() {
        nodes.insert(
            format!("n{}", node.id),
            GraphNodeJson {
                id: format!("n{}", node.id),
                side: if node.attack { "attack" } else { "defense" }.to_string(),
                sfen: node.sfen.clone(),
                outcome: node.outcome.as_str().to_string(),
                distance: node.distance,
                moves: node
                    .moves
                    .iter()
                    .map(|m| GraphMoveJson {
                        usi: m.usi().to_string(),
                        outcome: m.outcome().as_str().to_string(),
                        distance: m.distance(),
                        next: m.next().map(|id| format!("n{id}")),
                        refutation: m.refutation().map(|usi| usi.to_string()),
                        // Rule F: `prune_solver_graph` already dropped every
                        // flagged reply, so this is always `false` here.
                        useless: m.useless(),
                    })
                    .collect(),
            },
        );
    }
    SolverGraphJson {
        root: format!("n{}", graph.root),
        nodes,
    }
}

/// Converts the complete (unpruned) search graph to the published shape.
pub fn full_graph_to_json(graph: &crate::solver::Graph) -> SolverGraphJson {
    let mut nodes = IndexMap::with_capacity(graph.len());
    // The graph keeps no positions; this visits them in id order.
    graph.for_each_position(|node, position| {
        let id = format!("n{}", node.id());
        nodes.insert(
            id.clone(),
            GraphNodeJson {
                id,
                side: node.side_label().to_string(),
                sfen: to_sfen(position),
                outcome: node.outcome().as_str().to_string(),
                distance: node.distance(),
                moves: node
                    .moves()
                    .iter()
                    .map(|m| GraphMoveJson {
                        usi: m.usi().to_string(),
                        outcome: m.outcome().as_str().to_string(),
                        distance: m.distance(),
                        next: m.next().map(|id| format!("n{id}")),
                        refutation: m.refutation().map(|usi| usi.to_string()),
                        useless: m.useless(),
                    })
                    .collect(),
            },
        );
    });
    SolverGraphJson {
        root: format!("n{}", graph.root),
        nodes,
    }
}

/// Parses `normalizedSfen`, or the source SFEN when the record has not been
/// normalised yet.
pub fn record_position(record: &PipelineRecord) -> Result<Position> {
    let sfen = record
        .normalized_sfen
        .as_deref()
        .unwrap_or(&record.source_sfen);
    Ok(parse_sfen(sfen)?)
}
