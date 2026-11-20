//! Catalog records: the one shape `catalog.db` takes in (`catalog-record/2`,
//! `tools/catalog/record.py`, `docs/catalog/INGEST.md`), written by the
//! generators themselves (`--emit-record`, owner 2026-10-12).
//!
//! The record of a diagram is built from what the **publishing gate**
//! ([`crate::stages::solve_stage_reported`]) decided about it: the reasons, the
//! designated main line ([`crate::exact::designated_main_line`], the `Clean`-first
//! walk -- never a smallest-USI walk), the spare hand at its end, rule G's
//! pieces, the yozume location on the published graph, and the solution graph
//! for a sound diagram only (a negative keeps the facts, not the graph).
//! [`build`] mirrors `tools/catalog/adapters/common.py` `judge_solved` field for
//! field; `tools/catalog/tests/test_emit_record.py` runs both on the same gate
//! output and compares them.
//!
//! The gate's facts ride on the record as [`GateFacts`] in
//! [`PipelineRecord::gate`], a field serde skips, so every `PipelineRecord` the
//! commands write is byte for byte what they write without `--emit-record`.  The
//! facts are only collected while [`set_capture`] is on (the CLI turns it on with
//! `--emit-record`); a record's facts are the ones of its **last** gate solve, and
//! [`build`] refuses facts of another diagram (`normalizedSfen` differs) or of a
//! record a later stage rejected.
//!
//! Since 2026-10-27 (owner, `docs/experiments/emit-record-fields/README.md`)
//! a record also carries when and at what cost it was judged and reduced
//! (`judgement.solvedAt` / `wallMs` / `cpuMs`, `reduce.reducedAt` / `wallMs` /
//! `cpuMs`, [`crate::record_clock`]), the minimality certificate of its
//! reduction (`reduce.minimality`, `budgetOutChecks`, `budget`) and the build
//! and machine that wrote it (`producer.solverBuild`, `producer.host`).

use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Map, Value};

use crate::exact::TsumeRules;
use crate::jsonl::{InertPiece, PipelineRecord, SolverGraphJson, PIPELINE_VERSION};
use crate::record_clock::{Cost, SOLVER_BUILD};
use crate::solver::Solver;
use crate::stages::{solve_stage_reported, SolveStageOptions, SolveStageReport};

/// `tools/catalog/record.py` `RECORD_VERSION`.
pub const RECORD_VERSION: &str = "catalog-record/2";
/// `mainLine.source` of every line this module writes: the gate's designated
/// main line (`record.py` `DESIGNATED_SOURCES` looks for `designated`, and the
/// `gate` prefix turns on the koma-amari consistency check).
pub const LINE_SOURCE: &str = "gate:designated (tsume-solver --emit-record)";
/// `catalog_db.YOZUME_PROFILE_VERSION`.
pub const YOZUME_PROFILE_VERSION: u32 = 1;
/// `adapters/common.py` `YOZUME_DETAIL_VERSION`: the `rejudge.profile` shape
/// (4 since `PIPELINE_VERSION` 0.17.0, merged promotion twins apart).
pub const YOZUME_DETAIL_VERSION: u32 = 4;
/// Pseudo-reason of rule D: a gate pass the reducer could still shrink.
pub const RULE_D: &str = "rule_d_reducible";
/// The gate proved there is no forced mate within the bound: not a tsume, never
/// catalogued (owner 2026-10-12), so no record is written for it.
pub const NO_MATE_PROVEN: &str = "no_forced_mate_within_bound";

const RULE_B: [&str; 2] = ["first_move_not_unique", "yozume_in_main_line"];
const NO_MATE: [&str; 4] = [
    "no_forced_mate_within_bound",
    "target_mate_ply_changed",
    "minimum_length_not_proven",
    "defense_coverage_incomplete",
];
const SEARCH: [&str; 3] = [
    "incomplete_response_graph",
    "solver_budget_exhausted",
    // Rule G clause (e) out of budget (`PIPELINE_VERSION` 0.15.0).
    crate::inert::RULE_G_BUDGET_EXHAUSTED,
];
const RULE_A: [&str; 2] = ["attacker_king_present", "attacker_king_required"];
const RULE_C: [&str; 1] = ["koma_amari"];
const RULE_G: [&str; 3] = ["inert_attacker_piece", "inert_defender_piece", "dead_piece"];
/// `record.py` `ROUTES`.
pub const ROUTES: [&str; 6] = [
    "extend",
    "mine",
    "distill",
    "model_sampled",
    "reduce",
    "import",
];

static CAPTURE: AtomicBool = AtomicBool::new(false);

/// Turns the collection of [`GateFacts`] on or off; set it before any solve starts.
pub fn set_capture(on: bool) {
    CAPTURE.store(on, Ordering::Relaxed);
}

/// Whether [`crate::stages::solve_stage_reported`] records [`GateFacts`].
pub fn capture_on() -> bool {
    CAPTURE.load(Ordering::Relaxed)
}

/// What one gate solve decided about a record's diagram, kept beside the record
/// (never serialised) for [`build`].
#[derive(Clone, Debug, Default)]
pub struct GateFacts {
    /// The `normalizedSfen` the gate solved.
    pub sfen: String,
    /// The designated main line (empty when rules B and C were never evaluated).
    pub main_line: Vec<String>,
    pub leftover_hand: usize,
    /// Proven minimum mate length, when there was one.
    pub mate_ply: Option<u32>,
    /// The gate's node budget (the last field of `solveVersion`).
    pub node_budget: u64,
    /// The gate's `--max-ply`.
    pub max_ply: u32,
    /// `rejection.reasons` of the gate, empty on acceptance.
    pub reasons: Vec<String>,
    /// `rejection.inertPieces` of the gate.
    pub inert_pieces: Vec<InertPiece>,
    /// The record carried a `solution` after the gate.
    pub had_solution: bool,
    /// `solution.stats.nodes` after the gate.
    pub nodes: Option<u64>,
    /// `catalog_db.yozume_profile` (v1) of that solution along `main_line`.
    pub yozume_profile: Option<Value>,
    /// `adapters/common.py` `yozume_detail` (v4) of that solution along `main_line`.
    pub yozume_detail: Option<Value>,
    /// The record as the gate accepted it before rule D took pieces off it
    /// ([`crate::stages::reduce_accepted`]), with its own facts: a sound but
    /// not minimal diagram, written as its own (negative) record.
    pub reduced_from: Option<Box<PipelineRecord>>,
    /// The record the catalogue is to see instead of the written one, when a
    /// generator published it from another bound than the mate length the
    /// catalogue judges at: the same diagram judged at its mate length, with
    /// its own facts (`distill`, module docs "The gate at the mate length").
    pub view: Option<Box<PipelineRecord>>,
    /// When the gate finished and what it took (`judgement.solvedAt`,
    /// `wallMs`, `cpuMs`).
    pub cost: Option<Cost>,
}

/// Records the facts of the gate solve that just ran on `record`.
pub fn capture(
    record: &mut PipelineRecord,
    report: &SolveStageReport,
    options: &SolveStageOptions,
    cost: Cost,
) {
    let rejection = record
        .rejection
        .as_ref()
        .filter(|rejection| rejection.stage == "solve");
    let reasons = rejection.map_or_else(Vec::new, |r| r.reasons.clone());
    let inert_pieces = rejection
        .and_then(|r| r.inert_pieces.clone())
        .unwrap_or_default();
    let solution = record.solution.as_deref();
    let graph = solution.map(|solution| &solution.graph);
    let line = &report.main_line;
    record.gate = Some(Box::new(GateFacts {
        sfen: record.normalized_sfen.clone().unwrap_or_default(),
        main_line: line.clone(),
        leftover_hand: report.leftover_hand,
        mate_ply: report.mate_ply,
        node_budget: options.node_budget,
        max_ply: options.max_ply,
        reasons,
        inert_pieces,
        had_solution: solution.is_some(),
        nodes: solution.map(|solution| solution.stats.nodes),
        yozume_profile: graph.and_then(|graph| yozume_profile(graph, line)),
        yozume_detail: graph.and_then(|graph| yozume_detail(graph, line)),
        reduced_from: None,
        view: None,
        cost: Some(cost),
    }));
}

/// The correct moves of a node: `MATE` at the node's own distance.
fn correct(node: &crate::jsonl::GraphNodeJson) -> Vec<&str> {
    node.moves
        .iter()
        .filter(|m| m.outcome == "MATE" && m.distance == node.distance)
        .map(|m| m.usi.as_str())
        .collect()
}

/// Walks `line` from the root; calls `visit(ply, node, usi)` on every node met.
fn walk<'a>(
    graph: &'a SolverGraphJson,
    line: &[String],
    mut visit: impl FnMut(usize, &'a crate::jsonl::GraphNodeJson, &str),
) {
    let mut node_id = Some(graph.root.as_str());
    for (ply, usi) in line.iter().enumerate() {
        let Some(node) = node_id.and_then(|id| graph.nodes.get(id)) else {
            break;
        };
        visit(ply, node, usi);
        node_id = node
            .moves
            .iter()
            .find(|m| m.usi == *usi)
            .and_then(|m| m.next.as_deref());
        if node_id.is_none() {
            break;
        }
    }
}

/// Rule B's merged promotion twin ([`crate::exact::is_merged_promotion_twin`])
/// read off a published graph: two correct moves `m` / `m+` (the same four
/// USI bytes, one with `+`), and the longest replies of their two children --
/// the published graph keeps exactly the main-line candidates `P` there, rule
/// F's and 変同駒余り's replies are pruned -- the same non-empty set, every one
/// a board move onto the twins' destination.  The root only while
/// [`crate::exact::ROOT_TWIN_MERGES`] is on.  `adapters/common.py` mirrors it.
fn is_merged_twin_node(graph: &SolverGraphJson, node: &crate::jsonl::GraphNodeJson) -> bool {
    if node.side != "attack" || (node.id == graph.root && !crate::exact::ROOT_TWIN_MERGES) {
        return false;
    }
    let moves: Vec<&crate::jsonl::GraphMoveJson> = node
        .moves
        .iter()
        .filter(|m| m.outcome == "MATE" && m.distance == node.distance)
        .collect();
    let [a, b] = moves.as_slice() else {
        return false;
    };
    let (ua, ub) = (a.usi.as_bytes(), b.usi.as_bytes());
    let twin = ua.len() >= 4
        && ub.len() >= 4
        && ua[1] != b'*'
        && ua[..4] == ub[..4]
        && (ua.len() == 5) != (ub.len() == 5);
    if !twin {
        return false;
    }
    let to = &ua[2..4];
    let replies = |m: &crate::jsonl::GraphMoveJson| -> Option<Vec<&str>> {
        let child = graph.nodes.get(m.next.as_deref()?)?;
        let standing: Vec<&crate::jsonl::GraphMoveJson> =
            child.moves.iter().filter(|r| !r.useless).collect();
        let longest = standing
            .iter()
            .map(|r| r.distance.map_or(-1i64, i64::from))
            .max()?;
        let mut out = Vec::new();
        for r in standing {
            if r.distance.map_or(-1i64, i64::from) != longest {
                continue;
            }
            let u = r.usi.as_bytes();
            if u.len() < 4 || u[1] == b'*' || &u[2..4] != to {
                return None;
            }
            out.push(r.usi.as_str());
        }
        out.sort_unstable();
        Some(out)
    };
    match (replies(a), replies(b)) {
        (Some(pa), Some(pb)) => !pa.is_empty() && pa == pb,
        _ => false,
    }
}

fn is_yozume_node(graph: &SolverGraphJson, node: &crate::jsonl::GraphNodeJson) -> bool {
    node.side == "attack"
        && node.distance.unwrap_or(0) >= 3
        && correct(node).len() >= 2
        && !is_merged_twin_node(graph, node)
}

/// `catalog_db.yozume_profile` (v1): along the main line, the attack nodes of
/// distance 3 or more with two or more correct moves.  `None` without a line.
pub fn yozume_profile(graph: &SolverGraphJson, line: &[String]) -> Option<Value> {
    if line.is_empty() {
        return None;
    }
    let (mut count, mut first, mut most, mut last) = (0u64, None::<usize>, 1usize, false);
    walk(graph, line, |ply, node, _| {
        if node.side == "attack" {
            let k = correct(node).len();
            if is_yozume_node(graph, node) {
                count += 1;
                first.get_or_insert(ply);
                most = most.max(k);
            }
            if node.distance == Some(1) && k >= 2 {
                last = true;
            }
        }
    });
    let anywhere = count > 0 || graph.nodes.values().any(|node| is_yozume_node(graph, node));
    Some(json!({
        "count": count,
        "finalAlternatives": last,
        "firstPly": first,
        "inVariation": anywhere,
        "maxCorrect": most,
        "v": YOZUME_PROFILE_VERSION,
    }))
}

/// `adapters/common.py` `yozume_detail` (v4): where the yozume is -- every
/// attack node on the main line of distance 3 or more with several correct
/// moves (the line's move and the others), and how many such nodes the graph
/// has off the line.  `None` without a line.
///
/// v4 (`PIPELINE_VERSION` 0.17.0): a merged promotion twin
/// ([`crate::exact::is_merged_promotion_twin`], 「取られる成り駒の非限定」) is not
/// yozume.  The main line's ones are listed apart in `mergedTwins`, each with
/// `"kind": "mergedTwin"`, and the graph's others are counted in
/// `offMainLineMergedTwins`.
pub fn yozume_detail(graph: &SolverGraphJson, line: &[String]) -> Option<Value> {
    if line.is_empty() {
        return None;
    }
    let mut on = Vec::new();
    let mut merged = Vec::new();
    let mut seen = std::collections::HashSet::new();
    walk(graph, line, |ply, node, usi| {
        seen.insert(node.id.as_str());
        if node.side == "attack" {
            let moves = correct(node);
            if moves.len() >= 2 && is_merged_twin_node(graph, node) {
                let alternatives: Vec<&str> = moves.into_iter().filter(|m| *m != usi).collect();
                merged.push(json!({
                    "alternatives": alternatives, "kind": "mergedTwin", "move": usi, "ply": ply,
                }));
            } else if moves.len() >= 2 && node.distance.unwrap_or(0) >= 3 {
                let mut alternatives: Vec<&str> = moves.into_iter().filter(|m| *m != usi).collect();
                alternatives.sort_unstable();
                on.push(json!({ "alternatives": alternatives, "move": usi, "ply": ply }));
            }
        }
    });
    let off_line = |id: &String| !seen.contains(id.as_str());
    let off = graph
        .nodes
        .iter()
        .filter(|(id, node)| off_line(id) && is_yozume_node(graph, node))
        .count();
    let off_merged = graph
        .nodes
        .iter()
        .filter(|(id, node)| off_line(id) && is_merged_twin_node(graph, node))
        .count();
    Some(json!({
        "mainLine": on,
        "mergedTwins": merged,
        "offMainLine": off,
        "offMainLineMergedTwins": off_merged,
        "scope": "graph",
        "v": YOZUME_DETAIL_VERSION,
    }))
}

/// `record.py` `status_of`.
pub fn status_of(reasons: &[String], proven: bool) -> &'static str {
    let has = |set: &[&str]| reasons.iter().any(|r| set.contains(&r.as_str()));
    if has(&SEARCH) && reasons.iter().all(|r| SEARCH.contains(&r.as_str())) {
        return "unknown";
    }
    if has(&NO_MATE) || !proven {
        return "nomate";
    }
    if has(&RULE_B) {
        return "yozume";
    }
    if !reasons.is_empty() {
        return "defect";
    }
    "sound"
}

/// `record.py` `rule_f_verdict_of` over sorted reasons (rule D is not the gate's).
pub fn rule_f_verdict_of(reasons: &[String]) -> String {
    let gate: Vec<&str> = reasons
        .iter()
        .map(String::as_str)
        .filter(|r| *r != RULE_D)
        .collect();
    if gate.is_empty() {
        "pass".to_string()
    } else {
        format!("fail:{}", gate.join(","))
    }
}

/// The rule columns of schema v2.1 (`tools/catalog/schema.py`
/// `RULES_FROM_COLUMNS`, `record.py` `rules_of`) from the verdict, `has_yozume`
/// and the reduce status: `1` broken, `0` checked and clean, `null` unchecked.
pub fn rules_of(
    gate_reasons: Option<&[String]>,
    has_yozume: Option<u8>,
    reduce_status: &str,
) -> Value {
    let known = |set: &[&str]| -> bool {
        gate_reasons.is_some_and(|reasons| reasons.iter().any(|r| set.contains(&r.as_str())))
    };
    let checked = gate_reasons.is_some_and(|reasons| {
        reasons.iter().all(|r| {
            RULE_A.contains(&r.as_str())
                || RULE_B.contains(&r.as_str())
                || RULE_C.contains(&r.as_str())
                || RULE_G.contains(&r.as_str())
        })
    });
    let rule = |set: &[&str]| -> Value {
        if known(set) {
            json!(1)
        } else if checked {
            json!(0)
        } else {
            Value::Null
        }
    };
    let has =
        |reason: &str| gate_reasons.is_some_and(|reasons| reasons.iter().any(|r| r == reason));
    let dual = if known(&RULE_B) || has_yozume == Some(1) {
        json!(1)
    } else if has_yozume == Some(0) && checked {
        json!(0)
    } else {
        Value::Null
    };
    let kind = match (has("first_move_not_unique"), has("yozume_in_main_line")) {
        (true, true) => json!("both"),
        (true, false) => json!("first_move"),
        (false, true) => json!("main_line"),
        _ if has_yozume == Some(1) => json!("region"),
        _ => Value::Null,
    };
    let reducible = match reduce_status {
        "reducible" => json!(1),
        "minimal" | "reduced" => json!(0),
        _ => Value::Null,
    };
    let futile = if gate_reasons.is_some() {
        json!(0)
    } else {
        Value::Null
    };
    json!({
        "dualKind": kind,
        "futileInterpositionJudged": futile,
        "hasAttackerKing": rule(&RULE_A),
        "hasDual": dual,
        "hasInertPiece": rule(&RULE_G),
        "hasKomaAmari": rule(&RULE_C),
        "isReducible": reducible,
    })
}

/// Where a record's catalogue fields that the gate does not decide come from.
#[derive(Clone, Debug)]
pub struct RecordContext {
    /// `S-<commit>[-tag]`: the binary; `solveVersion` is
    /// `<solver_id>/<PIPELINE_VERSION>/<gate node budget>`.
    pub solver_id: String,
    /// `producer.producedBy` (`docs/catalog/CATALOG_ORIGINS.md` §3.4).
    pub produced_by: String,
    /// `producer.route` (one of [`ROUTES`]).
    pub route: String,
    /// `producer.generation`.
    pub generation: u64,
    /// `reduce.version` of a record that carries a `reduction`
    /// (`D-v5/<PIPELINE_VERSION>/<reduce node budget>`); `None` leaves its reduce
    /// status `unknown`.
    pub reduce_version: Option<String>,
    /// The command reduced the diagrams itself (`reduce`, `reduce-accepted`,
    /// `extend`, `distill`), so a record's `reduction` is this run's and the
    /// record carries its certificate and cost ([`REDUCE_FACTS`]); `false`
    /// for `solve`, which only passes an input's `reduction` on (the fields
    /// are then `null`).
    pub certify: bool,
}

/// `true` when `version` is a rule D v5 version, `D-v5/<x.y.z>/<budget>`
/// (`reduce_of` reads a `reduction` as v5).
pub fn valid_reduce_version(version: &str) -> bool {
    let Some(rest) = version.strip_prefix("D-v5/") else {
        return false;
    };
    let mut parts = rest.split('/');
    let (Some(pipeline), Some(budget), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    let numbers = pipeline.split('.').collect::<Vec<_>>();
    numbers.len() == 3
        && numbers
            .iter()
            .chain(std::iter::once(&budget))
            .all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// `true` when `id` is a solver id `record.py` accepts (`S-<7 hex>[tag]`).
pub fn valid_solver_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("S-") else {
        return false;
    };
    let bytes = rest.as_bytes();
    bytes.len() >= 7
        && bytes[..7]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        && bytes[7..]
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || b"+._-".contains(b))
}

/// What [`build`] adds beyond the gate's facts and the record's own fields.
#[derive(Clone, Debug, Default)]
pub struct Extras {
    /// Reasons beside the gate's (rule D's [`RULE_D`] for a reducible diagram).
    pub reasons: Vec<String>,
    /// `reduce` block; `None` derives it from the record's `reduction`.
    pub reduce: Option<Value>,
    /// `defects.removablePieces`.
    pub removable: Option<Value>,
    pub links: Vec<Value>,
    pub superseded_by: Option<String>,
}

/// Why [`build`] wrote nothing.
pub type Skip = &'static str;

/// The `reduce` keys of the minimality certificate and the reduction's cost
/// (2026-10-27): `minimality` (`verified` / `unverified`), `budgetOutChecks`,
/// `budget` (the decided ladder rung's budget), `reducedAt`, `wallMs`, `cpuMs`.
/// [`build`] fills each one the block does not hold yet from the record
/// ([`reduce_facts`]); `null` when the command did not reduce it.
pub const REDUCE_FACTS: [&str; 6] = [
    "budget",
    "budgetOutChecks",
    "cpuMs",
    "minimality",
    "reducedAt",
    "wallMs",
];

/// The [`REDUCE_FACTS`] of `record`, in that order, when the command reduced
/// it ([`RecordContext::certify`]; else all `null`): the certificate from its
/// `reduction` when this run made it (only this process's stamp sets
/// [`crate::jsonl::Reduction::decided_budget`], which is never read from an
/// input: an input's own older `reduction`, kept by a refused input or a
/// `reduce-accepted` fallback, certifies nothing here), the cost from
/// [`PipelineRecord::reduce_cost`] (`null` when rule D never ran on it).
pub fn reduce_facts(record: &PipelineRecord, ctx: &RecordContext) -> [Value; 6] {
    if !ctx.certify {
        return Default::default();
    }
    let reduction = fresh_reduction(record);
    let cost = record.reduce_cost.as_ref();
    [
        json!(reduction.and_then(|reduction| reduction.decided_budget)),
        json!(reduction.and_then(|reduction| reduction.budget_out_checks)),
        json!(cost.and_then(|cost| cost.cpu_ms)),
        json!(reduction.and_then(|reduction| reduction.minimality.as_deref())),
        json!(cost.map(|cost| cost.at.as_str())),
        json!(cost.map(|cost| cost.wall_ms)),
    ]
}

/// `record`'s `reduction` when this run's rule D wrote it (see [`reduce_facts`]).
pub fn fresh_reduction(record: &PipelineRecord) -> Option<&crate::jsonl::Reduction> {
    record
        .reduction
        .as_ref()
        .filter(|reduction| reduction.decided_budget.is_some())
}

/// Puts the [`REDUCE_FACTS`] `facts` into the `reduce` block where it does
/// not hold them yet.
fn complete_reduce(reduce: &mut Value, facts: [Value; 6]) {
    if let Value::Object(map) = reduce {
        for (key, value) in REDUCE_FACTS.iter().zip(facts) {
            map.entry(key.to_string()).or_insert(value);
        }
    }
}

/// The `reduce` cost keys (`reducedAt`, `wallMs`, `cpuMs`) of `record`'s
/// reduction, for a block [`build`] makes from another record's (a refused
/// input's, a reducible source's).
pub fn reduce_cost_into(reduce: &mut Value, record: &PipelineRecord, ctx: &RecordContext) {
    let facts = reduce_facts(record, ctx);
    if let Value::Object(map) = reduce {
        for (key, value) in REDUCE_FACTS.iter().zip(facts) {
            if matches!(*key, "reducedAt" | "wallMs" | "cpuMs") {
                map.insert(key.to_string(), value);
            }
        }
    }
}

/// The `judgement` keys of the gate's cost: `solvedAt`, `wallMs`, `cpuMs`.
fn judgement_cost(facts: &GateFacts) -> (Value, Value, Value) {
    match facts.cost.as_ref() {
        Some(cost) => (json!(cost.at), json!(cost.wall_ms), json!(cost.cpu_ms)),
        None => (Value::Null, Value::Null, Value::Null),
    }
}

/// The `reduce` block a record judged `status` may carry (owner 2026-11-11,
/// `tools/catalog/negshape.py` `reduce_for_status`): on a negative status
/// (`defect`, `yozume`, `nomate`) a `minimal` / `reduced` claim becomes
/// `unknown` (version and kind dropped; the source, the pieces, the
/// certificate and the cost stay as facts of the run), since minimality is
/// defined only for a diagram the acceptance test accepts (`docs/rules.md`,
/// rule D v4 strict entry); a `reducible` block drops its kind. A `solve
/// --record-reduce-version` over records that carry an older reduction, or a
/// reduced output the publishing gate rejects, no longer claims minimality on
/// a negative record (`tools/catalog/record.py` refuses one).
pub fn reduce_for_status(status: &str, mut reduce: Value) -> Value {
    if !matches!(status, "defect" | "yozume" | "nomate") {
        return reduce;
    }
    if let Value::Object(map) = &mut reduce {
        match map.get("status").and_then(Value::as_str) {
            Some("minimal" | "reduced") => {
                map.insert("status".to_string(), json!("unknown"));
                map.insert("version".to_string(), Value::Null);
                map.insert("kind".to_string(), Value::Null);
            }
            Some("reducible") => {
                map.insert("kind".to_string(), Value::Null);
            }
            _ => {}
        }
    }
    reduce
}

/// Whether a negative record of a reducible gate pass keeps `solution`'s graph
/// (owner 2026-11-11 10:0x, `tools/catalog/negshape.py`
/// `record_may_keep_graph`): `line` is one of the paths of Π of the graph
/// ([`crate::exact::main_line_paths`] over [`crate::stored_paths::graph_from_json`],
/// what `main-line-paths` reads). A graph over the cap has no Π and is not kept.
pub fn keeps_graph(graph: &SolverGraphJson, line: &[String]) -> bool {
    let Ok(graph) = crate::stored_paths::graph_from_json(graph) else {
        return false;
    };
    let Some(root) = graph.node(graph.root) else {
        return false;
    };
    let attacker = root.position().side;
    crate::exact::main_line_paths(&graph, attacker, crate::exact::MAIN_LINE_PATHS_CAP).is_some_and(
        |paths| {
            paths.iter().any(|path| {
                path.len() == line.len()
                    && path.iter().zip(line).all(|(usi, mv)| usi.as_str() == mv)
            })
        },
    )
}

/// The empty `reduce` block.
pub fn reduce_unknown() -> Value {
    json!({
        "intermediate": null, "kind": null, "rank": null, "rejection": null,
        "removed": null, "source": null, "status": "unknown", "version": null,
    })
}

/// The `reduce` block of a record from its `reduction` (rule D v5): nothing
/// removed -> `minimal`; pieces removed -> `reduced` with the source hash, kind,
/// rank and pieces.  `unknown` without a version, before v5 (no `kind`), when
/// the path cap stopped the main-line reduction, or without a source hash.
///
/// `unknown` also for a reduction that says `minimality: "unverified"` (owner
/// 2026-10-22, `docs/rules.md` "The minimality certificate"): one of its
/// checks ran out of budget, so it proves no minimality and the record claims
/// none.  `catalog-record/2` gets no field for it (the owner kept the shape,
/// 2026-10-17); `tools/catalog/record.py` refuses a sound record without a
/// claim (`reduce:sound-but-unknown`), so such a diagram does not enter the
/// catalogue as minimal, side file or not.
pub fn reduce_of(record: &PipelineRecord, version: Option<&str>) -> Value {
    let (Some(reduction), Some(version)) = (record.reduction.as_ref(), version) else {
        return reduce_unknown();
    };
    if reduction.minimality.as_deref() == Some("unverified") {
        return reduce_unknown();
    }
    let Some(kind) = reduction.kind.as_deref() else {
        return reduce_unknown();
    };
    if reduction.pi_capped {
        return reduce_unknown();
    }
    if reduction.removed.is_empty() {
        return json!({
            "intermediate": null, "kind": kind, "rank": reduction.rank, "rejection": null,
            "removed": null, "source": null, "status": "minimal", "version": version,
        });
    }
    let Some(source) = reduction.source_canonical_hash.as_deref() else {
        return reduce_unknown();
    };
    json!({
        "intermediate": null, "kind": kind, "rank": reduction.rank, "rejection": null,
        "removed": reduction.removed, "source": source, "status": "reduced", "version": version,
    })
}

fn producer_of(record: &PipelineRecord, ctx: &RecordContext) -> Value {
    // a generator may name the producer per record (`extra.catalogProducer`)
    let own = record
        .extra
        .get("catalogProducer")
        .and_then(Value::as_object);
    let pick = |key: &str| own.and_then(|own| own.get(key)).cloned();
    let ranker = record
        .provenance
        .extra
        .get("ranker")
        .and_then(|ranker| ranker.get("id").or_else(|| ranker.get("rankerId")))
        .cloned()
        .unwrap_or(Value::Null);
    json!({
        "generation": pick("generation").unwrap_or(json!(ctx.generation)),
        "host": crate::record_clock::host(),
        "interesting": pick("interesting").unwrap_or(json!(false)),
        "producedBy": pick("producedBy").unwrap_or(json!(ctx.produced_by)),
        "rankerId": pick("rankerId").unwrap_or(ranker),
        "route": pick("route").unwrap_or(json!(ctx.route)),
        // the binary's own commit, whatever `--record-solver-id` says
        "solverBuild": SOLVER_BUILD,
    })
}

/// The catalog record of `record` from its gate facts, or why there is none.
pub fn build(record: &PipelineRecord, ctx: &RecordContext, extras: &Extras) -> Result<Value, Skip> {
    let facts = record.gate.as_deref().ok_or("notGated")?;
    if record.normalized_sfen.as_deref() != Some(facts.sfen.as_str()) {
        return Err("staleGate");
    }
    if record
        .rejection
        .as_ref()
        .is_some_and(|rejection| rejection.stage != "solve")
    {
        return Err("rejectedAfterGate");
    }
    let hash = record.canonical_hash.as_deref().ok_or("noCanonicalHash")?;
    // the catalogue takes generated and third-party diagrams only (`record.py`)
    if !matches!(
        record.provenance.origin_class.as_deref(),
        Some(crate::jsonl::ORIGIN_CLASS_OWN_GENERATED | crate::jsonl::ORIGIN_CLASS_THIRD_PARTY)
    ) {
        return Err("originClassNotCatalogued");
    }
    let mut reasons: Vec<String> = facts
        .reasons
        .iter()
        .chain(&extras.reasons)
        .cloned()
        .collect();
    reasons.sort();
    reasons.dedup();
    if reasons.iter().any(|r| r == NO_MATE_PROVEN) {
        return Err("noMate");
    }
    let proven = !facts.main_line.is_empty() || facts.had_solution;
    let status = status_of(&reasons, proven);
    let solve_version = format!(
        "{}/{}/{}",
        ctx.solver_id, PIPELINE_VERSION, facts.node_budget
    );
    let Some(ply) = record.target_mate_ply.or(facts.mate_ply) else {
        if status == "unknown" {
            return Ok(budget_out(record, ctx, facts, hash, reasons, solve_version));
        }
        return Err("noMatePly");
    };
    let has = |set: &[&str]| reasons.iter().any(|r| set.contains(&r.as_str()));
    let line_ok = !facts.main_line.is_empty() && facts.main_line.len() == ply as usize;
    let gate_reasons: Vec<String> = reasons.iter().filter(|r| *r != RULE_D).cloned().collect();

    let main_line = if line_ok {
        json!({ "leftoverHand": facts.leftover_hand, "moves": facts.main_line, "source": LINE_SOURCE })
    } else {
        Value::Null
    };
    let mut defects = Map::new();
    let (mut yozume, mut koma, mut inert) = (Value::Null, Value::Null, Value::Null);
    let mut has_yozume = None;
    if matches!(status, "sound" | "defect" | "yozume") {
        let b = u8::from(has(&RULE_B));
        has_yozume = Some(b);
        let profile = if line_ok {
            facts.yozume_profile.clone()
        } else {
            None
        };
        yozume = json!({ "hasYozume": b, "profile": profile });
        if has(&RULE_C) && facts.leftover_hand > 0 {
            koma = json!({ "leftoverHand": facts.leftover_hand });
        }
        inert = if !facts.inert_pieces.is_empty() {
            json!(facts.inert_pieces)
        } else if !has(&RULE_G[..2]) {
            json!([])
        } else {
            Value::Null
        };
    }
    defects.insert("inertPieces".to_string(), inert);
    defects.insert("komaAmariDetail".to_string(), koma);
    defects.insert(
        "removablePieces".to_string(),
        extras.removable.clone().unwrap_or(Value::Null),
    );
    defects.insert("yozume".to_string(), yozume);

    let mut reduce = reduce_for_status(
        status,
        extras
            .reduce
            .clone()
            .unwrap_or_else(|| reduce_of(record, ctx.reduce_version.as_deref())),
    );
    complete_reduce(&mut reduce, reduce_facts(record, ctx));
    let reduce_status = reduce["status"].as_str().unwrap_or("unknown").to_string();
    // The graph of a sound record, and of a reducible gate pass (the final
    // reduce block says `reducible`, the gate gave no reason) whose line is in
    // Π of the graph (owner 2026-11-11 10:0x, [`keeps_graph`]).
    let reducible_pass = gate_reasons.is_empty()
        && reasons.iter().any(|r| r == RULE_D)
        && reduce_status == "reducible";
    let keep_graph = status == "sound"
        || (reducible_pass
            && line_ok
            && record
                .solution
                .as_deref()
                .is_some_and(|solution| keeps_graph(&solution.graph, &facts.main_line)));
    let (solution, absent) = match record.solution.as_deref() {
        Some(solution) if keep_graph => (
            serde_json::to_value(solution).map_err(|_| "solutionJson")?,
            Value::Null,
        ),
        _ if facts.had_solution => (
            Value::Null,
            json!(format!(
                "negative ({}): graph not stored; re-solve with solve --max-ply {} \
                 --node-budget 16000000 --emit-main-line",
                reasons.join(","),
                ply
            )),
        ),
        _ if !reasons.is_empty() => (
            Value::Null,
            json!(format!(
                "gate rejection ({}): the solve stage keeps no solution",
                reasons.join(",")
            )),
        ),
        _ => (Value::Null, json!("not kept by the producer")),
    };
    let verdict = rule_f_verdict_of(&reasons);
    let detail = if line_ok {
        facts.yozume_detail.clone()
    } else {
        None
    };
    let region = if status == "nomate" {
        Value::Null
    } else {
        json!(u8::from(has(&RULE_B)))
    };
    let rejudge = json!({
        "maxPly": ply,
        "pipelineVersion": PIPELINE_VERSION,
        "profile": detail,
        "reasons": gate_reasons,
        "regionYozume": region,
        "solveVersion": solve_version,
        "verdict": rule_f_verdict_of(&gate_reasons),
    });
    let (solved_at, wall_ms, cpu_ms) = judgement_cost(facts);
    let judgement = json!({
        "cpuMs": cpu_ms,
        "nodes": facts.nodes,
        "pipelineVersion": PIPELINE_VERSION,
        "reasons": reasons,
        "rejudge": rejudge,
        "ruleFVerdict": verdict,
        "solveVersion": solve_version,
        "solvedAt": solved_at,
        "status": status,
        "wallMs": wall_ms,
    });
    let rules = rules_of(Some(&gate_reasons), has_yozume, &reduce_status);
    let provenance = serde_json::to_value(&record.provenance).map_err(|_| "provenanceJson")?;
    Ok(json!({
        "recordVersion": RECORD_VERSION,
        "canonicalHash": hash,
        "sfen": facts.sfen,
        "matePly": ply,
        "targetMatePly": record.target_mate_ply,
        "judgement": judgement,
        "rules": rules,
        "mainLine": main_line,
        "defects": defects,
        "solution": solution,
        "solutionAbsent": absent,
        "reduce": reduce,
        "provenance": provenance,
        "producer": producer_of(record, ctx),
        "links": extras.links,
        "supersededBy": extras.superseded_by,
    }))
}

/// The record of a diagram the gate ran out of budget on before it proved a
/// mate length (composition main 2026-10-27, for the escalation queue): the
/// diagram, the reasons and the bound it was searched at, with no line, graph
/// or defects.  `matePly` is null, so the ingest refuses it with or without
/// `--allow-unknown`.  With `--emit-unresolved` it goes to that file, as every
/// record of status `unknown` does, instead of the records' file.
fn budget_out(
    record: &PipelineRecord,
    ctx: &RecordContext,
    facts: &GateFacts,
    hash: &str,
    reasons: Vec<String>,
    solve_version: String,
) -> Value {
    let (solved_at, wall_ms, cpu_ms) = judgement_cost(facts);
    json!({
        "recordVersion": RECORD_VERSION,
        "canonicalHash": hash,
        "sfen": facts.sfen,
        "matePly": null,
        "maxPly": facts.max_ply,
        "nodeBudget": facts.node_budget,
        "targetMatePly": record.target_mate_ply,
        "judgement": {
            "cpuMs": cpu_ms,
            "nodes": facts.nodes,
            "pipelineVersion": PIPELINE_VERSION,
            "reasons": reasons,
            "rejudge": null,
            "ruleFVerdict": null,
            "solveVersion": solve_version,
            "solvedAt": solved_at,
            "status": "unknown",
            "wallMs": wall_ms,
        },
        "provenance": serde_json::to_value(&record.provenance).unwrap_or(Value::Null),
        "producer": producer_of(record, ctx),
    })
}

/// The records of one generator output (`extend`, `distill`, `reduce-accepted`
/// paths that run [`crate::stages::reduce_accepted`]): the record itself and,
/// when rule D took pieces off it, the diagram the gate accepted before that
/// (sound but not minimal: `rule_d_reducible`, reducible, linked to the
/// reduced one).  Skips are returned for the caller's counters.  A record that
/// carries a [`GateFacts::view`] is written as that view.
pub fn generated(record: &PipelineRecord, ctx: &RecordContext) -> Vec<Result<Value, Skip>> {
    let record = record
        .gate
        .as_deref()
        .and_then(|facts| facts.view.as_deref())
        .unwrap_or(record);
    let mut out = Vec::with_capacity(2);
    if let Some(source) = record
        .gate
        .as_deref()
        .and_then(|facts| facts.reduced_from.as_deref())
    {
        out.push(reducible_source(source, record, ctx));
    }
    out.push(build(record, ctx, &Extras::default()));
    out
}

/// The record of `source`, a gate pass that rule D reduced to `reduced`.
pub fn reducible_source(
    source: &PipelineRecord,
    reduced: &PipelineRecord,
    ctx: &RecordContext,
) -> Result<Value, Skip> {
    let reduction = reduced.reduction.as_ref().ok_or("noReduction")?;
    let out_hash = reduced.canonical_hash.clone().ok_or("noCanonicalHash")?;
    let kind = reduction
        .kind
        .clone()
        .unwrap_or_else(|| "mainline".to_string());
    let gate_pass = source
        .gate
        .as_deref()
        .is_some_and(|facts| facts.reasons.is_empty());
    let removed = json!(reduction.removed);
    let link = json!({
        "kind": format!("reduce-{kind}"),
        "ops": { "removed": removed, "rank": reduction.rank },
        "outHash": out_hash,
    });
    let extras = if gate_pass && kind == "mainline" {
        // The reduction that took pieces off `source` is `reduced`'s: its
        // certificate (a reducible claim stands on it like a minimal one
        // does) and its cost.
        let mut certified = json!({
            "intermediate": null, "kind": null, "rank": null, "rejection": null,
            "removed": null, "source": null, "status": "reducible",
            "version": ctx.reduce_version.clone().ok_or("noReduceVersion")?,
        });
        complete_reduce(&mut certified, reduce_facts(reduced, ctx));
        Extras {
            reasons: vec![RULE_D.to_string()],
            reduce: Some(certified),
            removable: Some(removed),
            links: vec![link],
            superseded_by: Some(out_hash),
        }
    } else {
        let mut reduce = reduce_unknown();
        reduce_cost_into(&mut reduce, reduced, ctx);
        Extras {
            reduce: Some(reduce),
            links: vec![link],
            ..Extras::default()
        }
    };
    build(source, ctx, &extras)
}

/// `written` with the `unwritten` records put back where they were judged
/// (each carries the number of written records judged before it).
pub fn in_judged_order<'a>(
    written: &'a [PipelineRecord],
    unwritten: &'a [(usize, PipelineRecord)],
) -> Vec<&'a PipelineRecord> {
    let mut out = Vec::with_capacity(written.len() + unwritten.len());
    let mut pending = unwritten.iter().peekable();
    for (index, record) in written.iter().enumerate() {
        while let Some((_, earlier)) = pending.next_if(|(before, _)| *before <= index) {
            out.push(earlier);
        }
        out.push(record);
    }
    out.extend(pending.map(|(_, record)| record));
    out
}

/// `reduce.version` of rule D v5 at this pipeline version and budget.
pub fn rule_d_version(node_budget: u64) -> String {
    format!("D-v5/{}/{}", PIPELINE_VERSION, node_budget)
}

/// `record` judged again by the publishing gate `gate` on a copy (its
/// rejection and solution cleared first), for its catalog record.
fn judged_copy(
    record: &PipelineRecord,
    gate: &SolveStageOptions,
    solver: &mut Solver,
) -> anyhow::Result<PipelineRecord> {
    let mut copy = record.clone();
    copy.rejection = None;
    copy.solution = None;
    solve_stage_reported(&mut copy, gate, solver)?;
    Ok(copy)
}

/// The records of one `reduce` input (`--emit-record`): each output diagram
/// and, when pieces came off, the input's own diagram (a sound but not minimal
/// one, or the yozume input of the rule B entry), each judged by the publishing
/// gate on a copy at the reducer's bound and budget; an input the reducer
/// refused is written with the refusal.  The reducer's verdict is never the
/// record's: only the gate's is.
pub fn reduce_records(
    input: &PipelineRecord,
    main: &PipelineRecord,
    extra: &[PipelineRecord],
    gate: &SolveStageOptions,
    ctx: &RecordContext,
    solver: &mut Solver,
) -> anyhow::Result<Vec<Result<Value, Skip>>> {
    let judged = |record: &PipelineRecord, solver: &mut Solver| judged_copy(record, gate, solver);
    if let Some(rejection) = main.rejection.as_ref() {
        if rejection.stage != "reduce" || !input.is_accepted() {
            return Ok(vec![Err("rejectedBeforeReduce")]);
        }
        let copy = judged(input, solver)?;
        let mut reduce = reduce_unknown();
        reduce["rejection"] = json!({ "stage": rejection.stage, "reasons": rejection.reasons });
        // as the adapters wrote it: the refusal's rule D version
        // (`reduce_rejections.reduce_version`)
        reduce["version"] = json!(ctx.reduce_version);
        // what the refused reduction took (no certificate: nothing reduced)
        reduce_cost_into(&mut reduce, main, ctx);
        let extras = Extras {
            reduce: Some(reduce),
            ..Extras::default()
        };
        return Ok(vec![build(&copy, ctx, &extras)]);
    }
    let mut out = Vec::with_capacity(2 + extra.len());
    let main_judged = judged(main, solver)?;
    let removed = main
        .reduction
        .as_ref()
        .is_some_and(|reduction| !reduction.removed.is_empty());
    if removed && input.is_accepted() {
        let input_judged = judged(input, solver)?;
        out.push(reducible_source(&input_judged, &main_judged, ctx));
    }
    out.push(build(&main_judged, ctx, &Extras::default()));
    for record in extra {
        let copy = judged(record, solver)?;
        out.push(build(&copy, ctx, &Extras::default()));
    }
    Ok(out)
}

/// The records of one `reduce-accepted` input (`--emit-record`): when rule D
/// took pieces off, the reduced record re-solved by the command and the input
/// as a reducible one ([`generated`]); otherwise (`unchanged`,
/// `fallback`) the record judged again on a copy by the same gate, at the
/// input's bound and `node_budget`.  Then every length-preserving reduction,
/// which the command solved.  An input that was not an accepted, solved
/// record gets none.
pub fn reduce_accepted_records(
    input: &PipelineRecord,
    record: &PipelineRecord,
    extra: &[PipelineRecord],
    node_budget: u64,
    ctx: &RecordContext,
    solver: &mut Solver,
) -> anyhow::Result<Vec<Result<Value, Skip>>> {
    let Some(bound) = input
        .solution
        .as_ref()
        .filter(|_| input.is_accepted())
        .map(|solution| solution.stats.max_ply)
    else {
        return Ok(vec![Err("rejectedBeforeReduce")]);
    };
    let mut out = Vec::with_capacity(2 + extra.len());
    if record.gate.is_some() {
        out.extend(generated(record, ctx));
    } else {
        let gate = SolveStageOptions {
            max_ply: bound,
            node_budget,
            rules: TsumeRules::puzzle(),
        };
        let copy = judged_copy(record, &gate, solver)?;
        // A fallback is the input as read: its own `reduction`, if any, is not
        // this run's and claims nothing here.
        let extras = if fresh_reduction(&copy).is_some() {
            Extras::default()
        } else {
            let mut reduce = reduce_unknown();
            reduce_cost_into(&mut reduce, &copy, ctx);
            Extras {
                reduce: Some(reduce),
                ..Extras::default()
            }
        };
        out.push(build(&copy, ctx, &extras));
    }
    for child in extra {
        out.push(build(child, ctx, &Extras::default()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonl::{GraphMoveJson, GraphNodeJson};
    use indexmap::IndexMap;

    fn mv(usi: &str, outcome: &str, distance: Option<u32>, next: Option<&str>) -> GraphMoveJson {
        GraphMoveJson {
            usi: usi.to_string(),
            outcome: outcome.to_string(),
            distance,
            next: next.map(str::to_string),
            refutation: None,
            useless: false,
        }
    }

    fn node(
        id: &str,
        side: &str,
        distance: u32,
        moves: Vec<GraphMoveJson>,
    ) -> (String, GraphNodeJson) {
        (
            id.to_string(),
            GraphNodeJson {
                id: id.to_string(),
                side: side.to_string(),
                sfen: String::new(),
                outcome: "MATE".to_string(),
                distance: Some(distance),
                moves,
            },
        )
    }

    /// A 5-ply graph with two correct moves at the root (distance 5), one
    /// yozume node off the line, and two mates at the last move.
    fn graph() -> SolverGraphJson {
        let nodes: IndexMap<String, GraphNodeJson> = [
            node(
                "0",
                "attack",
                5,
                vec![
                    mv("2b2a", "MATE", Some(5), Some("1")),
                    mv("3c3b", "MATE", Some(5), Some("9")),
                    mv("4d4c", "MATE", Some(7), None),
                ],
            ),
            node(
                "1",
                "defense",
                4,
                vec![mv("1a1b", "MATE", Some(4), Some("2"))],
            ),
            node(
                "2",
                "attack",
                3,
                vec![mv("5e5d", "MATE", Some(3), Some("3"))],
            ),
            node(
                "3",
                "defense",
                2,
                vec![mv("1b1c", "MATE", Some(2), Some("4"))],
            ),
            node(
                "4",
                "attack",
                1,
                vec![
                    mv("G*1d", "MATE", Some(1), None),
                    mv("S*1d", "MATE", Some(1), None),
                ],
            ),
            node(
                "9",
                "attack",
                3,
                vec![
                    mv("6f6e", "MATE", Some(3), None),
                    mv("7g7f", "MATE", Some(3), None),
                ],
            ),
        ]
        .into_iter()
        .collect();
        SolverGraphJson {
            root: "0".to_string(),
            nodes,
        }
    }

    fn line() -> Vec<String> {
        ["2b2a", "1a1b", "5e5d", "1b1c", "G*1d"]
            .map(String::from)
            .to_vec()
    }

    #[test]
    fn the_yozume_profile_is_catalog_db_s() {
        let profile = yozume_profile(&graph(), &line()).unwrap();
        assert_eq!(
            profile,
            json!({"count": 1, "finalAlternatives": true, "firstPly": 0, "inVariation": true,
                   "maxCorrect": 2, "v": 1})
        );
        assert_eq!(yozume_profile(&graph(), &[]), None);
    }

    #[test]
    fn the_yozume_detail_names_the_alternatives_and_counts_the_rest() {
        let detail = yozume_detail(&graph(), &line()).unwrap();
        assert_eq!(
            detail,
            json!({"mainLine": [{"alternatives": ["3c3b"], "move": "2b2a", "ply": 0}],
                   "mergedTwins": [], "offMainLine": 1, "offMainLineMergedTwins": 0,
                   "scope": "graph", "v": 4})
        );
    }

    /// A merged promotion twin (0.17.0) is listed apart and is no yozume.
    #[test]
    fn a_merged_promotion_twin_is_listed_apart_from_yozume() {
        let nodes: IndexMap<String, crate::jsonl::GraphNodeJson> = [
            node(
                "0",
                "attack",
                3,
                vec![
                    mv("3c2b+", "MATE", Some(3), Some("1")),
                    mv("3c2b", "MATE", Some(3), Some("2")),
                ],
            ),
            node(
                "1",
                "defense",
                2,
                vec![mv("1a2b", "MATE", Some(2), Some("3"))],
            ),
            node(
                "2",
                "defense",
                2,
                vec![mv("1a2b", "MATE", Some(2), Some("3"))],
            ),
            node("3", "attack", 1, vec![mv("G*1b", "MATE", Some(1), None)]),
        ]
        .into_iter()
        .collect();
        let graph = SolverGraphJson {
            root: "0".to_string(),
            nodes,
        };
        let line = ["3c2b+", "1a2b", "G*1b"].map(String::from).to_vec();
        assert_eq!(
            yozume_detail(&graph, &line).unwrap(),
            json!({"mainLine": [], "mergedTwins": [{"alternatives": ["3c2b"], "kind": "mergedTwin",
                   "move": "3c2b+", "ply": 0}], "offMainLine": 0, "offMainLineMergedTwins": 0,
                   "scope": "graph", "v": 4})
        );
        assert_eq!(yozume_profile(&graph, &line).unwrap()["count"], json!(0));
        assert_eq!(
            yozume_profile(&graph, &line).unwrap()["inVariation"],
            json!(false)
        );
    }

    #[test]
    fn status_and_verdict_follow_record_py() {
        let r = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(status_of(&r(&[]), true), "sound");
        assert_eq!(status_of(&r(&[]), false), "nomate");
        assert_eq!(status_of(&r(&["koma_amari"]), true), "defect");
        assert_eq!(
            status_of(&r(&["koma_amari", "yozume_in_main_line"]), true),
            "yozume"
        );
        assert_eq!(status_of(&r(&["target_mate_ply_changed"]), true), "nomate");
        assert_eq!(status_of(&r(&["solver_budget_exhausted"]), true), "unknown");
        assert_eq!(rule_f_verdict_of(&r(&[RULE_D])), "pass");
        assert_eq!(rule_f_verdict_of(&r(&["a", RULE_D, "b"])), "fail:a,b");
    }

    #[test]
    fn rules_follow_schema_sql() {
        let r = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let clean = rules_of(Some(&r(&[])), Some(0), "minimal");
        assert_eq!(
            clean,
            json!({"dualKind": null, "futileInterpositionJudged": 0, "hasAttackerKing": 0,
                   "hasDual": 0, "hasInertPiece": 0, "hasKomaAmari": 0, "isReducible": 0})
        );
        let failed = rules_of(Some(&r(&["target_mate_ply_changed"])), None, "unknown");
        assert_eq!(failed["hasKomaAmari"], Value::Null);
        assert_eq!(failed["futileInterpositionJudged"], json!(0));
        let both = rules_of(
            Some(&r(&["first_move_not_unique", "yozume_in_main_line"])),
            Some(1),
            "unknown",
        );
        assert_eq!(both["dualKind"], json!("both"));
        assert_eq!(both["hasDual"], json!(1));
        assert_eq!(rules_of(None, None, "reducible")["isReducible"], json!(1));
    }

    #[test]
    fn a_negative_record_claims_no_minimality() {
        let claim = |status: &str| {
            json!({"intermediate": null, "kind": "mainline", "rank": 0, "rejection": null,
                   "removed": [{"piece": "bP", "square": 3}], "source": "ab", "status": status,
                   "version": "D-v5/0.16.0/10000000", "minimality": "verified"})
        };
        for status in ["defect", "yozume", "nomate"] {
            let out = reduce_for_status(status, claim("reduced"));
            assert_eq!(out["status"], json!("unknown"));
            assert_eq!(out["version"], Value::Null);
            assert_eq!(out["kind"], Value::Null);
            // the facts of the run stay
            assert_eq!(out["source"], json!("ab"));
            assert_eq!(out["minimality"], json!("verified"));
            assert_eq!(
                reduce_for_status(status, claim("minimal"))["status"],
                json!("unknown")
            );
            let reducible = reduce_for_status(status, claim("reducible"));
            assert_eq!(reducible["status"], json!("reducible"));
            assert_eq!(reducible["kind"], Value::Null);
            assert_eq!(reducible["version"], json!("D-v5/0.16.0/10000000"));
        }
        for status in ["sound", "unknown"] {
            assert_eq!(
                reduce_for_status(status, claim("minimal")),
                claim("minimal")
            );
        }
    }

    #[test]
    fn reduce_versions() {
        assert!(valid_reduce_version("D-v5/0.13.0/4000000"));
        assert!(!valid_reduce_version("D-v4/0.8.0/250000"));
        assert!(!valid_reduce_version("D-v5/0.13/1"));
        assert!(!valid_reduce_version("D-v5/0.13.0/"));
        assert!(!valid_reduce_version(""));
    }

    #[test]
    fn solver_ids() {
        assert!(valid_solver_id("S-dd32a41"));
        assert!(valid_solver_id("S-dd32a41-vast.2"));
        assert!(!valid_solver_id("S-dd32a4"));
        assert!(!valid_solver_id("dd32a41"));
        assert!(!valid_solver_id("S-dd32a41/0.13.0"));
    }
}
