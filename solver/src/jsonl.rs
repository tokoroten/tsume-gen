//! JSONL `PipelineRecord` documents.
//!
//! Field names and values match `tools/puzzle-pipeline/src/types.ts` so the Rust
//! stages are drop-in replacements inside the TypeScript pipeline:
//!
//! ```text
//! … import | tsume-solver validate | tsume-solver reduce | tsume-solver solve | … compile
//! ```
//!
//! Unknown fields are preserved through `extra`, so records written by a newer
//! TypeScript pipeline round-trip without loss.  JSON *key order* is the order of
//! the TypeScript interface declarations rather than the order a particular
//! TypeScript object literal happened to build, which JSON does not treat as
//! significant.

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::validate::Issue;

/// Schema version understood by this build.
pub const PIPELINE_SCHEMA_VERSION: u32 = 1;
/// `packages/versions` `SOLVER_VERSION`.
pub const SOLVER_VERSION: &str = "bounded-and-or-v1";
/// `tools/puzzle-pipeline/src/types.ts` `GENERATOR_VERSION`.
pub const GENERATOR_VERSION: &str = "constrained-random-v1";
/// `tools/puzzle-pipeline/src/types.ts` `PIPELINE_VERSION`.
///
/// `0.2.0` was the frozen tsume-rules revision: no attacker king, no yozume on
/// the designated main line, no koma-amari, and the rule D reduction candidate
/// list.  `0.3.0` adds rule F (無駄合): a useless interposition is neither a
/// defence nor part of the mate length, so published mate lengths, the designated
/// main line and the runtime response graph all ignore it.  `0.4.0` adds rule G
/// (無駄駒): a board piece that takes no part in the solution is a defect, so the
/// reducer's acceptance test and the `solve` stage both refuse it
/// (`inert_attacker_piece`, `inert_defender_piece`, `inertPieceCount`).  `0.5.0`
/// is rule D v3, the inert sweep: instead of refusing a diagram whose inert
/// pieces can only go together, the reducer takes the whole inert set off at once
/// and iterates to a fixpoint, on entry and inside every greedy step.  `0.6.0`
/// adds origin provenance (`docs/legal-data.md` §3, owner decision 2026-09-13):
/// every root record states `originClass`, a non-empty `origins` set and its
/// `rootSourceId`, every derivation appends a `lineage` entry, and a record that
/// states none of it is rejected with `missing_origin` rather than defaulted to
/// `own`.  `0.7.0` relaxes rule B by mate length (owner decision 2026-09-14): a
/// puzzle whose proven minimum mate is **9 plies or longer** may offer several
/// correct moves at an attack node, so neither `first_move_not_unique` nor
/// `yozume_in_main_line` is reported for it, while `matePly <= 7` stays strict.
/// Rules A, C, F and G keep their definitions and their evaluation scope - rule C
/// is still judged on the canonical main line alone - and `solution` gains
/// `yozumeAllowed`, `attackerBranchNodes` and `maxAttackerBranching`.  On input
/// those three default to `false` / `0` / `1`, so a record written before
/// `0.7.0` still parses; every record `0.7.0` emits states them, so the written
/// format - and therefore this version - does not move.  `0.8.0` makes the
/// reducer entry strict (owner decision 2026-09-18): an input that is not
/// already a publishable puzzle is rejected up front instead of repaired.
/// `0.9.0` redefines the designated main line (owner decision 2026-09-19,
/// `docs/rules.md`): rules B and C are judged on the main-line region -- every
/// correct move and every equally long defence that is not 変同駒余り -- instead
/// of one path chosen by USI order, so a verdict no longer depends on the board
/// orientation; the final move may have several mates; and
/// `attackerBranchNodes` / `maxAttackerBranching` count over that region.  The
/// USI order only picks the displayed line.  `0.9.1` leaves an equally long
/// 変同駒余り reply out of the **runtime** graph too ([`crate::exact::prune_solver_graph`]):
/// the runtime defender answers with the longest reply of the lowest USI, which
/// without this could be such a reply, and the player would then mate with
/// material to spare on a puzzle rule C accepts.  `0.10.0` reads rule F
/// classically (owner decision 2026-10-05, `docs/rules.md`): at a defence node
/// whose every legal reply is an interposition drop on the single sliding check
/// line, a drop the checker captures (with check) into mate -- or into another
/// such position whose drops are all futile -- is 無駄合 as well, and when every
/// reply is, the check itself is mate.  The composition rules solve with
/// [`crate::solver::UselessInterposition::Classical`] instead of `Prune`.
/// `0.11.0` makes the classical ply bound count rule-F plies (owner decision
/// 2026-10-07, `docs/rules.md` rule F, "The ply bound counts rule-F plies"): an
/// F1 無駄合 and the checker's capture of it consume none of the bound, because
/// a credit search reads the defence node after the capture at the bound the
/// node's other replies set.  A mate in N with any number of nested 無駄合 is
/// proven at bound N, at its rule-F length, and its correct moves no longer
/// depend on the bound; `keep` and `prune` are unchanged.  `0.12.0` enforces
/// rule B at every mate length again (owner decision 2026-10-09): the length
/// condition of `0.7.0` is gone, so a 9-ply-or-longer puzzle with two correct
/// moves at the root (`first_move_not_unique`) or at an attack node of the
/// main-line region of distance 3 or more (`yozume_in_main_line`) is rejected
/// like a shorter one.  `solution.yozumeAllowed` is always `false` from here on
/// and stays only for format compatibility, and the generation-only
/// `--strict-rule-b` flags of P8-016 became the default and were removed.
/// `0.13.0` is rule D v5 (owner decision 2026-10-09, `docs/rules.md`
/// "Reduction v5"): the `reduce` stage emits the **main-line-preserving**
/// reduction -- every step keeps the displayed main line and adds no path to
/// the main-line region -- swept to its fixpoint, and on request
/// (`--length-k K`) up to K **length-preserving** reductions with other
/// displayed main lines, each swept to its own fixpoint, as extra records.
/// `reduction` gains `kind`, `rank` and `sourceCanonicalHash` (and `piCapped`,
/// `entry` when they apply), and a diagram is minimal when no
/// main-line-preserving step from it is accepted.  `0.14.0` drops the
/// displayed-main-line condition from the main-line-preserving step (owner
/// decision 2026-10-27, `docs/rules.md` "Reduction v5"): a step is accepted
/// when the result passes the acceptance test and adds no path to the
/// main-line region (`Π(y) ⊆ Π(z)`).  The displayed main line's tie-break (the
/// smallest USI) depends on the board's orientation, so under 0.13.0 a diagram
/// and its mirror image could get different minimality verdicts (047a130e: a
/// gold that only supports one of two final mates).  The reduce
/// transformation strings and the catalogue's `D-v5/...` reduce version keep
/// their names.  `0.15.0` adds rule G clause (e) (owner decision 2026-10-29,
/// `docs/rules.md` rule G): a piece (a)-(d) leave inert is active when, taken
/// off to the defender's hand at an attack node of the solution tree, it lets
/// in a check its presence alone made illegal that mates within the node's
/// distance (余詰防ぎの駒); a question that runs out of budget leaves it
/// undecided and the gate reports `rule_g_budget_exhausted`.  `0.16.0`
/// (owner decision 2026-10-30) reads rule G clause (d) on the **set** of
/// replies that refute a wrong check within the attack node's distance (not
/// the first refuting reply in USI order), and builds rule G's solution tree at
/// the proven mate length, not the gate's bound: rule G's verdict no longer
/// depends on the left-right orientation of the diagram or on `--max-ply`.
/// `0.17.0` (owner decision 2026-11-16) counts a **merged promotion twin** as
/// one correct move for rule B, at the root and deeper: two correct moves `m` /
/// `m+` (same from- and to-squares) whose main-line candidate replies are the
/// same non-empty set, every one capturing the moved piece on its destination
/// (`exact::is_merged_promotion_twin`); `solution.capturedPromoAltNodes`
/// counts such nodes (「取られる成り駒の非限定」).  The displayed main line breaks
/// ties on the USI without its `+`, the promotion first (`exact::main_line_key`),
/// so a final move that mates promoted and unpromoted shows the promotion.
pub const PIPELINE_VERSION: &str = "0.17.0";

/// `true` for a `false` flag, so an unset boolean is omitted like an absent
/// TypeScript optional.
pub fn is_false(value: &bool) -> bool {
    !*value
}

/// The `maxAttackerBranching` a record written before `0.7.0` implies.
///
/// Strict rule B — the only reading such a record was ever judged under — leaves
/// exactly **one** correct move at every attack node of the canonical main line,
/// so the default is `1`, not `0`.  Together with `false` for `yozumeAllowed`
/// and `0` for `attackerBranchNodes` it restates what the old record already
/// meant rather than inventing a value, which is why a pre-0.7.0 `matePly <= 7`
/// record round-trips to the very bytes `0.7.0` writes for it.
fn default_max_attacker_branching() -> u64 {
    1
}

/// `skip_serializing_if` of [`Solution::captured_promo_alt_nodes`].
fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

/// Serialises an integral `f64` without a trailing `.0`, like `JSON.stringify`.
pub fn js_number<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        serializer.serialize_i64(*value as i64)
    } else {
        serializer.serialize_f64(*value)
    }
}

/// The origin id of project-authored and project-generated roots
/// (`docs/legal-data.md` §3 `origins`).
pub const OWN_ORIGIN: &str = "own";

/// `originClass` values.  Fixed when a root record is created; no later stage
/// changes one.
pub const ORIGIN_CLASS_OWN_AUTHORED: &str = "own_authored";
pub const ORIGIN_CLASS_OWN_GENERATED: &str = "own_generated";
pub const ORIGIN_CLASS_THIRD_PARTY: &str = "third_party";

/// Every accepted `originClass`, in the order `docs/legal-data.md` §3 lists them.
pub const ORIGIN_CLASSES: [&str; 3] = [
    ORIGIN_CLASS_OWN_AUTHORED,
    ORIGIN_CLASS_OWN_GENERATED,
    ORIGIN_CLASS_THIRD_PARTY,
];

/// One derivation step of `provenance.lineage`: the record the step consumed,
/// and the tool and version that performed it.  Matches the TypeScript
/// `LineageEntry` (`tools/puzzle-pipeline/src/types.ts`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LineageEntry {
    pub source_id: String,
    pub tool: String,
    pub version: String,
    /// Fields other steps wrote into their entry (the catalogue's `reduce`
    /// step adds `fromCanonicalHash` / `reducedLineChanged` / `removed`).
    /// Passed through untouched, as the TypeScript pipeline does; before
    /// 2026-09-25 they were dropped on every validate / solve / reduce pass.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    pub source_type: String,
    pub generator_version: String,
    pub generator_config_hash: String,
    /// The distribution the record was actually drawn from, per record - one of
    /// `king-zone` / `slider-pressure` / `mate-shell` for the constrained-random
    /// source, whatever the caller *asked* for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator_distribution: Option<String>,
    /// What the caller asked for: the `--distribution` option as spelled, or
    /// `weighted-mix` when it was left to the mix.  Separate from
    /// [`Self::generator_distribution`] since `mate-mining-v2`, because recording
    /// only the option made every candidate of a weighted-mix run look alike.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator_distribution_option: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_position_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rng_seed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_options_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_play_plies: Option<u32>,
    pub mate_solver_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discovered_mate_ply: Option<u32>,
    pub created_at: String,
    pub historical_reachability_checked: bool,
    pub transformations: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    /// Which regime this record's **root** falls under (`docs/legal-data.md` §3).
    /// Typed `Option` rather than required because a record arriving on stdin may
    /// lack it - which is exactly what `validate` rejects (`missing_origin`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_class: Option<String>,
    /// Sorted dataset ids that contributed to this record through any derivation.
    /// `["own"]` for project work; never defaulted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origins: Option<Vec<String>>,
    /// `sourceId` of the first record in the chain.  Never changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_source_id: Option<String>,
    /// One entry per derivation step, oldest first, ending with the direct
    /// parent.  `Some(vec![])` on a root, which serialises as `[]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<Vec<LineageEntry>>,
    /// Third-party rows only: the acquired snapshot's id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_dataset_id: Option<String>,
    /// Third-party rows only: sha256 of the acquired source file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file_sha256: Option<String>,
    /// Third-party rows only: ISO-8601 download date of the snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_accessed_at: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Provenance {
    /// A minimal provenance block for generated candidates.
    pub fn generated(source_type: &str, generator_version: &str, config_hash: String) -> Self {
        Self {
            source_type: source_type.to_string(),
            generator_version: generator_version.to_string(),
            generator_config_hash: config_hash,
            generator_distribution: None,
            generator_distribution_option: None,
            proposal_index: None,
            proposal_fingerprint: None,
            source_position_hash: None,
            rng_seed: None,
            engine_name: None,
            engine_version: None,
            engine_options_hash: None,
            self_play_plies: None,
            mate_solver_version: SOLVER_VERSION.to_string(),
            discovered_mate_ply: None,
            created_at: crate::util::build_timestamp(),
            historical_reachability_checked: false,
            transformations: Vec::new(),
            source_file: None,
            source_line: None,
            source_hash: None,
            origin_class: None,
            origins: None,
            root_source_id: None,
            lineage: None,
            source_dataset_id: None,
            source_file_sha256: None,
            source_accessed_at: None,
            extra: Map::new(),
        }
    }

    /// Stamps the origin fields of a **root** record (`import`, `mine`, a bare
    /// SFEN handed to `distill`): the class, the one dataset it came from, its own
    /// id as `rootSourceId`, and an empty lineage.
    pub fn set_root_origin(&mut self, origin_class: &str, origin: &str, source_id: &str) {
        self.origin_class = Some(origin_class.to_string());
        self.origins = Some(vec![origin.to_string()]);
        self.root_source_id = Some(source_id.to_string());
        self.lineage = Some(Vec::new());
        if origin_class == ORIGIN_CLASS_THIRD_PARTY {
            self.source_dataset_id = Some(origin.to_string());
        }
    }

    /// `true` when the three fields every stage boundary requires are present and
    /// the origin set is non-empty (`docs/legal-data.md` §3).
    pub fn is_root_complete(&self) -> bool {
        self.origin_class.is_some()
            && self
                .origins
                .as_ref()
                .is_some_and(|origins| !origins.is_empty())
            && self
                .root_source_id
                .as_ref()
                .is_some_and(|id| !id.is_empty())
    }

    /// Appends one derivation step, oldest first.  The origin set, the class and
    /// the root id are inherited by the caller's `clone`; a derivation only ever
    /// *adds* to the lineage, never rewrites what is already there.
    pub fn append_lineage(&mut self, source_id: &str, tool: &str, version: &str) {
        self.lineage
            .get_or_insert_with(Vec::new)
            .push(LineageEntry {
                source_id: source_id.to_string(),
                tool: tool.to_string(),
                version: version.to_string(),
                extra: Map::new(),
            });
    }

    /// Sorted union of this record's origin set with another's, for a derivation
    /// that consumes several inputs.  The set may only grow.
    pub fn merge_origins(&mut self, other: &[String]) {
        let mut merged = self.origins.take().unwrap_or_default();
        merged.extend_from_slice(other);
        merged.sort();
        merged.dedup();
        self.origins = Some(merged);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Validation {
    pub valid: bool,
    pub issues: Vec<Issue>,
    pub strict_defender_material: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovedPiece {
    pub square: usize,
    pub piece: String,
}

/// One board piece rule G found no use for (`rejection.inertPieces`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InertPiece {
    pub square: usize,
    /// `piece_label` spelling, e.g. `bR`, `w+P`.
    pub piece: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reduction {
    /// Decimal string of the base seed, as in TypeScript (`bigint.toString()`).
    pub seed: String,
    pub attempts: u32,
    pub original_board_pieces: usize,
    pub final_board_pieces: usize,
    pub removed: Vec<RemovedPiece>,
    pub target_mate_ply: u32,
    /// Rule D v5 (`PIPELINE_VERSION` 0.13.0): `"mainline"` for the
    /// main-line-preserving reduction, `"length"` for a length-preserving one.
    /// Absent on a record reduced before 0.13.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// `0` for the main-line-preserving reduction, `1, 2, …` for the
    /// length-preserving ones, best first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    /// `canonicalHash` of the diagram the reducer started from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_canonical_hash: Option<String>,
    /// The input's main-line region had more paths than the cap
    /// (`exact::MAIN_LINE_PATHS_CAP`), so the main-line-preserving reduction
    /// took nothing off.  Written only when `true`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub pi_capped: bool,
    /// `"rule-b-relaxed"` when the input came in through the relaxed rule B
    /// entry (on by default; `reduce --no-rule-b-entry` turns it off);
    /// `"rule-cg-relaxed"` through the opt-in C/G entry (`reduce --cg-entry`,
    /// RL tooling only, never in the public pipeline).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    /// `"verified"` when no acceptance check behind this reduction ended out of
    /// budget (every step the reducer refused was refused for a decided
    /// reason, so the diagram is a fixpoint of its kind), `"unverified"` when
    /// at least one did (a piece may have stayed for lack of budget; owner
    /// 2026-10-22, `docs/rules.md` "Reduction v5", [`crate::reduce::BudgetOutTally`]).
    /// Written on every reduction since; absent on older records, which say
    /// nothing either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimality: Option<String>,
    /// `0` with `"verified"`.  Since 2026-10-25 the first check that ends out
    /// of budget stops the reducer ([`crate::reduce::ReducedSet::stopped`]:
    /// the record is the unreduced input), so an `"unverified"` record of a
    /// current reducer says `1`; older records count the distinct diagrams
    /// whose check ended out of budget.  Written with `minimality`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_out_checks: Option<u64>,
    /// `budgetOutChecks` by the reducer's stage (`greedy`, `sweep`, `king`,
    /// `repair`), on older `"unverified"` records only: a record the early
    /// stop ended names no stage (which check stopped it may depend on
    /// `--jobs`; `--emit-budget-outs` names it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_out_stages: Option<std::collections::BTreeMap<String, u64>>,
    /// The node budget of the highest `--reduce-budget-ladder` rung any check
    /// behind this reduction needed to be decided (the plain node budget
    /// without a ladder; the full budget when a check ran out), for the
    /// catalog record's `reduce.budget` (`--emit-record`).  Never serialised
    /// or read: the record's JSON is the same with it or without it.
    #[serde(skip)]
    pub decided_budget: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolveStatsJson {
    pub nodes: u64,
    pub cache_hits: u64,
    pub budget_exhausted: bool,
    pub max_ply: u32,
    pub exhaustive_within_bound: bool,
    /// Written only by the `solve` stage on an accepted record
    /// ([`crate::solver::non_futile_node_count`], owner decision 2026-09-25); the
    /// puzzle rating reads it (`DIFFICULTY_MODEL_VERSION` 3).  Absent on records
    /// solved before it existed, which then rate on `nodes`.  Last, as in the
    /// TypeScript `solveStage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_non_futile: Option<u64>,
    /// On a solution the construction wrote (with `nodes_kind`): the nodes a
    /// fresh classical df-pn prober expands to prove the mate within exactly
    /// `matePly` ([`crate::reduce::proof_nodes`]); independent of the gate's
    /// bound, and the count the rating maps (`DIFFICULTY_MODEL_VERSION` 4).
    /// Absent when that proof did not finish within the gate's budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_proof: Option<u64>,
    /// `"prober"` on a solution the publishing gate's construction wrote
    /// ([`crate::stages::NODES_KIND_PROBER`], solver v3, owner decision
    /// 2026-10-09): `nodes` is then the classical df-pn prober's count and the
    /// rating maps it per mate length (`DIFFICULTY_MODEL_VERSION` 4);
    /// `"prober-leaf3"` with the prober's leaf search on, 2026-10-26 to 10-27
    /// ([`crate::stages::NODES_KIND_PROBER_LEAF3`], its own table, owner
    /// 2026-10-27; no longer written); `"prober-v2"` when, besides, the one-thread prober
    /// keeps its children's numbers between descents, the default since
    /// 2026-10-27 ([`crate::stages::NODES_KIND_PROBER_V2`]); `"prober-v3"` when the one-thread prober
    /// also uses the 1+ε threshold, the default since 2026-10-28
    /// ([`crate::stages::NODES_KIND_PROBER_V3`], the only one-thread kind
    /// since 1.0.0 with the leaf search on).  A parallel prober (`--prober-threads` above 1) appends
    /// `"-par"` (`"prober-par"`, `"prober-v2-par"`, until 1.0.0 also
    /// `"prober-leaf3-par"`; it
    /// never uses the 1+ε threshold, [`crate::stages::nodes_kind_for`]): its
    /// counts depend on the threads.  Each kind has its own rating table.  Absent
    /// when `nodes` is the full-width search's.  Last, as in the TypeScript
    /// `solveStage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_kind: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphMoveJson {
    pub usi: String,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refutation: Option<String>,
    /// 無駄合 (`docs/rules.md` rule F).  Only the *complete* search graph ever
    /// carries it: the pruned runtime graph drops flagged replies entirely, so the
    /// key is absent from every published record, exactly as in TypeScript.
    #[serde(default, skip_serializing_if = "is_false")]
    pub useless: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNodeJson {
    pub id: String,
    /// `"attack"` or `"defense"`.
    pub side: String,
    pub sfen: String,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<u32>,
    pub moves: Vec<GraphMoveJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolverGraphJson {
    pub root: String,
    /// Keyed by node id; insertion order is preserved.
    pub nodes: IndexMap<String, GraphNodeJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DifficultyFeatures {
    pub mate_ply: u32,
    pub total_search_nodes: u64,
    pub solution_tree_node_count: u64,
    pub solution_tree_depth: u32,
    pub defender_branch_count: u64,
    pub max_defender_branching: u64,
    #[serde(serialize_with = "js_number")]
    pub mean_defender_branching: f64,
    pub legal_checking_moves_at_root: u64,
    pub plausible_wrong_checks: u64,
    #[serde(serialize_with = "js_number")]
    pub false_move_difficulty: f64,
    /// Defender drops that really are resistance (rule F: 無駄合 excluded).
    pub number_of_interpositions: u64,
    /// Defender drops the solver flagged 無駄合 (`docs/rules.md` rule F).
    pub number_of_useless_interpositions: u64,
    /// Board pieces that take no part in the solution (`docs/rules.md` rule G,
    /// 無駄駒).  Always `0` on an accepted record: a publishable puzzle has none.
    pub inert_piece_count: u64,
    pub number_of_drops: u64,
    pub number_of_sacrifices: u64,
    pub number_of_promotions: u64,
    pub king_move_count: u64,
    pub attacker_piece_count: u64,
    pub board_piece_count: u64,
    pub visual_density_near_king: u64,
    pub edge_distance: u64,
    pub transposition_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Solution {
    pub mate_ply: u32,
    pub unique_first_move: bool,
    pub first_move: String,
    /// Composition rule B: rule B was not enforced for this puzzle.  `true` only
    /// on a record of 9 plies or longer written by `0.7.0`-`0.11.0` (owner
    /// decision 2026-09-14); since `0.12.0` (owner decision 2026-10-09) rule B
    /// has no length condition and every record written says `false`.  Kept for
    /// format compatibility: the catalogue and the dataset export read it.
    ///
    /// These three fields sit between `firstMove` and `graph` because serde emits
    /// struct fields in declaration order while `JSON.stringify` emits object keys
    /// in insertion order: `tools/puzzle-pipeline/src/stages.ts` `solveStage`
    /// builds them in exactly this place, and the two must agree byte for byte.
    ///
    /// All three default on input so that a record written before `0.7.0` — the
    /// whole pre-existing 7-ply-and-below stock a 7 -> 9 run reads — still parses
    /// instead of failing with `missing field yozumeAllowed`
    /// (`docs/handoffs/closed/handoff-from-mac-2026-09-15.md` §3.1).  They are always
    /// serialised, so the emitted format is unchanged.
    #[serde(default)]
    pub yozume_allowed: bool,
    /// Attack nodes of the main-line region (`exact::MainLine::branching_nodes`,
    /// since `0.9.0`; the canonical main line before), the root and the final
    /// moves included, with more than one correct move.  `0` is the default for a
    /// pre-0.7.0 record.
    #[serde(default)]
    pub attacker_branch_nodes: u64,
    /// The largest number of correct moves at one attack node of the main-line
    /// region (the canonical main line before `0.9.0`); `1` for a single-solution
    /// puzzle, and `1` — not `0` — for a
    /// pre-0.7.0 record (see [`default_max_attacker_branching`]).
    #[serde(default = "default_max_attacker_branching")]
    pub max_attacker_branching: u64,
    /// Attack nodes of the main-line region, the root included, whose two
    /// correct moves are a merged promotion twin (`exact::MainLine::captured_promo_alt_nodes`,
    /// rule B since `0.17.0`): the flag 「取られる成り駒の非限定」.  Written only
    /// when it is not `0`, so a record without the flag keeps its bytes (the
    /// TypeScript `solveStage` adds the key in the same place on the same
    /// condition).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub captured_promo_alt_nodes: u64,
    pub graph: SolverGraphJson,
    pub stats: SolveStatsJson,
    pub defense_coverage_verified: bool,
    pub features: DifficultyFeatures,
    /// Written by the TypeScript `compile` stage; preserved on round-trip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rejection {
    pub stage: String,
    pub reasons: Vec<String>,
    /// Composition rule G: the board pieces that take no part in the solution,
    /// attacker entries first.  Present exactly when the rejection carries
    /// `inert_attacker_piece` or `inert_defender_piece`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inert_pieces: Option<Vec<InertPiece>>,
}

/// One JSONL pipeline document.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineRecord {
    pub schema_version: u32,
    pub stage: String,
    pub source_id: String,
    pub source_sfen: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_sfen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_mate_ply: Option<u32>,
    /// Rule A opt-out: keep the attacker's king on the board.  Absent means `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_attacker_king: Option<bool>,
    pub original_candidate_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_mirrored: Option<bool>,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<Validation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduction: Option<Reduction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solution: Option<Box<Solution>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection: Option<Rejection>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
    /// What the publishing gate last decided about this diagram, for
    /// `--emit-record` ([`crate::catalog_record`]).  Never serialised or read:
    /// the record's JSON is the same with it or without it.
    #[serde(skip)]
    pub gate: Option<Box<crate::catalog_record::GateFacts>>,
    /// What rule D on this record took ([`crate::record_clock::Cost`]), for
    /// `--emit-record`'s `reduce.reducedAt` / `wallMs` / `cpuMs`.  Never
    /// serialised or read.
    #[serde(skip)]
    pub reduce_cost: Option<crate::record_clock::Cost>,
}

impl PipelineRecord {
    /// The imported-stage record `distill` and `extend` build for a bare SFEN
    /// input line (one helper since 2026-10-27; the two copies were the same
    /// but for `tool` and `version`).
    ///
    /// `tool` names the source id (`<tool>-<ordinal>-<hash12>`) and the
    /// transformation (`<tool>:sfen-input`); `version` is the generator version
    /// in the provenance.  A bare SFEN carries no provenance of its own, so the
    /// caller has to say where the position came from (`origin`,
    /// `origin_class`; `docs/legal-data.md` §3).  Nothing here defaults to
    /// `own`: `main.rs` refuses bare-SFEN input that does not name an origin.
    pub fn imported_from_sfen(
        position: &crate::position::Position,
        ordinal: usize,
        origin: &str,
        origin_class: &str,
        tool: &str,
        version: &str,
    ) -> PipelineRecord {
        let sfen = crate::sfen::to_sfen(position);
        // Hash the pre-2026-10-04 spelling so a seed keeps its id (`to_legacy_sfen_key`).
        let hash = crate::util::sha256_text(&crate::sfen::to_legacy_sfen_key(position));
        let source_id = format!("{tool}-{ordinal}-{}", &hash[..12]);
        let source_type = if origin_class == ORIGIN_CLASS_THIRD_PARTY {
            "third_party_practical_mate"
        } else {
            "original_seed"
        };
        let mut provenance = Provenance::generated(source_type, version, hash.clone());
        provenance.source_position_hash = Some(hash.clone());
        provenance.proposal_fingerprint = Some(hash);
        provenance.set_root_origin(origin_class, origin, &source_id);
        provenance.transformations = vec![format!("{tool}:sfen-input")];
        PipelineRecord::new_imported(source_id, sfen, provenance, Map::new())
    }

    /// A new imported-stage record with nothing judged yet: its own id as the
    /// original candidate, the given provenance and `extra` (the generators'
    /// records: a bare SFEN input line, a mined candidate).
    pub fn new_imported(
        source_id: String,
        source_sfen: String,
        provenance: Provenance,
        extra: Map<String, Value>,
    ) -> PipelineRecord {
        PipelineRecord {
            schema_version: PIPELINE_SCHEMA_VERSION,
            stage: "imported".to_string(),
            source_id: source_id.clone(),
            source_sfen,
            normalized_sfen: None,
            target_mate_ply: None,
            allow_attacker_king: None,
            original_candidate_id: source_id,
            canonical_hash: None,
            canonical_mirrored: None,
            provenance,
            validation: None,
            reduction: None,
            solution: None,
            rejection: None,
            extra,
            gate: None,
            reduce_cost: None,
        }
    }

    /// A diagram derived from this record (a distill variant, an extend
    /// candidate): a copy at stage `imported` with id `source_id`, position
    /// `source_sfen` and target `target_mate_ply`, everything judged about
    /// this record cleared (normalized SFEN, canonical hash, validation,
    /// reduction, solution, rejection), the original candidate kept (this
    /// record's id when it names none), and the step appended to the lineage
    /// (`docs/legal-data.md` §3: the copy keeps the origin set).  The rest
    /// (`provenance.extra`, transformations, `extra`) is the caller's.
    pub fn derived(
        &self,
        source_id: String,
        source_sfen: String,
        target_mate_ply: u32,
        tool: &str,
        version: &str,
    ) -> PipelineRecord {
        let mut out = self.clone();
        out.stage = "imported".to_string();
        out.source_id = source_id;
        if out.original_candidate_id.is_empty() {
            out.original_candidate_id = self.source_id.clone();
        }
        out.provenance
            .append_lineage(&self.source_id, tool, version);
        out.source_sfen = source_sfen;
        out.normalized_sfen = None;
        out.canonical_hash = None;
        out.canonical_mirrored = None;
        out.target_mate_ply = Some(target_mate_ply);
        out.validation = None;
        out.reduction = None;
        out.solution = None;
        out.rejection = None;
        out
    }

    /// This record refused by a generator at stage `stage` (`distill`,
    /// `extend`) with `reasons`: a copy at stage `solved` without a solution.
    /// The generator adds its own `extra` block.
    pub fn refused(&self, stage: &str, reasons: Vec<String>) -> PipelineRecord {
        let mut out = self.clone();
        out.stage = "solved".to_string();
        out.solution = None;
        out.rejection = Some(Rejection {
            stage: stage.to_string(),
            reasons,
            inert_pieces: None,
        });
        out
    }

    /// Rule A opt-out, defaulting to `false` when the field is absent.
    pub fn allows_attacker_king(&self) -> bool {
        self.allow_attacker_king.unwrap_or(false)
    }

    /// `isRecordAccepted`: no rejection and validation not explicitly invalid.
    pub fn is_accepted(&self) -> bool {
        self.rejection.is_none()
            && self
                .validation
                .as_ref()
                .map(|validation| validation.valid)
                .unwrap_or(true)
    }

    /// Appends provenance transformation strings, as every TypeScript stage does.
    pub fn add_transformations(&mut self, entries: &[&str]) {
        for entry in entries {
            self.provenance.transformations.push((*entry).to_string());
        }
    }
}

/// Parses one JSONL line, rejecting unsupported schema versions like
/// `readJsonLine`.
pub fn read_json_line(line: &str) -> Result<PipelineRecord> {
    let record: PipelineRecord =
        serde_json::from_str(line).context("Invalid JSONL pipeline record")?;
    if record.schema_version != PIPELINE_SCHEMA_VERSION {
        bail!("Unsupported pipeline schema {}", record.schema_version);
    }
    Ok(record)
}

/// Serialises one record as a single JSON line.
pub fn write_json_line(record: &PipelineRecord) -> Result<String> {
    Ok(serde_json::to_string(record)?)
}

#[cfg(test)]
mod lineage_tests {
    use super::*;

    #[test]
    fn lineage_entries_keep_fields_they_do_not_know() {
        let line = r#"{"sourceId":"a","tool":"reduce","version":"v","fromCanonicalHash":"h","reducedLineChanged":false,"removed":1}"#;
        let entry: LineageEntry = serde_json::from_str(line).unwrap();
        assert_eq!(entry.extra.len(), 3);
        let back: Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(back, serde_json::from_str::<Value>(line).unwrap());
    }

    #[test]
    fn plain_lineage_entries_serialise_as_before() {
        let mut provenance = Provenance::generated("t", "g", "c".to_string());
        provenance.append_lineage("a", "distill", "v");
        let text = serde_json::to_string(&provenance.lineage).unwrap();
        assert_eq!(text, r#"[{"sourceId":"a","tool":"distill","version":"v"}]"#);
    }
}
