//! The command line: the clap types of every command and their shared option groups.

use super::*;

/// `--version`: the crate's version and the commit the binary was built from
/// (`build.rs`, the same id as the records' `S-<commit>`), e.g.
/// `tsume-solver 1.0.0 (ec2ef78)`.  Versioning: `README.md`, "Versions".
const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("TSUME_SOLVER_BUILD"),
    ")"
);

#[derive(Parser)]
#[command(
    name = "tsume-solver",
    about = "Offline strict-tsume solver and pipeline stages",
    version = VERSION
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

// `clap` parses this once at start-up, so the size of a variant is irrelevant and
// boxing individual option fields would fight the derive.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
pub(crate) enum Command {
    /// Print every legal move of a position as sorted USI, one per line.
    Movegen {
        /// Four-field SFEN.  Omit with --stdin.
        sfen: Option<String>,
        /// Read one SFEN per line from stdin and print one JSON object per line.
        #[arg(long)]
        stdin: bool,
    },
    /// Count legal move sequences of exactly <DEPTH> plies.
    Perft { sfen: String, depth: u32 },
    /// Solve one SFEN (or a JSON-lines stream) and print the full solve result.
    SolveSfen {
        sfen: Option<String>,
        #[arg(long, default_value_t = 11)]
        max_ply: u32,
        #[arg(long, default_value_t = 100_000)]
        node_budget: u64,
        /// Attacking side; defaults to the side to move.
        #[arg(long, value_parser = ["b", "w"])]
        attacker: Option<String>,
        /// Disable the strict-tsume defender remainder.
        /// Experimental and hidden from `--help`; using it prints a note on stderr
        /// and tags the `--emit-record` records `+x`.
        // Experimental since 2026-10-22.
        // docs/experiments/solver-cli-audit/README.md §7.4, §8.
        #[arg(long = "no-strict-material", hide = true)]
        no_strict_material: bool,
        /// `pruned` (default; the full graph of a deep solve runs to gigabytes),
        /// `full`, or `none`.
        // `pruned` is the default since 2026-10-22.
        #[arg(long, default_value = "pruned", value_parser = ["full", "pruned", "none"])]
        graph: String,
        /// 無駄合 policy (rule F): `classical` (default: prune plus the classical
        /// 無駄合 reading, a check answered only by futile interposition drops is
        /// mate; what the composition rules use), `prune`, or `keep` (the bare
        /// solver contract).  A stdin request may override it per line with
        /// `"uselessInterposition"`.
        // `classical` is the default since 2026-10-22.
        // Rule F: docs/rules.md.  Classical since PIPELINE_VERSION 0.10.0.  `keep` is packages/solver's default, which
        // diff-solve.ts names explicitly.
        #[arg(long, default_value = "classical", value_parser = ["keep", "prune", "classical"])]
        useless_interposition: String,
        /// Read `{"sfen":…,"maxPly":…,"nodeBudget":…,"strictMaterial":…,
        /// "uselessInterposition":…,"creditBudget":…}` JSON objects, one per
        /// line, from stdin; every field but `sfen` falls back to the flag
        /// above.
        #[arg(long)]
        stdin: bool,
        #[command(flatten)]
        position: PositionParallel,
    },
    /// Strict-tsume normalisation, structural validation and canonical hash.
    Validate {
        #[command(flatten)]
        parallel: Parallel,
    },
    /// Seeded multi-order greedy piece removal.
    Reduce {
        #[arg(long, default_value_t = 8)]
        attempts: u32,
        /// Mate-search ply bound (required).
        // Required since 2026-10-22: a forgotten bound silently judged at 11
        // (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        max_ply: u32,
        /// Node budget of one search (required).
        // Required since 2026-10-22: a forgotten budget silently ran one or two
        // orders of magnitude below the catalogue's (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        node_budget: u64,
        /// Overrides the seed derived from `sourceId`.
        #[arg(long)]
        seed: Option<u64>,
        /// Rule D v5: up to K length-preserving reductions (other displayed
        /// main lines) as extra records after each input's main-line-preserving
        /// one.  `0` (the default) keeps one output record per input.
        // The frozen number the catalogue runs pass is 2 (docs/rules.md).
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u64).range(0..=tsume_solver::reduce::LENGTH_K as u64))]
        length_k: u64,
        /// Turns rule D v5's rule B entry off: by default an input
        /// refused for rule B alone is let in, its record becomes its best
        /// length-preserving reduction (and up to K_y - 1 more follow), each
        /// step judged by the full acceptance test;
        /// one with no step is rejected `yozume_not_reducible`.
        // owner decision 2026-10-09 17:50 JST.
        /// Experimental and hidden from `--help`; using it prints a note on stderr
        /// and tags the `--emit-record` records `+x`.
        // Experimental since 2026-10-22.
        // docs/experiments/solver-cli-audit/README.md §7.4, §8.
        #[arg(long = "no-rule-b-entry", hide = true)]
        no_rule_b_entry: bool,
        /// The C/G entry, the rule C / G repair.  Off by default and not part of
        /// the frozen rule D: an
        /// input refused for rules C and/or G alone (`koma_amari`, `inert_*`;
        /// together with rule B's reasons when the rule B entry is on) is
        /// repaired -- spare attacker hand units handed back, inert pieces
        /// swept, each step judged with only those defects set aside -- until
        /// it passes the full test, then reduced like a rule B entry input to
        /// its fixpoint: length-preserving outputs only (`reduction.entry`
        /// `rule-cg-relaxed`, `reduction.minimality` as for every reduction),
        /// or `cg_not_repairable`.  Every output is gated again by `solve`.
        // owner decision 2026-10-10, owner-approved as a formal option 2026-11-10:
        // the production judge repairs koma-amari and uninvolved pieces with it;
        // the public pipeline never passes it (docs/experiments/cg-repair/README.md).
        #[arg(long = "cg-entry")]
        cg_entry: bool,
        // owner 2026-11-22: the workshop's pinned pieces and square rules.
        /// Squares whose piece is never taken off: comma-separated board
        /// squares 0-80 (`rank * 9 + file`, file 0 is USI file 9) and `81 + k`
        /// for every attacker hand unit of type `k` (`P L N S G B R`).  The
        /// verdicts are unchanged; the result is minimal among the other
        /// pieces (`reduction.minimality` proves that much).  Absent: the
        /// reducer as it was, byte for byte.  Not with `--emit-record`: a
        /// record's minimality is over every piece.
        #[arg(long = "keep-squares", value_parser = parse_keep_squares)]
        keep_squares: Option<tsume_solver::reduce::KeepSquares>,
        // owner-approved 2026-09-25.
        /// Concurrent acceptance tests per record:
        /// the attempts and speculative greedy steps of one record run on this
        /// many threads, each search on its own diagram with a solver of its
        /// own.  Candidates are committed in the sequential order, so the
        /// output does not depend on this value.  Memory is about `jobs` times
        /// one reduce's.  Not with `--prober-threads` > 1.  `--threads` still
        /// schedules whole records.
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[command(flatten)]
        ladder: BudgetLadder,
        #[command(flatten)]
        negatives: NegativesArgs,
        #[command(flatten)]
        budget_outs: BudgetOutsArgs,
        #[command(flatten)]
        record: RecordArgs,
        #[command(flatten)]
        parallel: Parallel,
        #[command(flatten)]
        position: PositionParallel,
    },
    /// Exact solve, uniqueness/coverage gates, features and runtime graph.
    Solve {
        /// Mate-search ply bound (required).
        // Required since 2026-10-22: a forgotten bound silently judged at 11
        // (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        max_ply: u32,
        /// Node budget of one search (required).
        // Required since 2026-10-22: a forgotten budget silently ran one or two
        // orders of magnitude below the catalogue's (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        node_budget: u64,
        /// Accepted and ignored: the gate's designated main line
        /// ([`tsume_solver::exact::designated_main_line`], the Clean-first walk
        /// of the full graph) is always written as `extra.designatedMainLine`
        /// (USI, empty when rules B and C were never evaluated) and
        /// `extra.designatedMainLineLeftoverHand` on every record the gate
        /// solved, accepted or rejected, yozume included.
        // Ignored since 2026-10-22; kept (and listed in --help) so callers that
        // also drive older binaries, where it is still needed, keep one command
        // line.
        // owner 2026-10-11: the catalogue keeps a main line for every row
        // (docs/catalog/INGEST.md).
        #[arg(long)]
        emit_main_line: bool,
        #[command(flatten)]
        record: RecordArgs,
        #[command(flatten)]
        parallel: Parallel,
        #[command(flatten)]
        position: PositionParallel,
    },
    /// Measurement only: probe each input line (a record or a bare SFEN) for a
    /// forced mate within `--max-ply` and its length; one small JSON object per
    /// line, nothing published.
    ///
    /// `dfpn` (the df-pn prober), `exact-probe` (`exact::minimum_mate_validate`,
    /// rules off) and `cdfpn` (the classical df-pn prober under rule F
    /// classical, `cdfpn::entry_probe`; its `pv` is always empty).
    ///
    /// `cdfpn` rows also carry `disproofKind` (null unless `DISPROVEN`):
    /// `checksExhausted` -- no mate at any bound (every checking line runs out
    /// of checks, a proof; a deeper `--max-ply` cannot mate) -- or `horizon`
    /// -- some checking line reached `--max-ply` (or no bound-free refutation
    /// was found; a deeper mate may exist).  Measurement only: the other
    /// fields are unchanged.
    // docs/generation.md §5.5.  cdfpn is the RL entry since 2026-10-10; split
    // out of `solve --engine` on 2026-10-22 (docs/experiments/solver-cli-audit/README.md §7.6, §8).
    Probe {
        /// `dfpn`, `exact-probe` or `cdfpn`.
        #[arg(long, value_parser = ["dfpn", "exact-probe", "cdfpn"])]
        engine: String,
        /// Mate-search ply bound.
        #[arg(long)]
        max_ply: u32,
        /// Node budget per input line.
        #[arg(long)]
        node_budget: u64,
        #[command(flatten)]
        parallel: Parallel,
        #[command(flatten)]
        position: PositionParallel,
    },
    /// Rule D over records a generator already accepted (`extend`, `distill`
    /// output): the standard `reduce` stage, then a re-solve only when pieces
    /// came off.
    ///
    /// Each record is reduced and re-solved under the bound its own solve used
    /// (`solution.stats.maxPly`), so a mixed stream keeps every record's gate.
    /// A record the reducer or the re-solve rejects (a node budget, in practice)
    /// is written back unchanged with `extra.reduceAccepted.outcome = "fallback"`
    /// and the reasons.  Rejected input records pass through untouched.
    ///
    /// `--emit-record`: the side record of every output diagram
    /// and, when pieces came off, of the input as a reducible one, judged by
    /// the publishing gate at the record's bound and `--node-budget` (an
    /// `unchanged` or `fallback` record, which the command does not re-solve,
    /// is judged again on a copy for its record).  The record's `route`
    /// defaults to `reduce`; an input record's own override or
    /// `--record-route` names the generator.
    // The override: extra.catalogProducer (producer.route).
    ReduceAccepted {
        /// Node budget of one search (required).
        // Required since 2026-10-22: a forgotten budget silently ran one or two
        // orders of magnitude below the catalogue's (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        node_budget: u64,
        /// Randomised greedy orders; the fewest-piece result wins.
        #[arg(long, default_value_t = 8)]
        attempts: u32,
        /// Rule D v5: also emit up to K length-preserving reductions, each
        /// solved by the same gate, after the record (only those it accepts).
        /// `0` (the default) keeps one output record per input.
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u64).range(0..=tsume_solver::reduce::LENGTH_K as u64))]
        length_k: u64,
        // owner-approved 2026-09-25.
        /// Concurrent acceptance tests per record:
        /// the attempts and speculative greedy steps of one record run on this
        /// many threads, each search on its own diagram with a solver of its
        /// own.  Candidates are committed in the sequential order, so the
        /// output does not depend on this value.  Memory is about `jobs` times
        /// one reduce's.  Not with `--prober-threads` > 1.  `--threads` still
        /// schedules whole records.
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[command(flatten)]
        ladder: BudgetLadder,
        #[command(flatten)]
        budget_outs: BudgetOutsArgs,
        #[command(flatten)]
        record: RecordArgs,
        #[command(flatten)]
        parallel: Parallel,
        #[command(flatten)]
        position: PositionParallel,
    },
    // Spec 9.1.7, spec 10.
    /// Generate tsume puzzles from mating positions.
    ///
    /// Reads SFEN lines or JSONL PipelineRecords (any stage), one position per
    /// record, and writes solved-stage records for the best strict-valid variants.
    Distill {
        /// Base seed (required).
        // Required since 2026-10-22: the time-based default broke reproducibility
        // (spec §35).
        #[arg(long)]
        seed: u64,
        /// Randomised greedy operator orders per source.
        // Spec 10.4.
        #[arg(long, default_value_t = 16)]
        attempts: u32,
        // docs/generation.md §2.6.
        /// Opt in to `distill-v2`: rule B relaxed
        /// on chain steps and one `relocate` (operator f, the rule-C repair).
        /// **Off by default** -- without it `distill` reproduces `distill-v1`
        /// byte for byte and stamps `distill-v1` in the provenance.
        #[arg(long)]
        distill_v2: bool,
        /// Accept variants whose minimum mate length differs from the source's
        /// (shorter or longer; the bound is `--max-ply`).  Each emitted record
        /// carries the variant's own `matePly`, and the provenance records
        /// `allowLengthChange: true`.  **Off by default**: without it the length
        /// stays the source's.
        // owner 2026-10-29, for sources whose fixed-length distill found nothing;
        // without it the output is byte for byte the same as before the flag
        // came back.
        #[arg(long)]
        allow_length_change: bool,
        /// Origin dataset id of **bare SFEN** input lines, e.g. `own` or
        /// `yaneuraou-2020-12-25` (`solver/README.md`, "Provenance").  Required together
        /// with `--origin-class` when any input line is a bare SFEN; JSONL input
        /// carries its own provenance and ignores both flags.
        #[arg(long)]
        origin: Option<String>,
        /// Origin class of bare SFEN input lines.  Required with `--origin`.
        #[arg(long, value_parser = tsume_solver::jsonl::ORIGIN_CLASSES)]
        origin_class: Option<String>,
        /// Mate-search ply bound (required).
        // Required since 2026-10-22: a forgotten bound silently judged at 11
        // (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        max_ply: u32,
        #[arg(long, default_value_t = 250_000)]
        node_budget: u64,
        /// Upper bound on search nodes spent per source position.
        #[arg(long, default_value_t = 4_000_000)]
        max_nodes: u64,
        /// Delete far board pieces while the source-length mate survives, before
        /// the variant search (`crate::prestrip`).  Off by default.
        #[arg(long, default_value_t = false)]
        prestrip: bool,
        /// Engine behind the per-step existence/minimal-length checks: `exact`
        /// (default) or `dfpn`.  The
        /// emitted record's verdict, graph and stats always come from the exact
        /// solver.
        // docs/generation.md §5.5.
        #[arg(long, default_value = "exact", value_parser = ["exact", "dfpn"])]
        probe: String,
        /// Node budget of one df-pn probe.  Ignored with `--probe exact`.
        #[arg(long, default_value_t = 1_000_000)]
        dfpn_node_budget: u64,
        /// Write the run counters (wall, measured CPU, records) to this JSON file.
        #[arg(long)]
        stats_out: Option<String>,
        #[command(flatten)]
        negatives: NegativesArgs,
        #[command(flatten)]
        budget_outs: BudgetOutsArgs,
        #[command(flatten)]
        record: RecordArgs,
        #[command(flatten)]
        parallel: Parallel,
    },
    /// Verification of the df-pn rule B pre-filter (`src/rule_b_probe.rs`).
    ///
    /// Reads `{"id":…,"sfen":…,"matePly":N}` JSON lines (any `id` value is passed
    /// through; `sourceId`, `normalizedSfen` / `sourceSfen` and `targetMatePly`
    /// are accepted too, and a bare SFEN line takes `--mate-ply`) and writes one
    /// `{"id","verdict":"reject"|"inconclusive","reason","nodes","plyReached"}`
    /// line per input, in input order.  The rules are distill's final rules
    /// (`TsumeRules::puzzle()`, rule F classical), so the walk enforces rule B at
    /// every length (since rules version 0.12.0).  Publishes nothing.
    RuleBProbe {
        /// df-pn node budget of one position, over every probe of its walk.
        #[arg(long, default_value_t = 200_000)]
        node_budget: u64,
        /// Target mate length for lines that carry none.
        #[arg(long)]
        mate_ply: Option<u32>,
        /// Write the run counters (verdicts, reasons, nodes, wall and CPU) here.
        #[arg(long)]
        stats_out: Option<String>,
        #[command(flatten)]
        parallel: Parallel,
    },
    // Catalog schema v2.5 `n_main_paths` / `has_final_alt`.
    /// `|Π|` and the final-move alternative of stored solution graphs, without
    /// solving.
    ///
    /// Reads JSONL with `canonicalHash` and `solution` (the solution object, or
    /// a string holding its JSON) and writes one
    /// `{"canonicalHash","nMainPaths","hasFinalAlt","error"}` line per input, in
    /// input order.  `nMainPaths` is the number of root-to-end paths of the
    /// main-line region exactly as the reducer counts them (rule D,
    /// "Reduction v5"), `null` over `--cap`; `hasFinalAlt` is 1 when an attack
    /// node of the region at distance 1 has two or more correct moves.  A line
    /// that cannot be read gets `error` (and `nMainPaths: null, hasFinalAlt: 0`).
    MainLinePaths {
        /// Path cap: over this many paths `nMainPaths` is `null`.
        #[arg(long, default_value_t = tsume_solver::exact::MAIN_LINE_PATHS_CAP)]
        cap: usize,
        #[command(flatten)]
        parallel: Parallel,
    },
    // 1.8.0: the catalog's main_line toward this build's tie-break,
    // PIPELINE_VERSION 0.17.0.
    /// The designated main line of stored solution graphs, without solving
    /// (with this build's tie-break, e.g. a promotion before its
    /// non-promoting twin).
    ///
    /// Reads JSONL with `canonicalHash` and `solution` (the solution object, or
    /// a string holding its JSON) and writes one
    /// `{"canonicalHash","mainLine","leftoverHand","complete","error"}` line per
    /// input, in input order.  `mainLine` is `exact::designated_main_line` of
    /// the stored graph -- the walk `solve --emit-main-line` writes as
    /// `extra.designatedMainLine` -- so it reads only the main-line region the
    /// stored graph holds whole.  A line that cannot be read gets `error` (and
    /// `mainLine: null`).
    StoredMainLine {
        #[command(flatten)]
        parallel: Parallel,
    },
    /// Reverse method (逆算法): enumerate 2-ply predecessors and gate every one.
    ///
    /// Reads solved-stage JSONL records (or bare SFENs) of mate length k and writes
    /// solved-stage records for every `k + 2` predecessor, accepted or rejected.
    Extend {
        /// Node budget of one candidate's publishing gate (required).
        // Required since 2026-10-22 (docs/experiments/solver-cli-audit/README.md §7.3).
        #[arg(long)]
        node_budget: u64,
        /// Ply bound used to measure k when the input record does not carry it.
        #[arg(long, default_value_t = 11)]
        source_max_ply: u32,
        /// `accepted` (default), `all`, or `candidates` (enumerate only, no
        /// search: the input of a candidate ranker).
        // The ranker: tools/reverse-ranker/score.py.
        #[arg(long, default_value = "accepted", value_parser = ["accepted", "all", "candidates"])]
        emit: String,
        /// JSONL verification order: one `{"parentSourceId", "candidateIndex",
        /// "score"}` row per scored candidate.  Candidates are gated in
        /// descending score (ties by `candidateIndex`); unscored ones follow in
        /// enumeration order.  Requires `--ranker-id`.
        #[arg(long)]
        order: Option<String>,
        /// Model id written to `provenance.extra.ranker.id`, e.g. the sha256 of
        /// the LightGBM `model.txt`.  Required with `--order`.
        #[arg(long)]
        ranker_id: Option<String>,
        /// Candidates *verified* per source; 0 verifies every one.
        #[arg(long, default_value_t = 0)]
        max_verify: usize,
        /// Stop a source after this many accepted candidates; 0 never stops.
        #[arg(long, default_value_t = 0)]
        stop_after_accepted: usize,
        /// Engine that screens each candidate before the exact publishing gate:
        /// `exact` (default) runs the gate on every candidate, `dfpn` puts the
        /// df-pn prober in front of it.  The prober can only reject; accepted
        /// records are byte-identical either way.
        #[arg(long, default_value = "exact", value_parser = ["exact", "dfpn"])]
        probe: String,
        /// df-pn node budget of one candidate probe.  Only read with `--probe dfpn`.
        #[arg(long, default_value_t = 1_000_000)]
        dfpn_node_budget: u64,
        /// Origin dataset id of **bare SFEN** input lines, e.g. `own` or
        /// `yaneuraou-2020-12-25` (`solver/README.md`, "Provenance").  Required together
        /// with `--origin-class` when any input line is a bare SFEN; JSONL input
        /// carries its own provenance and ignores both flags.
        #[arg(long)]
        origin: Option<String>,
        /// Origin class of bare SFEN input lines.  Required with `--origin`.
        #[arg(long, value_parser = tsume_solver::jsonl::ORIGIN_CLASSES)]
        origin_class: Option<String>,
        /// Write the run counters to this JSON file.
        #[arg(long)]
        stats_out: Option<String>,
        #[command(flatten)]
        budget_outs: BudgetOutsArgs,
        #[command(flatten)]
        record: RecordArgs,
        #[command(flatten)]
        parallel: Parallel,
    },
    /// Mine mating positions: random position -> AI play -> mate detection.
    Mine {
        /// Candidate-generation family.
        // Spec 9.1.1.
        #[arg(long, default_value = "random_selfplay", value_parser = ["random_direct", "random_selfplay", "standard_selfplay"])]
        family: String,
        /// Candidate records to emit.
        #[arg(long, default_value_t = 100)]
        count: usize,
        /// Base seed (required).
        // Required since 2026-10-22: the time-based default broke reproducibility
        // (spec §35).
        #[arg(long)]
        seed: u64,
        /// Self-play plies before/while mining, as MIN..MAX or a single number.
        #[arg(long, default_value = "5..40")]
        plies: String,
        /// Mate-search ply bound.
        #[arg(long, default_value_t = 7)]
        max_ply: u32,
        /// First ply eligible for mining; defaults to 0 for random_direct, 1 otherwise.
        #[arg(long)]
        check_from: Option<u32>,
        #[arg(long, default_value_t = 200_000)]
        node_budget: u64,
        #[arg(long)]
        max_games: Option<u64>,
        /// Strip far board pieces while the mate length holds, before emitting.
        #[arg(long)]
        prestrip: bool,
        /// Engine behind mate detection: `exact` (default) or `dfpn`.
        /// `strictValid` always comes from the exact solver.
        // docs/generation.md §5.5.
        #[arg(long, default_value = "exact", value_parser = ["exact", "dfpn"])]
        probe: String,
        /// Node budget of one df-pn probe.  Ignored with `--probe exact`.
        #[arg(long, default_value_t = 1_000_000)]
        dfpn_node_budget: u64,
        /// `RandomPositionGeneratorConfig` JSON.
        // Spec 9.1.2.
        #[arg(long)]
        config: Option<String>,
        /// Restrict the proposal distribution.
        #[arg(long, value_parser = ["king-zone", "slider-pressure", "mate-shell", "uniform"])]
        distribution: Option<String>,
        /// `builtin` or `random`.
        // `usi:<engine command>` went on 2026-10-27.
        #[arg(long, default_value = "builtin")]
        policy: String,
        /// Write the run counters to this JSON file.
        #[arg(long)]
        stats_out: Option<String>,
        #[command(flatten)]
        parallel: Parallel,
    },
}

// owner order 2026-10-10 22:30 JST, docs/experiments/reduce-negatives/README.md.
/// `--emit-negatives` ([`tsume_solver::negatives`]):
/// the diagrams the reducer judged and refused, written to a side file.  Off by
/// default; the main output is byte-identical with it on or off.
impl Command {
    /// The prober options of a command that takes them.
    pub(crate) fn position(&self) -> Option<PositionParallel> {
        match self {
            Command::SolveSfen { position, .. }
            | Command::Reduce { position, .. }
            | Command::Solve { position, .. }
            | Command::Probe { position, .. }
            | Command::ReduceAccepted { position, .. } => Some(*position),
            _ => None,
        }
    }
}

#[derive(Args, Clone, Debug)]
pub(crate) struct NegativesArgs {
    /// Write the negatives (one `"type":"negative"` row per picked diagram,
    /// one `"type":"source"` summary row per input) to this JSONL file, in
    /// input order.  Rows are deduplicated by canonical hash over the run.
    #[arg(long)]
    pub(crate) emit_negatives: Option<String>,
}

/// Negatives written per input at most (closest to a sound diagram first).
// Fixed since `--negatives-per-source` went on 2026-10-27 (never given).
pub(crate) const NEGATIVES_PER_SOURCE: usize = 4;

impl NegativesArgs {
    /// The finalize options, or `None` when the feature is off.
    pub(crate) fn setup(
        &self,
        command: &'static str,
        node_budget: u64,
    ) -> Option<(String, tsume_solver::negatives::FinalizeOptions)> {
        self.emit_negatives.clone().map(|path| {
            (
                path,
                tsume_solver::negatives::FinalizeOptions {
                    cap: NEGATIVES_PER_SOURCE,
                    node_budget,
                    command,
                },
            )
        })
    }
}

// owner 2026-10-17.
/// `--emit-budget-outs` ([`tsume_solver::reduce::budget_outs_begin`]):
/// every reducer check whose final answer ran out of its budget, per input,
/// written to a side file.  Off by default; the main output, the records and
/// `--emit-negatives` are byte-identical with it on or off.
#[derive(Args, Clone, Debug)]
pub(crate) struct BudgetOutsArgs {
    /// Write one JSONL line per input, in input order: its `sourceId` and
    /// `canonicalHash`, `budgetOutChecks` (the reducer checks whose final
    /// answer on the last rung carried `solver_budget_exhausted`,
    /// `incomplete_response_graph` or `minimum_length_not_proven`) and one
    /// entry per such check: stage, removal, diagram, budget, nodes, reasons,
    /// thread mode (`serial`, or `parallel` with `--prober-threads` > 1) and
    /// final outcome (always `still-out`; `rerun` is always null).
    // Always still-out since 2026-10-27, when the one-thread re-run went.
    #[arg(long)]
    pub(crate) emit_budget_outs: Option<String>,
}

// owner 2026-10-12; docs/catalog/INGEST.md
// §6, tools/catalog/record.py, tools/catalog/ingest.py.
/// `--emit-record`: one side record of every diagram the publishing gate
/// judged, written to a side file in input order.  Off by default; the main
/// output is byte-identical with it on or off ([`tsume_solver::catalog_record`]).
#[derive(Args, Clone, Debug)]
pub(crate) struct RecordArgs {
    /// Write a side record of every judged diagram (verdict, main line,
    /// solver and rules versions) to this JSONL file, in input order.
    #[arg(long)]
    pub(crate) emit_record: Option<String>,
    /// Write the records of status `unknown` (the gate ran out of budget,
    /// with or without a target length: its reasons are only
    /// `solver_budget_exhausted` / `incomplete_response_graph`) to this JSONL
    /// file instead of the `--emit-record` file, in input order.  Without it
    /// they stay in the records' file.  A gate whose prober or construction
    /// ran out of budget lands here too.
    // 2026-10-27, for the escalation queue.  Since 1.4.0 (owner 2026-11-05)
    // the prober / construction budget-outs too: the one-thread full-width
    // search no longer solves them again on the same budget.
    #[arg(long)]
    pub(crate) emit_unresolved: Option<String>,
    /// The binary's id, `S-<7-hex commit>[tag]` (e.g. `S-dd32a41`): the record's
    /// `solveVersion` is `<id>/<rules version>/<gate node budget>`.
    /// Defaults to `S-<the commit the binary was built from>` (`+dirty` when
    /// the crate's files differed from it; the record's `solverBuild`); required
    /// only for a build that does not know its commit (`solverBuild`
    /// `unknown`).  The tags `+x` and `+pdfpn` are appended either way.
    #[arg(long)]
    pub(crate) record_solver_id: Option<String>,
    // docs/catalog/CATALOG_ORIGINS.md §3.4.
    /// The record's `producedBy`: who or what made the diagrams.  Required
    /// with `--emit-record`.
    // producer.producedBy.  An input record's extra.catalogProducer object
    // overrides it (and the route, the generation, interesting, rankerId) per
    // record.
    #[arg(long)]
    pub(crate) record_produced_by: Option<String>,
    /// The record's `route`.  `extend`, `distill` and `reduce` default to their
    /// own name; `solve` needs it.
    #[arg(long, value_parser = tsume_solver::catalog_record::ROUTES)]
    pub(crate) record_route: Option<String>,
    /// The record's `generation` (0 when not given).
    #[arg(long)]
    pub(crate) record_generation: Option<u64>,
    /// `solve` only: the `reduce.version` of input records that carry a
    /// `reduction` (`D-v5/<rules version>/<reduce node budget>`).  Without it
    /// their reduce status stays `unknown`.
    /// `reduce`, `extend` and `distill` write their own.
    #[arg(long)]
    pub(crate) record_reduce_version: Option<String>,
}

impl RecordArgs {
    /// `true` when any of the options was given.
    pub(crate) fn given(&self) -> bool {
        self.emit_record.is_some()
            || self.emit_unresolved.is_some()
            || self.record_solver_id.is_some()
            || self.record_produced_by.is_some()
            || self.record_route.is_some()
            || self.record_reduce_version.is_some()
            || self.record_generation.is_some()
    }

    /// The side files and the record context, or `None` when the feature is
    /// off.  Turns the gate's fact collection on.  `other_outputs`: the
    /// command's other output files, which the side files must not be.
    /// `reduce_version`: the command's own rule D version, for a command that
    /// reduces (the records then carry the certificate and the cost of their
    /// reduction, [`RecordContext::certify`]).
    pub(crate) fn setup(
        &self,
        command: &str,
        default_route: Option<&str>,
        reduce_version: Option<String>,
        other_outputs: &[Option<&str>],
    ) -> Result<Option<(RecordFiles, RecordContext)>> {
        let Some(path) = self.emit_record.clone() else {
            if self.given() {
                bail!("{command}: the --record-* and --emit-unresolved options need --emit-record <path>");
            }
            return Ok(None);
        };
        let key = output_key(&path);
        if other_outputs
            .iter()
            .flatten()
            .any(|other| output_key(other) == key)
        {
            bail!("{command}: --emit-record {path} is also another output of the command");
        }
        if let Some(unresolved) = self.emit_unresolved.as_deref() {
            let unresolved_key = output_key(unresolved);
            if unresolved_key == key
                || other_outputs
                    .iter()
                    .flatten()
                    .any(|other| output_key(other) == unresolved_key)
            {
                bail!("{command}: --emit-unresolved {unresolved} is also another output of the command");
            }
        }
        let certify = reduce_version.is_some();
        let mut solver_id = match self.record_solver_id.clone() {
            Some(id) => id,
            None => tsume_solver::record_clock::default_solver_id().with_context(|| {
                format!(
                    "this build does not know its commit (solverBuild {}): \
                     --emit-record needs --record-solver-id S-<commit>[tag]",
                    tsume_solver::record_clock::SOLVER_BUILD
                )
            })?,
        };
        if !catalog_record::valid_solver_id(&solver_id) {
            bail!("--record-solver-id must be S-<7 hex digits>[A-Za-z0-9+._-]*, got {solver_id}");
        }
        if EXPERIMENTAL.load(std::sync::atomic::Ordering::Relaxed) {
            solver_id.push_str(EXPERIMENTAL_TAG);
        }
        let produced_by = self
            .record_produced_by
            .clone()
            .filter(|label| !label.is_empty())
            .context("--emit-record needs --record-produced-by <label>")?;
        let route = self
            .record_route
            .clone()
            .or_else(|| default_route.map(str::to_string))
            .with_context(|| format!("{command} --emit-record needs --record-route"))?;
        if let Some(version) = self.record_reduce_version.as_deref() {
            if !catalog_record::valid_reduce_version(version) {
                bail!("--record-reduce-version must be D-v5/<pipeline>/<budget>, got {version}");
            }
        }
        let reduce_version = match (reduce_version, self.record_reduce_version.clone()) {
            (Some(_), Some(_)) => {
                bail!("{command} writes its own reduce version; drop --record-reduce-version")
            }
            (own, given) => own.or(given),
        };
        catalog_record::set_capture(true);
        Ok(Some((
            RecordFiles {
                records: path,
                unresolved: self.emit_unresolved.clone(),
            },
            RecordContext {
                solver_id,
                produced_by,
                route,
                generation: self.record_generation.unwrap_or(0),
                reduce_version,
                certify,
            },
        )))
    }
}

#[derive(Args, Clone, Copy)]
pub(crate) struct PositionParallel {
    /// The exact search's transposition table: `exact` (default) or
    /// `compact` (opt-in): 22 bytes per node
    // compact: owner instruction 2026-10-15.
    /// (128-bit key, id, verdict) and no search graph, the solution region
    /// rebuilt by a second pass; the output is identical.
    #[arg(long, default_value = "exact", value_parser = parse_table_mode)]
    pub(crate) table: TableMode,
    /// Threads per "mate within b?" question of the classical df-pn prober
    /// (the solver's prefilter, construction and `stats.nodesProof`): the
    /// prober asked and `P - 1` helper threads of its own run a parallel
    /// df-pn with virtual proof numbers over one lock-free shared
    /// transposition table.  `1` (default) is the one-thread prober.
    // Opt-in since 2026-10-16 (docs/experiments/par-dfpn/README.md); 1 is
    // byte-identical to before.
    /// With `P > 1` every resolved verdict (status, reasons, mate length,
    /// solution, main line) is the one-thread prober's; what may differ is
    /// the prober node counts (`stats.nodes` and `stats.nodesProof` count
    /// every thread's expansions) and whether a budget suffices; a search
    /// that runs out stays a budget-out (never re-run on one thread; the
    /// reducers climb `--reduce-budget-ladder` on the same threads).  The
    /// `--emit-record` records carry the tag `+pdfpn` after the solver id.
    /// `T / P` records run at a time (`--threads T`).
    #[arg(long, default_value_t = 1)]
    pub(crate) prober_threads: usize,
}

pub(crate) fn parse_table_mode(text: &str) -> std::result::Result<TableMode, String> {
    TableMode::parse(text).ok_or_else(|| format!("expected exact or compact, got {text}"))
}

impl PositionParallel {
    /// One record at a time on one thread (`validate`, which runs no search).
    pub(crate) const SINGLE: Self = Self {
        table: TableMode::Exact,
        prober_threads: 1,
    };

    /// The `--emit-record` records' solver id: `+pdfpn` appended with a parallel
    /// prober (counts and budget-outs may vary with the threads, verdicts do
    /// not), so the run stays visible in every record's `solveVersion`
    /// (`S-<commit>[tag]/<pipeline>/<budget>`).
    // `+par` (the position pool, removed 2026-10-27) and `+pbu` (fcd5851) are
    // no longer written; readers still accept them.
    pub(crate) fn tag_records(
        self,
        records: Option<(RecordFiles, RecordContext)>,
    ) -> Option<(RecordFiles, RecordContext)> {
        records.map(|(path, mut ctx)| {
            if self.prober_threads > 1 {
                ctx.solver_id.push_str(PDFPN_TAG);
            }
            (path, ctx)
        })
    }

    /// Refuses (exit 2) what these options cannot run: the leaf search off
    /// (`TSUME_CDFPN_LEAF=0`, the diff checks only) runs on one prober thread
    /// (1.0.0: the parallel loop always keeps its children's numbers).
    /// [`run`] calls it before a command opens any output (Codex review of
    /// 1.0.0 R1: refused later, `--emit-record` and the other side files were
    /// already truncated).
    pub(crate) fn refuse_unsupported(self) {
        if self.prober_threads > 1 && !tsume_solver::cdfpn::leaf_search() {
            eprintln!("tsume-solver: TSUME_CDFPN_LEAF=0 (tests only) runs on one prober thread; drop --prober-threads");
            std::process::exit(2);
        }
    }

    /// `reduce --jobs`: the given value (at least 1).  With
    /// `--prober-threads P > 1` it is 1 (each check's prober already runs
    /// on `P` threads; `K` workers would each hold a prober of `P` threads
    /// and its tables), and a larger `--jobs` is refused.
    pub(crate) fn reduce_jobs(self, jobs: usize) -> usize {
        if self.prober_threads > 1 {
            if jobs > 1 {
                eprintln!(
                    "tsume-solver: --jobs > 1 cannot be combined with --prober-threads > 1                      (every worker would run a prober of that many threads); use --jobs 1"
                );
                std::process::exit(2);
            }
            return 1;
        }
        jobs.max(1)
    }

    /// Threads one record holds: the prober's (`T / this` records run at a
    /// time).
    pub(crate) fn threads_per_record(self) -> usize {
        self.prober_threads.max(1)
    }

    /// Sets up `--prober-threads` for every prober built from now on
    /// ([`ClassicalProber::configured`]): called by [`PositionParallel::solver`]
    /// and by the `cdfpn` probe engine, which builds no solver.
    pub(crate) fn configure_prober(self) {
        tsume_solver::cdfpn::set_prober_threads(
            self.prober_threads,
            tsume_solver::cdfpn::PAR_TABLE_BITS,
        );
    }

    /// A solver with these options.
    pub(crate) fn solver(self) -> Solver {
        self.configure_prober();
        let mut solver = Solver::new();
        solver.set_table(self.table);
        solver
    }
}

/// The solver-id tag of a run with `--prober-threads` > 1: the prober node
/// counts (`stats.nodes` of a `nodesKind: "prober"` solution and
/// `stats.nodesProof`) count every thread's expansions and whether a budget
/// suffices may depend on the threads; verdicts never.
pub(crate) const PDFPN_TAG: &str = "+pdfpn";

#[derive(Args, Clone, Copy)]
pub(crate) struct Parallel {
    /// Worker threads.  Defaults to every logical core; 0 means the same thing.
    /// Output order never depends on this.
    #[arg(long, default_value_t = default_threads())]
    pub(crate) threads: usize,
}

/// Logical cores, the default worker count for every parallel command.
///
/// `available_parallelism` already honours the cgroup/affinity limits a container
/// or `start /affinity` imposes, so it is the right default rather than the raw
/// core count.  It falls back to 1 on the platforms that cannot report it.
pub(crate) fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |value| value.get())
}

/// Parses `MIN..MAX`, `MIN..=MAX` or a single number.
pub(crate) fn parse_ply_range(text: &str) -> Result<(u32, u32)> {
    let text = text.trim();
    let (low, high) = match text.split_once("..") {
        None => (text, text),
        Some((low, high)) => (low, high.trim_start_matches('=')),
    };
    let low: u32 = low.trim().parse().context("--plies expects integers")?;
    let high: u32 = if high.trim().is_empty() {
        low
    } else {
        high.trim().parse().context("--plies expects integers")?
    };
    if low > high {
        bail!("--plies MIN must not exceed MAX");
    }
    Ok((low, high))
}

// docs/experiments/solver-cli-audit/README.md §5.
/// The CLI surface snapshot:
/// every subcommand's every option -- name, value or flag, default, possible
/// values, hidden -- as one deterministic line each, compared with
/// `tests/fixtures/cli-surface.txt`.  Adding, removing or re-defaulting an
/// option fails this test until the fixture (and the audit's tables) follow:
/// `TSUME_UPDATE_CLI_SURFACE=1 cargo test --release --bin tsume-solver cli_surface`
/// rewrites it.
/// `--keep-squares`: [`tsume_solver::reduce::KeepSquares::parse`].
fn parse_keep_squares(
    spelled: &str,
) -> std::result::Result<tsume_solver::reduce::KeepSquares, String> {
    tsume_solver::reduce::KeepSquares::parse(spelled)
        .ok_or_else(|| format!("not a list of squares 0-87: {spelled:?}"))
}

#[cfg(test)]
mod cli_surface {
    use clap::CommandFactory;

    fn surface() -> String {
        let mut out = String::new();
        let mut command = super::Cli::command();
        command.build();
        for sub in command.get_subcommands() {
            let name = sub.get_name();
            if sub.is_hide_set() {
                out.push_str(&format!("{name} (hidden)\n"));
            } else {
                out.push_str(&format!("{name}\n"));
            }
            for arg in sub.get_arguments() {
                let id = arg.get_id().as_str();
                if id == "help" || id == "version" {
                    continue;
                }
                let label = match arg.get_long() {
                    Some(long) => format!("--{long}"),
                    None => format!("<{id}>"),
                };
                let takes_value = arg.get_action().takes_values() || arg.get_long().is_none();
                let mut line = format!("  {label} {}", if takes_value { "value" } else { "flag" });
                let defaults: Vec<String> = arg
                    .get_default_values()
                    .iter()
                    .map(|value| value.to_string_lossy().into_owned())
                    .collect();
                if !defaults.is_empty() && takes_value {
                    // The worker count follows the machine.
                    let shown = if id == "threads" {
                        "<logical cores>".to_string()
                    } else {
                        defaults.join(",")
                    };
                    line.push_str(&format!(" default={shown}"));
                }
                let values: Vec<String> = arg
                    .get_possible_values()
                    .iter()
                    .map(|value| value.get_name().to_string())
                    .collect();
                if !values.is_empty() && takes_value {
                    line.push_str(&format!(" values={}", values.join("|")));
                }
                if arg.is_required_set() {
                    line.push_str(" required");
                }
                if arg.is_hide_set() {
                    line.push_str(" hidden");
                }
                // Old spellings kept as hidden aliases (stage 4, 2026-10-22).
                if let Some(aliases) = arg.get_all_aliases() {
                    for alias in aliases {
                        line.push_str(&format!(" alias=--{alias}"));
                    }
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    }

    #[test]
    fn cli_surface_matches_the_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/cli-surface.txt"
        );
        let actual = surface();
        if std::env::var_os("TSUME_UPDATE_CLI_SURFACE").is_some() {
            std::fs::write(path, &actual).expect("write the fixture");
            return;
        }
        let expected = std::fs::read_to_string(path)
            .expect("tests/fixtures/cli-surface.txt (TSUME_UPDATE_CLI_SURFACE=1 writes it)")
            .replace("\r\n", "\n");
        assert!(
            actual == expected,
            "the CLI surface changed; update the CLI notes and \
             rewrite the fixture with TSUME_UPDATE_CLI_SURFACE=1:\n{actual}"
        );
    }
}
