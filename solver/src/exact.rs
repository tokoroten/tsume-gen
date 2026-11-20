//! Exact validation and graph pruning.
//!
//! Ports `exactValidate`, `completeDefenseCoverage`, `pruneSolverGraph` and its
//! helpers from `tools/puzzle-pipeline/src/solve.ts`.

use std::collections::{HashMap, HashSet};

use crate::inert::{analyze_inert_pieces_counted, InertReport, RULE_G_BUDGET_EXHAUSTED};
use crate::jsonl::InertPiece;
use crate::movegen::legal_moves;
use crate::piece::PT_K;
use crate::position::{Move, Position, Usi, NO_SQUARE};
use crate::solver::{
    Graph, GraphMove, Node, Outcome, SolveOptions, SolveResult, Solver, UselessInterposition,
};

/// The frozen orthodox composition rules A-C and G of `docs/rules.md`, evaluated
/// on top of the forced-mate gate.
///
/// All of them are *validation-level*: the solver and the move generator are
/// untouched by them.  Rule A is a property of the diagram, rules B and C are
/// properties of the **main-line region** `R` (see [`designated_main_line`];
/// owner decision 2026-09-19: the set of equally long defences, not one path), and
/// rule G is a property of the whole solution tree plus the defender's recorded
/// refutations (see [`crate::inert`]).
///
/// | rule | requirement | rejection reason |
/// | --- | --- | --- |
/// | A | the attacker's king is absent from the board | `attacker_king_present` |
/// | B | the root and every attack node of `R` at distance 3 or more have exactly one correct move (the final move is exempt) | `first_move_not_unique` at the root, `yozume_in_main_line` deeper |
/// | C | `Clean(root)`: the attacker mates with an empty hand against every main-line candidate reply | `koma_amari` |
/// | G | every board piece takes part in the solution ([`crate::inert`]) | `inert_attacker_piece`, `inert_defender_piece` (`rule_g_budget_exhausted` when clause (e) ran out of budget) |
///
/// Rule B applies at **every** mate length (`PIPELINE_VERSION` 0.12.0, owner
/// decision 2026-10-09).  From 0.7.0 to 0.11.0 it carried a length condition --
/// not enforced from 9 plies up (owner decision 2026-09-14) -- which is gone.  The
/// root code sits outside `composition_rules`, because it is part of the bare
/// contract.
///
/// `composition_rules` mirrors the `compositionRules` option of
/// `tools/puzzle-pipeline/src/solve.ts` and is **off by default**, so the bare
/// contract of [`exact_validate`] stays "is this an exact, minimum mate whose
/// first move is unique".  That is what `data/golden/puzzles.jsonl` is a
/// conformance set for: it deliberately contains koma-amari and yozume diagrams.
/// The publishing gates -- the reducer's acceptance test and
/// [`crate::stages::solve_stage`] -- turn the rules on; the reducer's *entry* check
/// leaves them off, because shedding an attacker king or spare hand material is
/// exactly the repair the reduction performs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TsumeRules {
    /// Enforce rules A, B and C (`compositionRules` in the TypeScript reference).
    pub composition_rules: bool,
    /// Rule A escape hatch: an attacker king may stay on the board (the shinshogi
    /// "double king" genre, `PipelineRecord.allowAttackerKing`).
    pub allow_attacker_king: bool,
    /// Rule G escape hatch, **Rust only**: accept a diagram that carries board
    /// pieces the solution never uses while still enforcing A, B and C.
    ///
    /// Same reason as [`TsumeRules::allow_koma_amari`]: there is no TypeScript
    /// counterpart and no parity stage uses it.  A position lifted out of a
    /// self-played game is *made* of pieces the mate does not need, and removing
    /// them is exactly what [`crate::distill`] and the reducer exist for, so a
    /// chain that had to satisfy rule G at every step could never take its first
    /// step.  Distillation therefore checks the relaxed rules while stepping and
    /// the full rules on every variant it records.  When this is set the analysis
    /// is skipped entirely, so [`ExactValidation::inert_pieces`] stays empty.
    pub allow_inert_pieces: bool,
    /// Rule C escape hatch, **Rust only**: accept a main line that leaves material
    /// in the attacker's hand while still enforcing A and B.
    ///
    /// There is no TypeScript counterpart and no parity stage uses it.  It exists
    /// for [`crate::distill`], whose operators drain the attacker's hand one piece
    /// at a time: a chain that must satisfy rule C at *every* step can never take
    /// its first step away from a two-spare-piece hand, so distillation checks the
    /// relaxed rules while stepping and the full rules on every variant it records.
    pub allow_koma_amari: bool,
    /// Rule B escape hatch, **Rust only**: accept a diagram whose main line has
    /// more than one correct move somewhere (`first_move_not_unique` at the root,
    /// `yozume_in_main_line` deeper) while still enforcing A, C and G.
    ///
    /// There is no TypeScript counterpart and no parity stage uses it.  It exists
    /// for the same reason as [`TsumeRules::allow_koma_amari`], and the measurement
    /// behind it is in `reports/distill-rule-c-2026-09-14.md` §2: **every one of the
    /// 15 eleven-ply mates this repository has mined carries yozume on its own main
    /// line, and 58 of the 64 nine-ply ones do.**  A dense mined diagram is *made*
    /// of alternative mating moves, and thinning it is exactly what removes them,
    /// so a chain that must satisfy rule B at every step can never take its first
    /// step -- the identical valley the relaxations above exist for.  Distillation
    /// therefore checks the relaxed rules while stepping and the full rules,
    /// rule B included, on every variant it records.
    ///
    /// Generation-internal only.  Since `PIPELINE_VERSION` 0.12.0 the published
    /// rule B has no length condition, and the `--strict-rule-b` tightening of
    /// P8-016 (owner decision 2026-10-04) that enforced it at every length for
    /// generation became the default and was removed; this stepping hatch is
    /// the one relaxation of rule B left.
    pub allow_yozume: bool,
    /// Rule F policy (`uselessInterposition` in `tools/puzzle-pipeline/src/solve.ts`).
    ///
    /// `None` rides with [`TsumeRules::composition_rules`] -- classical when they
    /// are on (`PIPELINE_VERSION` 0.10.0, owner decision 2026-10-05; prune before),
    /// keep when they are off -- which is what makes the published mate length,
    /// the designated main line and therefore rules B and C ignore 無駄合.  `Some`
    /// overrides it: the reducer's non-strict *entry* check passes the composition
    /// mode explicitly even though its composition rules are off, because the
    /// length it records as `targetMatePly` has to be measured the way its
    /// acceptance test measures it.
    pub useless_interposition: Option<UselessInterposition>,
}

impl TsumeRules {
    /// Composition rules off: the bare solver contract.
    pub const fn off() -> Self {
        Self {
            composition_rules: false,
            allow_attacker_king: false,
            allow_koma_amari: false,
            allow_inert_pieces: false,
            allow_yozume: false,
            useless_interposition: None,
        }
    }

    /// A publishable puzzle: A, B, C and G all enforced.
    pub const fn puzzle() -> Self {
        Self {
            composition_rules: true,
            allow_attacker_king: false,
            allow_koma_amari: false,
            allow_inert_pieces: false,
            allow_yozume: false,
            useless_interposition: None,
        }
    }

    /// An intermediate diagram of a distillation chain, or a raw mined candidate:
    /// A, C and G relaxed, B enforced.
    pub const fn candidate() -> Self {
        Self {
            composition_rules: true,
            allow_attacker_king: true,
            allow_koma_amari: true,
            allow_inert_pieces: true,
            // Rule B stays on here: `mine` uses this rule set for its candidate
            // gate and its `strictValid` flag, and relaxing B would change what
            // `mine` emits.  Distillation opts in with `allowing_yozume`.
            allow_yozume: false,
            useless_interposition: None,
        }
    }

    /// Rule A relaxed or enforced.
    pub const fn allowing_attacker_king(self, allow: bool) -> Self {
        Self {
            allow_attacker_king: allow,
            ..self
        }
    }

    /// Rule C relaxed or enforced.
    pub const fn allowing_koma_amari(self, allow: bool) -> Self {
        Self {
            allow_koma_amari: allow,
            ..self
        }
    }

    /// Rule G relaxed or enforced.
    pub const fn allowing_inert_pieces(self, allow: bool) -> Self {
        Self {
            allow_inert_pieces: allow,
            ..self
        }
    }

    /// Rule B relaxed or enforced.
    pub const fn allowing_yozume(self, allow: bool) -> Self {
        Self {
            allow_yozume: allow,
            ..self
        }
    }

    /// Rule F policy overridden, or left to ride with the composition rules.
    pub const fn allowing_useless_interposition(self, mode: Option<UselessInterposition>) -> Self {
        Self {
            useless_interposition: mode,
            ..self
        }
    }

    /// The rule F policy this rule set actually solves with.
    pub const fn useless_mode(self) -> UselessInterposition {
        match self.useless_interposition {
            Some(mode) => mode,
            None if self.composition_rules => UselessInterposition::Classical,
            None => UselessInterposition::Keep,
        }
    }

    /// A stable discriminator, for callers that memoise verdicts per rule set.
    pub const fn key(self) -> u8 {
        (self.composition_rules as u8)
            | ((self.allow_attacker_king as u8) << 1)
            | ((self.allow_koma_amari as u8) << 2)
            | ((self.allow_inert_pieces as u8) << 3)
            | (match self.useless_interposition {
                None => 0u8,
                Some(UselessInterposition::Keep) => 1,
                Some(UselessInterposition::Prune) => 2,
                Some(UselessInterposition::Classical) => 3,
            } << 4)
            | ((self.allow_yozume as u8) << 6)
    }
}

/// Whether a df-pn **disproof** at a bound ([`crate::dfpn`]), by the prober
/// [`dfpn_useless_mode`] picks, settles that `rules` find no mate within that
/// bound either.
///
/// True whenever `rules` read rule F as `Keep` or `Prune`, where rule F changes a
/// mate's length but never its existence at a bound.  Under
/// [`UselessInterposition::Classical`] (the composition reading) it is **false
/// for every prober** since `PIPELINE_VERSION` 0.11.0: the exact solver counts
/// the bound in rule-F plies (an F1 無駄合 and its capture consume none of it,
/// `docs/rules.md` rule F, "The ply bound counts rule-F plies"), while the df-pn
/// prober applies F2 but not that credit search, so a `Classical` prober's
/// disproof can still be a classical mate within the bound (stage 1 of
/// `docs/experiments/rule-f-bound/README.md` §4.4; under 0.10.0 it was true for a
/// `Classical` prober).  Callers that take a disproof as a verdict must hand it
/// to the exact solver when this is false.  The prober's own mode does not
/// enter (the parameter that carried it was dropped in the refactor of
/// 2026-10-08, `docs/experiments/rule-f-bound/REFACTOR.md`).
///
/// Proofs only carry over in one direction.  Existence at a bound is the same
/// under `Keep` and `Prune`, and a mate within a bound under either is a
/// `Classical` mate within it too, so a `Keep` prober's **proof** is sound for
/// every rule set.  A `Classical` prober's proof is sound only for `Classical`
/// rules: `8k/6G2/9/7N1/9/9/9/9/R8 b - 1` is proven at bound 1 under `Classical`
/// (every answer to 9i1i is a futile drop) and refuted at bound 1 under `Keep`
/// and `Prune`.  Use [`dfpn_useless_mode`] to pick the prober for a rule set.
///
/// This is about **existence** at a bound only: under `Prune` and `Classical`
/// the rule-F *distance* can be shorter than the smallest proving bound, so "proven
/// at `N`, disproven at `N - 2`" does not settle a distance of `N`.
pub fn dfpn_disproof_is_conclusive(rules: TsumeRules) -> bool {
    rules.useless_mode() != UselessInterposition::Classical
}

/// The rule-F mode a df-pn prober should run in to answer existence questions
/// for `rules`: `Classical` when `rules` read rule F classically (its proofs are
/// sound for them and it proves more than a `Keep` prober; its disproofs are not
/// conclusive since 0.11.0, [`dfpn_disproof_is_conclusive`]), `Keep` otherwise
/// (F1 changes no existence, so `Keep` *is* the `Prune` answer).
pub fn dfpn_useless_mode(rules: TsumeRules) -> UselessInterposition {
    match rules.useless_mode() {
        UselessInterposition::Classical => UselessInterposition::Classical,
        UselessInterposition::Keep | UselessInterposition::Prune => UselessInterposition::Keep,
    }
}

/// What the main-line analysis of a solved graph found ([`designated_main_line`]).
///
/// Since 2026-09-19 (owner decision) rules B and C are judged on the main-line
/// **region** `R` -- every node reachable from the root through every correct
/// move `C(n)` and every main-line candidate reply `P(m)` -- and the counters
/// below are counts over `R`.  Only [`MainLine::moves`] is a single path, and it
/// is for display: the verdicts never depend on it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MainLine {
    /// The displayed line, attacker and defender moves alternating: inside `R`,
    /// the `Clean` child first, then [`main_line_key`] (the smallest USI, a
    /// promotion before its non-promoting twin, `PIPELINE_VERSION` 0.17.0).
    pub moves: Vec<Usi>,
    /// Attack nodes of `R` *below the root* with mate distance 3 or more that have
    /// more than one correct move (rule B, `yozume_in_main_line`).  Distance-1
    /// nodes -- the final move -- are exempt and not counted.
    ///
    /// Any of them is rule B's defect at every mate length since
    /// `PIPELINE_VERSION` 0.12.0, unless [`TsumeRules::allow_yozume`] is set.
    pub yozume_nodes: usize,
    /// Pieces left in the attacker's hand at the terminal mate of the displayed
    /// line.  `> 0` exactly when rule C fails ([`MainLine::clean`] is `false`),
    /// because the displayed line of a refused diagram is the witness of the
    /// refusal.
    pub leftover_hand: usize,
    /// `Clean(root)`: rule C holds.
    pub clean: bool,
    /// Attack nodes of `R` -- the **root and the final moves included** -- with
    /// more than one correct move.  Nothing reads it to make a decision; it
    /// answers "how branched is the puzzle's main line", which is what a
    /// ply-threshold relaxation of rule B is priced against
    /// (`reports/distill-rule-c-2026-09-14.md` §4.4).
    pub branching_nodes: usize,
    /// The largest number of correct moves at any attack node of `R`, `0` when
    /// there is no attack node at all.
    pub max_correct: usize,
    /// `false` when the displayed line could not reach a terminal mate.
    pub complete: bool,
    /// Attack nodes of `R` -- the root included -- whose two correct moves are
    /// a **merged promotion twin** ([`is_merged_promotion_twin`], rule B,
    /// `PIPELINE_VERSION` 0.17.0): one correct move for rule B, the flag
    /// 「取られる成り駒の非限定」 (`capturedPromoAltNodes`).  Still counted in
    /// [`MainLine::branching_nodes`] and [`MainLine::max_correct`].
    pub captured_promo_alt_nodes: usize,
    /// The root itself is a merged promotion twin, so its two correct moves
    /// count as one for rule B's root test (`first_move_not_unique`); see
    /// [`ROOT_TWIN_MERGES`].
    pub root_merged_twin: bool,
}

/// Result of [`exact_validate`].
#[derive(Clone, Debug)]
pub struct ExactValidation {
    pub accepted: bool,
    pub reasons: Vec<String>,
    pub result: SolveResult,
    pub unique_first_move: bool,
    pub first_move: Option<Usi>,
    pub defense_coverage_verified: bool,
    /// Present when rules B or C were evaluated.
    pub main_line: Option<MainLine>,
    /// Rule G: the board pieces that take no part in the solution, attacker
    /// entries first.  Empty unless the composition rules were on and the position
    /// is a proven mate; the `solve` stage publishes it as
    /// `rejection.inertPieces`.
    pub inert_pieces: Vec<InertPiece>,
    /// Attack nodes of the main-line region `R` ([`MainLine::branching_nodes`]),
    /// the root and the final moves included, with more than one correct move.
    /// Since 2026-09-19 a puzzle rule B accepts can count final-move alternatives
    /// here.  Measured only when the composition rules were on and the position is
    /// a proven mate.
    pub attacker_branch_nodes: usize,
    /// The largest number of correct moves at one attack node of `R`: `1` for a
    /// single-solution puzzle, `0` when no main line was analysed.
    pub max_attacker_branching: usize,
    /// Search nodes spent *outside* [`ExactValidation::result`]: the extra solve the
    /// graph-free path needs for the designated main line.
    ///
    /// Kept separate on purpose, because `result.stats` is published puzzle metadata
    /// and must stay bit-identical to the TypeScript stage.  Callers that budget
    /// their own search (the distiller) add this to their node accounting.
    pub extra_nodes: u64,
    /// Search nodes of the minimality search (no mate one ply shorter, whose
    /// failure is `minimum_length_not_proven`), also outside
    /// [`ExactValidation::result`] and never published: the reducer's
    /// budget-out log counts them (`reduce --emit-budget-outs`).  0 when it
    /// did not run.
    pub shorter_nodes: u64,
}

/// Default node budget of `exactValidate`.
pub const DEFAULT_NODE_BUDGET: u64 = 500_000;
/// Default ply bound used by every pipeline stage.
pub const DEFAULT_MAX_PLY: u32 = 11;

/// Publishable-puzzle gate: forced mate, unchanged target ply, unique minimum
/// first move, minimum length proven by re-solving one ply shorter, and complete
/// defense coverage.
///
/// `build_graph` only affects what the returned [`SolveResult`] carries; the
/// verdict, reasons and stats are identical either way.
pub fn exact_validate(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    max_ply: u32,
    node_budget: u64,
    build_graph: bool,
) -> ExactValidation {
    exact_validate_with_rules(
        solver,
        position,
        target_mate_ply,
        max_ply,
        node_budget,
        build_graph,
        TsumeRules::off(),
    )
}

/// [`exact_validate`] with an explicit rule set.
///
/// Rules B and C need the designated main line, which needs a graph.  When the
/// caller did not ask for one, this runs **one** extra solve at the proven mate
/// distance to obtain it -- and only for a candidate that has already passed every
/// other gate, so the cheap inner loops of the reducer and the distiller pay for
/// the main line exactly on success and never on rejection.
#[allow(clippy::too_many_arguments)]
pub fn exact_validate_with_rules(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    max_ply: u32,
    node_budget: u64,
    build_graph: bool,
    rules: TsumeRules,
) -> ExactValidation {
    // Rule F rides with the composition rules unless the caller overrode it: the
    // publishable length is the one that ignores 無駄合.
    let useless = rules.useless_mode();
    let options = SolveOptions::default()
        .max_ply(max_ply)
        .node_budget(node_budget)
        .strict_material(true)
        .build_graph(build_graph)
        .useless_interposition(useless);
    let result = solver.solve(position, &options);
    let mut reasons: Vec<String> = Vec::new();
    if result.outcome != Outcome::Mate || result.distance.is_none() {
        reasons.push(
            if result.outcome == Outcome::Unknown {
                "solver_budget_exhausted"
            } else {
                "no_forced_mate_within_bound"
            }
            .to_string(),
        );
    }
    // `graphHasUnknown(graph)` is equivalent to `stats.budgetExhausted`: UNKNOWN
    // is only ever produced by the budget check, and the node that triggers it is
    // always recorded in the graph.  Checking the flag keeps the fast
    // graph-free path (used by the reducer) identical to the TypeScript result.
    if result.stats.budget_exhausted {
        reasons.push("incomplete_response_graph".to_string());
    }
    if let Some(graph) = result.graph.as_ref() {
        debug_assert_eq!(result.stats.budget_exhausted, graph_has_unknown(graph));
    }
    // An `UNKNOWN` root (the budget ran out before the root was settled) proves
    // nothing about the length, the first move or the coverage, so the gate
    // reports only the budget-out (`solver_budget_exhausted`,
    // `incomplete_response_graph`) and not the three reasons below, which used to
    // follow from the empty mate set alone (2026-10-15; `exactValidate` in
    // `tools/puzzle-pipeline/src/solve.ts` does the same).  A settled root of an
    // exhausted search (`REFUTED`, or `MATE` beside an exhausted credit oracle)
    // is judged as before.
    let unknown = result.outcome == Outcome::Unknown;
    if let Some(target) = target_mate_ply {
        if !unknown && result.distance != Some(target) {
            reasons.push("target_mate_ply_changed".to_string());
        }
    }
    let minimum_moves: Vec<&GraphMove> = match result.distance {
        None => Vec::new(),
        Some(distance) => result
            .moves
            .iter()
            .filter(|m| m.outcome() == Outcome::Mate && m.distance() == Some(distance))
            .collect(),
    };
    let unique_first_move = minimum_moves.len() == 1;
    // Rule B at the root, at every mate length since `PIPELINE_VERSION` 0.12.0
    // (owner decision 2026-10-09; 0.7.0-0.11.0 exempted 9 plies and longer).  It
    // sits outside `composition_rules`: the root code is part of the bare
    // contract.  The one way to skip it is `allow_yozume`, the Rust-only stepping
    // hatch distill-v2 opts into, which has no TypeScript counterpart and no
    // parity stage.
    let rule_b_skipped = rules.allow_yozume;
    // The graph-free path buys its solution tree with one extra search at the
    // same bound: here when the root's two correct moves are a promotion twin
    // (whether they merge reads the tree), else below for rules B, C and G.
    let mut extra: Option<SolveResult> = None;
    let mut extra_nodes = 0u64;
    let root_twin = ROOT_TWIN_MERGES
        && matches!(minimum_moves.as_slice(), [a, b] if is_promotion_twin(a.mv(), b.mv()));
    let mut root_merged = false;
    if root_twin && !rule_b_skipped && !unknown {
        if result.graph.is_none() {
            let solved = solver.solve(
                position,
                &SolveOptions::default()
                    .max_ply(max_ply)
                    .node_budget(node_budget)
                    .strict_material(true)
                    .build_graph(true)
                    .useless_interposition(useless),
            );
            extra_nodes = solved.stats.nodes;
            extra = Some(solved);
        }
        root_merged = match (result.graph.as_ref(), extra.as_ref()) {
            (Some(graph), _) => root_merged_twin(graph, result.attacker),
            (None, Some(solved)) if solved.outcome == Outcome::Mate => solved
                .graph
                .as_ref()
                .is_some_and(|graph| root_merged_twin(graph, solved.attacker)),
            _ => false,
        };
    }
    // Rule B at the root: one correct move, or a merged promotion twin (0.17.0).
    if !unique_first_move && !root_merged && !rule_b_skipped && !unknown {
        reasons.push("first_move_not_unique".to_string());
    }
    // The first move: the smallest USI, as before 0.17.0 -- except a merged
    // root twin, whose first move is the main line's, the promotion.
    let first_move = if root_merged {
        minimum_moves
            .iter()
            .map(|m| m.usi())
            .min_by_key(|&usi| main_line_key(usi))
    } else {
        minimum_moves.first().map(|m| m.usi())
    };
    let mut shorter_nodes = 0;
    if let Some(distance) = result.distance {
        if distance > 1 {
            // Minimality is proven against the same ply bound *and* the same
            // rule-F policy: a shorter mate whose own 無駄合 branches outrun this
            // bound is not provable here either.
            let shorter = solver.solve(
                position,
                &SolveOptions::default()
                    .max_ply(distance - 1)
                    .node_budget(node_budget)
                    .strict_material(true)
                    .build_graph(false)
                    .useless_interposition(useless),
            );
            shorter_nodes = shorter.stats.nodes;
            if shorter.outcome != Outcome::Refuted || shorter.stats.budget_exhausted {
                reasons.push("minimum_length_not_proven".to_string());
            }
        }
    }
    let defense_coverage_verified = result.outcome == Outcome::Mate
        && match &result.graph {
            Some(graph) => complete_defense_coverage(graph),
            // Without a graph the property still holds by construction: a defense
            // node can only be MATE after listing every legal reply (or with no
            // legal reply at all while in check), and only MATE nodes are visited.
            None => true,
        };
    if !defense_coverage_verified && !unknown {
        reasons.push("defense_coverage_incomplete".to_string());
    }
    // The composition rules are appended after the mate gate, in rule order, which
    // is the order `tools/puzzle-pipeline/src/solve.ts` uses: `rejection.reasons` is
    // published metadata, so the sequence is part of the contract.
    let mut main_line = None;
    let mut inert_pieces: Vec<InertPiece> = Vec::new();
    // An UNKNOWN root is a budget-out and nothing else (2026-10-15): not even
    // rule A, which would otherwise read as a judged defect.
    if rules.composition_rules && !unknown {
        // Rule A is a property of the diagram, so it costs nothing.
        if !rules.allow_attacker_king
            && result.root_position.king_square(result.attacker) != NO_SQUARE
        {
            reasons.push("attacker_king_present".to_string());
        }
        // Rules B, C and G read the solution tree, which only exists for a mate.
        if result.outcome == Outcome::Mate && result.distance.is_some() {
            // The graph-free fast path has no tree to read, so buy one with a single
            // extra search.  It runs at the *same* `max_ply` as the search that
            // already happened, so rules B and C read the graph the TypeScript
            // gate (which always has one) reads; rule G reads its own tree at the
            // proven distance (below).
            if result.graph.is_none() && extra.is_none() {
                let solved = solver.solve(
                    position,
                    &SolveOptions::default()
                        .max_ply(max_ply)
                        .node_budget(node_budget)
                        .strict_material(true)
                        .build_graph(true)
                        .useless_interposition(useless),
                );
                extra_nodes += solved.stats.nodes;
                extra = Some(solved);
            }
            let solution_tree = match (result.graph.as_ref(), extra.as_ref()) {
                (Some(graph), _) => Some((graph, result.attacker)),
                (None, Some(extra)) if extra.outcome == Outcome::Mate => {
                    extra.graph.as_ref().map(|graph| (graph, extra.attacker))
                }
                _ => None,
            };
            // Rule G reads the solution tree of the proven mate length, never
            // the caller's bound (`PIPELINE_VERSION` 0.16.0, owner decision
            // 2026-10-30): when the gate searched wider, the tree is built again
            // at the proven distance.  `None` when that search ran out of budget.
            let rule_g_extra = match (solution_tree, result.distance) {
                (Some(_), Some(distance)) if !rules.allow_inert_pieces && max_ply > distance => {
                    let extra = solver.solve(
                        position,
                        &SolveOptions::default()
                            .max_ply(distance)
                            .node_budget(node_budget)
                            .strict_material(true)
                            .build_graph(true)
                            .useless_interposition(useless),
                    );
                    extra_nodes += extra.stats.nodes;
                    Some(extra)
                }
                _ => None,
            };
            let measured = solution_tree.map(|(graph, attacker)| {
                (
                    designated_main_line(graph, attacker),
                    if rules.allow_inert_pieces {
                        (InertReport::default(), 0)
                    } else {
                        match rule_g_extra.as_ref() {
                            None => analyze_inert_pieces_counted(graph, attacker, node_budget),
                            Some(extra) => match extra.graph.as_ref() {
                                Some(graph)
                                    if extra.outcome == Outcome::Mate
                                        && !extra.stats.budget_exhausted =>
                                {
                                    analyze_inert_pieces_counted(graph, attacker, node_budget)
                                }
                                _ => (
                                    InertReport::all_undecided(&result.root_position, attacker),
                                    0,
                                ),
                            },
                        }
                    },
                )
            });
            match measured {
                Some((line, (inert, inert_nodes))) => {
                    extra_nodes += inert_nodes;
                    // Rule B deeper than the root.
                    if line.yozume_nodes > 0 && !rule_b_skipped {
                        reasons.push("yozume_in_main_line".to_string());
                    }
                    // Rule C is `Clean(root)` over the main-line region: the
                    // attacker must be able to mate without spare material against
                    // every main-line candidate reply (2026-09-19).
                    if !rules.allow_koma_amari && !line.clean {
                        reasons.push("koma_amari".to_string());
                    }
                    // Rule G: a board piece the solution never uses (無駄駒).  Both
                    // sides are reported, so one rejection names every square.
                    if !inert.attacker.is_empty() {
                        reasons.push("inert_attacker_piece".to_string());
                    }
                    if !inert.defender.is_empty() {
                        reasons.push("inert_defender_piece".to_string());
                    }
                    if !inert.undecided.is_empty() {
                        reasons.push(RULE_G_BUDGET_EXHAUSTED.to_string());
                    }
                    inert_pieces = inert.all();
                    main_line = Some(line);
                }
                // Only reachable on the graph-free path, when the extra search ran
                // out of budget.  Rules B and C are then *undecided*, and this gate
                // fails safe rather than open: silently accepting would publish a
                // yozume or koma-amari diagram.  The TypeScript reference cannot
                // reach this branch at all, because `solveTsume` always returns a
                // graph, so the publishing gate (`solve_stage`, which always asks
                // for one) stays bit-identical to it.
                None => reasons.push("incomplete_response_graph".to_string()),
            }
        }
    }
    let accepted = reasons.is_empty();
    let mut seen = HashSet::new();
    reasons.retain(|reason| seen.insert(reason.clone()));
    let attacker_branch_nodes = main_line.as_ref().map_or(0, |line| line.branching_nodes);
    let max_attacker_branching = main_line.as_ref().map_or(0, |line| line.max_correct);
    ExactValidation {
        accepted,
        reasons,
        result,
        unique_first_move,
        first_move,
        defense_coverage_verified,
        main_line,
        inert_pieces,
        attacker_branch_nodes,
        max_attacker_branching,
        extra_nodes,
        shorter_nodes,
    }
}

/// The correct moves `C(n)` of an attack node: `MATE` at exactly the node's own
/// distance, so a longer mate is never "correct" and therefore never yozume.
pub fn correct_moves(node: Node<'_>) -> Vec<&GraphMove> {
    node.moves()
        .iter()
        .filter(|m| m.outcome() == Outcome::Mate && m.distance() == node.distance())
        .collect()
}

/// The longest replies `L(m)` of a defence node: every reply rule F did not flag
/// 無駄合 whose mate distance is the largest among them.  More than one when the
/// defender has several equally long defences (変同).
pub fn longest_replies(node: Node<'_>) -> Vec<&GraphMove> {
    let longest = node
        .moves()
        .iter()
        .filter(|m| !m.useless())
        .map(|m| m.distance().map_or(-1i64, |d| d as i64))
        .max();
    let Some(longest) = longest else {
        return Vec::new();
    };
    node.moves()
        .iter()
        .filter(|m| !m.useless() && m.distance().map_or(-1i64, |d| d as i64) == longest)
        .collect()
}

/// A defence node the main line ends on: no reply at all, or -- only under
/// [`UselessInterposition::Classical`], the composition reading -- only replies flagged
/// 無駄合 (a check answered by futile interposition drops alone is mate).  Under
/// `Keep` and `Prune` a defence node with replies always keeps at least one
/// unflagged reply, so this is exactly `moves.is_empty()` there.
pub fn is_mated_leaf(node: Node<'_>) -> bool {
    !node.attack() && node.moves().iter().all(|m| m.useless())
}

/// Pieces the attacker still holds at a position.
fn hand_count(position: &Position, attacker: u8) -> usize {
    position.hands[attacker as usize]
        .iter()
        .map(|&count| count as usize)
        .sum()
}

/// `Amari` and `Clean` of every node of the shortest solution graph
/// (`docs/rules.md`, "Designated main line", owner decision 2026-09-19).
///
/// * `Amari(x)`: the attacker can mate at the minimum length **and** keep
///   material in hand.  Terminal: `left > 0`; attack node: some child of `C(n)`;
///   defence node: every child of `L(m)`.
/// * `P(m)`: the replies of `L(m)` whose child is not `Amari`, or all of `L(m)`
///   when every one is (変同駒余り is not the main line).
/// * `Clean(x)`: rule C.  Terminal: `left == 0`; attack node: some child of
///   `C(n)`; defence node: every child of `P(m)`.
///
/// Both are memoised per node id; the mate distance strictly decreases along
/// every edge read here, so the recursion is bounded by the mate length.
pub struct SolutionFacts<'g> {
    graph: &'g Graph,
    attacker: u8,
    amari: HashMap<u32, bool>,
    clean: HashMap<u32, bool>,
}

impl<'g> SolutionFacts<'g> {
    pub fn new(graph: &'g Graph, attacker: u8) -> Self {
        Self {
            graph,
            attacker,
            amari: HashMap::new(),
            clean: HashMap::new(),
        }
    }

    /// `Amari(id)`.  A missing node is no evidence of spare material.
    pub fn amari(&mut self, id: u32) -> bool {
        if let Some(&known) = self.amari.get(&id) {
            return known;
        }
        let graph = self.graph;
        let value = match graph.node(id) {
            None => false,
            Some(node) if node.attack() => {
                let children: Vec<Option<u32>> =
                    correct_moves(node).iter().map(|m| m.next()).collect();
                children
                    .into_iter()
                    .any(|next| next.is_some_and(|next| self.amari(next)))
            }
            Some(node) if is_mated_leaf(node) => hand_count(&node.position(), self.attacker) > 0,
            Some(node) => {
                let children: Vec<Option<u32>> =
                    longest_replies(node).iter().map(|m| m.next()).collect();
                children
                    .into_iter()
                    .all(|next| next.is_some_and(|next| self.amari(next)))
            }
        };
        self.amari.insert(id, value);
        value
    }

    /// `P(m)` of a defence node, in recorded (USI) order.
    pub fn candidate_replies(&mut self, node: Node<'g>) -> Vec<&'g GraphMove> {
        let longest = longest_replies(node);
        let clean: Vec<&GraphMove> = longest
            .iter()
            .copied()
            .filter(|m| !m.next().is_some_and(|next| self.amari(next)))
            .collect();
        if clean.is_empty() {
            longest
        } else {
            clean
        }
    }

    /// `Clean(id)`, rule C.  A missing node is no evidence of a defect, which is
    /// what the pre-2026-09-19 walk did when it could not reach a terminal.
    pub fn clean(&mut self, id: u32) -> bool {
        if let Some(&known) = self.clean.get(&id) {
            return known;
        }
        let graph = self.graph;
        let value = match graph.node(id) {
            None => true,
            Some(node) if node.attack() => {
                let children: Vec<Option<u32>> =
                    correct_moves(node).iter().map(|m| m.next()).collect();
                children
                    .into_iter()
                    .any(|next| next.is_none_or(|next| self.clean(next)))
            }
            Some(node) if is_mated_leaf(node) => hand_count(&node.position(), self.attacker) == 0,
            Some(node) => {
                let children: Vec<Option<u32>> = self
                    .candidate_replies(node)
                    .iter()
                    .map(|m| m.next())
                    .collect();
                children
                    .into_iter()
                    .all(|next| next.is_none_or(|next| self.clean(next)))
            }
        };
        self.clean.insert(id, value);
        value
    }

    /// The edges that span the main-line region `R` out of `node`: `C(n)` at an
    /// attack node, `P(m)` at a defence node.
    pub fn region_moves(&mut self, node: Node<'g>) -> Vec<&'g GraphMove> {
        if node.attack() {
            correct_moves(node)
        } else {
            self.candidate_replies(node)
        }
    }
}

/// The displayed main line's tie-break key (`PIPELINE_VERSION` 0.17.0, owner
/// decision 2026-11-16): the USI with its `+` taken off, then `0` for a
/// promotion and `1` for a move that does not promote.
///
/// Every USI move has four bytes before the optional `+`, so only a move and
/// its promotion twin (the same from- and to-squares) trade places against the
/// plain USI order: `5c4d+` < `5c4d` < `5c4e`.  The final move therefore shows
/// the promotion whenever both mate,
/// and so does a merged promotion twin ([`is_merged_promotion_twin`]).
/// Mirrors `mainLineKey` in `tools/puzzle-pipeline/src/solve.ts`.
pub fn main_line_key(usi: Usi) -> ([u8; 4], u8) {
    let bytes = usi.as_bytes();
    let mut head = [0u8; 4];
    head.copy_from_slice(&bytes[..4]);
    (head, if bytes.len() > 4 { 0 } else { 1 })
}

/// Whether two moves are a promotion twin: the same from- and to-squares on
/// the board, one promoting and one not.
pub fn is_promotion_twin(a: Move, b: Move) -> bool {
    !a.is_drop() && !b.is_drop() && a.from == b.from && a.to == b.to && a.promote != b.promote
}

/// Rule B's **merged promotion twin** (`docs/rules.md`, rule B, `PIPELINE_VERSION`
/// 0.17.0, owner decision 2026-11-16): the attack node `node` has exactly two
/// correct moves, a board move `m` and its promotion twin `m+`, and they merge
/// -- the main-line candidate replies `P(m)` and `P(m+)` are the same non-empty
/// set of USI moves, and every one of them captures the moved piece on its
/// destination square (a board move to that square).  A capture puts the same
/// unpromoted piece in the defender's hand from either twin, so every reply
/// leads to the same position: rule B counts the node as one correct move, and
/// the catalogue flags it 「取られる成り駒の非限定」.
///
/// Only the **immediate** reply counts: a twin the
/// defender captures one move later (the king steps away first) stays yozume.
/// A node whose moves have no recorded child does not merge.
pub fn is_merged_promotion_twin<'g>(facts: &mut SolutionFacts<'g>, node: Node<'g>) -> bool {
    if !node.attack() {
        return false;
    }
    let correct = correct_moves(node);
    let [a, b] = correct.as_slice() else {
        return false;
    };
    let (ma, mb) = (a.mv(), b.mv());
    if !is_promotion_twin(ma, mb) {
        return false;
    }
    let graph = facts.graph;
    let replies = |facts: &mut SolutionFacts<'g>, m: &GraphMove| -> Option<Vec<Usi>> {
        let child = graph.node(m.next()?)?;
        let mut usi: Vec<Usi> = Vec::new();
        for reply in facts.candidate_replies(child) {
            let mv = reply.mv();
            if mv.is_drop() || mv.to != ma.to {
                return None;
            }
            usi.push(reply.usi());
        }
        usi.sort_unstable();
        Some(usi)
    };
    let (Some(pa), Some(pb)) = (replies(facts, a), replies(facts, b)) else {
        return false;
    };
    !pa.is_empty() && pa == pb
}

/// Whether rule B's root test merges a promotion twin as it merges one deeper
/// (owner decision 2026-11-16: the root is included).  The root case is kept
/// behind this one switch so it can be turned off on its own; with it off, a
/// root twin is `first_move_not_unique` as before 0.17.0.
pub const ROOT_TWIN_MERGES: bool = true;

/// Whether the root of `graph` is a merged promotion twin that rule B's root
/// test counts as one correct move ([`ROOT_TWIN_MERGES`],
/// [`is_merged_promotion_twin`]).
pub fn root_merged_twin(graph: &Graph, attacker: u8) -> bool {
    if !ROOT_TWIN_MERGES {
        return false;
    }
    let Some(root) = graph.node(graph.root) else {
        return false;
    };
    let mut facts = SolutionFacts::new(graph, attacker);
    is_merged_promotion_twin(&mut facts, root)
}

/// What the main-line analysis of a solved graph found (`docs/rules.md`,
/// "Designated main line", owner decision 2026-09-19).
///
/// Rules B and C are judged on the **set** `R` of positions reachable from the
/// root through every correct move `C(n)` and every main-line candidate reply
/// `P(m)`, never on one path, so the verdict does not depend on the board
/// orientation.  The USI order only picks the single line [`MainLine::moves`]
/// that is displayed.
pub fn designated_main_line(graph: &Graph, attacker: u8) -> MainLine {
    let mut facts = SolutionFacts::new(graph, attacker);
    let mut out = MainLine {
        clean: facts.clean(graph.root),
        ..MainLine::default()
    };
    if graph.node(graph.root).is_none() {
        return out;
    }

    // 1. Counters over the region R (distinct nodes; transpositions count once).
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = vec![graph.root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = graph.node(id) else {
            continue;
        };
        let edges = facts.region_moves(node);
        if node.attack() {
            let correct = edges.len();
            if correct > 1 {
                out.branching_nodes += 1;
                let merged = is_merged_promotion_twin(&mut facts, node);
                if merged && (id != graph.root || ROOT_TWIN_MERGES) {
                    out.captured_promo_alt_nodes += 1;
                    if id == graph.root {
                        out.root_merged_twin = true;
                    }
                } else if id != graph.root && node.distance().is_some_and(|d| d >= 3) {
                    // Rule B below the root: the final move (distance 1) may
                    // have several mates; any other attack node of R may not,
                    // unless its two are a merged promotion twin (0.17.0).
                    out.yozume_nodes += 1;
                }
            }
            out.max_correct = out.max_correct.max(correct);
        }
        for m in edges {
            if let Some(next) = m.next() {
                stack.push(next);
            }
        }
    }

    // 2. The displayed line: inside R, at every node the child whose `Clean`
    //    equals the node's own, ties on [`main_line_key`] (the smallest USI,
    //    a promotion before its non-promoting twin).  On a puzzle rule C
    //    accepts this is "prefer the Clean child"; on one it refuses it is the
    //    witness of the refusal, so `leftover_hand > 0` exactly when rule C fails.
    let mut walked: HashSet<u32> = HashSet::new();
    let mut current = graph.root;
    loop {
        if !walked.insert(current) {
            return out;
        }
        let Some(node) = graph.node(current) else {
            return out;
        };
        if is_mated_leaf(node) {
            // Checkmate: the attacker's hand is whatever it still holds here.
            out.leftover_hand = hand_count(&node.position(), attacker);
            out.complete = true;
            return out;
        }
        let want = facts.clean(current);
        let edges = facts.region_moves(node);
        let chosen = edges
            .iter()
            .copied()
            .filter(|m| m.next().is_none_or(|next| facts.clean(next) == want))
            .min_by_key(|m| main_line_key(m.usi()));
        let Some(chosen) = chosen else {
            return out;
        };
        out.moves.push(chosen.usi());
        match chosen.next() {
            Some(next) => current = next,
            None => return out,
        }
    }
}

/// Rule D v5's path cap (`PIPELINE_VERSION` 0.13.0, `docs/rules.md`): a main-line
/// region with more root-to-terminal paths than this is not enumerated, and a
/// main-line-preserving step to such a diagram is refused.
pub const MAIN_LINE_PATHS_CAP: usize = 512;

/// `Π(graph)` of rule D v5 (`docs/rules.md`, "Reduction v5"): the USI sequences
/// of every path of the main-line region `R` from the root to its end -- every
/// correct move `C(n)` at an attack node, every main-line candidate reply `P(m)`
/// at a defence node (so rule F's useless replies never appear) -- sorted and
/// without repeats.  `None` when there are more than `cap` of them.
///
/// A path ends at a node rule F mates ([`is_mated_leaf`]), at a node the region
/// leaves no edge out of, at a move without a recorded child, or at a node id
/// the graph does not hold.  The set does not depend on the order the walk takes
/// the edges in, and neither does the cap, which counts every path.  Mirrors
/// `mainLinePaths` in `tools/puzzle-pipeline/src/solve.ts`.
pub fn main_line_paths(graph: &Graph, attacker: u8, cap: usize) -> Option<Vec<Vec<Usi>>> {
    fn walk<'g>(
        facts: &mut SolutionFacts<'g>,
        graph: &'g Graph,
        id: u32,
        prefix: &mut Vec<Usi>,
        out: &mut Vec<Vec<Usi>>,
        cap: usize,
    ) -> bool {
        let edges = match graph.node(id) {
            Some(node) if !is_mated_leaf(node) => facts.region_moves(node),
            _ => Vec::new(),
        };
        if edges.is_empty() {
            out.push(prefix.clone());
            return out.len() <= cap;
        }
        for m in edges {
            prefix.push(m.usi());
            let fits = match m.next() {
                Some(next) => walk(facts, graph, next, prefix, out, cap),
                None => {
                    out.push(prefix.clone());
                    out.len() <= cap
                }
            };
            prefix.pop();
            if !fits {
                return false;
            }
        }
        true
    }
    let mut facts = SolutionFacts::new(graph, attacker);
    let mut out: Vec<Vec<Usi>> = Vec::new();
    let mut prefix: Vec<Usi> = Vec::new();
    if !walk(&mut facts, graph, graph.root, &mut prefix, &mut out, cap) {
        return None;
    }
    out.sort();
    out.dedup();
    Some(out)
}

/// `true` when some attack node of the main-line region `R` at mate distance 1
/// has two or more correct moves: a final-move alternative anywhere in `R`, not
/// only on the displayed line (the one kind of branching rule B allows below the
/// root, see [`designated_main_line`]).  Walks `R` exactly as
/// [`main_line_paths`] does: [`SolutionFacts::region_moves`], stopping at
/// [`is_mated_leaf`], each node once.
pub fn has_final_move_alternative(graph: &Graph, attacker: u8) -> bool {
    let mut facts = SolutionFacts::new(graph, attacker);
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = vec![graph.root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = graph.node(id) else {
            continue;
        };
        if is_mated_leaf(node) {
            continue;
        }
        let edges = facts.region_moves(node);
        if node.attack() && node.distance() == Some(1) && edges.len() >= 2 {
            return true;
        }
        stack.extend(edges.iter().filter_map(|m| m.next()));
    }
    false
}

/// The pre-2026-09-19 single line: at every attack node the correct move with the
/// smallest USI, at every defence node [`longest_resistance`].
///
/// It is **not** the designated main line any more and decides nothing.  It is
/// kept for the measurement-mode `probe --engine exact-probe`, whose `pv` must be
/// the line the df-pn prober extracts ([`crate::dfpn`]): the prober has no graph
/// and no hands to judge `Amari`, so both engines keep reporting this
/// orientation-dependent but cheap shortest line.
pub fn usi_first_line(graph: &Graph) -> Vec<Usi> {
    let mut out = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut current = graph.root;
    loop {
        if !seen.insert(current) {
            return out;
        }
        let Some(node) = graph.node(current) else {
            return out;
        };
        let chosen = if node.attack() {
            correct_moves(node).into_iter().min_by_key(|m| m.usi())
        } else {
            longest_resistance(node.moves())
        };
        let Some(chosen) = chosen else {
            return out;
        };
        out.push(chosen.usi());
        match chosen.next() {
            Some(next) => current = next,
            None => return out,
        }
    }
}

/// The attacker king's square, when the attacker still has one on the board.
pub fn attacker_king_square(position: &Position, attacker: u8) -> Option<usize> {
    let square = position.king_square(attacker);
    if square == NO_SQUARE {
        None
    } else {
        Some(square as usize)
    }
}

/// Removes the attacker's king from the board (the rule D pre-pass).
///
/// A king never enters a hand, so the strict remainder is unaffected and the
/// defender's hand does not change.  `None` when there is no attacker king.
pub fn without_attacker_king(position: &Position, attacker: u8) -> Option<Position> {
    let square = attacker_king_square(position, attacker)?;
    debug_assert_eq!(crate::piece::ptype(position.board[square]), PT_K);
    let mut out = position.clone();
    out.board[square] = crate::piece::EMPTY;
    out.refresh();
    crate::strict_material::apply(&mut out, attacker);
    Some(out)
}

/// Finds the minimum mate length by iterative deepening (spec 11.2), returning
/// the bound that was searched last together with its validation.
///
/// A mate distance measured from an attacker-to-move root is always odd -- the
/// mate falls on an attacker move -- so only odd bounds are tried.  The first
/// bound that reports `MATE` reports the *minimum* length, because every shorter
/// bound has already been refuted.
///
/// Compared with one search at `max_ply` this is dramatically cheaper whenever a
/// short mate exists, which is the normal case for mining and distillation: at a
/// bound of 11 the solver explores every wrong first check to depth 11 even when
/// the answer is a mate in 1.  For positions with no mate the cost is the same
/// search at `max_ply` plus the cheap shorter bounds.  The verdict is identical
/// either way: [`exact_validate`] at the returned bound already proves minimum
/// length by re-solving one ply shorter.
pub fn minimum_mate_validate(
    solver: &mut Solver,
    position: &Position,
    max_ply: u32,
    node_budget: u64,
    build_graph: bool,
) -> (u32, ExactValidation) {
    minimum_mate_validate_with_rules(
        solver,
        position,
        max_ply,
        node_budget,
        build_graph,
        TsumeRules::off(),
    )
}

/// [`minimum_mate_validate`] with an explicit rule set.
pub fn minimum_mate_validate_with_rules(
    solver: &mut Solver,
    position: &Position,
    max_ply: u32,
    node_budget: u64,
    build_graph: bool,
    rules: TsumeRules,
) -> (u32, ExactValidation) {
    let mut bound = 1;
    loop {
        let validation = exact_validate_with_rules(
            solver,
            position,
            None,
            bound,
            node_budget,
            build_graph,
            rules,
        );
        let resolved = validation.result.outcome == Outcome::Mate
            || validation.result.stats.budget_exhausted
            || bound + 2 > max_ply;
        if resolved {
            return (bound, validation);
        }
        bound += 2;
    }
}

/// Whether a validated diagram carries **yozume on its main line** in the sense of
/// rule B, regardless of its length: the root has more than one correct move, or
/// an attack node of the main-line region `R` at distance 3 or more does (the
/// final move is exempt).  This is the `has_yozume` fact of `docs/TODO.md` P8-015.
///
/// `None` when it cannot be read off `validation`: no proven mate, or rules B and
/// C were not evaluated (composition rules off, or the main-line solve ran out of
/// budget).
pub fn validation_has_yozume(validation: &ExactValidation) -> Option<bool> {
    if validation.result.outcome != Outcome::Mate || validation.result.distance.is_none() {
        return None;
    }
    let line = validation.main_line.as_ref()?;
    Some((!validation.unique_first_move && !line.root_merged_twin) || line.yozume_nodes > 0)
}

/// Measures [`validation_has_yozume`] for `position` (Rust only, P8-016): one
/// validation under [`TsumeRules::candidate`] with rules A, C, G relaxed and
/// rule B's hatch open, so only the measurement matters, at the caller's bound
/// and budget -- pass the ones the caller's own gate uses so the main-line region
/// is the one that gate would read.  Returns the verdict and the search nodes it
/// cost.
///
/// Like the publishing gate ([`crate::stages::solve_stage_reported`]), the
/// measurement is first made by solver v3's construction
/// ([`crate::reduce::constructed_validation`], on whenever the gate's is,
/// `TSUME_SOLVE_CONSTRUCT`).  When the prober or the construction runs out of
/// budget the measurement is `None` (`parent_yozume_unknown`): since solver
/// 1.4.0 (owner 2026-11-05) the full-width validation is not run again on the
/// same budget; it decides only with the construction off.  The contract is
/// the gate's: the verdict is the full-width one wherever both searches stay
/// within their budget; where one runs out, the answer may be `None` on one
/// side and decided on the other (`docs/experiments/solver-v3/README.md` §3.4,
/// `docs/experiments/extend-speedup/README.md` §3.7).  The node count is what
/// the construction's validation reports when it decided (the classical df-pn
/// prober's nodes; 0 when the prober refuted a mate within the bound, whose cost
/// that validation does not carry); callers report it only
/// as run telemetry (`extend --stats-out` `parentYozumeNodes`, beside the wall
/// and CPU seconds), never in a record.
pub fn measure_main_line_yozume(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    max_ply: u32,
    node_budget: u64,
) -> (Option<bool>, u64) {
    let rules = TsumeRules::candidate()
        .allowing_attacker_king(true)
        .allowing_yozume(true);
    if crate::stages::solve_construct_on() {
        if let Some(validation) = crate::reduce::constructed_validation(
            position,
            target_mate_ply,
            max_ply,
            node_budget,
            rules,
        ) {
            let nodes = validation.result.stats.nodes + validation.extra_nodes;
            return (validation_has_yozume(&validation), nodes);
        }
    }
    // Counted with the gates' full-width searches (`tests/fast_budget_out.rs`).
    crate::reduce::count_full_width_gate();
    let validation = exact_validate_with_rules(
        solver,
        position,
        target_mate_ply,
        max_ply,
        node_budget,
        false,
        rules,
    );
    let nodes = validation.result.stats.nodes + validation.extra_nodes;
    (validation_has_yozume(&validation), nodes)
}

/// `true` when any node or edge of the graph is unresolved.
///
/// On a compact table's sparse graph this is the whole graph's answer, which
/// its search recorded ([`Graph::compact_facts`]).
pub fn graph_has_unknown(graph: &Graph) -> bool {
    if let Some(facts) = graph.compact_facts() {
        return facts.has_unknown;
    }
    graph.nodes().any(|node| {
        node.outcome() == Outcome::Unknown
            || node.moves().iter().any(|m| m.outcome() == Outcome::Unknown)
    })
}

/// Every mating defense node in the solution tree lists every legal reply.
///
/// Rule F keeps its 無駄合 replies in the search graph (flagged, with their
/// subtree) precisely so this check still sees them: a flagged reply counts as
/// covered, and coverage keeps descending through it.
///
/// A compact table's sparse graph ([`Graph::is_sparse`]) holds the solution
/// region only; the property holds by construction there, exactly as on the
/// graph-free path of [`exact_validate_with_rules`]: a defence node is `MATE`
/// only after listing every legal reply.
pub fn complete_defense_coverage(graph: &Graph) -> bool {
    if graph.is_sparse() {
        return true;
    }
    let mut seen = HashSet::new();
    visit_coverage(graph, graph.root, &mut seen)
}

fn visit_coverage(graph: &Graph, id: u32, seen: &mut HashSet<u32>) -> bool {
    if !seen.insert(id) {
        return true;
    }
    let Some(node) = graph.node(id) else {
        return false;
    };
    if !node.attack() && node.outcome() == Outcome::Mate {
        let mut expected: Vec<Usi> = legal_moves(&node.position())
            .iter()
            .map(|m| m.to_usi())
            .collect();
        expected.sort_unstable();
        let mut recorded: Vec<Usi> = node.moves().iter().map(|m| m.usi()).collect();
        recorded.sort_unstable();
        if expected != recorded {
            return false;
        }
    }
    for m in node.moves().iter() {
        if m.outcome() == Outcome::Mate {
            if let Some(next) = m.next() {
                if !visit_coverage(graph, next, seen) {
                    return false;
                }
            }
        }
    }
    true
}

/// Number of graph nodes reachable from `start`.
///
/// On a compact table's sparse graph ([`Graph::is_sparse`]) this counts the
/// nodes the graph holds; the whole graph's counts are in
/// [`Graph::compact_facts`] (`features::extract_with` reads them).
pub fn reachable_node_count(graph: &Graph, start: Option<u32>) -> usize {
    let Some(start) = start else {
        return 0;
    };
    if graph.is_sparse() {
        // Search ids are not storage indices here: a set, not a bitmap.
        let mut seen: HashSet<u32> = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            let Some(node) = graph.node(id) else {
                continue;
            };
            if !seen.insert(id) {
                continue;
            }
            stack.extend(node.moves().iter().filter_map(GraphMove::next));
        }
        return seen.len();
    }
    // A bit per node instead of a hash set: the same count, and the difficulty
    // features call this once per wrong root move over graphs of millions of
    // nodes.
    let mut seen = vec![0u64; graph.len().div_ceil(64)];
    let mut count = 0usize;
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        let Some(node) = graph.node(id) else {
            continue;
        };
        let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
        if seen[word] & bit != 0 {
            continue;
        }
        seen[word] |= bit;
        count += 1;
        for m in node.moves().iter() {
            if let Some(next) = m.next() {
                stack.push(next);
            }
        }
    }
    count
}

/// A node of the pruned, runtime-shaped graph.
#[derive(Clone, Debug)]
pub struct PrunedNode {
    pub id: u32,
    pub attack: bool,
    pub sfen: String,
    pub outcome: Outcome,
    pub distance: Option<u32>,
    pub moves: Vec<GraphMove>,
}

/// The pruned graph, with nodes in prune-visit order (the order the TypeScript
/// object literal receives its keys).
#[derive(Clone, Debug)]
pub struct PrunedGraph {
    pub root: u32,
    pub nodes: Vec<PrunedNode>,
}

/// A move that stays inside the proven solution tree.
fn is_solution_move(node: Node<'_>, m: &GraphMove) -> bool {
    !node.attack() || (m.outcome() == Outcome::Mate && m.distance() == node.distance())
}

/// Defender answer that resists longest; ties break on the lowest USI.
///
/// Until 2026-09-19 this was also the designated main line's reply.  It no longer
/// is ([`designated_main_line`] judges the set [`longest_replies`] and prefers
/// replies that are not 変同駒余り), and it deliberately keeps its old meaning for
/// its two remaining readers: [`selected_reply`] -- the one answer the runtime
/// plays after a wrong or longer check, which lies outside the shortest solution
/// graph the new rule is defined on (rule G's clause (d) read it through
/// `PIPELINE_VERSION` 0.15.0 and reads the set of refuting replies since
/// 0.16.0) -- and [`usi_first_line`].
/// Rule F: a reply the solver flagged 無駄合 is not resistance at all, so it can
/// never be chosen.
pub fn longest_resistance(moves: &[GraphMove]) -> Option<&GraphMove> {
    let mut best: Option<&GraphMove> = None;
    for m in moves.iter() {
        if m.useless() {
            continue;
        }
        best = match best {
            None => Some(m),
            Some(incumbent) => {
                let candidate = m.distance().map_or(-1i64, |d| d as i64);
                let current = incumbent.distance().map_or(-1i64, |d| d as i64);
                if candidate > current || (candidate == current && m.usi() < incumbent.usi()) {
                    Some(m)
                } else {
                    Some(incumbent)
                }
            }
        };
    }
    best
}

/// The single defender answer the runtime plays after a non-solution move.
///
/// Through `PIPELINE_VERSION` 0.15.0 this was also the "recorded refutation
/// reply" rule G's clause (d) read; since 0.16.0 [`crate::inert`] reads every
/// reply that refutes the wrong check within the node's distance instead.
pub(crate) fn selected_reply(graph: &Graph, m: &GraphMove) -> Option<Move> {
    let defense = m.next().and_then(|id| graph.node(id));
    if let Some(defense) = defense {
        let replies = defense.moves();
        if !defense.attack() && !replies.is_empty() {
            if m.outcome() == Outcome::Refuted {
                return replies
                    .iter()
                    .find(|reply| reply.outcome() == Outcome::Refuted)
                    .map(GraphMove::mv)
                    .or_else(|| {
                        replies
                            .iter()
                            .find(|reply| reply.refutation_move().is_some())
                            .and_then(GraphMove::refutation_move)
                    })
                    .or_else(|| longest_resistance(replies).map(GraphMove::mv));
            }
            return longest_resistance(replies).map(GraphMove::mv);
        }
    }
    m.refutation_move()
}

/// Reduces a search graph to what the runtime needs (spec 12.2): the solution
/// tree, every legal attacker check at each attack node, and exactly one
/// defender answer per wrong move.  Idempotent.
///
/// At a defence node of the solution tree, an equally long reply outside the
/// main-line candidates `P(m)` (変同駒余り: the attacker can answer it and keep
/// material in hand) is left out (owner decision 2026-09-19, `PIPELINE_VERSION`
/// 0.9.0).  The runtime defender plays the longest reply with the smallest USI;
/// without this it could pick such a reply and let the player finish with
/// pieces in hand on a puzzle rule C accepts.
pub fn prune_solver_graph(graph: &Graph) -> PrunedGraph {
    let mut out = PrunedGraph {
        root: graph.root,
        nodes: Vec::new(),
    };
    let attacker = graph
        .node(graph.root)
        .map_or(0, |root| root.position().side);
    let mut facts = SolutionFacts::new(graph, attacker);
    let mut index: HashMap<u32, usize> = HashMap::new();
    visit_prune(graph, graph.root, &mut out, &mut index, &mut facts);
    out
}

fn visit_prune<'g>(
    graph: &'g Graph,
    id: u32,
    out: &mut PrunedGraph,
    index: &mut HashMap<u32, usize>,
    facts: &mut SolutionFacts<'g>,
) {
    if index.contains_key(&id) {
        return;
    }
    let Some(node) = graph.node(id) else {
        return;
    };
    let slot = out.nodes.len();
    index.insert(id, slot);
    // 変同駒余り replies: in L(m) but not in P(m).
    let off_line: Vec<Usi> = if node.attack() {
        Vec::new()
    } else {
        let candidates: Vec<Usi> = facts
            .candidate_replies(node)
            .iter()
            .map(|m| m.usi())
            .collect();
        longest_replies(node)
            .iter()
            .map(|m| m.usi())
            .filter(|usi| !candidates.contains(usi))
            .collect()
    };
    out.nodes.push(PrunedNode {
        id,
        attack: node.attack(),
        sfen: node.sfen(),
        outcome: node.outcome(),
        distance: node.distance(),
        moves: Vec::new(),
    });
    for m in node.moves().iter() {
        // Rule F: the runtime defender never plays a 無駄合, so the runtime graph
        // does not carry one.  The flagged reply stays in the search graph.
        if !node.attack() && (m.useless() || off_line.contains(&m.usi())) {
            continue;
        }
        if let (true, Some(next)) = (is_solution_move(node, m), m.next()) {
            out.nodes[slot].moves.push(*m);
            visit_prune(graph, next, out, index, facts);
            continue;
        }
        let reply = selected_reply(graph, m);
        out.nodes[slot].moves.push(GraphMove::new(
            m.mv(),
            m.outcome(),
            m.distance(),
            None,
            reply,
            false,
        ));
    }
}
