//! One diagram's gate: the fast validation (the prefilter, the construction
//! and the proof on the thread's prober, memoised), and the constructed gate
//! the publishing gate shares.

use super::*;

thread_local! {
    /// One prober per thread; every candidate starts it afresh, so a verdict
    /// does not depend on which thread checks which candidate.
    pub(super) static PROBER: RefCell<ClassicalProber> = RefCell::new(ClassicalProber::configured());
}

/// Drops this thread's prober and its tables (a fresh one is built on the
/// next question).  Every question starts the prober afresh ([`fresh`]), so
/// nothing a later question answers or counts depends on the old tables: this
/// only hands their memory back.  The browser gate (`wasm32`, whose linear
/// memory never shrinks) calls it before the exact search, so the exact
/// table can reuse the prober's memory instead of stacking on top of it
/// ([`crate::stages`], [`super::chain`]).  It exists only on `wasm32`, so the
/// native library and its public API are unchanged.
#[cfg(target_arch = "wasm32")]
pub(crate) fn release_prober() {
    PROBER.with(|cell| *cell.borrow_mut() = ClassicalProber::configured());
}

/// CPU seconds the helper threads of this thread's prober consumed so far
/// (`--prober-threads` > 1; `Some(0.0)` with one prober thread), or `None`
/// where they cannot be read (macOS, or the prober is in use).  Measurement
/// only ([`crate::record_clock`]).
pub fn prober_helper_cpu_seconds() -> Option<f64> {
    if crate::cdfpn::prober_threads() <= 1 {
        return Some(0.0);
    }
    PROBER.with(|cell| {
        cell.try_borrow()
            .ok()
            .and_then(|prober| prober.helper_cpu_seconds())
    })
}

/// What the fast path (solver v3) made of one gate.
#[derive(Clone)]
pub(super) enum Fast {
    /// Decided: the reasons [`exact_validate_with_rules`] gives, in its order,
    /// the mate length and the inert pieces it reports.
    Decided {
        reasons: Vec<&'static str>,
        distance: Option<u32>,
        inert_pieces: Vec<InertPiece>,
        /// For a mate, what the reasons were read from.
        built: Option<Box<BuiltGate>>,
    },
    /// A mate within the bound, and the construction is off (the reducer with
    /// `TSUME_REDUCE_CONSTRUCT=0`, the pre-v3 reducer): the exact gate decides,
    /// and may build its graph in its first search.  Also the construction's
    /// budget-out under the test-only [`set_exact_fallback`].
    Mate,
    /// The fast path did not run (the prefilter off, or a rule F other than
    /// classical), or the prefilter alone ran out of budget with the
    /// construction off: the exact gate decides, as before.  Also the fast
    /// path's budget-out under the test-only [`set_exact_fallback`].
    Unknown,
    /// The fast path ran out of its node budget with the construction on: the
    /// prober did not settle "mate within `max_ply`", or it proved the mate and
    /// the construction ran out.  A budget-out like the exact gate's on an
    /// `UNKNOWN` root ([`fast_budget_out_reasons`]): the reducer's ladder climbs
    /// to the next rung, the last rung's answer is logged (`--emit-budget-outs`)
    /// and counts against `reduction.minimality`, the publishing gate writes
    /// status `unknown` (`--emit-unresolved`).  The one-thread full-width search
    /// is **not** run on the same budget again (owner 2026-11-05, in line with
    /// the owner's rule of 2026-10-23: no re-run on one thread;
    /// solver 1.4.0).  `nodes`: the prober nodes it spent (prefilter and
    /// construction), the gate's `stats.nodes`.
    BudgetOut { nodes: u64 },
}

/// The reasons of a fast-path budget-out ([`Fast::BudgetOut`]): exactly the
/// exact gate's on an `UNKNOWN` root (`exact_validate_with_rules`, 2026-10-15:
/// the budget-out and nothing else), so every reader of a budget-out
/// (`budget_out`, `catalog_record::status_of`, the TypeScript reader) takes it
/// as one, and no new kind of answer appears.
pub(super) fn fast_budget_out_reasons() -> Vec<String> {
    vec![
        "solver_budget_exhausted".to_string(),
        "incomplete_response_graph".to_string(),
    ]
}

/// The graph a decided mate was read from, and what reading it gave.
#[derive(Clone)]
pub(super) struct BuiltGate {
    pub(super) graph: crate::solver::Graph,
    pub(super) root: Position,
    /// Prober nodes of the whole gate: the prefilter's question and the
    /// construction's.
    pub(super) prober_nodes: u64,
    /// Prober nodes of the prefilter's question alone: a fresh prober proving
    /// "mate within `max_ply`" on the gate's budget, which is exactly
    /// [`proof_nodes`] when the mate length is `max_ply`.
    pub(super) prefilter_nodes: u64,
    pub(super) transpositions: u64,
    pub(super) unique: bool,
    pub(super) first_move: Option<crate::position::Usi>,
    pub(super) main_line: Option<crate::exact::MainLine>,
}

/// The gate [`exact_validate_with_rules`] at `options.max_ply` with `rules`
/// and the target length `target`, decided without the full-width search when
/// the node budget allows (solver v3, `docs/experiments/solver-v3/README.md`).
///
/// 1. The classical df-pn prober asks "mate within `max_ply`?" (item 2a).
///    `REFUTED`: the exact search finds no mate either (or runs out of budget),
///    and its reasons depend on nothing else -- the mate gate's, the coverage
///    gate's and rule A's.
/// 2. `MATE`: [`crate::construct::build`] builds the part of the full-width
///    graph the gate reads (item 1) and the reasons are read off it with the
///    same analyses: the mate length against `target`, rule B at the root
///    (the minimum length is proven by the construction itself), then rules
///    A, B, C and G.
///
/// Both run on a node budget of their own (the exact search's).  With the
/// construction on (`construct`, the default), running out is a budget-out
/// ([`Fast::BudgetOut`]; until solver 1.3.0 it left the gate to the one-thread
/// exact search on the same budget, owner 2026-11-05).  With it off, a
/// prefilter that runs out leaves the gate to the exact search, as the pre-v3
/// reducer did.  Only rule F classical is covered.  Where one of the two
/// searches would run out of budget and the other not, the verdict can
/// differ: a budget-out on one side, a decision on the other.
#[allow(clippy::too_many_arguments)]
pub(super) fn fast_validate(
    position: &Position,
    target: Option<u32>,
    rules: TsumeRules,
    max_ply: u32,
    budget: u64,
    construct: bool,
    gate: bool,
) -> Fast {
    // The two cases the body answers `Unknown` without a search.
    if (!gate && !PREFILTER_ON.load(Ordering::Relaxed))
        || rules.useless_mode() != UselessInterposition::Classical
    {
        return Fast::Unknown;
    }
    if !FAST_MEMO.load(Ordering::Relaxed) {
        return fast_validate_uncached(position, target, rules, max_ply, budget, construct, gate);
    }
    let key = FastKey {
        root: strict_material::normalize(position, position.side),
        target,
        rules: rules.key(),
        max_ply,
        budget,
        construct,
        caps: prober_caps(),
        prober_threads: PROBER.with(|cell| cell.borrow().threads()),
        exact_fallback: exact_fallback_on(),
    };
    let hit = LAST_FAST.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|(stored, _, _)| *stored == key)
            .map(|(_, fast, nodes)| (fast.clone(), *nodes))
    });
    if let Some((fast, nodes)) = hit {
        SPEEDUP_HITS[2].fetch_add(1, Ordering::Relaxed);
        // The budget-out log counts a reused answer's nodes again, so its
        // entries do not depend on which thread asked first.
        charge_spent(nodes);
        return fast;
    }
    let before = spent();
    let fast = fast_validate_uncached(position, target, rules, max_ply, budget, construct, gate);
    let nodes = spent().wrapping_sub(before);
    LAST_FAST.with(|cell| *cell.borrow_mut() = Some((key, fast.clone(), nodes)));
    fast
}

/// Everything [`fast_validate`]'s answer depends on besides the switches its
/// early `Unknown` reads: the normalised diagram (the attacker is its side to
/// move), the target, the rules, the bound, the budget, the construction and
/// the prober's table caps.  `gate` only decides that early `Unknown`.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct FastKey {
    pub(super) root: Position,
    pub(super) target: Option<u32>,
    pub(super) rules: u8,
    pub(super) max_ply: u32,
    pub(super) budget: u64,
    pub(super) construct: bool,
    /// The prober's caps as [`prober_caps`] sets them (a test's, or the
    /// defaults) and its threads: neither changes a resolved answer, but each
    /// may change whether the budget suffices.
    pub(super) caps: [usize; 3],
    pub(super) prober_threads: usize,
    /// The test-only [`set_exact_fallback`]: it decides whether a budget-out
    /// is answered `BudgetOut` or left to the exact gate.
    pub(super) exact_fallback: bool,
}

thread_local! {
    /// The thread's last [`fast_validate`] answer.  The answer is a pure
    /// function of [`FastKey`] (the prober is reset for every question), so a
    /// repeated question -- the reducer's entry check right after the
    /// publishing gate in `extend`, the gate of a reduced diagram right after
    /// the reducer's last check of it -- takes it from here.  One entry only.
    /// A hit adds nothing to [`PREFILTER_STATS`] (`TSUME_V3_STATS`), which
    /// counts the questions computed.
    /// With the nodes it counted (the budget-out log).
    pub(super) static LAST_FAST: RefCell<Option<(FastKey, Fast, u64)>> = const { RefCell::new(None) };
}

/// [`fast_validate`] without the memo.
#[allow(clippy::too_many_arguments)]
pub(super) fn fast_validate_uncached(
    position: &Position,
    target: Option<u32>,
    rules: TsumeRules,
    max_ply: u32,
    budget: u64,
    construct: bool,
    gate: bool,
) -> Fast {
    // `set_prefilter` (tests and measurement only) is a switch of the
    // reducer only; the publishing gate (`gate`, [`constructed_validation`])
    // never reads it, as the TypeScript `solveStage` does not.
    if (!gate && !PREFILTER_ON.load(Ordering::Relaxed))
        || rules.useless_mode() != UselessInterposition::Classical
    {
        return Fast::Unknown;
    }
    let attacker = position.side;
    let root = strict_material::normalize(position, attacker);
    let king = rules.composition_rules
        && !rules.allow_attacker_king
        && root.king_square(attacker) != NO_SQUARE;
    let started = crate::clock::Instant::now();
    let (verdict, prefilter_nodes) = PROBER.with(|cell| {
        let prober = &mut *cell.borrow_mut();
        fresh(prober, attacker);
        PREFILTER_STATS[0].fetch_add(1, Ordering::Relaxed);
        let mut work = root.clone();
        let verdict = prober.mate_within(&mut work, max_ply, budget);
        PREFILTER_STATS[3].fetch_add(prober.stats().nodes, Ordering::Relaxed);
        charge_spent(prober.stats().nodes);
        crate::posprof::add(crate::posprof::K::PrefilterNodes, prober.stats().nodes);
        (verdict, prober.stats().nodes)
    });
    crate::posprof::charge_since(crate::posprof::K::PrefilterNs, started);
    charge_phase(0, started);
    match verdict {
        Outcome::Unknown => {
            PREFILTER_STATS[2].fetch_add(1, Ordering::Relaxed);
            // The prefilter alone (the construction off) is the pre-v3
            // reducer's: the exact gate decides.  With the construction on,
            // a budget-out (solver 1.4.0).
            if !construct || exact_fallback_on() {
                return Fast::Unknown;
            }
            return Fast::BudgetOut {
                nodes: prefilter_nodes,
            };
        }
        Outcome::Refuted => {
            PREFILTER_STATS[1].fetch_add(1, Ordering::Relaxed);
            let mut reasons = vec!["no_forced_mate_within_bound"];
            if target.is_some() {
                reasons.push("target_mate_ply_changed");
            }
            if !rules.allow_yozume {
                reasons.push("first_move_not_unique");
            }
            reasons.push("defense_coverage_incomplete");
            if king {
                reasons.push("attacker_king_present");
            }
            return Fast::Decided {
                reasons,
                distance: None,
                inert_pieces: Vec::new(),
                built: None,
            };
        }
        Outcome::Mate => {}
    }
    if !construct {
        return Fast::Mate;
    }
    let started = crate::clock::Instant::now();
    let built = PROBER.with(|cell| {
        let prober = &mut *cell.borrow_mut();
        let built = crate::construct::build(prober, &root, attacker, max_ply, budget);
        built.map(|built| (built, prober.stats().nodes))
    });
    // The construction's nodes, for the budget-out log: the gate's prober
    // nodes past the prefilter's when it built, else what the builder's
    // prober counted past the prefilter's.
    charge_spent(match &built {
        Some((_, nodes)) => nodes.saturating_sub(prefilter_nodes),
        None => PROBER
            .with(|cell| cell.borrow().stats().nodes)
            .saturating_sub(prefilter_nodes),
    });
    charge_phase(1, started);
    crate::posprof::charge_since(crate::posprof::K::ConstructNs, started);
    if crate::posprof::on() {
        PROBER.with(|cell| {
            let [table, intervals] = cell.borrow().table_sizes();
            crate::posprof::max(crate::posprof::K::MainTable, table as u64);
            crate::posprof::max(crate::posprof::K::MainIntervals, intervals as u64);
        });
    }
    let Some((built, prober_nodes)) = built else {
        PREFILTER_STATS[5].fetch_add(1, Ordering::Relaxed);
        if exact_fallback_on() {
            return Fast::Mate;
        }
        // The mate is proven, its graph is not: a budget-out (solver 1.4.0).
        return Fast::BudgetOut {
            nodes: PROBER.with(|cell| cell.borrow().stats().nodes),
        };
    };
    PREFILTER_STATS[4].fetch_add(1, Ordering::Relaxed);
    PREFILTER_STATS[6].fetch_add(built.nodes, Ordering::Relaxed);
    let graph = &built.graph;
    let mut reasons: Vec<&'static str> = Vec::new();
    if target.is_some_and(|target| target != built.distance) {
        reasons.push("target_mate_ply_changed");
    }
    let correct: Vec<crate::position::Usi> = graph
        .node(graph.root)
        .map(|root| correct_moves(root).iter().map(|m| m.usi()).collect())
        .unwrap_or_default();
    let unique = correct.len() == 1;
    // Rule B at the root: one correct move, or a merged promotion twin
    // (`PIPELINE_VERSION` 0.17.0, `exact::root_merged_twin`).
    let root_merged = !unique && root_merged_twin(graph, attacker);
    if !unique && !rules.allow_yozume && !root_merged {
        reasons.push("first_move_not_unique");
    }
    let mut inert_pieces = Vec::new();
    let mut main_line = None;
    if rules.composition_rules {
        if king {
            reasons.push("attacker_king_present");
        }
        let line = designated_main_line(graph, attacker);
        let inert = if rules.allow_inert_pieces {
            InertReport::default()
        } else if max_ply > built.distance {
            // Rule G reads the solution tree of the proven mate length, never
            // the gate's bound (`PIPELINE_VERSION` 0.16.0, owner decision
            // 2026-10-30): build it again at the proven distance, on the same
            // prober (`exact_validate_with_rules` does the same).
            let started = crate::clock::Instant::now();
            let tight = PROBER.with(|cell| {
                let prober = &mut *cell.borrow_mut();
                let before = prober.stats().nodes;
                let tight =
                    crate::construct::build(prober, &root, attacker, built.distance, budget);
                charge_spent(prober.stats().nodes.saturating_sub(before));
                tight
            });
            charge_phase(1, started);
            match tight {
                Some(tight) => {
                    let (inert, nodes) =
                        analyze_inert_pieces_counted(&tight.graph, attacker, budget);
                    charge_spent(nodes);
                    inert
                }
                None => InertReport::all_undecided(&root, attacker),
            }
        } else {
            let (inert, nodes) = analyze_inert_pieces_counted(graph, attacker, budget);
            charge_spent(nodes);
            inert
        };
        if line.yozume_nodes > 0 && !rules.allow_yozume {
            reasons.push("yozume_in_main_line");
        }
        if !rules.allow_koma_amari && !line.clean {
            reasons.push("koma_amari");
        }
        if !inert.attacker.is_empty() {
            reasons.push("inert_attacker_piece");
        }
        if !inert.defender.is_empty() {
            reasons.push("inert_defender_piece");
        }
        if !inert.undecided.is_empty() {
            reasons.push(RULE_G_BUDGET_EXHAUSTED);
        }
        inert_pieces = inert.all();
        main_line = Some(line);
    }
    Fast::Decided {
        reasons,
        distance: Some(built.distance),
        inert_pieces,
        built: Some(Box::new(BuiltGate {
            root,
            prober_nodes,
            prefilter_nodes,
            transpositions: built.transpositions,
            unique,
            // A merged root twin shows its promotion (0.17.0), else the smallest USI.
            first_move: if root_merged {
                correct
                    .iter()
                    .copied()
                    .min_by_key(|&usi| main_line_key(usi))
            } else {
                correct.first().copied()
            },
            main_line,
            graph: built.graph,
        })),
    }
}

/// The thread's prober, emptied for `attacker`, with the caps of
/// [`set_cdfpn_caps`] (the defaults unless a test lowered them).
pub(super) fn fresh(prober: &mut ClassicalProber, attacker: u8) {
    prober.reset(attacker);
    let [table, intervals, dominance] = prober_caps();
    prober.set_caps_for_tests(table, intervals, dominance);
}

/// The prober's table caps: [`set_cdfpn_caps`]'s, else the defaults.
pub(super) fn prober_caps() -> [usize; 3] {
    let caps = CDFPN_CAPS.each_ref().map(|cap| cap.load(Ordering::Relaxed));
    if caps == [0; 3] {
        [
            crate::cdfpn::CDFPN_TABLE_CAP,
            crate::cdfpn::CDFPN_INTERVAL_CAP,
            crate::cdfpn::CDFPN_DOMINANCE_CAP,
        ]
    } else {
        caps
    }
}

/// `stats.nodesProof` (solver v3, `DIFFICULTY_MODEL_VERSION` 4): the nodes a
/// fresh classical df-pn prober expands to prove `position` mate within exactly
/// `mate_ply` rule-F plies.  Unlike the gate's own prober count it does not
/// depend on the gate's bound (a looser bound makes the gate ask more), only on
/// the diagram and its mate length, as `nodesNonFutile` does for the full-width
/// search.  `None` when `budget` nodes do not prove it.  The TypeScript twin is
/// `proofNodes` in `tools/puzzle-pipeline/src/solve.ts`.
pub fn proof_nodes(position: &Position, mate_ply: u32, budget: u64) -> Option<u64> {
    let attacker = position.side;
    let mut root = strict_material::normalize(position, attacker);
    let started = crate::clock::Instant::now();
    let nodes = PROBER.with(|cell| {
        let mut prober = cell.borrow_mut();
        fresh(&mut prober, attacker);
        (prober.mate_within(&mut root, mate_ply, budget) == Outcome::Mate)
            .then(|| prober.stats().nodes)
    });
    charge_phase(2, started);
    nodes
}

/// Solver v3 item 1 in the publishing gate (on by default since the owner
/// decision 2026-10-09, `crate::stages::set_solve_construct`): [`exact_validate_with_rules`]
/// with a graph, decided by the fast path when its budget allows, as the
/// [`ExactValidation`](crate::exact::ExactValidation) the `solve` stage
/// publishes from.  `None` only when the fast path does not run (rule F other
/// than classical, or the test-only [`set_exact_fallback`] after a
/// budget-out): the caller runs the exact gate.  When the fast path runs out
/// of budget the answer is the exact gate's budget-out
/// (`budget_out_validation`, `solver_budget_exhausted`): since solver 1.4.0
/// (owner 2026-11-05) the full-width gate is no longer run again on the same
/// budget.
///
/// The verdict, the reasons, the mate length, the main line and its counters,
/// rule G's pieces and the pruned runtime graph's structure are the
/// full-width gate's wherever that gate stays within its budget.  What
/// differs is what the full-width search counted: `stats.nodes` is the
/// prober's nodes, `stats.cache_hits` the construction's transpositions, and
/// the graph holds the solution tree and the recorded answer to every wrong
/// check, not the wrong checks' subtrees (so `nodesNonFutile` and the features
/// counted over the whole graph change), with the construction's node ids.
/// Mirrors `constructedValidate` in `tools/puzzle-pipeline/src/solve.ts`.
pub fn constructed_validation(
    position: &Position,
    target: Option<u32>,
    max_ply: u32,
    node_budget: u64,
    rules: TsumeRules,
) -> Option<crate::exact::ExactValidation> {
    constructed_gate(position, target, max_ply, node_budget, rules).map(|gate| gate.validation)
}

/// What [`constructed_gate`] decided, and what it knows besides.
pub struct ConstructedGate {
    /// [`constructed_validation`]'s answer.
    pub validation: crate::exact::ExactValidation,
    /// The prefilter's prober nodes for "mate within `max_ply`" (a fresh
    /// prober, the gate's budget), `None` when the prefilter refuted the mate:
    /// `stats.nodesProof` itself ([`proof_nodes`]) when the mate length is
    /// `max_ply`, since that asks the identical question (finding 3 of the
    /// 2026-10-15 parallel review: no second serial proof).  `None` on a
    /// budget-out too.
    pub prefilter_nodes: Option<u64>,
    /// The fast path ran out of budget (`Fast::BudgetOut`): `validation` is
    /// the exact gate's `UNKNOWN`-root answer (`budget_out_validation`), and
    /// the caller must not run the full-width gate again on the same budget
    /// (solver 1.4.0, owner 2026-11-05).
    pub budget_out: bool,
}

/// `stats.nodesProof` of a constructed mate of length `mate_ply` at the gate
/// bound `max_ply` on the gate's `budget`: [`proof_nodes`], taken from the
/// prefilter's count (`prefilter_nodes`, [`ConstructedGate::prefilter_nodes`])
/// when the length is the bound, since the prefilter asked that identical
/// question of a fresh prober (same value, no second serial proof).
pub fn proof_nodes_after_gate(
    position: &Position,
    prefilter_nodes: Option<u64>,
    max_ply: u32,
    mate_ply: u32,
    budget: u64,
) -> Option<u64> {
    match prefilter_nodes {
        Some(nodes) if mate_ply == max_ply => Some(nodes),
        _ => proof_nodes(position, mate_ply, budget),
    }
}

/// [`constructed_validation`] with the prefilter's node count.
pub fn constructed_gate(
    position: &Position,
    target: Option<u32>,
    max_ply: u32,
    node_budget: u64,
    rules: TsumeRules,
) -> Option<ConstructedGate> {
    let attacker = position.side;
    let (reasons, distance, inert_pieces, built) =
        match fast_validate(position, target, rules, max_ply, node_budget, true, true) {
            Fast::Decided {
                reasons,
                distance,
                inert_pieces,
                built,
            } => (reasons, distance, inert_pieces, built),
            Fast::BudgetOut { nodes } => {
                return Some(ConstructedGate {
                    validation: budget_out_validation(position, max_ply, nodes),
                    prefilter_nodes: None,
                    budget_out: true,
                });
            }
            Fast::Mate | Fast::Unknown => return None,
        };
    let mut prefilter = None;
    let (result, unique, first_move, main_line) = match built {
        Some(built) => {
            let BuiltGate {
                graph,
                root,
                prober_nodes,
                prefilter_nodes,
                transpositions,
                unique,
                first_move,
                main_line,
            } = *built;
            prefilter = Some(prefilter_nodes);
            let moves = graph
                .node(graph.root)
                .map(|node| node.moves().to_vec())
                .unwrap_or_default();
            let result = crate::solver::SolveResult {
                outcome: Outcome::Mate,
                distance,
                stats: crate::solver::SolveStats {
                    nodes: prober_nodes,
                    cache_hits: transpositions,
                    budget_exhausted: false,
                    max_ply,
                    exhaustive_within_bound: true,
                },
                moves,
                graph: Some(graph),
                root_position: root,
                attacker,
            };
            (result, unique, first_move, main_line)
        }
        None => {
            let result = crate::solver::SolveResult {
                outcome: Outcome::Refuted,
                distance: None,
                stats: crate::solver::SolveStats {
                    nodes: 0,
                    cache_hits: 0,
                    budget_exhausted: false,
                    max_ply,
                    exhaustive_within_bound: true,
                },
                moves: Vec::new(),
                graph: None,
                root_position: strict_material::normalize(position, attacker),
                attacker,
            };
            (result, false, None, None)
        }
    };
    let attacker_branch_nodes = main_line.as_ref().map_or(0, |line| line.branching_nodes);
    let max_attacker_branching = main_line.as_ref().map_or(0, |line| line.max_correct);
    let validation = crate::exact::ExactValidation {
        accepted: reasons.is_empty(),
        reasons: reasons.into_iter().map(str::to_string).collect(),
        defense_coverage_verified: result.outcome == Outcome::Mate,
        result,
        unique_first_move: unique,
        first_move,
        main_line,
        inert_pieces,
        attacker_branch_nodes,
        max_attacker_branching,
        extra_nodes: 0,
        shorter_nodes: 0,
    };
    Some(ConstructedGate {
        validation,
        prefilter_nodes: prefilter,
        budget_out: false,
    })
}

/// The gate's answer on a fast-path budget-out ([`Fast::BudgetOut`]): the
/// [`ExactValidation`](crate::exact::ExactValidation) the exact gate gives on an
/// `UNKNOWN` root -- outcome `UNKNOWN`, no length, no first move, no main
/// line, `stats.budgetExhausted`, the reasons [`fast_budget_out_reasons`] --
/// with the prober's `nodes` as `stats.nodes`.  Nothing in it reads as a
/// judged defect, so the `solve` stage writes it as a rejection of status
/// `unknown` (`catalog_record::status_of`, `--emit-unresolved`) and keeps an
/// earlier `solution` as on any failed re-solve.
pub(super) fn budget_out_validation(
    position: &Position,
    max_ply: u32,
    nodes: u64,
) -> crate::exact::ExactValidation {
    let attacker = position.side;
    crate::exact::ExactValidation {
        accepted: false,
        reasons: fast_budget_out_reasons(),
        result: crate::solver::SolveResult {
            outcome: Outcome::Unknown,
            distance: None,
            stats: crate::solver::SolveStats {
                nodes,
                cache_hits: 0,
                budget_exhausted: true,
                max_ply,
                exhaustive_within_bound: false,
            },
            moves: Vec::new(),
            graph: None,
            root_position: strict_material::normalize(position, attacker),
            attacker,
        },
        unique_first_move: false,
        first_move: None,
        defense_coverage_verified: false,
        main_line: None,
        inert_pieces: Vec::new(),
        attacker_branch_nodes: 0,
        max_attacker_branching: 0,
        extra_nodes: 0,
        shorter_nodes: 0,
    }
}
