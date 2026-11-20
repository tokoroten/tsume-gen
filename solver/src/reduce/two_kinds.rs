//! Reduction v5, the two kinds (`docs/rules.md`): the main-line-preserving
//! reduction and up to K length-preserving ones, from the rule B entry.

use super::*;

/// The number of length-preserving reductions rule D v5 emits from an input
/// without yozume (`K`, owner decision 2026-10-09, `docs/rules.md`).
pub const LENGTH_K: usize = 2;

/// The number of length-preserving reductions rule D v5 emits from an input the
/// rule B entry let in (`K_y`, owner decision 2026-10-09 17:50 JST,
/// `docs/experiments/yozume-reduce/REPORT.md` §5).
pub const LENGTH_K_YOZUME: usize = 2;

/// The rejection of an input the rule B entry let in when no reduction step
/// removes its yozume (the first attempt took no step).
pub const YOZUME_NOT_REDUCIBLE: &str = "yozume_not_reducible";

/// The rejection of an input the C/G entry ([`TwoKindsOptions::cg_entry`]) let
/// in when no attempt reaches a diagram that passes the full acceptance test.
pub const CG_NOT_REPAIRABLE: &str = "cg_not_repairable";

/// The rejection of an input a relaxed entry (rule B, C/G) let in when the
/// early stop ended it ([`ReducedSet::stopped`]): such an input is no puzzle
/// as it stands, so a budget-out leaves nothing to emit.  The budget reason
/// the solver's own budget-outs carry, so readers count it as undecided.
pub const REDUCE_BUDGET_OUT: &str = "solver_budget_exhausted";

/// The rule C and rule G reasons the C/G entry repairs: spare attacker hand
/// material (`koma_amari`) and inert board pieces (`inert_*`).
pub const RULE_CG_REASONS: [&str; 3] =
    ["koma_amari", "inert_attacker_piece", "inert_defender_piece"];

/// `reasons` qualify for the C/G entry: at least one rule C / G reason, and
/// nothing else but rule B's when `with_rule_b` (the rule B entry is on too).
pub fn rule_cg_entry_reasons(reasons: &[String], with_rule_b: bool) -> bool {
    reasons
        .iter()
        .any(|reason| RULE_CG_REASONS.contains(&reason.as_str()))
        && reasons.iter().all(|reason| {
            RULE_CG_REASONS.contains(&reason.as_str())
                || (with_rule_b && crate::stages::RULE_B_REASONS.contains(&reason.as_str()))
        })
}

/// How an input entered [`reduce_two_kinds`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    /// It passed the strict entry of rule D v4.
    Strict,
    /// The rule B entry ([`TwoKindsOptions::rule_b_entry`]).
    RuleB,
    /// The C/G entry ([`TwoKindsOptions::cg_entry`]).
    RuleCg,
}

impl EntryKind {
    /// `reduction.entry` of the records it yields.
    pub const fn label(self) -> Option<&'static str> {
        match self {
            EntryKind::Strict => None,
            EntryKind::RuleB => Some("rule-b-relaxed"),
            EntryKind::RuleCg => Some("rule-cg-relaxed"),
        }
    }

    /// The rejection when the relaxed entry finds no output.
    pub(super) const fn not_reducible(self) -> &'static str {
        match self {
            EntryKind::RuleCg => CG_NOT_REPAIRABLE,
            _ => YOZUME_NOT_REDUCIBLE,
        }
    }
}

/// What [`reduce_two_kinds`] emits besides the main-line-preserving reduction.
/// The default emits one record per input (`length_k` 0) with the rule B entry
/// on, as the `reduce` stage does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoKindsOptions {
    /// Length-preserving reductions from an input without yozume (at most).
    /// `0` (the default) emits the main-line-preserving reduction alone, one
    /// output per input; the frozen number is [`LENGTH_K`].
    pub length_k: usize,
    /// Length-preserving reductions from an input the rule B entry let in (at
    /// most), also capped at `1 + length_k` so `length_k` 0 keeps one output per
    /// input; the frozen number is [`LENGTH_K_YOZUME`].
    pub length_k_yozume: usize,
    /// The rule B entry (V1 of `docs/experiments/yozume-reduce/REPORT.md`, owner
    /// decision 2026-10-09 17:50 JST): an input the strict entry refuses for rule
    /// B alone (`first_move_not_unique`, `yozume_in_main_line`) is let in and
    /// yields length-preserving reductions only, each step judged by the full
    /// acceptance test.  On by default (it is part of rule D v5); the workshop
    /// gate turns it off.
    pub rule_b_entry: bool,
    /// The C/G entry (owner decision 2026-10-10, `reduce --cg-entry`,
    /// `docs/experiments/ply13/README.md` §13): an input the strict entry
    /// refuses for rules C and/or G alone (`koma_amari`, `inert_*`; with
    /// [`TwoKindsOptions::rule_b_entry`] also together with rule B's reasons)
    /// is let in and **repaired**: spare attacker hand material is handed back
    /// and inert pieces are swept, each step judged by the acceptance test with
    /// only the input's own defects (C, G and, when it had them, B) set aside,
    /// until the diagram passes the full test; from there every step keeps the
    /// full test, as the rule B entry's do.  Yields length-preserving
    /// reductions only (counted as the rule B entry's), each a publishable
    /// puzzle at the input's length, or `cg_not_repairable`.
    ///
    /// **Off by default and outside the frozen rule D**: the `reduce` stage of
    /// the public pipeline never sets it, so its bytes, the TypeScript reducer
    /// and `PIPELINE_VERSION` are untouched.  It is Rust-only tooling for the
    /// RL loop's background repair, whose outputs the publishing gate (`solve`)
    /// judges again before anything adopts them.
    pub cg_entry: bool,
}

impl Default for TwoKindsOptions {
    fn default() -> Self {
        Self {
            length_k: 0,
            length_k_yozume: LENGTH_K_YOZUME,
            rule_b_entry: true,
            cg_entry: false,
        }
    }
}

impl TwoKindsOptions {
    /// The frozen numbers of `docs/rules.md` (K = 2, K_y = 2), rule B entry on.
    pub const fn frozen() -> Self {
        Self {
            length_k: LENGTH_K,
            length_k_yozume: LENGTH_K_YOZUME,
            rule_b_entry: true,
            cg_entry: false,
        }
    }

    /// How many length-preserving reductions an input emits at most: never
    /// more than the frozen [`LENGTH_K`] / [`LENGTH_K_YOZUME`].
    pub(super) fn length_cap(&self, rule_b_entry: bool) -> usize {
        let length_k = self.length_k.min(LENGTH_K);
        if rule_b_entry {
            self.length_k_yozume.min(LENGTH_K_YOZUME).min(1 + length_k)
        } else {
            length_k
        }
    }
}

/// What [`reduce_two_kinds`] made of one input.
#[derive(Clone, Debug)]
pub struct ReducedSet {
    /// The main-line-preserving reduction, the input itself when nothing could
    /// come off.  `None` for an input a relaxed entry (rule B, or the opt-in
    /// C/G entry) let in.
    pub main_line: Option<Reduced>,
    /// The input's main-line region has more than the cap's paths
    /// ([`MAIN_LINE_PATHS_CAP`]), so the main-line-preserving reduction took
    /// nothing off.
    pub paths_capped: bool,
    /// The length-preserving reductions, rank 1 first.
    pub length: Vec<Reduced>,
    /// The rule B reasons the relaxed entry let the input in with; empty for an
    /// input that passed the strict entry.  Also set for the C/G entry, whose
    /// reasons are the input's rule C / G (and B) reasons.
    pub rule_b_entry_reasons: Vec<String>,
    /// How the input entered.
    pub entry: EntryKind,
    /// The greedy passes behind the outputs, with [`set_trace_passes`] on
    /// (measurement only, never written to a record).
    pub trace: Option<TwoKindsTrace>,
    /// The budget-outs behind [`ReducedSet::main_line`] (`reduction.minimality`,
    /// `budgetOutChecks`, `budgetOutStages`; owner 2026-10-22).
    pub main_line_budget_outs: BudgetOutTally,
    /// The budget-outs behind every length-preserving reduction (one tally for
    /// all of them: the attempts' chains share their checks).
    pub length_budget_outs: BudgetOutTally,
    /// The early stop ([`stopped`], owner 2026-10-25): a check that counts
    /// ended out of budget and the reducer stopped there.  The record is then
    /// a budget-out: [`ReducedSet::main_line`] is the input itself (nothing
    /// removed), no length-preserving reduction, both tallies
    /// [`BudgetOutTally::stopped`].  An input of a relaxed entry is rejected
    /// with [`REDUCE_BUDGET_OUT`] instead.
    pub stopped: bool,
    /// The node budget of the highest `--reduce-budget-ladder` rung a check
    /// of this call needed to be decided (the full budget when one ran out,
    /// the node budget without a ladder): `reduce.budget` of the catalog
    /// records (`--emit-record`).  The checks the reduction consumed only, so
    /// the same for every `--jobs`, with `--emit-negatives` or without; under
    /// a parallel prober, which rung decides may depend on the threads.
    pub decided_budget: u64,
}

/// Outcome of [`reduce_two_kinds`].
#[derive(Clone, Debug)]
pub enum ReduceSetResult {
    /// The input never qualified, with the entry check's reasons.
    Rejected {
        reasons: Vec<String>,
    },
    Reduced(Box<ReducedSet>),
}

/// [`reduce_two_kinds`] with the TypeScript seed schedule.
pub fn reduce_two_kinds_seeded(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    kinds: &TwoKindsOptions,
    base_seed: u64,
) -> ReduceSetResult {
    let mut attempts = SeededAttempts::new(base_seed);
    reduce_two_kinds(
        solver,
        position,
        target_mate_ply,
        options,
        kinds,
        &mut attempts,
    )
}

/// Rule D v5 (`PIPELINE_VERSION` 0.13.0, owner decision 2026-10-09,
/// `docs/rules.md` "Reduction v5"): from one input, the **main-line-preserving**
/// reduction and up to `K` **length-preserving** reductions with pairwise
/// different displayed main lines.
///
/// * Entry: the strict entry of v4 (`options.strict_entry` is not read).  With
///   [`TwoKindsOptions::rule_b_entry`] an input refused for rule B alone is let
///   in and yields length-preserving reductions only.  With the opt-in
///   [`TwoKindsOptions::cg_entry`] (not part of rule D v5) an input refused for
///   rules C / G alone (or with rule B's reasons, the rule B entry being on) is
///   let in the same way, repaired first by [`repair_pass`] and stepped by
///   [`StepRule::Repair`] until it passes the full test; `cg_not_repairable`
///   when no attempt gets there.
/// * The attacker-king pre-pass runs once per kind, judged by that kind's rule.
///   The entry sweep of v3 is not run for the strict and rule B entries: they
///   guarantee the input carries no inert piece, and so does an accepted king
///   deletion.  The C/G entry's [`repair_pass`] runs it.
/// * The shuffles (`options.attempts` of them) are drawn once and shared by both
///   kinds.
/// * Main line: every attempt steps by [`StepRule::MainLine`]; the best attempt
///   ([`beats_best`]) is then swept in the frozen candidate order until a whole
///   pass removes nothing.  An input whose main-line region has more paths than
///   the cap is its own result.
/// * Length: every attempt steps by [`StepRule::Length`] and is swept to its own
///   fixpoint; the results are grouped by displayed main line, the group of the
///   input's line is dropped (inputs of the strict entry only), each group keeps
///   its best diagram, and the groups are taken best first.
///
/// `options.jobs` above 1 runs the chains on a pool of threads ([`workers`]);
/// the output is byte-identical for every value ([`ReduceOptions::jobs`]).
///
/// Every output says whether its minimality is verified: the budget-outs
/// behind it ([`BudgetOutTally`], [`ReducedSet::main_line_budget_outs`] and
/// [`ReducedSet::length_budget_outs`]), read off this call's own budget-out
/// log, which then goes on to the thread's (`--emit-budget-outs`) as before.
///
/// The first check that counts ([`BudgetOut::counts`]) and ends out of budget
/// stops the call (owner 2026-10-25, [`stopped`]): no other check of this
/// input runs, and the input comes back as a budget-out
/// ([`ReducedSet::stopped`]; rejected [`REDUCE_BUDGET_OUT`] behind a relaxed
/// entry).  Whether a record stops is the same for every `--jobs`: every chain runs to its fixpoint unless one stops, each
/// chain's committed checks are the sequential loop's, so some chain meets a
/// counted budget-out exactly when the sequential loop does.
pub fn reduce_two_kinds(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    kinds: &TwoKindsOptions,
    attempts: &mut dyn AttemptRng,
) -> ReduceSetResult {
    let _stop = StopScope::arm();
    let (mut result, log) = with_frame(|| {
        reduce_two_kinds_unlogged(solver, position, target_mate_ply, options, kinds, attempts)
    });
    if let ReduceSetResult::Reduced(set) = &mut result {
        if set.stopped {
            set.main_line_budget_outs = BudgetOutTally::stopped();
            set.length_budget_outs = BudgetOutTally::stopped();
        } else {
            set.main_line_budget_outs = BudgetOutTally::of(&log.entries, true);
            set.length_budget_outs = BudgetOutTally::of(&log.entries, false);
        }
        set.decided_budget = log.rung;
    }
    absorb_budget_outs(log);
    result
}

/// [`reduce_two_kinds`] before the tallies.
pub(super) fn reduce_two_kinds_unlogged(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    kinds: &TwoKindsOptions,
    attempts: &mut dyn AttemptRng,
) -> ReduceSetResult {
    // `--emit-negatives` only: buffers this call's events and hands them over,
    // in chain order, whichever way the call returns.
    let mut negatives = NegativeScope::open();
    let attacker = position.side;
    let structural = validate_position(position, ValidateOptions::pipeline());
    if !structural.is_empty() {
        return ReduceSetResult::Rejected {
            reasons: structural.into_iter().map(|issue| issue.code).collect(),
        };
    }
    let entry = entry_verdict(
        solver,
        position,
        target_mate_ply,
        options.rules,
        options,
        true,
    );
    if entry.distance.is_some() || !entry.reasons.is_empty() {
        negative_event(0, || crate::negatives::Event {
            kind: crate::negatives::EventKind::Entry,
            tool: "reduce",
            position: position.clone(),
            target: target_mate_ply.or(entry.distance),
            reasons: entry.reasons.clone(),
            accepted: entry.reasons.is_empty(),
            parent_sound: false,
            step: None,
            path: Vec::new(),
            chain: 0,
            parent: None,
            keeps_line: false,
        });
    }
    let Some(distance) = entry.distance else {
        return ReduceSetResult::Rejected {
            reasons: entry.reasons,
        };
    };
    let relaxed = !entry.reasons.is_empty();
    let entry_kind = if !relaxed {
        EntryKind::Strict
    } else if kinds.rule_b_entry && crate::stages::rule_b_only(&entry.reasons) {
        EntryKind::RuleB
    } else if kinds.cg_entry && rule_cg_entry_reasons(&entry.reasons, kinds.rule_b_entry) {
        EntryKind::RuleCg
    } else {
        return ReduceSetResult::Rejected {
            reasons: entry.reasons,
        };
    };
    // The C/G entry also needs at least one attempt: its outputs are the
    // attempts' chains, and with none the repair pre-pass alone would reach the
    // output by one path and not by the other (`jobs`, the early stop).
    if relaxed
        && (kinds.length_cap(true) == 0
            || (entry_kind == EntryKind::RuleCg && options.attempts == 0))
    {
        return ReduceSetResult::Rejected {
            reasons: entry.reasons,
        };
    }
    let target = target_mate_ply.unwrap_or(distance);
    negatives.target = Some(target);
    let original_board_pieces = position.board_piece_count();
    let original_attacker_hand = attacker_hand_total(position, attacker);
    let input_line = entry.main_line.clone();
    let length_k = kinds.length_cap(relaxed);
    // The early stop ([`stopped`]): the input as a budget-out.
    let paths_capped = !relaxed && !matches!(entry.paths, MainLinePaths::Paths(_));
    let stop = |negatives: &NegativeScope| {
        negatives.discard();
        if relaxed {
            return ReduceSetResult::Rejected {
                reasons: vec![REDUCE_BUDGET_OUT.to_string()],
            };
        }
        ReduceSetResult::Reduced(Box::new(ReducedSet {
            main_line: Some(Reduced {
                original_board_pieces,
                final_board_pieces: original_board_pieces,
                original_attacker_hand,
                final_attacker_hand: original_attacker_hand,
                attacker_king_removed: false,
                position: position.clone(),
                removed: Vec::new(),
                target_mate_ply: target,
            }),
            paths_capped,
            length: Vec::new(),
            rule_b_entry_reasons: Vec::new(),
            entry: EntryKind::Strict,
            trace: None,
            main_line_budget_outs: BudgetOutTally::default(),
            length_budget_outs: BudgetOutTally::default(),
            decided_budget: 0,
            stopped: true,
        }))
    };

    // The main-line kind starts from the input with the rule of its own line and
    // paths.  Without paths (over the cap) no step can be accepted, so the input
    // is the result as it stands.
    let mut main_start: Option<ChainState> = None;
    let mut main_fixed: Option<ChainState> = None;
    if !relaxed {
        let state = |rule| {
            let mut state =
                ChainState::new(position.clone(), Vec::new(), rule, input_line.clone(), true);
            state.neg_chain = NEG_MAIN_CHAINS;
            state
        };
        match &entry.paths {
            MainLinePaths::Paths(paths) => {
                main_start = Some(state(StepRule::MainLine {
                    paths: paths.clone(),
                }));
            }
            _ => main_fixed = Some(state(StepRule::Length)),
        }
    }
    // The C/G entry steps by the repair rule until the diagram passes the full
    // test ([`StepRule::Repair`]); every other chain by the length rule.
    let length_rule = match entry_kind {
        EntryKind::RuleCg => StepRule::Repair {
            allow_rule_b: entry
                .reasons
                .iter()
                .any(|reason| crate::stages::RULE_B_REASONS.contains(&reason.as_str())),
        },
        _ => StepRule::Length,
    };
    let mut length_start = (length_k > 0).then(|| {
        let mut state = ChainState::new(
            position.clone(),
            Vec::new(),
            length_rule,
            input_line.clone(),
            !relaxed,
        );
        state.neg_chain = NEG_LENGTH_CHAINS;
        state
    });

    // The attacker-king pre-pass, judged per kind.  The king is only ever still
    // on the board when the record declares the double-king genre: anything else
    // carrying one fails the entry with `attacker_king_present`.
    if let Some(square) = attacker_king_square(position, attacker) {
        let removal = RemovedPiece {
            square,
            piece: piece_label(position, square),
        };
        let kingless = without_attacker_king(position, attacker).expect("attacker king");
        let check = {
            let _stage = Stage::enter("king", None, vec![removal.clone()]);
            check_reduction(solver, &kingless, target, options, main_start.is_some())
        };
        if stopped() {
            return stop(&negatives);
        }
        // `--emit-negatives`: the deletion's verdict, whichever kind takes it
        // (a sound kingless diagram also proves the input not minimal).
        negative_event(0, || crate::negatives::Event {
            kind: crate::negatives::EventKind::Refused,
            tool: "reduce",
            position: kingless.clone(),
            target: Some(target),
            reasons: check.reasons.clone(),
            accepted: check.accepted,
            parent_sound: !relaxed,
            step: Some(removal.clone()),
            path: Vec::new(),
            chain: 0,
            parent: Some(position.hash()),
            keeps_line: false,
        });
        if !check.accepted && !options.rules.allow_attacker_king {
            return ReduceSetResult::Rejected {
                reasons: vec!["attacker_king_required".to_string()],
            };
        }
        if let Some(state) = length_start.as_mut() {
            // The length rule accepts exactly `check.accepted`; the C/G
            // entry's repair rule also takes a deletion that leaves only the
            // input's own defects.
            if state.rule.accepts_step(&check) {
                state.commit(
                    removal.clone(),
                    SweepResult {
                        position: kingless.clone(),
                        removed: Vec::new(),
                        check: check.clone(),
                    },
                );
            }
        }
        if let Some(state) = main_start.as_mut() {
            if state.rule.accepts_step(&check) {
                state.commit(
                    removal,
                    SweepResult {
                        position: kingless,
                        removed: Vec::new(),
                        check,
                    },
                );
            }
        }
    }

    if stopped() {
        return stop(&negatives);
    }

    // The C/G entry's repair pre-pass: before any shuffled pass, take the
    // input's own defects off in a fixed order (inert pieces, then spare hand
    // units), so every attempt starts from the repaired diagram when the
    // pre-pass alone repairs it.
    if entry_kind == EntryKind::RuleCg {
        if let Some(state) = length_start.as_mut() {
            repair_pass(solver, state, target, options, attacker);
        }
        if stopped() {
            return stop(&negatives);
        }
    }

    let mut candidates = reduction_candidates(position, attacker);
    if !options.hand_transfer {
        candidates.retain(|candidate| !matches!(candidate, Candidate::Hand(_)));
    }
    // One shuffle per attempt, shared by both kinds.
    let orders: Vec<Vec<Candidate>> = (0..options.attempts)
        .map(|attempt| attempts.rng_for_attempt(attempt).shuffle(&candidates))
        .collect();
    // The rule B entry (`docs/experiments/yozume-reduce/REPORT.md` §3.6): until a
    // step is accepted every attempt tests the same single removals on the same
    // diagram, so either every attempt takes a step or none does.  The first
    // attempt's pass decides; the others are not run when it takes none, which
    // leaves the output as it would be.
    //
    // The C/G entry stops early on the same argument, but only when the first
    // pass took no step at all: a step its repair rule took need not yet pass
    // the full test, and another attempt may still get there.
    let mut length_first: Option<ChainState> = None;
    if relaxed && RULE_B_EARLY_STOP.load(Ordering::Relaxed) {
        let mut chain = length_start
            .as_ref()
            .expect("a relaxed entry emits length-preserving reductions")
            .for_attempt(0);
        let mut changed = false;
        if let Some(order) = orders.first() {
            changed = greedy_pass(solver, &mut chain, order, target, options, attacker);
        }
        if stopped() {
            return stop(&negatives);
        }
        let stuck = match entry_kind {
            EntryKind::RuleCg => !chain.accepted && !changed,
            _ => !chain.accepted,
        };
        if stuck {
            return ReduceSetResult::Rejected {
                reasons: vec![entry_kind.not_reducible().to_string()],
            };
        }
        length_first = Some(chain);
    }
    let (main_best, lengths) = if options.jobs > 1 {
        workers::with_workers(solver, target, options, attacker, |runner| {
            run_two_kinds_parallel(
                runner,
                main_start,
                length_start,
                length_first,
                &orders,
                &candidates,
                attacker,
            )
        })
    } else {
        run_two_kinds_sequential(
            solver,
            main_start,
            length_start,
            length_first,
            &orders,
            &candidates,
            target,
            options,
            attacker,
        )
    };
    if stopped() {
        return stop(&negatives);
    }
    let main_out = main_best.or(main_fixed);
    let tracing = TRACE_PASSES.load(Ordering::Relaxed);
    let mut trace = tracing.then(|| TwoKindsTrace {
        main_line: main_out
            .as_ref()
            .and_then(|state| state.trace.as_deref().cloned()),
        length_attempts: lengths
            .iter()
            .filter_map(|state| state.trace.as_deref().cloned())
            .collect(),
        length: Vec::new(),
    });

    // The length-preserving results, grouped by displayed main line.
    let mut groups: Vec<ChainState> = Vec::new();
    for state in lengths {
        if !state.accepted || (!relaxed && state.line == input_line) {
            continue;
        }
        match groups.iter_mut().find(|group| group.line == state.line) {
            Some(group) => {
                if beats_best(&state.board, &group.board, attacker) {
                    *group = state;
                }
            }
            None => groups.push(state),
        }
    }
    groups.sort_by(|a, b| {
        if beats_best(&a.board, &b.board, attacker) {
            std::cmp::Ordering::Less
        } else if beats_best(&b.board, &a.board, attacker) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let main_key = main_out
        .as_ref()
        .map(|state| to_legacy_sfen_key(&state.board));
    let king = attacker_king_square(position, attacker).is_some();
    let reduced = |state: ChainState| Reduced {
        original_board_pieces,
        final_board_pieces: state.board.board_piece_count(),
        original_attacker_hand,
        final_attacker_hand: attacker_hand_total(&state.board, attacker),
        attacker_king_removed: king && attacker_king_square(&state.board, attacker).is_none(),
        position: state.board,
        removed: state.removed,
        target_mate_ply: target,
    };
    let chosen: Vec<ChainState> = groups
        .into_iter()
        .filter(|group| Some(to_legacy_sfen_key(&group.board)) != main_key)
        .take(length_k)
        .collect();
    if let Some(trace) = trace.as_mut() {
        trace.length = chosen
            .iter()
            .filter_map(|state| state.trace.as_deref().cloned())
            .collect();
    }
    let length: Vec<Reduced> = chosen.into_iter().map(reduced).collect();
    if relaxed && length.is_empty() {
        return ReduceSetResult::Rejected {
            reasons: vec![entry_kind.not_reducible().to_string()],
        };
    }
    ReduceSetResult::Reduced(Box::new(ReducedSet {
        main_line: main_out.map(reduced),
        paths_capped,
        length,
        rule_b_entry_reasons: if relaxed { entry.reasons } else { Vec::new() },
        entry: entry_kind,
        trace,
        main_line_budget_outs: BudgetOutTally::default(),
        length_budget_outs: BudgetOutTally::default(),
        decided_budget: 0,
        stopped: false,
    }))
}

/// The chains of [`reduce_two_kinds`], one search at a time: the main-line
/// attempts, the best of them to its fixpoint, then every length attempt to its
/// own fixpoint.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_two_kinds_sequential(
    solver: &mut Solver,
    main_start: Option<ChainState>,
    length_start: Option<ChainState>,
    length_first: Option<ChainState>,
    orders: &[Vec<Candidate>],
    candidates: &[Candidate],
    target: u32,
    options: &ReduceOptions,
    attacker: u8,
) -> (Option<ChainState>, Vec<ChainState>) {
    let main_best = main_start.map(|start| {
        let mut best = start.clone();
        for (attempt, order) in orders.iter().enumerate() {
            if stopped() {
                break;
            }
            let mut chain = start.for_attempt(attempt);
            greedy_pass(solver, &mut chain, order, target, options, attacker);
            if beats_best(&chain.board, &best.board, attacker) {
                best = chain;
            } else {
                best.inherit_settled(&chain);
            }
        }
        while greedy_pass(solver, &mut best, candidates, target, options, attacker) {}
        best
    });
    let mut lengths = Vec::new();
    // A relaxed entry's (rule B, C/G) first attempt has had its shuffled pass
    // already (`length_first`, [`reduce_two_kinds`]).
    let mut first = length_first;
    if let Some(start) = length_start {
        for (attempt, order) in orders.iter().enumerate() {
            if stopped() {
                break;
            }
            let mut chain = match first.take() {
                Some(chain) => chain,
                None => {
                    let mut chain = start.for_attempt(attempt);
                    greedy_pass(solver, &mut chain, order, target, options, attacker);
                    chain
                }
            };
            while greedy_pass(solver, &mut chain, candidates, target, options, attacker) {}
            lengths.push(chain);
        }
    }
    (main_best, lengths)
}

/// [`run_two_kinds_sequential`] on the worker pool: every attempt of both kinds
/// at once, then every fixpoint sweep at once.  Each chain ends exactly where
/// the sequential loop leaves it, so the result is the same.
pub(super) fn run_two_kinds_parallel(
    runner: &mut workers::Runner<'_>,
    main_start: Option<ChainState>,
    length_start: Option<ChainState>,
    length_first: Option<ChainState>,
    orders: &[Vec<Candidate>],
    candidates: &[Candidate],
    attacker: u8,
) -> (Option<ChainState>, Vec<ChainState>) {
    let mut passes: Vec<(&[Candidate], ChainState)> = Vec::new();
    if let Some(start) = &main_start {
        for (attempt, order) in orders.iter().enumerate() {
            passes.push((order.as_slice(), start.for_attempt(attempt)));
        }
    }
    // A relaxed entry's (rule B, C/G) first attempt has had its shuffled pass
    // already.
    let skip = usize::from(length_first.is_some());
    if let Some(start) = &length_start {
        for (attempt, order) in orders.iter().enumerate().skip(skip) {
            passes.push((order.as_slice(), start.for_attempt(attempt)));
        }
    }
    let mut results = runner.passes(passes).into_iter().map(|(state, _)| state);
    let main_best = main_start.map(|start| {
        let mut best = start;
        for chain in results.by_ref().take(orders.len()) {
            if beats_best(&chain.board, &best.board, attacker) {
                best = chain;
            } else {
                best.inherit_settled(&chain);
            }
        }
        best
    });
    let mut swept: Vec<ChainState> = main_best.into_iter().collect();
    let has_main = !swept.is_empty();
    swept.extend(length_first);
    swept.extend(results);
    let mut swept = runner.sweep_to_fixpoint(candidates, swept).into_iter();
    let main_best = if has_main { swept.next() } else { None };
    (main_best, swept.collect())
}

/// What the entry check found (`exactValidate` at the strict or lenient entry
/// rules): its reasons, the mate length and, for rule D v5, the displayed main
/// line and the main-line paths of the input.
#[derive(Clone)]
pub(super) struct EntryVerdict {
    pub(super) reasons: Vec<String>,
    pub(super) distance: Option<u32>,
    pub(super) main_line: Option<Vec<Usi>>,
    pub(super) paths: MainLinePaths,
}

/// The entry check of both reducers.  `want_paths` measures `Π` of an input the
/// check accepts (rule D v5); the verdict is the same either way.
pub(super) fn entry_verdict(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    rules: TsumeRules,
    options: &ReduceOptions,
    want_paths: bool,
) -> EntryVerdict {
    on_ladder(
        options.node_budget,
        solver,
        |verdict: &EntryVerdict| &verdict.reasons,
        None,
        Checked {
            kind: "entry",
            position,
            target: target_mate_ply,
            paths: want_paths,
        },
        |budget, solver| {
            entry_verdict_at(
                solver,
                position,
                target_mate_ply,
                rules,
                options,
                want_paths,
                budget,
            )
        },
    )
}

/// [`entry_verdict`] on the node budget `budget`.
pub(super) fn entry_verdict_at(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    rules: TsumeRules,
    options: &ReduceOptions,
    want_paths: bool,
    budget: u64,
) -> EntryVerdict {
    // The entry check publishes its reasons, and a refutation the prefilter
    // proves is reported with the mate gate's reasons even where the exact
    // search would have run out of budget, so the entry takes the fast path
    // only together with the construction (item 1), never with the prefilter
    // alone.
    let fast = if CONSTRUCT_ON.load(Ordering::Relaxed) {
        fast_validate(
            position,
            target_mate_ply,
            rules,
            options.max_ply,
            budget,
            true,
            false,
        )
    } else {
        Fast::Unknown
    };
    match fast {
        Fast::Decided {
            reasons,
            distance,
            built,
            ..
        } => {
            let (main_line, graph) = built_line_and_graph(built);
            let paths = if want_paths && reasons.is_empty() {
                measure_paths(graph.as_ref(), position.side)
            } else {
                MainLinePaths::Unmeasured
            };
            EntryVerdict {
                reasons: reasons.into_iter().map(str::to_string).collect(),
                distance,
                main_line,
                paths,
            }
        }
        // The fast path ran out of budget: the entry's budget-out (the ladder
        // climbs; on the last rung the input is rejected
        // `solver_budget_exhausted`), never the one-thread full-width search
        // on the same budget (solver 1.4.0, owner 2026-11-05).
        Fast::BudgetOut { .. } => EntryVerdict {
            reasons: fast_budget_out_reasons(),
            distance: None,
            main_line: None,
            paths: MainLinePaths::Unmeasured,
        },
        // The construction off (`TSUME_REDUCE_CONSTRUCT=0`), or the test-only
        // `set_exact_fallback`: the exact gate decides, as before.
        Fast::Mate | Fast::Unknown => {
            count_full_width_gate();
            let initial = exact_validate_with_rules(
                solver,
                position,
                target_mate_ply,
                options.max_ply,
                budget,
                want_paths,
                rules,
            );
            charge_spent(initial.result.stats.nodes + initial.extra_nodes + initial.shorter_nodes);
            let paths = if want_paths && initial.reasons.is_empty() {
                measure_paths(initial.result.graph.as_ref(), initial.result.attacker)
            } else {
                MainLinePaths::Unmeasured
            };
            EntryVerdict {
                reasons: initial.reasons,
                distance: initial.result.distance,
                main_line: initial.main_line.map(|line| line.moves),
                paths,
            }
        }
    }
}

/// The displayed main line and the graph of a mate the fast path decided.
pub(super) fn built_line_and_graph(
    built: Option<Box<BuiltGate>>,
) -> (Option<Vec<Usi>>, Option<Graph>) {
    match built {
        Some(built) => {
            let BuiltGate {
                graph, main_line, ..
            } = *built;
            (main_line.map(|line| line.moves), Some(graph))
        }
        None => (None, None),
    }
}

/// `Π` of a solved graph under the current cap.
pub(super) fn measure_paths(graph: Option<&Graph>, attacker: u8) -> MainLinePaths {
    match graph {
        None => MainLinePaths::Unmeasured,
        Some(graph) => match main_line_paths(graph, attacker, paths_cap()) {
            Some(paths) => MainLinePaths::Paths(Arc::new(paths)),
            None => MainLinePaths::Capped,
        },
    }
}
