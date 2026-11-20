//! Rule D: seeded greedy piece reduction (spec 10; `docs/rules.md`, "Reduction v5").
//!
//! Port of `reduceStage` in `tools/puzzle-pipeline/src/stages.ts`, split so the
//! search part is reusable: the seed derivation lives in
//! [`crate::stages::reduce_stage`], the algorithm here only needs a source of
//! randomness.
//!
//! Each attempt shuffles the reduction candidates, greedily applies one whenever
//! the result still passes [`crate::exact::exact_validate_with_rules`] against the
//! target ply, and the attempt with the fewest board pieces wins; ties break on
//! fewer attacker hand pieces and then on the smaller SFEN string.
//!
//! # Rule D: what a reduction candidate is
//!
//! The candidate list is built **before** shuffling, in this order:
//!
//! 1. the attacker king's square, if the attacker still has a king (delete it);
//! 2. every non-king board piece square, ascending (remove it; the defender gains
//!    it, which is what the strict convention forces);
//! 3. one entry per *unit* of attacker hand material, piece types in `P L N S G B R`
//!    order, encoded as square `81 + k` (move one piece from the attacker's hand to
//!    the defender's) -- two golds contribute two `85` entries.
//!
//! Entry (1) is in the list so the shuffle consumes exactly as much randomness as
//! the TypeScript reference does, but deleting the attacker king is a **mandatory
//! pre-pass** applied before the greedy pass: rule A says a published puzzle has no
//! attacker king, so it is not something a random order may skip.  By the time the
//! shuffled order reaches that square it is already empty and the entry is a no-op.
//!
//! # Which rules each gate enforces
//!
//! Since rule D v4 (`PIPELINE_VERSION` 0.8.0, owner decision 2026-09-18) the
//! `reduce` stage has a **strict entry**: the input must already pass the full
//! acceptance test (rules A-C, F and G at the target length) or it is rejected
//! with the test's reasons, and nothing below repairs it.  Since 0.13.0 the
//! stage runs [`reduce_two_kinds`] (rule D v5), whose entry is always the strict
//! one, with two relaxations:
//!
//! | entry | admits | judged by |
//! | --- | --- | --- |
//! | strict (rule D v4) | inputs that pass the full test | every step: the full test |
//! | rule B entry (rule D v5, on by default) | inputs failing rule B alone | every step: the full test |
//! | C/G entry (`--cg-entry`, **opt-in, not part of the frozen rule D**) | inputs failing rules C and/or G alone, or together with rule B's reasons when the rule B entry is on | [`StepRule::Repair`] (the test with only those defects set aside) until a diagram passes the full test, then the full test |
//!
//! The C/G entry (owner decision 2026-10-10, `docs/experiments/ply13/README.md`
//! §13) is the repair the strict entry refuses: a pre-pass in a fixed order
//! (the inert sweep, then spare attacker hand units one at a time,
//! [`repair_pass`]) and then the shuffled passes, every step judged by the
//! repair rule until one reaches the full test.  Off by default, so the public
//! pipeline's bytes, the TypeScript reducer and `PIPELINE_VERSION` do not move;
//! its outputs are judged again by the publishing gate before anything adopts
//! them.
//!
//! The older **length-preserving reducer** [`reduce_with`] keeps a lenient entry
//! for [`crate::distill`]'s baseline only (`ReduceOptions::strict_entry`
//! `false`): the entry check then runs [`TsumeRules::off`] with rule F's
//! measurement, the attacker-king pre-pass and the entry sweep below repair the
//! input, and the greedy steps use the caller's rules.  The `reduce` stage never
//! takes that path.
//!
//! | gate of [`reduce_with`] | rules |
//! | --- | --- |
//! | entry check | `options.rules` (strict entry), or [`TsumeRules::off`] with rule F's measurement (lenient entry, `distill` only) |
//! | attacker-king pre-pass | the caller's rules, normally [`TsumeRules::puzzle`] |
//! | greedy acceptance | the same |
//! | one sweep round | the same, minus rule G -- see below |
//!
//! # Rule D v3: the inert sweep
//!
//! Rule G gives the reducer a reason to take a piece off that rule D alone would
//! have kept, but its acceptance test is all-or-nothing, so a diagram carrying
//! *two* inert pieces used to be irreducible: shedding either one still leaves the
//! other, and the step is refused.  [`sweep_inert_pieces`] is the answer.  One
//! round removes the pieces rule G reported -- all of them at once when that
//! passes, otherwise one at a time in ascending square order -- and then looks
//! again, because the inert set *grows* as material comes off: a piece that was
//! active only as the single blocker on a slider's line (rule G clause c) is inert
//! the moment that slider leaves the board.  At most [`INERT_SWEEP_ITERATIONS`]
//! rounds, so the fixpoint is never chased forever.
//!
//! Rule G is deliberately **excluded** from the test one round applies: a round's
//! result only has to be a legal, exact, minimum, unique mate respecting rules A,
//! B, C and F.  Whether the diagram it returns is finally rule-G-clean is the
//! caller's business, read off [`SweepResult::check`] without solving again.
//!
//! The sweep runs inside every greedy step, whose removal plus the sweep it makes
//! possible is judged as a whole with rule G back on -- a step that only trades
//! one inert piece for another is no step at all.  An *entry* sweep (on the
//! input, before any shuffled pass) runs only where the input may carry inert
//! pieces: in [`reduce_with`] (it is a no-op behind the strict entry, and does
//! the repair behind `distill`'s lenient one) and in the C/G entry's
//! [`repair_pass`].  [`reduce_two_kinds`] does not run it for the strict and
//! rule B entries: their inputs carry no inert piece.
//!
//! There is deliberately **no** final re-check in [`reduce_with`]: when no
//! candidate is accepted the result is the unreduced diagram, and
//! [`crate::stages::solve_stage`] is the gate that refuses to publish it.  This
//! mirrors `reduceStage` in `tools/puzzle-pipeline/src/stages.ts` step for step,
//! because the two implementations have to agree byte for byte.
//!
//! Two valleys survive rule D v3 for the full-test steps, both out of the
//! sweep's reach because neither is about an inert *board* piece.  A diagram
//! carrying two spare hand pieces cannot be repaired by full-test steps --
//! handing back the first one still leaves koma-amari, so the step is refused
//! and the second is never reached -- and neither can one whose attacker king
//! may only go together with spare material, since the mandatory pre-pass judges
//! that deletion on its own.  Behind the strict entry neither input gets in at
//! all.  [`crate::distill`] gets past both (it checks the relaxed rules while
//! stepping and the full rules on every variant it records), and so does the
//! C/G entry for the first one (each spare unit is judged with rule C set
//! aside, the full test only at the end).

use crate::canonical::{piece_label, remove_board_piece_to_defender};
use crate::exact::{
    attacker_king_square, correct_moves, designated_main_line, exact_validate_with_rules,
    main_line_key, main_line_paths, root_merged_twin, without_attacker_king, TsumeRules,
    MAIN_LINE_PATHS_CAP,
};
use crate::inert::{analyze_inert_pieces_counted, InertReport, RULE_G_BUDGET_EXHAUSTED};
use crate::jsonl::{InertPiece, RemovedPiece};
use crate::piece::{ptype, EMPTY, HAND_PIECES, PIECE_LETTERS, PT_K};
use crate::position::Position;
use crate::position::{Usi, NO_SQUARE};
use crate::rng::{Rng, RngExt, SplitMix64};
use crate::sfen::to_legacy_sfen_key;
use crate::solver::{Graph, Outcome, Solver, UselessInterposition};
use crate::strict_material;
use crate::validate::{validate_position, ValidateOptions};

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::cdfpn::ClassicalProber;

mod budget;
mod candidates;
mod chain;
mod config;
mod diagnostics;
mod gate;
mod two_kinds;
mod workers;

pub use self::{
    budget::*, candidates::*, chain::*, config::*, diagnostics::*, gate::*, two_kinds::*,
};

// ---------------------------------------------------------------------------
// Budget-out measurement (`--emit-budget-outs`, owner 2026-10-17).  Measurement only: nothing below
// changes what a check answers or what the reducer does with the answer.
// ---------------------------------------------------------------------------

/// Reducer configuration.  Defaults match `reduceStage`.
#[derive(Clone, Copy, Debug)]
pub struct ReduceOptions {
    pub attempts: u32,
    pub max_ply: u32,
    pub node_budget: u64,
    /// Rules the *final* result must satisfy.  The intermediate gates relax rules
    /// A and C as described in the module docs.
    pub rules: TsumeRules,
    /// Offer rule D entry 3, the attacker-hand-to-defender transfer.
    ///
    /// Always `true` in the `reduce` stage: rule D is frozen.  It exists so
    /// [`crate::distill`] can run the same reduction *without* the transfer and
    /// measure what rule C costs when nothing can drain the attacker's hand.
    pub hand_transfer: bool,
    /// Strict entry (rule D v4, owner decision 2026-09-18): the input must
    /// already satisfy `rules` or it is rejected up front, without asking
    /// whether removing pieces could repair it.
    ///
    /// Always `true` in the `reduce` stage.  It exists as a flag (rather than
    /// unconditional) so [`crate::distill`] keeps its lenient entry: its
    /// baseline reuses this reducer on mined mates that routinely carry yozume
    /// or spare material, and judging them publishable-before-repair would
    /// reject nearly every baseline.
    pub strict_entry: bool,
    /// Rule D's goal is "no piece the puzzle can do without", not the fewest
    /// pieces (owner 2026-09-19).  With this on, the
    /// best attempt is swept again in the frozen candidate order until a whole
    /// pass removes nothing, so the result is guaranteed 1-minimal: no single
    /// board piece or attacker hand unit can come off.  One attempt plus this
    /// sweep is the cheap way to that guarantee; more attempts only look for a
    /// smaller 1-minimal diagram.
    ///
    /// `false` in the `reduce` stage, whose bytes are frozen against the
    /// TypeScript reference.
    pub until_fixpoint: bool,
    /// Concurrent acceptance tests per record (`reduce --jobs K`, 2026-09-25,
    /// owner-approved).  `1` (the default) is the sequential path, unchanged.
    ///
    /// Above 1 the greedy chains run on a pool of `jobs` threads, each with its
    /// own [`Solver`], and independent candidate diagrams are checked
    /// concurrently: the attempts side by side, and within one chain the next
    /// few candidates speculatively.  Every check is a pure function of
    /// (diagram, target, options) -- [`Solver::solve`] clears its table per
    /// call -- and a chain commits results strictly in its own order, so the
    /// output is byte-identical for every value (`tests/reduce_jobs.rs`).
    /// Each check searches its own diagram on its worker's thread; memory is
    /// about `jobs` times one reduce's.  The CLI refuses `jobs` above 1 with
    /// a parallel prober (`--prober-threads`).  See [`workers`].
    pub jobs: usize,
}

impl Default for ReduceOptions {
    fn default() -> Self {
        Self {
            attempts: 8,
            max_ply: 11,
            node_budget: 250_000,
            rules: TsumeRules::puzzle(),
            hand_transfer: true,
            strict_entry: true,
            until_fixpoint: false,
            jobs: 1,
        }
    }
}

/// A successful reduction.
#[derive(Clone, Debug)]
pub struct Reduced {
    pub position: Position,
    pub removed: Vec<RemovedPiece>,
    pub original_board_pieces: usize,
    pub final_board_pieces: usize,
    pub target_mate_ply: u32,
    /// Attacker hand material before and after, for the record.
    pub original_attacker_hand: usize,
    pub final_attacker_hand: usize,
    /// `true` when the mandatory pre-pass removed the attacker's king.
    pub attacker_king_removed: bool,
}

/// Outcome of a reduction attempt set.
#[derive(Clone, Debug)]
pub enum ReduceResult {
    /// The input position never qualified, with `exactValidate` reasons.
    Rejected {
        reasons: Vec<String>,
    },
    Reduced(Box<Reduced>),
}

/// Supplies the random stream for each attempt.
pub trait AttemptRng {
    fn rng_for_attempt(&mut self, attempt: u32) -> &mut dyn Rng;
}

/// TypeScript-parity attempts: a fresh `SplitMix64(baseSeed + attempt)` each time.
///
/// `baseSeed + attempt` is the shape [`crate::rng::derive_stream_seed`] exists to
/// replace, and it stays here on purpose: the schedule is frozen against
/// `tools/puzzle-pipeline/src/stages.ts`, `scripts/diff-reduce.ts` holds the two
/// implementations to 0 mismatches, and the reducer draws one short attempt series
/// per record rather than stepping a base seed across runs.  Changing it is a rule D
/// change on both sides and a `PIPELINE_VERSION` bump.
pub struct SeededAttempts {
    base_seed: u64,
    current: SplitMix64,
}

impl SeededAttempts {
    pub fn new(base_seed: u64) -> Self {
        Self {
            base_seed,
            current: SplitMix64::new(base_seed),
        }
    }
}

impl AttemptRng for SeededAttempts {
    fn rng_for_attempt(&mut self, attempt: u32) -> &mut dyn Rng {
        self.current = SplitMix64::new(self.base_seed.wrapping_add(attempt as u64));
        &mut self.current
    }
}

/// Draws every attempt from one shared stream, for callers that do not need
/// TypeScript parity (mining loops, time-seeded exploration).
pub struct SharedRng<'a>(pub &'a mut dyn Rng);

impl AttemptRng for SharedRng<'_> {
    fn rng_for_attempt(&mut self, _attempt: u32) -> &mut dyn Rng {
        self.0
    }
}

/// Reduces with the TypeScript seed schedule.
pub fn reduce_seeded(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    base_seed: u64,
) -> ReduceResult {
    let mut attempts = SeededAttempts::new(base_seed);
    reduce_with(solver, position, target_mate_ply, options, &mut attempts)
}

/// Reduces using one shared random stream.
pub fn reduce(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    rng: &mut dyn Rng,
) -> ReduceResult {
    let mut attempts = SharedRng(rng);
    reduce_with(solver, position, target_mate_ply, options, &mut attempts)
}

/// Reduces with a caller-supplied per-attempt random source.
///
/// This is the **length-preserving** reducer of rule D v3 / v4: every step is
/// judged by the acceptance test alone, and the best attempt wins.  Since
/// `PIPELINE_VERSION` 0.13.0 the `reduce` stage runs [`reduce_two_kinds`]
/// instead; this one stays for [`crate::distill`], whose baseline reuses it with
/// a lenient entry, and for the tests that pin its bytes.
pub fn reduce_with(
    solver: &mut Solver,
    position: &Position,
    target_mate_ply: Option<u32>,
    options: &ReduceOptions,
    attempts: &mut dyn AttemptRng,
) -> ReduceResult {
    let attacker = position.side;
    // Strict entry (rule D v4, owner decision 2026-09-18): the structural gate
    // runs before any search, so an impossible diagram never reaches the
    // solver, and the composition gate then demands the input already be a
    // publishable puzzle.  Both mirror `check_reduction`, which every greedy
    // step passes through.  `strict_entry` is off only for callers that reuse
    // this reducer as a repair-capable baseline ([`crate::distill`]).
    let structural = validate_position(position, ValidateOptions::pipeline());
    if !structural.is_empty() {
        return ReduceResult::Rejected {
            reasons: structural.into_iter().map(|issue| issue.code).collect(),
        };
    }
    // Rule B is enforced at every length (`PIPELINE_VERSION` 0.12.0), so a
    // removal that would create yozume is refused by every gate below, and a
    // diagram that already carries yozume fails the strict entry.  This is what
    // `--strict-rule-b` (P8-016, owner decision 2026-10-04) did for a yozume-free
    // input; the flag became the default and was removed.
    // Entry check only, so the composition rules used to be deliberately off:
    // shedding an attacker king or spare hand material was exactly the repair
    // performed below.  Rule F is not a rejection rule but a measurement, so it
    // stays on: the length recorded as `targetMatePly` must be the one the
    // acceptance test will measure, or every reduction step would report
    // `target_mate_ply_changed`.
    //
    // Rule F rides with the composition rules (`TsumeRules::useless_mode`), so
    // the strict entry passes `options.rules` straight through: rule D v4 keeps
    // the measurement the acceptance test uses.
    let entry_rules = if options.strict_entry {
        options.rules
    } else {
        TsumeRules::off().allowing_useless_interposition(Some(options.rules.useless_mode()))
    };
    let entry = entry_verdict(
        solver,
        position,
        target_mate_ply,
        entry_rules,
        options,
        false,
    );
    let Some(initial_distance) = entry.distance else {
        return ReduceResult::Rejected {
            reasons: entry.reasons,
        };
    };
    if !entry.reasons.is_empty() {
        return ReduceResult::Rejected {
            reasons: entry.reasons,
        };
    }
    let target = target_mate_ply.unwrap_or(initial_distance);
    let original_board_pieces = position.board_piece_count();
    let original_attacker_hand = attacker_hand_total(position, attacker);

    // Mandatory pre-pass, before any shuffled pass: an orthodox diagram carries no
    // attacker king, so deleting it is attempted first and its failure is fatal.
    let mut start = position.clone();
    let mut pre_pass: Vec<RemovedPiece> = Vec::new();
    let mut attacker_king_removed = false;
    if let Some(square) = attacker_king_square(position, attacker) {
        let label = piece_label(position, square);
        let kingless = without_attacker_king(position, attacker).expect("attacker king");
        let stage = Stage::enter(
            "king",
            None,
            vec![RemovedPiece {
                square,
                piece: label.clone(),
            }],
        );
        let accepted = accepted_reduction(solver, &kingless, target, options);
        drop(stage);
        if accepted {
            start = kingless;
            pre_pass.push(RemovedPiece {
                square,
                piece: label,
            });
            attacker_king_removed = true;
        } else if !options.rules.allow_attacker_king {
            return ReduceResult::Rejected {
                reasons: vec!["attacker_king_required".to_string()],
            };
        }
    }

    // Rule D v3: the entry sweep.  A mined diagram is *made* of pieces the mate
    // does not need, and rule G's all-or-nothing acceptance test would refuse every
    // single step of such a source; sweeping once here puts the greedy pass on a
    // diagram whose inert material has already gone.
    let entry_check = {
        let _stage = Stage::enter("start", None, Vec::new());
        check_reduction(solver, &start, target, options, false)
    };
    let entry_sweep = sweep_inert_pieces(
        solver,
        &start,
        target,
        options,
        entry_check,
        &StepRule::Length,
    );
    start = entry_sweep.position;
    pre_pass.extend(entry_sweep.removed);

    // The candidate list is derived from the *pre-reduction* diagram, so its
    // length -- and with it the shuffle -- never depends on what the pre-pass or
    // the entry sweep already took off.
    let mut candidates = reduction_candidates(position, attacker);
    if !options.hand_transfer {
        candidates.retain(|candidate| !matches!(candidate, Candidate::Hand(_)));
    }
    let start_state = ChainState::new(start, pre_pass, StepRule::Length, None, true);
    let best = if options.jobs > 1 {
        // The shuffles consume the random stream exactly as the sequential loop
        // does -- one per attempt, in attempt order, independent of any result
        // -- so drawing them all up front leaves a shared stream in the same
        // state and gives every attempt the same order.
        let orders: Vec<Vec<Candidate>> = (0..options.attempts)
            .map(|attempt| attempts.rng_for_attempt(attempt).shuffle(&candidates))
            .collect();
        workers::with_workers(solver, target, options, attacker, |runner| {
            let results = runner.passes(
                orders
                    .iter()
                    .map(|order| (order.as_slice(), start_state.clone()))
                    .collect(),
            );
            // The sequential fold, in attempt order.
            let mut best = start_state.clone();
            for (state, _) in results {
                if beats_best(&state.board, &best.board, attacker) {
                    best = state;
                } else {
                    best.inherit_settled(&state);
                }
            }
            if options.until_fixpoint {
                best = runner.sweep_to_fixpoint(&candidates, vec![best]).remove(0);
            }
            best
        })
    } else {
        let mut best = start_state.clone();
        for attempt in 0..options.attempts {
            let order = attempts.rng_for_attempt(attempt).shuffle(&candidates);
            let mut candidate = start_state.clone();
            greedy_pass(solver, &mut candidate, &order, target, options, attacker);
            if beats_best(&candidate.board, &best.board, attacker) {
                best = candidate;
            } else {
                best.inherit_settled(&candidate);
            }
        }
        if options.until_fixpoint {
            // A step refused early in a pass may become possible once later pieces
            // are gone, so one pass alone does not prove 1-minimality; repeat until
            // a pass takes nothing.  Each pass removes at least one piece or ends
            // the loop, so it runs at most once per candidate plus one.
            while greedy_pass(solver, &mut best, &candidates, target, options, attacker) {}
        }
        best
    };
    ReduceResult::Reduced(Box::new(Reduced {
        original_board_pieces,
        final_board_pieces: best.board.board_piece_count(),
        original_attacker_hand,
        final_attacker_hand: attacker_hand_total(&best.board, attacker),
        attacker_king_removed,
        position: best.board,
        removed: best.removed,
        target_mate_ply: target,
    }))
}
