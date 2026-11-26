//! Tsume generation from a mating position (spec 9.1.7, spec 10).
//!
//! `distill` takes any position in which the side to move has a forced mate and
//! searches the space of *variants* of that position for strict-valid tsume
//! puzzles.  It is a generalisation of the reducer in [`crate::reduce`]: the
//! reducer only ever deletes a board piece and hands it to the defender, while
//! distillation also decides **who keeps the absent material**.
//!
//! That extra degree of freedom is the point.  Under the frozen strict convention
//! (`docs/rules.md`) every non-king piece missing from the board and from the
//! attacker's hand goes to the *defender's* hand, so a fully stocked defender can
//! interpose or capture its way out of almost every distant check.  Empirically
//! this makes 5-ply-and-longer mates extremely rare in random search.  Therefore
//! the allocation of absent material is a first-class variant operator here, not
//! an afterthought:
//!
//! | operator | effect | bound |
//! | --- | --- | --- |
//! | a `remove-to-defender` | delete a board piece, defender gains it (the reducer's step) | none |
//! | b `keep-in-attacker-hand` | delete a board piece, the **attacker** gains it in hand | `max_attacker_hand` |
//! | c `hand-to-attacker` | move one piece from the defender's implicit hand to the attacker's hand | `max_attacker_hand` |
//! | d `hand-to-defender` | move one piece from the attacker's hand to the defender's | none |
//! | e `delete-attacker-king` | remove the attacker's king (rule A), a mandatory pre-pass | once |
//! | f `relocate` | move one board piece to another square; material is untouched | `max_relocate` (**`distill-v2`, off by default**), and only on the rule-C repair path |
//!
//! Operators b and c both deny a piece to the defender and simultaneously give the
//! attacker a drop, which is exactly what rescues long forcing lines.  They are
//! bounded because an attacker holding half a shogi set produces brute-force
//! puzzles, not elegant ones.  Operator d is their inverse and exists for rule C:
//! a puzzle may not end with material still in the attacker's hand, so draining
//! the hand is as much a reduction as clearing the board.  Operator e is not
//! optional -- rule A says a published diagram has no attacker king.
//!
//! # Operator f, and why rule C needs it (`reports/distill-rule-c-2026-09-14.md`)
//!
//! Operators a-e conserve one quantity exactly.  Write the attacker's hand at the
//! terminal mate of the designated main line as
//!
//! ```text
//! leftover = root hand + captures on the line - drops on the line
//! ```
//!
//! Rule C says `leftover == 0`.  Operators b, c and d move material between the
//! root hand and the defender's, and the *tightening* pass below drains whatever
//! the line does not spend, so between them they settle the `root hand` term.  They
//! cannot touch `captures`: which squares the mating line captures on is a property
//! of the geometry, and a-e never move a piece between squares.  Measured on the
//! 64 nine-ply and 15 eleven-ply mates this repository has mined, `captures`
//! dominates -- removing the root hand would leave 55 of the 79 sources still
//! failing rule C, removing the captures would leave 25 -- and the captured
//! material is overwhelmingly ordinary defender *board* pieces (194 of 203
//! captures), not interposed 合駒.
//!
//! Operator f is the answer to that term, and it is deliberately *targeted* rather
//! than offered to the greedy chain: a blind "move any piece anywhere" operator
//! multiplies the candidate list by an order of magnitude and crowds out the
//! reduction operators that actually shrink a diagram.  [`repair_rule_c`] runs only
//! on a variant that already satisfies rules A, B and G and fails **only** rule C,
//! reads the squares its own main line captures on, and tries moving each of those
//! pieces to a bounded set of empty squares ([`relocate_destinations`]).  Every
//! result is put through the same full exact gate as any other variant.
//!
//! # `distill-v1` is the default; `distill-v2` is opt-in
//!
//! Operator f and the rule-B step relaxation below are both **off** unless asked
//! for ([`DistillOptions::max_relocate`], [`DistillOptions::relax_step_yozume`];
//! `--distill-v2` turns on both).  With both off this module takes the same search
//! decisions and writes the same bytes `distill-v1` did for the same arguments,
//! down to the absent JSON keys and the `distill-v1` version string
//! ([`DistillOptions::version`]).  `reports/distill-rule-c-2026-09-14.md` §3.5 is
//! the byte-for-byte check against a binary built from the base commit.
//!
//! # The frozen rules
//!
//! Acceptance is [`crate::exact::TsumeRules`].  Rules A and C are relaxed on the
//! *intermediate* steps of a chain and enforced on every recorded variant, for the
//! reason spelled out in [`crate::reduce`]: an operator that drains one piece at a
//! time can never take its first step if the invariant it is working towards is
//! also its entry condition.
//!
//! Every accepted variant is a strict-valid puzzle in the full
//! [`crate::exact::exact_validate`] sense: forced mate, unique minimum first move,
//! minimum length proven by re-solving one ply shorter, complete defence coverage
//! and structural validity.  Variants are ranked by [`crate::quality`] and the
//! best few are emitted as solved-stage `PipelineRecord`s that the TypeScript
//! `compile` stage consumes unchanged.
//!
//! # The gate at the mate length (2026-10-16)
//!
//! A record is published from a solve two plies wider than its mate when
//! `--max-ply` allows it ([`score_variant`]), and rule D runs at that bound too.
//! Rules A-C and F do not depend on the bound, but rule G does: it reads the
//! defender's recorded refutation of every wrong check, and which reply that is
//! depends on how many plies the bound leaves (`exact_validate_with_rules`).  The
//! catalogue judges every row at its own mate length (`solve --max-ply
//! <matePly>`), so a record the wider solve accepts can carry a piece rule G calls
//! inert there.  `docs/experiments/distill-rule-g/README.md` is the case that
//! found it: a 7-ply record of `distill --max-ply 9` whose defender knight on 8a
//! refutes a wrong check only at bound 9.
//!
//! Every accepted record therefore also passes the gate at its mate length, at
//! the run's node budget, and with rule D on (the default) is minimal there: no
//! main-line-preserving step of rule D is accepted from it at that bound.
//!
//! * The ranking takes a variant scored at the wider bound only if the gate at
//!   its mate length accepts it as well ([`mate_length_gate`]).  For the baseline
//!   reducer's result, which is recorded among the variants unjudged (its lenient
//!   entry may hand back a diagram judged with rule G set aside), the scoring
//!   and this gate are the first judgements by the final rules.
//! * A record stays at the wider bound only when it is accepted there and passes
//!   [`mate_length_verdict`]; otherwise the variant is published again at the
//!   mate length (`distill.mateLengthGate` says why) and written as a rejection
//!   when it fails there.
//! * A record published at its mate length whose rule D could not be finished
//!   there (a budget) is written as a rejection of stage `reduce`.
//! * `--emit-record` sees every record as judged at its mate length.
//!
//! With `reduce` off (library only; the CLI's `--no-reduce` went on 2026-10-22)
//! nothing is checked for minimality.  A record that passes is
//! written exactly as before.
//!
//! Search is randomised (K greedy orders over a shuffled operator list; the
//! optional beam search, never used, went on 2026-10-27 and `"beam"` is
//! recorded as `0`), but every emitted record carries the seed and all
//! parameters in `provenance.distill`, so any single result can be reproduced
//! exactly.

use anyhow::Result;
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::canonical::canonicalize;
use crate::dfpn::{ProbeEngine, ProbeOptions, Prober, Verdict};
use crate::exact::{
    attacker_king_square, dfpn_disproof_is_conclusive, dfpn_useless_mode,
    exact_validate_with_rules, minimum_mate_validate_with_rules, without_attacker_king,
    ExactValidation, MainLine, TsumeRules,
};
use crate::features;
use crate::jsonl::{PipelineRecord, Rejection};
use crate::piece::*;
use crate::position::Position;
use crate::quality::{self, Quality, QualityInput, QualityWeights};
use crate::reduce::{self, ReduceOptions, ReduceResult};
use crate::rng::{RngExt, SplitMix64};
use crate::sfen::{legacy_sfen_key_of, square_to_usi, to_sfen};
use crate::solver::{Solver, UselessInterposition};
use crate::stages::{
    reduce_accepted, solve_stage, solve_stage_reported, validate_stage, ReduceAcceptedOutcome,
    SolveStageOptions,
};
use crate::strict_material;
use crate::util::sha256_text;
use crate::validate::{validate_position, ValidateOptions};

/// Recorded in `provenance.distill.version` and in the `lineage` entry, and the
/// generator version of a bare-SFEN input record.
///
/// **`distill-v1` is still the default.** The `distill-v2` behaviour -- rule B
/// relaxed on chain steps, and operator f, the rule-C repair -- is opt-in behind
/// [`DistillOptions::relax_step_yozume`] and [`DistillOptions::max_relocate`],
/// and [`DistillOptions::version`] stamps `distill-v2` exactly when one of them is
/// on.  With neither, every byte this module writes is the byte `distill-v1`
/// wrote for the same arguments: the search takes the same decisions, the extra
/// counters and JSON keys are suppressed, and the version string does not move.
pub const DISTILL_VERSION: &str = "distill-v1";

/// The version stamped when any `distill-v2` path is enabled; see
/// [`DISTILL_VERSION`] and `reports/distill-rule-c-2026-09-14.md`.
pub const DISTILL_VERSION_V2: &str = "distill-v2";

/// Which variant operator was applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// Operator a: the removed piece goes to the defender's hand.
    RemoveToDefender,
    /// Operator b: the removed piece goes to the attacker's hand.
    KeepInAttackerHand,
    /// Operator c: one piece moves from the defender's hand to the attacker's.
    HandToAttacker,
    /// Operator d: one piece moves from the attacker's hand to the defender's.
    HandToDefender,
    /// Operator e: the attacker's king leaves the board (rule A).
    DeleteAttackerKing,
    /// Operator f: one board piece moves to another square; no hand changes.
    Relocate,
}

impl OpKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            OpKind::RemoveToDefender => "remove-to-defender",
            OpKind::KeepInAttackerHand => "keep-in-attacker-hand",
            OpKind::HandToAttacker => "hand-to-attacker",
            OpKind::HandToDefender => "hand-to-defender",
            OpKind::DeleteAttackerKing => "delete-attacker-king",
            OpKind::Relocate => "relocate",
        }
    }
    /// `true` when the operator consumes the attacker-hand budget.
    pub const fn uses_attacker_hand(self) -> bool {
        matches!(self, OpKind::KeepInAttackerHand | OpKind::HandToAttacker)
    }
    /// `true` when the operator names a board square rather than a piece type.
    pub const fn is_square_op(self) -> bool {
        matches!(
            self,
            OpKind::RemoveToDefender
                | OpKind::KeepInAttackerHand
                | OpKind::DeleteAttackerKing
                | OpKind::Relocate
        )
    }
}

/// One operator applied to a variant, with the piece it moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedOp {
    pub kind: OpKind,
    /// The board square, for operators a, b, e and f (operator f's *origin*).
    pub square: Option<usize>,
    /// The destination square of operator f, and `None` for every other operator.
    pub to: Option<usize>,
    /// Base piece type moved (`PT_*`).
    pub piece: u8,
    /// `true` when the piece belonged to the defender.
    pub defender_piece: bool,
}

impl AppliedOp {
    /// The `provenance.transformations` entry, e.g. `distill:remove-to-defender:7c`,
    /// `distill:keep-in-attacker-hand:G` or `distill:relocate:7c-6d`.
    pub fn transformation(&self) -> String {
        match self.kind {
            OpKind::RemoveToDefender | OpKind::DeleteAttackerKing => format!(
                "distill:{}:{}",
                self.kind.as_str(),
                self.square.map(square_to_usi).unwrap_or_default()
            ),
            OpKind::Relocate => format!(
                "distill:{}:{}-{}",
                self.kind.as_str(),
                self.square.map(square_to_usi).unwrap_or_default(),
                self.to.map(square_to_usi).unwrap_or_default()
            ),
            OpKind::KeepInAttackerHand | OpKind::HandToAttacker | OpKind::HandToDefender => {
                format!(
                    "distill:{}:{}",
                    self.kind.as_str(),
                    PIECE_LETTERS[self.piece as usize] as char
                )
            }
        }
    }

    /// The `distill.operators[]` entry.  `to` / `toUsi` appear only for operator f,
    /// which is the only operator that has a destination -- so a `distill-v1` run,
    /// which cannot produce one, writes exactly the keys it always wrote.
    fn to_json(self) -> Value {
        let mut value = json!({
            "op": self.kind.as_str(),
            "square": self.square,
            "squareUsi": self.square.map(square_to_usi),
            "piece": (PIECE_LETTERS[self.piece as usize] as char).to_string(),
            "defenderPiece": self.defender_piece,
        });
        if let Some(to) = self.to {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("to".to_string(), json!(to));
            map.insert("toUsi".to_string(), json!(square_to_usi(to)));
        }
        value
    }
}

/// A candidate operator, before the moved piece is known.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    kind: OpKind,
    square: Option<usize>,
    /// Operator f's destination square.
    to: Option<usize>,
    piece: u8,
}

/// `distill` configuration.
#[derive(Clone, Copy, Debug)]
pub struct DistillOptions {
    /// Effective seed of this source.  Every randomised decision derives from it.
    pub seed: u64,
    /// The seed the caller passed on the command line.  [`distill_record`] derives
    /// a per-record `seed` from it so one `--seed` reproduces a whole stream *and*
    /// each record on its own; both are recorded.
    pub base_seed: u64,
    /// Randomised greedy orders over the operator list (spec 10.4).
    pub attempts: u32,
    /// Distinct accepted puzzles emitted per source, best first.
    pub max_variants: usize,
    /// Upper bound on operator b + c applications along one chain.
    pub max_attacker_hand: usize,
    /// Relax rule B on chain *steps* ([`DistillOptions::step_rules`]).  `false`
    /// reproduces `distill-v1`'s stepping, which enforced B at every step.
    pub relax_step_yozume: bool,
    /// Upper bound on operator f (relocate) applications along one chain.  `0`
    /// switches the rule-C repair off entirely and reproduces `distill-v1`'s
    /// variant set.
    pub max_relocate: usize,
    /// Upper bound on *distinct variants* the rule-C repair is attempted on per
    /// source.  Each attempt costs one graph solve plus at most
    /// `|captures| x |destinations|` validations, so this is what keeps a noisy
    /// diagram whose every chain ends in koma-amari from spending the whole
    /// per-source budget on repairs.
    pub max_repairs: u64,
    /// Rule A opt-out: keep the attacker's king on the board.
    pub allow_attacker_king: bool,
    /// Offer operator d (attacker hand to defender).  Turning it off is what the
    /// generator comparison measures rule C against.
    pub hand_to_defender: bool,
    /// Accept variants whose minimum mate length differs from the source
    /// (`--allow-length-change`, off by default).
    pub allow_length_change: bool,
    pub max_ply: u32,
    /// Node budget of the decisive searches: the initial mate probe and the solve
    /// behind every emitted record.
    pub node_budget: u64,
    /// Node budget of one inner-loop trial.  Deliberately smaller than
    /// `node_budget`: a trial that cannot be resolved cheaply is simply not
    /// explored, which costs recall, never correctness.
    pub search_node_budget: u64,
    /// Upper bound on search nodes spent exploring one source.  This is the knob
    /// that keeps a noisy 14-piece diagram with a 5-ply mate from eating minutes:
    /// cheap sources get every attempt, expensive ones get truncated.
    pub max_nodes: u64,
    /// Distinct variants that get a full graph solve for ranking.
    pub max_scored: usize,
    /// Probe each board piece of an emitted variant for provable irrelevance
    /// (spec 10.6 compactness).
    pub irrelevance_probe: bool,
    /// Upper bound on exact validations spent searching one source.  Reached only
    /// by large noisy diagrams, where it trades a little search for a predictable
    /// cost per record.
    pub max_validations: u64,
    /// Which engine answers the per-step "is there still a mate of exactly this
    /// length" question ([`crate::dfpn`]).  [`ProbeEngine::Exact`] is the default
    /// and reproduces every earlier run byte for byte; [`ProbeEngine::Dfpn`] puts
    /// the df-pn prober in front of the exact gate, which decides nothing on its
    /// own but rejects a dead end without paying for a full exact search and finds
    /// the source's length on a board the exact solver cannot finish.
    pub probe: ProbeEngine,
    /// Node budget of one **decisive** df-pn probe: the source's minimum mate
    /// length.  Ignored with [`ProbeEngine::Exact`].
    ///
    /// One inner-loop probe gets `min(dfpn_node_budget, search_node_budget)`
    /// instead, for exactly the reason [`DistillOptions::search_node_budget`]
    /// exists: an inner-loop trial that cannot be settled cheaply is not worth
    /// settling, and a probe that runs to a million nodes on every one of a
    /// thousand candidates would spend the whole per-source `max_nodes` budget on
    /// the pre-filter instead of on the search it is supposed to protect.
    pub dfpn_node_budget: u64,
    pub weights: QualityWeights,
    /// Rule D behind the variant search: every emitted variant goes through the
    /// standard `reduce` stage (strict entry) under its own solve bound, budget
    /// and final rules, and is re-solved when pieces came off.  Two variants of
    /// one source that reduce to the same diagram are emitted once.
    ///
    /// The search above only drops pieces along its randomised chains and ranks
    /// by quality, where a removable piece costs a little score but is never a
    /// rejection; measured on the catalog (2026-09-19) half the 5- and 7-ply
    /// distill output reduced further.  Off (library only since 2026-10-22,
    /// when the CLI's `--no-reduce` went) reproduces the earlier bytes.
    pub reduce: bool,
    /// Prestrip before the variant search ([`crate::prestrip`], `--prestrip`,
    /// 2026-10-01).  Off by default.  Right after the rule-A pre-pass, board pieces
    /// are deleted farthest from the defender king first while the base keeps a
    /// mate of exactly the source length (df-pn first under `--probe dfpn`, the
    /// exact solver otherwise, all charged to `max_nodes`), and the variant search
    /// starts from what is left.  Each deletion is recorded as an ordinary
    /// `remove-to-defender` operator, and every recorded variant still passes the
    /// same final gate.
    pub prestrip: bool,
    /// Which pieces prestrip tries, in which order ([`crate::prestrip::Order`],
    /// default `attacker-first` since 2026-10-02; the CLI's `--prestrip-order`
    /// went on 2026-10-22).  Ignored without `prestrip`.  Recorded in the
    /// options JSON unless it is the original distance order, which reproduces
    /// the bytes of earlier `--prestrip` runs.
    pub prestrip_order: crate::prestrip::Order,
}

impl Default for DistillOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            base_seed: 0,
            attempts: 16,
            max_variants: 3,
            max_attacker_hand: 3,
            // Both off: the default run is `distill-v1`, byte for byte.
            relax_step_yozume: false,
            max_relocate: 0,
            max_repairs: 48,
            allow_attacker_king: false,
            hand_to_defender: true,
            allow_length_change: false,
            max_ply: crate::exact::DEFAULT_MAX_PLY,
            node_budget: 250_000,
            search_node_budget: 60_000,
            max_nodes: 4_000_000,
            max_scored: 12,
            irrelevance_probe: true,
            max_validations: 4_000,
            probe: ProbeEngine::Exact,
            dfpn_node_budget: 1_000_000,
            weights: quality::DEFAULT_WEIGHTS,
            reduce: true,
            prestrip: false,
            prestrip_order: crate::prestrip::Order::AttackerFirst,
        }
    }
}

impl DistillOptions {
    fn reduce_options(&self, target: u32) -> ReduceOptions {
        ReduceOptions {
            keep: crate::reduce::KeepSquares::NONE,
            attempts: 1,
            // The target length is the tightest exhaustive bound for the reducer's
            // acceptance test, exactly as in [`Search::accept`].
            max_ply: target,
            node_budget: self.search_node_budget,
            rules: self.final_rules(),
            hand_transfer: self.hand_to_defender,
            // The distill baseline reuses the reducer as a repair-capable pass
            // over mined mates: keep the lenient pre-v4 entry here.
            strict_entry: false,
            until_fixpoint: false,
            jobs: 1,
        }
    }

    /// Node budget of one inner-loop df-pn probe; see
    /// [`DistillOptions::dfpn_node_budget`].
    fn inner_dfpn_node_budget(&self) -> u64 {
        self.dfpn_node_budget.min(self.search_node_budget)
    }

    /// Rules every *recorded* variant must satisfy: `TsumeRules::puzzle()` --
    /// A, B, C and G -- exactly as `distill-v1` judges them, rule B at every mate
    /// length since `PIPELINE_VERSION` 0.12.0 (what `--strict-rule-b`, P8-016, did
    /// before it became the default and was removed).  The reductions run under
    /// these rules too, so reducing a yozume-free diagram never creates yozume.
    /// `distill-v2` relaxes rule B on chain *steps* only
    /// ([`DistillOptions::step_rules`]); nothing a flag can set moves this gate.
    pub fn final_rules(&self) -> TsumeRules {
        TsumeRules::puzzle().allowing_attacker_king(self.allow_attacker_king)
    }

    /// Rules used while stepping: A, C and G relaxed so the king, hand and
    /// removal operators can make progress, and -- since `distill-v2` -- B as
    /// well, for the reason measured in `reports/distill-rule-c-2026-09-14.md` §2
    /// and spelled out on [`TsumeRules::allow_yozume`]: a mined long mate carries
    /// yozume on its own main line (15 of 15 at eleven plies, 58 of 64 at nine),
    /// and thinning the diagram is what removes it, so a chain gated on rule B
    /// cannot take its first step.  Every *recorded* variant is still gated on
    /// [`DistillOptions::final_rules`], rule B included.
    ///
    /// Opt-in: `--distill-v2`.  Off by default, which is
    /// the `distill-v1` behaviour.
    pub fn step_rules(&self) -> TsumeRules {
        TsumeRules::candidate().allowing_yozume(self.relax_step_yozume)
    }

    /// `true` when any `distill-v2` path is enabled.
    pub fn is_v2(&self) -> bool {
        self.relax_step_yozume || self.max_relocate > 0
    }

    /// The version string this configuration stamps: [`DISTILL_VERSION_V2`] when a
    /// v2 path is on, [`DISTILL_VERSION`] otherwise.
    pub fn version(&self) -> &'static str {
        if self.is_v2() {
            DISTILL_VERSION_V2
        } else {
            DISTILL_VERSION
        }
    }

    /// The parameter block recorded in `provenance.distill`.
    ///
    /// The df-pn keys appear **only** under `--probe dfpn`, and the v2 keys only
    /// under a v2 configuration, so a default run keeps producing the very bytes it
    /// produced before either existed.
    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "version": self.version(),
            "seed": self.seed.to_string(),
            "baseSeed": self.base_seed.to_string(),
            "attempts": self.attempts,
            // The beam search went on 2026-10-27; the key stays at its only value.
            "beam": 0,
            "maxVariants": self.max_variants,
            "maxAttackerHand": self.max_attacker_hand,
            "allowAttackerKing": self.allow_attacker_king,
            "handToDefender": self.hand_to_defender,
            "allowLengthChange": self.allow_length_change,
            "maxPly": self.max_ply,
            "nodeBudget": self.node_budget,
            "searchNodeBudget": self.search_node_budget,
            "maxNodes": self.max_nodes,
            "maxScored": self.max_scored,
            "irrelevanceProbe": self.irrelevance_probe,
            "maxValidations": self.max_validations,
        });
        if self.probe.is_dfpn() {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("probe".to_string(), json!(self.probe.as_str()));
            map.insert("dfpnNodeBudget".to_string(), json!(self.dfpn_node_budget));
            // (`"yozumePrefilter": true` was written here when the rule B
            // pre-filter was on; it went on 2026-10-27, its CLI flag on
            // 2026-10-22.)
        }
        if self.reduce {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("reduce".to_string(), json!(true));
        }
        if self.prestrip {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("prestrip".to_string(), json!(true));
            if self.prestrip_order != crate::prestrip::Order::Distance {
                map.insert(
                    "prestripOrder".to_string(),
                    json!(self.prestrip_order.as_str()),
                );
            }
        }
        if self.is_v2() {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("relaxStepYozume".to_string(), json!(self.relax_step_yozume));
            map.insert("maxRelocate".to_string(), json!(self.max_relocate));
            map.insert("maxRepairs".to_string(), json!(self.max_repairs));
        }
        value
    }
}

/// One strict-valid variant found by the search.
#[derive(Clone, Debug)]
pub struct Variant {
    /// Strict-normalised, attacker to move, ready to publish.
    pub position: Position,
    /// The attacker's explicit hand; the defender holds the remainder.
    pub attacker_hand: [u8; 7],
    pub ops: Vec<AppliedOp>,
    pub extra_attacker: usize,
    /// Operator f applications along the chain that produced this variant.
    pub relocations: usize,
    pub mate_ply: u32,
    /// Ply bound used for the published solve; see [`score_variant`].
    pub solve_bound: u32,
    /// Attack nodes of the published main line with more than one correct move
    /// (root included), and the largest such count.  Both `0` for a rule-B-clean
    /// puzzle; they are what `--relax-rules b` is measured with.
    pub branching_nodes: usize,
    pub max_correct: usize,
    /// SHA-256 canonical hash, the dedup key.
    pub canonical_hash: String,
}

impl Variant {
    fn uses_attacker_hand(&self) -> bool {
        self.ops.iter().any(|op| op.kind.uses_attacker_hand())
    }
}

/// Outcome of distilling one position.
#[derive(Clone, Debug)]
pub enum DistillResult {
    /// The side to move has no forced mate within the ply bound.
    NoMate { reasons: Vec<String> },
    /// Rule A: the mate exists only while the attacker's king is on the board.
    AttackerKingRequired { source_mate_ply: u32 },
    /// A mate exists but no variant (including the source) is strict-valid.
    ///
    /// The counters ride along because this is the *interesting* outcome of a long
    /// mate: `distill-v1` published `distill.ruleRejections` only on records it
    /// accepted, which is exactly the population that does not answer "which rule
    /// is holding 9-ply and 11-ply back".
    NoValidVariant {
        source_mate_ply: u32,
        rule_rejections: RuleRejections,
        repair: RepairStats,
        validations: u64,
        search_nodes: u64,
    },
    Distilled {
        source_mate_ply: u32,
        source_board_pieces: usize,
        /// Best first.
        variants: Vec<(Variant, Quality)>,
        /// Distinct strict-valid variants found before the emit cap.  The
        /// baseline reduction's result is counted before any gate judged it (its
        /// lenient entry can hand back a diagram rule G was never asked about),
        /// so this can count one variant the ranking then refuses.
        distinct: usize,
        /// Final board piece count of the plain [`crate::reduce`] baseline.
        baseline_board_pieces: Option<usize>,
        /// Exact validations spent on this source.
        validations: u64,
        /// Search nodes spent on this source.
        search_nodes: u64,
        /// How often each frozen rule refused a trial variant of this source.
        rule_rejections: RuleRejections,
        /// What operator f did on this source.
        repair: RepairStats,
        /// The attacker king was on the source board and the pre-pass removed it.
        attacker_king_removed: bool,
    },
}

/// How often each of the frozen rules refused a trial variant.
///
/// This is what answers "did rule C cost us puzzles, and did the hand-transfer
/// operator win them back": run the same sources with and without operator d and
/// compare `koma_amari` here against the accepted count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuleRejections {
    /// Rule A, `attacker_king_present`.
    pub attacker_king_present: u64,
    /// Rule B, `yozume_in_main_line`.
    pub yozume_in_main_line: u64,
    /// Rule C, `koma_amari`.
    pub koma_amari: u64,
    /// Pieces left in the attacker's hand, summed over every rule-C refusal.
    pub koma_amari_leftover: u64,
    /// The part of that leftover a capture on the refused main line put there,
    /// `min(leftover, captures)` per refusal.  The rest came from the variant's
    /// root hand, which operators b, c, d and the tightening pass can reach and
    /// which operator f cannot help with.  This split is what
    /// `reports/distill-rule-c-2026-09-14.md` §2 measures the operator set against.
    pub koma_amari_from_captures: u64,
    /// Rule G, `inert_attacker_piece` or `inert_defender_piece` (either counts
    /// once per refusal).
    pub inert_piece: u64,
}

impl RuleRejections {
    /// Records one refused trial.  `position` is the diagram the validation was
    /// run on, needed to replay the main line for the rule-C split.
    fn observe(&mut self, validation: &ExactValidation, position: &Position) {
        let mut inert = false;
        for reason in validation.reasons.iter() {
            match reason.as_str() {
                "attacker_king_present" => self.attacker_king_present += 1,
                "yozume_in_main_line" => self.yozume_in_main_line += 1,
                "koma_amari" => {
                    self.koma_amari += 1;
                    if let Some(line) = validation.main_line.as_ref() {
                        let captures = main_line_capture_squares(position, line).len() as u64;
                        let leftover = line.leftover_hand as u64;
                        self.koma_amari_leftover += leftover;
                        self.koma_amari_from_captures += leftover.min(captures);
                    }
                }
                "inert_attacker_piece" | "inert_defender_piece" => inert = true,
                _ => {}
            }
        }
        if inert {
            self.inert_piece += 1;
        }
    }

    /// The `distill.ruleRejections` block.  The three `distill-v2` counters are
    /// written only under a v2 configuration, so a v1 run's accepted records keep
    /// the exact three keys they always had.
    fn to_json(self, v2: bool) -> Value {
        let mut value = json!({
            "attackerKingPresent": self.attacker_king_present,
            "yozumeInMainLine": self.yozume_in_main_line,
            "komaAmari": self.koma_amari,
        });
        if v2 {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert(
                "komaAmariLeftover".to_string(),
                json!(self.koma_amari_leftover),
            );
            map.insert(
                "komaAmariFromCaptures".to_string(),
                json!(self.koma_amari_from_captures),
            );
            map.insert("inertPiece".to_string(), json!(self.inert_piece));
        }
        value
    }
}

/// How much work the rule-C repair (operator f) did on one source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepairStats {
    /// Distinct variants the repair looked at.
    pub considered: u64,
    /// Of those, the ones that fail rule C **and nothing else**, so the repair
    /// actually had a main line to read.
    pub eligible: u64,
    /// Relocations tried.
    pub relocations: u64,
    /// Relocations that produced a strict-valid puzzle.
    pub accepted: u64,
}

impl RepairStats {
    fn to_json(self) -> Value {
        json!({
            "considered": self.considered,
            "eligible": self.eligible,
            "relocations": self.relocations,
            "accepted": self.accepted,
        })
    }
}

/// The squares the attacker captures on along `line`, in main-line order.
///
/// A capture is what puts material into the attacker's hand, so this is the term
/// of the rule-C balance that no operator but f can reach: which square the mating
/// move lands on is geometry, and operators a-e never move a piece between squares.
/// Kings are skipped; a mate line never captures one, and the hand encoding has no
/// slot for it.
fn main_line_capture_squares(position: &Position, line: &MainLine) -> Vec<usize> {
    let mut walk = position.clone();
    let mut out = Vec::new();
    for (index, usi) in line.moves.iter().enumerate() {
        let Ok(m) = crate::position::Move::from_usi(usi.as_str()) else {
            break;
        };
        let to = m.to as usize;
        if index % 2 == 0 && !m.is_drop() {
            let target = walk.board[to];
            if target != EMPTY && ptype(target) != PT_K {
                out.push(to);
            }
        }
        walk.make(m);
    }
    out
}

/// Rebuilds a strict-convention position from a board and the attacker's hand.
///
/// This is the canonical representation of a variant: the defender's hand is
/// never stored, it is *derived*, which makes the strict invariant impossible to
/// break by construction.
fn compose(board: &[u8; 81], attacker_hand: &[u8; 7]) -> Position {
    let mut position = Position::empty();
    position.board.copy_from_slice(board);
    position.side = BLACK;
    position.refresh();
    for (piece, &count) in attacker_hand.iter().enumerate() {
        if count > 0 {
            position.set_hand(BLACK, piece as u8, count as usize);
        }
    }
    strict_material::apply(&mut position, BLACK);
    position
}

fn variant_from(position: Position, ops: Vec<AppliedOp>, mate_ply: u32) -> Variant {
    let attacker_hand = position.hands[BLACK as usize];
    let extra_attacker = ops.iter().filter(|op| op.kind.uses_attacker_hand()).count();
    let relocations = ops.iter().filter(|op| op.kind == OpKind::Relocate).count();
    Variant {
        canonical_hash: canonicalize(&position).hash,
        position,
        attacker_hand,
        ops,
        extra_attacker,
        relocations,
        mate_ply,
        solve_bound: mate_ply,
        branching_nodes: 0,
        max_correct: 0,
    }
}

/// Chebyshev radius of the defender king's zone that operator f may move a piece
/// into.  Two is the smallest radius that still contains the squares a blocker or
/// a guard can stand on without touching the king, and it keeps the destination
/// list at a couple of dozen squares on a full board.
const RELOCATE_KING_RADIUS: i32 = 2;

/// Where operator f may move the piece standing on `from`, ascending by square
/// index so the candidate order is fixed.
///
/// Two bounded neighbourhoods, unioned: the defender king's zone (rule C fails on
/// captures made *around the king*, which is where a blocker has to move out of)
/// and the eight squares around `from` itself (the "nudge one square" repair).
/// Deliberately **not** the piece's whole attack line: a rook's file is 16 squares
/// of which only the two or three near the king can change the mating line, and
/// every destination costs a full validation.
fn relocate_destinations(position: &Position, from: usize) -> Vec<usize> {
    let king = position.king_square(WHITE);
    let (king_file, king_rank) = if king == crate::position::NO_SQUARE {
        (i32::MIN, i32::MIN)
    } else {
        (file_of(king as usize) as i32, rank_of(king as usize) as i32)
    };
    let from_file = file_of(from) as i32;
    let from_rank = rank_of(from) as i32;
    (0..81usize)
        .filter(|&square| position.board[square] == EMPTY)
        .filter(|&square| {
            let file = file_of(square) as i32;
            let rank = rank_of(square) as i32;
            let near_king = king_file != i32::MIN
                && (file - king_file).abs().max((rank - king_rank).abs()) <= RELOCATE_KING_RADIUS;
            let near_from = (file - from_file).abs().max((rank - from_rank).abs()) <= 1;
            near_king || near_from
        })
        .collect()
}

/// Every operator applicable to the source, one entry per usable budget slot.
fn candidates(source: &Position, options: &DistillOptions) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (square, &c) in source.board.iter().enumerate() {
        if c == EMPTY || ptype(c) == PT_K {
            continue;
        }
        out.push(Candidate {
            kind: OpKind::RemoveToDefender,
            square: Some(square),
            to: None,
            piece: ptype(c),
        });
        if options.max_attacker_hand > 0 {
            out.push(Candidate {
                kind: OpKind::KeepInAttackerHand,
                square: Some(square),
                to: None,
                piece: ptype(c),
            });
        }
    }
    for &piece in HAND_PIECES.iter() {
        let available = source.hands[WHITE as usize][piece as usize] as usize;
        for _ in 0..available.min(options.max_attacker_hand) {
            out.push(Candidate {
                kind: OpKind::HandToAttacker,
                square: None,
                to: None,
                piece,
            });
        }
    }
    if options.hand_to_defender {
        for &piece in HAND_PIECES.iter() {
            for _ in 0..source.hands[BLACK as usize][piece as usize] {
                out.push(Candidate {
                    kind: OpKind::HandToDefender,
                    square: None,
                    to: None,
                    piece,
                });
            }
        }
    }
    // Operator f is deliberately absent here: it is offered only by
    // [`repair_rule_c`], on the squares a variant's own main line captures on.
    out
}

/// Applies one candidate operator, or `None` when it no longer applies.
fn apply(base: &Variant, candidate: Candidate, options: &DistillOptions) -> Option<Variant> {
    if candidate.kind.uses_attacker_hand() && base.extra_attacker >= options.max_attacker_hand {
        return None;
    }
    if candidate.kind == OpKind::Relocate && base.relocations >= options.max_relocate {
        return None;
    }
    let mut board = base.position.board;
    let mut hand = base.attacker_hand;
    let mut op = AppliedOp {
        kind: candidate.kind,
        square: candidate.square,
        to: candidate.to,
        piece: candidate.piece,
        defender_piece: false,
    };
    match candidate.kind {
        OpKind::RemoveToDefender | OpKind::KeepInAttackerHand => {
            let square = candidate.square?;
            let c = board[square];
            if c == EMPTY || ptype(c) == PT_K {
                return None;
            }
            op.piece = ptype(c);
            op.defender_piece = color_of(c) == WHITE;
            board[square] = EMPTY;
            if candidate.kind == OpKind::KeepInAttackerHand {
                hand[ptype(c) as usize] = hand[ptype(c) as usize].checked_add(1)?;
            }
        }
        OpKind::HandToAttacker => {
            if base.position.hands[WHITE as usize][candidate.piece as usize] == 0 {
                return None;
            }
            op.defender_piece = true;
            hand[candidate.piece as usize] = hand[candidate.piece as usize].checked_add(1)?;
        }
        OpKind::HandToDefender => {
            let held = hand[candidate.piece as usize];
            if held == 0 {
                return None;
            }
            hand[candidate.piece as usize] = held - 1;
        }
        OpKind::Relocate => {
            let from = candidate.square?;
            let to = candidate.to?;
            let c = board[from];
            if c == EMPTY || ptype(c) == PT_K || from == to || board[to] != EMPTY {
                return None;
            }
            op.piece = ptype(c);
            op.defender_piece = color_of(c) == WHITE;
            board[from] = EMPTY;
            board[to] = c;
        }
        // Applied by the mandatory pre-pass, never from the shuffled list.
        OpKind::DeleteAttackerKing => return None,
    }
    let position = compose(&board, &hand);
    let mut ops = base.ops.clone();
    if candidate.kind == OpKind::HandToDefender {
        // Operator d is the inverse of b and c: when it takes back a piece one of
        // them just added, cancel the pair instead of recording both, so the
        // operator sequence of a variant always describes the shortest route to it.
        match ops
            .iter()
            .rposition(|prior| prior.kind.uses_attacker_hand() && prior.piece == candidate.piece)
        {
            Some(slot) if ops[slot].kind == OpKind::KeepInAttackerHand => {
                // The piece came off the board, so it ends up with the defender.
                ops[slot].kind = OpKind::RemoveToDefender;
            }
            Some(slot) => {
                ops.remove(slot);
            }
            None => ops.push(op),
        }
    } else {
        ops.push(op);
    }
    Some(variant_from(position, ops, base.mate_ply))
}

/// A necessary condition for rule C: the attacker cannot possibly empty a hand
/// that holds more pieces than the mate has attacker moves.
///
/// Cheap enough to run before every full-rule validation, which is the point: it
/// keeps the extra rule-C check off the hot path for variants that cannot pass it.
fn koma_amari_possible(variant: &Variant) -> bool {
    let held: usize = variant
        .attacker_hand
        .iter()
        .map(|&count| count as usize)
        .sum();
    held <= (variant.mate_ply as usize).div_ceil(2)
}

/// One distillation search over one source position.
///
/// The memo is what makes 16 randomised greedy orders affordable: different orders
/// reach the same intermediate diagrams constantly, and each verdict costs a full
/// exact validation.  Keyed by the 128-bit Zobrist key of the trial position plus
/// the target length, which together determine the verdict exactly.
struct Search<'a> {
    solver: &'a mut Solver,
    /// The df-pn prober, idle unless `options.probe` is [`ProbeEngine::Dfpn`].
    prober: &'a mut Prober,
    options: &'a DistillOptions,
    memo: std::collections::HashMap<(u128, u32, u8), Option<u32>>,
    validations: u64,
    nodes: u64,
    rule_rejections: RuleRejections,
    /// Variants [`repair_rule_c`] has already looked at, so the repair costs one
    /// graph solve per *distinct* diagram however many chains reach it.
    repaired: std::collections::HashSet<u128>,
    repair: RepairStats,
}

impl<'a> Search<'a> {
    fn new(solver: &'a mut Solver, prober: &'a mut Prober, options: &'a DistillOptions) -> Self {
        Self {
            solver,
            prober,
            options,
            memo: std::collections::HashMap::new(),
            validations: 0,
            nodes: 0,
            rule_rejections: RuleRejections::default(),
            repaired: std::collections::HashSet::new(),
            repair: RepairStats::default(),
        }
    }

    /// The designated main line of `position` with rule C alone set aside.
    ///
    /// `Some` exactly when the diagram is a mate of `target` plies satisfying every
    /// frozen rule *but* C, which is the only state [`repair_rule_c`] can do
    /// anything with.  The graph is asked for directly: the rule-B/C/G gate buys
    /// one anyway on the graph-free path and then throws it away, so this costs the
    /// same search and keeps the line.
    fn koma_amari_main_line(&mut self, position: &Position, target: u32) -> Option<MainLine> {
        if !validate_position(position, ValidateOptions::pipeline()).is_empty() {
            return None;
        }
        self.validations += 1;
        let validation = exact_validate_with_rules(
            self.solver,
            position,
            Some(target),
            target,
            self.options.search_node_budget,
            true,
            self.options.final_rules().allowing_koma_amari(true),
        );
        self.nodes += validation.result.stats.nodes + validation.extra_nodes;
        validation.accepted.then_some(())?;
        validation.main_line
    }

    /// The prestrip test: is `position` still a mate of exactly `target` plies?
    /// Existence only -- no uniqueness, yozume or material rule -- because the
    /// recorded variants are gated later anyway.  df-pn settles it when it can
    /// under `--probe dfpn`; the exact solver decides otherwise.  Every node is
    /// charged to the per-source budget, and a spent budget stops the stripping.
    fn prestrip_keeps(&mut self, position: &Position, target: u32) -> crate::prestrip::Keep {
        use crate::prestrip::Keep;
        if self.exhausted() {
            return Keep::Stop;
        }
        if self.options.probe.is_dfpn() {
            let rules = self.options.step_rules();
            // The prober reads rule F as far as it can (F2 under the classical
            // reading).  Its disproofs settle existence at a bound only when
            // `dfpn_disproof_is_conclusive` (never under the classical rules
            // since `PIPELINE_VERSION` 0.11.0, whose credit search it lacks).
            let mode = dfpn_useless_mode(rules);
            let options = ProbeOptions::default()
                .max_ply(target)
                .node_budget(self.options.inner_dfpn_node_budget())
                .useless(mode);
            let within = self.prober.prove(position, &options);
            self.nodes += within.nodes;
            let conclusive = dfpn_disproof_is_conclusive(rules);
            // "Proven at `target`, disproven at `target - 2`" settles a mate of
            // exactly `target` only when rule F changes no length; under the
            // classical reading F1 can make the distance shorter than the bound
            // that proves it, so the exact solver measures it instead.  (Step
            // rules are never `Prune`, so this is the same test.)
            let length_settled = conclusive;
            match within.verdict {
                Verdict::Disproven if conclusive => return Keep::No,
                Verdict::Disproven => {}
                Verdict::Proven if target < 3 => return Keep::Yes,
                Verdict::Proven => {
                    let shorter = self.prober.prove(position, &options.max_ply(target - 2));
                    self.nodes += shorter.nodes;
                    match shorter.verdict {
                        Verdict::Proven => return Keep::No,
                        Verdict::Disproven if conclusive && length_settled => return Keep::Yes,
                        Verdict::Disproven | Verdict::Unknown => {}
                    }
                }
                Verdict::Unknown => {}
            }
        }
        self.validations += 1;
        let (_, validation) = minimum_mate_validate_with_rules(
            self.solver,
            position,
            target,
            self.options.search_node_budget,
            false,
            self.options.step_rules(),
        );
        self.nodes += validation.result.stats.nodes + validation.extra_nodes;
        if validation.result.outcome == crate::solver::Outcome::Mate
            && validation.result.distance == Some(target)
        {
            Keep::Yes
        } else {
            Keep::No
        }
    }

    /// `true` once the per-source search budget is spent.
    fn exhausted(&self) -> bool {
        self.validations >= self.options.max_validations || self.nodes >= self.options.max_nodes
    }

    /// Structural validity plus the full exact gate, returning the proven mate
    /// length.
    ///
    /// This is the same predicate the reducer accepts on (`validate_position` +
    /// `exact_validate(...).accepted`), with the target ply relaxed when
    /// `allow_length_change` is set.
    ///
    /// When a target length is known the search bound is the target itself, which
    /// is the tightest exhaustive bound and keeps the inner loop affordable: a
    /// variant whose mate got longer is refuted at that bound, which is exactly the
    /// verdict the caller wants.  With `allow_length_change` (`--allow-length-change`) there is no target,
    /// so the full `max_ply` bound is searched by iterative deepening.
    fn accept(
        &mut self,
        position: &Position,
        target: Option<u32>,
        rules: TsumeRules,
    ) -> Option<u32> {
        let key = (position.hash(), target.unwrap_or(0), rules.key());
        if let Some(hit) = self.memo.get(&key) {
            return *hit;
        }
        let verdict = self.validate_uncached(position, target, rules);
        self.memo.insert(key, verdict);
        verdict
    }

    /// Acceptance of one chain *step*: rules A and C relaxed.
    fn accept_step(&mut self, position: &Position, target: Option<u32>) -> Option<u32> {
        self.accept(position, target, self.options.step_rules())
    }

    /// Acceptance of a *recorded* variant: every frozen rule enforced.
    fn accept_final(&mut self, position: &Position, target: u32) -> Option<u32> {
        self.accept(position, Some(target), self.options.final_rules())
    }

    fn validate_uncached(
        &mut self,
        position: &Position,
        target: Option<u32>,
        rules: TsumeRules,
    ) -> Option<u32> {
        if !validate_position(position, ValidateOptions::pipeline()).is_empty() {
            return None;
        }
        // The df-pn pre-filter, when it is on.  It can only *reject*, and only for
        // the two reasons the exact gate would have reached itself after a full
        // search: no mate within the target at all, or a mate that is already
        // shorter than the target (`target_mate_ply_changed`).  The prober runs in
        // the rule-F mode `rules` read existence in (`dfpn_useless_mode`: F2 under
        // the classical reading -- F1 changes a mate's length, never its existence
        // at a bound), so its verdict at a bound is the exact solver's verdict at
        // that bound, and the shorter-mate test is sound because a rule-F distance
        // is never longer than the bound that proves it.  (The rule B
        // pre-filter behind it, `yozume_prefilter`, went on 2026-10-27;
        // `rule-b-probe` keeps the probe itself.)
        if self.options.probe.is_dfpn() {
            if let Some(target) = target {
                if self.dfpn_rejects(position, target, rules) {
                    return None;
                }
            }
        }
        self.validations += 1;
        let validation = match target {
            Some(target) => exact_validate_with_rules(
                self.solver,
                position,
                Some(target),
                target,
                self.options.search_node_budget,
                false,
                rules,
            ),
            None if self.options.probe.is_dfpn() => {
                let probe = source_probe(
                    self.solver,
                    self.prober,
                    position,
                    self.options,
                    rules,
                    self.options.search_node_budget,
                    self.options.inner_dfpn_node_budget(),
                );
                self.nodes += probe.dfpn_nodes;
                probe.validation
            }
            None => {
                minimum_mate_validate_with_rules(
                    self.solver,
                    position,
                    self.options.max_ply,
                    self.options.search_node_budget,
                    false,
                    rules,
                )
                .1
            }
        };
        self.nodes += validation.result.stats.nodes + validation.extra_nodes;
        self.rule_rejections.observe(&validation, position);
        // `--emit-negatives` ([`crate::negatives`]): an observer only, so the
        // verdict below is the same either way.
        if crate::negatives::active() {
            crate::negatives::record(crate::negatives::Event {
                kind: crate::negatives::EventKind::Variant,
                tool: "distill",
                position: position.clone(),
                target: target.or(validation.result.distance),
                reasons: validation.reasons.clone(),
                accepted: validation.accepted,
                parent_sound: false,
                step: None,
                path: Vec::new(),
                chain: 0,
                parent: None,
                keeps_line: false,
            });
        }
        validation.accepted.then_some(())?;
        validation.result.distance
    }

    /// `true` when the prober alone settles that this diagram is not a mate of
    /// exactly `target` plies.  df-pn nodes are charged to the same `--max-nodes`
    /// budget as exact nodes; they are a different unit, and deliberately so -- the
    /// budget bounds *work*, not a comparable node count.
    fn dfpn_rejects(&mut self, position: &Position, target: u32, rules: TsumeRules) -> bool {
        let mode = dfpn_useless_mode(rules);
        let options = ProbeOptions::default()
            .max_ply(target)
            .node_budget(self.options.inner_dfpn_node_budget())
            .useless(mode);
        let within = self.prober.prove(position, &options);
        self.nodes += within.nodes;
        match within.verdict {
            // Conclusive only when `dfpn_disproof_is_conclusive` says so.
            Verdict::Disproven => return dfpn_disproof_is_conclusive(rules),
            Verdict::Unknown => return false,
            Verdict::Proven => {}
        }
        if target < 3 {
            return false;
        }
        let shorter = self.prober.prove(position, &options.max_ply(target - 2));
        self.nodes += shorter.nodes;
        shorter.verdict == Verdict::Proven
    }
}

/// What one "minimum mate length here" probe found.
struct SourceProbe {
    /// The length to distil against: the exact solver's rule-F-pruned length when
    /// it could prove one, and otherwise -- only under `--probe dfpn`, and only when
    /// the exact search ran out of budget -- the prober's real-ply length, so a
    /// board the exact solver cannot finish can still be distilled.
    mate_ply: Option<u32>,
    /// The exact validation behind it, for its `reasons`.
    validation: ExactValidation,
    dfpn_nodes: u64,
}

/// The mate length of `position` under `rules`.
///
/// With [`ProbeEngine::Exact`] this is exactly `minimum_mate_validate_with_rules`.
/// With [`ProbeEngine::Dfpn`] the prober finds the minimal **real-ply** length
/// first and the exact solver is asked once, at that bound -- which is the very
/// search the iterative deepening would have ended on, minus every refuted shorter
/// bound it pays for on the way.  When the prober disproves every bound there is no
/// mate and the exact solver is only asked for the cheapest bound there is; when
/// the prober runs out of budget the exact path runs unchanged.
fn source_probe(
    solver: &mut Solver,
    prober: &mut Prober,
    position: &Position,
    options: &DistillOptions,
    rules: TsumeRules,
    node_budget: u64,
    dfpn_node_budget: u64,
) -> SourceProbe {
    let mut dfpn_nodes = 0;
    // The prober reads rule F as far as it can (`dfpn_useless_mode`).  Its proofs
    // are sound for `rules`, and its smallest proving bound is an upper bound on
    // the rule-F distance, which the exact solver measures at that bound.  Its
    // disproofs settle "no mate" only when `dfpn_disproof_is_conclusive`; under
    // the classical reading (`PIPELINE_VERSION` 0.11.0, the exact solver's credit
    // search) they are not, and a disproof falls through to the exact path.
    let mode = dfpn_useless_mode(rules);
    if options.probe.is_dfpn() {
        let probe = prober.minimal_mate_ply(
            position,
            &ProbeOptions::default()
                .max_ply(options.max_ply)
                .node_budget(dfpn_node_budget)
                .useless(mode),
        );
        dfpn_nodes = probe.nodes;
        match probe.verdict {
            Verdict::Disproven if dfpn_disproof_is_conclusive(rules) => {
                let validation =
                    exact_validate_with_rules(solver, position, None, 1, node_budget, false, rules);
                return SourceProbe {
                    mate_ply: None,
                    validation,
                    dfpn_nodes,
                };
            }
            Verdict::Proven => {
                let bound = probe.mate_ply.expect("a proven probe carries its length");
                let validation = exact_validate_with_rules(
                    solver,
                    position,
                    None,
                    bound,
                    node_budget,
                    false,
                    rules,
                );
                // When the exact search could not finish, the proving bound stands
                // in for the length, as it always has outside the classical
                // reading.  Under it the bound can exceed the rule-F distance (F1
                // shortens what F2 lets the prober prove), so no length is claimed
                // and the record behaves as on the exact path.
                let fallback =
                    (rules.useless_mode() != UselessInterposition::Classical).then_some(bound);
                let mate_ply = mate_distance(&validation).or(fallback);
                return SourceProbe {
                    mate_ply,
                    validation,
                    dfpn_nodes,
                };
            }
            Verdict::Disproven | Verdict::Unknown => {}
        }
    }
    let (_, validation) = minimum_mate_validate_with_rules(
        solver,
        position,
        options.max_ply,
        node_budget,
        false,
        rules,
    );
    SourceProbe {
        mate_ply: mate_distance(&validation),
        validation,
        dfpn_nodes,
    }
}

/// The proven mate distance of a validation, or `None` when it proved none.
fn mate_distance(validation: &ExactValidation) -> Option<u32> {
    validation
        .result
        .distance
        .filter(|_| validation.result.outcome == crate::solver::Outcome::Mate)
}

/// The acceptance target for the next step from `base`.
fn target_for(base: &Variant, options: &DistillOptions) -> Option<u32> {
    if options.allow_length_change {
        None
    } else {
        Some(base.mate_ply)
    }
}

/// Number of board pieces whose removal alone still leaves a strict-valid puzzle
/// of the same length (spec 10.6).
fn irrelevant_pieces(search: &mut Search<'_>, variant: &Variant) -> usize {
    let mut count = 0usize;
    for (square, &c) in variant.position.board.iter().enumerate() {
        if c == EMPTY || ptype(c) == PT_K {
            continue;
        }
        let mut board = variant.position.board;
        board[square] = EMPTY;
        let trial = compose(&board, &variant.attacker_hand);
        if search.accept_final(&trial, variant.mate_ply).is_some() {
            count += 1;
        }
    }
    count
}

/// Hands back every attacker hand piece that operators b and c added but the
/// puzzle does not need (spec 10.6: no gratuitous material).
///
/// This is the inverse of operators b and c, and it is what keeps the search
/// honest: a greedy pass may take `hand-to-attacker` simply because it happens to
/// preserve validity, which would quietly add a spare piece to the attacker's hand
/// and, worse, reward it in the score by inventing extra meaningless checks.
/// Tightening removes every added piece that is provably unnecessary, so an
/// operator b or c survives in the final record only when the mate really depends
/// on denying that piece to the defender.  Handing back a piece taken from the
/// board turns operator b back into operator a.
fn tighten(search: &mut Search<'_>, variant: Variant) -> Variant {
    let mut variant = variant;
    loop {
        let mut improved = false;
        for &piece in HAND_PIECES.iter() {
            if variant.attacker_hand[piece as usize] == 0 {
                continue;
            }
            let Some(slot) = variant
                .ops
                .iter()
                .rposition(|op| op.kind.uses_attacker_hand() && op.piece == piece)
            else {
                continue;
            };
            let mut hand = variant.attacker_hand;
            hand[piece as usize] -= 1;
            let trial = compose(&variant.position.board, &hand);
            if search.accept_step(&trial, Some(variant.mate_ply)).is_none() {
                continue;
            }
            let mut ops = variant.ops.clone();
            if ops[slot].kind == OpKind::KeepInAttackerHand {
                // The piece came off the board, so it now goes to the defender.
                ops[slot].kind = OpKind::RemoveToDefender;
            } else {
                ops.remove(slot);
            }
            variant = variant_from(trial, ops, variant.mate_ply);
            improved = true;
            break;
        }
        if !improved {
            return variant;
        }
    }
}

/// Tightens a variant and records it, if every frozen rule accepts it.
///
/// The chain that produced it only had to satisfy the relaxed step rules, so this
/// is where rules A and C are actually decided.  [`koma_amari_possible`] keeps the
/// extra validation off the hot path for variants that cannot pass rule C anyway.
fn record_tightened(
    search: &mut Search<'_>,
    found: &mut IndexMap<String, Variant>,
    variant: Variant,
) {
    let variant = if variant.extra_attacker > 0 {
        tighten(search, variant)
    } else {
        variant
    };
    if found.contains_key(&variant.canonical_hash) {
        record_variant(found, variant);
        return;
    }
    if !koma_amari_possible(&variant) {
        return;
    }
    if search
        .accept_final(&variant.position, variant.mate_ply)
        .is_none()
    {
        repair_rule_c(search, found, &variant);
        return;
    }
    record_variant(found, variant);
}

/// Operator f: the rule-C repair.
///
/// Runs on a variant the full gate just refused.  It is worth anything only when
/// the refusal was rule C *alone* -- rules A and G are repaired by deleting pieces,
/// which the chain already does, and rule B is enforced on every chain step, so a
/// variant that reaches here and still fails B is not a diagram this operator can
/// mend.  [`Search::koma_amari_main_line`] establishes exactly that, and hands back
/// the line whose captures put the leftover material in hand.
///
/// Each capture square on that line is then vacated by moving its occupant to a
/// bounded set of empty squares ([`relocate_destinations`]), and every result goes
/// through the same [`Search::accept_final`] gate as any other variant.  Squares
/// ascend and destinations ascend, so the whole pass is deterministic and depends
/// on nothing but the variant.
///
/// It never recurses: a repaired variant is recorded through [`record_variant`],
/// not [`record_tightened`], and `max_relocate` bounds the chain besides.
fn repair_rule_c(
    search: &mut Search<'_>,
    found: &mut IndexMap<String, Variant>,
    variant: &Variant,
) {
    if search.options.max_relocate == 0
        || variant.relocations >= search.options.max_relocate
        || search.repair.considered >= search.options.max_repairs
        || search.exhausted()
        || !search.repaired.insert(variant.position.hash())
    {
        return;
    }
    search.repair.considered += 1;
    let Some(line) = search.koma_amari_main_line(&variant.position, variant.mate_ply) else {
        return;
    };
    if line.leftover_hand == 0 {
        // Rule C was not the refusal after all; nothing here to repair.
        return;
    }
    search.repair.eligible += 1;
    let mut squares = main_line_capture_squares(&variant.position, &line);
    squares.sort_unstable();
    squares.dedup();
    for square in squares {
        for destination in relocate_destinations(&variant.position, square) {
            if search.exhausted() {
                return;
            }
            let candidate = Candidate {
                kind: OpKind::Relocate,
                square: Some(square),
                to: Some(destination),
                piece: ptype(variant.position.board[square]),
            };
            let Some(moved) = apply(variant, candidate, search.options) else {
                continue;
            };
            search.repair.relocations += 1;
            if search
                .accept_final(&moved.position, moved.mate_ply)
                .is_some()
            {
                search.repair.accepted += 1;
                record_variant(found, moved);
            }
        }
    }
}

fn record_variant(found: &mut IndexMap<String, Variant>, variant: Variant) {
    match found.get(&variant.canonical_hash) {
        Some(existing) if existing.ops.len() <= variant.ops.len() => {}
        _ => {
            found.insert(variant.canonical_hash.clone(), variant);
        }
    }
}

/// Distils one position.  `position` may be either side to move; it is rotated so
/// the attacker is black and the strict convention is applied first.
pub fn distill_position(
    solver: &mut Solver,
    position: &Position,
    options: &DistillOptions,
) -> DistillResult {
    let _phase = crate::prof::enter(crate::prof::Phase::DistillSource);
    let source = crate::canonical::normalize_attacker(position);
    let source_board_pieces = source.board_piece_count();
    let step_rules = options.step_rules();

    // Rule A pre-pass.  It runs *first* because the diagram without the attacker
    // king is the one being distilled: its own minimum mate length is the target
    // for every operator below, and probing the source as well would cost a second
    // full search for a length nothing uses.
    let king_square = attacker_king_square(&source, BLACK).filter(|_| !options.allow_attacker_king);
    let base = match king_square {
        Some(_) => without_attacker_king(&source, BLACK).unwrap_or_else(|| source.clone()),
        None => source.clone(),
    };
    let mut prober = Prober::new();
    let probe = source_probe(
        solver,
        &mut prober,
        &base,
        options,
        step_rules,
        options.node_budget,
        options.dfpn_node_budget,
    );
    let Some(source_mate_ply) = probe.mate_ply else {
        // Distinguish "there was never a mate here" from "the mate needs the
        // attacker's own king on the board", which rule A forbids.
        if king_square.is_some() {
            let with_king = source_probe(
                solver,
                &mut prober,
                &source,
                options,
                TsumeRules::candidate(),
                options.node_budget,
                options.dfpn_node_budget,
            );
            if let Some(source_mate_ply) = with_king.mate_ply {
                return DistillResult::AttackerKingRequired { source_mate_ply };
            }
        }
        let mut reasons = probe.validation.reasons;
        if !reasons.iter().any(|reason| {
            reason == "no_forced_mate_within_bound" || reason == "solver_budget_exhausted"
        }) {
            reasons.insert(0, "no_forced_mate_within_bound".to_string());
        }
        reasons.retain(|reason| {
            reason == "no_forced_mate_within_bound" || reason == "solver_budget_exhausted"
        });
        return DistillResult::NoMate { reasons };
    };
    let attacker_king_removed = king_square.is_some();
    let mut king_ops: Vec<AppliedOp> = king_square
        .map(|square| {
            vec![AppliedOp {
                kind: OpKind::DeleteAttackerKing,
                square: Some(square),
                to: None,
                piece: PT_K,
                defender_piece: false,
            }]
        })
        .unwrap_or_default();
    let mut rng = SplitMix64::new(options.seed);
    let mut found: IndexMap<String, Variant> = IndexMap::new();
    let mut search = Search::new(solver, &mut prober, options);

    // `--prestrip`: thin the base cheaply before the expensive search below.
    crate::prof::switch(crate::prof::Phase::DistillPrestrip);
    let base = if options.prestrip {
        let (stripped, removed) = crate::prestrip::strip(&base, options.prestrip_order, |trial| {
            search.prestrip_keeps(trial, source_mate_ply)
        });
        king_ops.extend(removed.iter().map(|removal| AppliedOp {
            kind: OpKind::RemoveToDefender,
            square: Some(removal.square),
            to: None,
            piece: ptype(removal.code),
            defender_piece: color_of(removal.code) == WHITE,
        }));
        stripped
    } else {
        base
    };

    // The base itself is a candidate whenever it already publishes.
    crate::prof::switch(crate::prof::Phase::DistillBaseline);
    let source_variant = variant_from(base.clone(), king_ops, source_mate_ply);
    let base_publishes = search
        .accept_final(&source_variant.position, source_mate_ply)
        .is_some();
    if base_publishes {
        record_variant(&mut found, source_variant.clone());
    }

    // Baseline: the existing strict reducer, one randomised order.  Reusing it
    // keeps the "pieces removed" metric comparable with the `reduce` stage.
    let mut baseline_board_pieces = None;
    {
        let reduce_options = options.reduce_options(source_mate_ply);
        if let ReduceResult::Reduced(reduced) = reduce::reduce(
            search.solver,
            &base,
            Some(source_mate_ply),
            &reduce_options,
            &mut rng,
        ) {
            baseline_board_pieces = Some(reduced.final_board_pieces);
            let mut ops = source_variant.ops.clone();
            ops.extend(reduced.removed.iter().map(|removed| {
                AppliedOp {
                    kind: if removed.square >= crate::reduce::HAND_SQUARE_BASE {
                        OpKind::HandToDefender
                    } else {
                        OpKind::RemoveToDefender
                    },
                    square: (removed.square < crate::reduce::HAND_SQUARE_BASE)
                        .then_some(removed.square),
                    to: None,
                    piece: removed
                        .piece
                        .chars()
                        .last()
                        .and_then(|letter| {
                            PIECE_LETTERS
                                .iter()
                                .position(|&candidate| candidate == letter as u8)
                        })
                        .unwrap_or(PT_P as usize) as u8,
                    defender_piece: removed.piece.starts_with('w'),
                }
            }));
            // Recorded unjudged: with the lenient entry, a reduction that took no
            // greedy step hands back the entry sweep's diagram, which rule G
            // never judged.  The ranking below gates it ([`mate_length_gate`]);
            // gating it here instead would change `distinctVariants` of records
            // that were right all along.
            record_variant(
                &mut found,
                variant_from(reduced.position, ops, reduced.target_mate_ply),
            );
        }
    }

    crate::prof::switch(crate::prof::Phase::DistillVariants);
    let all = candidates(&base, options);
    for _ in 0..options.attempts {
        if search.exhausted() {
            break;
        }
        let order = rng.shuffle(&all);
        let mut state = source_variant.clone();
        let mut accepted_any = base_publishes;
        for candidate in order {
            let Some(next) = apply(&state, candidate, options) else {
                continue;
            };
            let target = target_for(&state, options);
            if let Some(mate_ply) = search.accept_step(&next.position, target) {
                state = Variant { mate_ply, ..next };
                accepted_any = true;
                // A chain is only rule-compliant at some of its steps: operator b
                // adds hand material that a later step may or may not spend, so the
                // *final* state can fail rule C where an intermediate one passed.
                // Recording every compliant step costs one memoised validation and
                // is what keeps recall from collapsing under rule C.
                record_tightened(&mut search, &mut found, state.clone());
            }
            if search.exhausted() {
                break;
            }
        }
        if accepted_any {
            record_tightened(&mut search, &mut found, state);
        }
    }

    let no_valid_variant = |search: &Search<'_>| DistillResult::NoValidVariant {
        source_mate_ply,
        rule_rejections: search.rule_rejections,
        repair: search.repair,
        validations: search.validations,
        search_nodes: search.nodes,
    };
    if found.is_empty() {
        return no_valid_variant(&search);
    }
    let distinct = found.len();

    // Rank cheaply, then spend graph solves on the short list only.
    crate::prof::switch(crate::prof::Phase::DistillRank);
    let mut shortlist: Vec<Variant> = found.into_values().collect();
    shortlist.sort_by(|a, b| {
        a.position
            .board_piece_count()
            .cmp(&b.position.board_piece_count())
            .then_with(|| a.extra_attacker.cmp(&b.extra_attacker))
            .then_with(|| a.canonical_hash.cmp(&b.canonical_hash))
    });
    shortlist.truncate(options.max_scored.max(options.max_variants).max(1));
    let mut scored: Vec<(Variant, Quality)> = Vec::with_capacity(shortlist.len());
    for mut variant in shortlist {
        let Some((quality, bound, branching, max_correct)) =
            score_variant(search.solver, &variant, None, options)
        else {
            continue;
        };
        // Scored at the wider bound: publishable only if the gate at the mate
        // length accepts it as well (rule G depends on the bound).  At the mate
        // length itself `score_variant` was that gate already.
        if bound > variant.mate_ply && !mate_length_gate(search.solver, &variant, options) {
            continue;
        }
        variant.solve_bound = bound;
        variant.branching_nodes = branching;
        variant.max_correct = max_correct;
        scored.push((variant, quality));
    }
    sort_by_quality(&mut scored);
    scored.truncate(options.max_variants.max(1));
    if options.irrelevance_probe {
        for entry in scored.iter_mut() {
            let count = irrelevant_pieces(&mut search, &entry.0);
            if let Some((quality, bound, branching, max_correct)) =
                score_variant(search.solver, &entry.0, Some(count), options)
            {
                entry.1 = quality;
                entry.0.solve_bound = bound;
                entry.0.branching_nodes = branching;
                entry.0.max_correct = max_correct;
            }
        }
        sort_by_quality(&mut scored);
    }
    if scored.is_empty() {
        return no_valid_variant(&search);
    }
    let validations = search.validations;
    let search_nodes = search.nodes;
    let rule_rejections = search.rule_rejections;
    let repair = search.repair;
    DistillResult::Distilled {
        source_mate_ply,
        source_board_pieces,
        variants: scored,
        distinct,
        baseline_board_pieces,
        validations,
        search_nodes,
        rule_rejections,
        repair,
        attacker_king_removed,
    }
}

fn sort_by_quality(scored: &mut [(Variant, Quality)]) {
    scored.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.canonical_hash.cmp(&b.0.canonical_hash))
    });
}

/// Solves `variant` with a graph and scores it, reporting the bound it used.
///
/// The bound is two plies wider than the mate when `max_ply` allows it.  Two plies
/// is exactly what the uniqueness margin needs: mate distances from an attacker
/// root are odd, so the next-best mating first move is at least `mate_ply + 2`, and
/// at a tighter bound it would be indistinguishable from a refuted move.  If the
/// wider search does not resolve inside the node budget the tight bound is used
/// instead, so a valid puzzle is never dropped for being expensive to over-solve.
type Scored = (Quality, u32, usize, usize);

fn score_variant(
    solver: &mut Solver,
    variant: &Variant,
    irrelevant: Option<usize>,
    options: &DistillOptions,
) -> Option<Scored> {
    let wide = options
        .max_ply
        .min(variant.mate_ply + 2)
        .max(variant.mate_ply);
    let mut bound = wide;
    let rules = options.final_rules();
    let mut validation = exact_validate_with_rules(
        solver,
        &variant.position,
        Some(variant.mate_ply),
        bound,
        options.node_budget,
        true,
        rules,
    );
    if !validation.accepted && bound > variant.mate_ply {
        bound = variant.mate_ply;
        validation = exact_validate_with_rules(
            solver,
            &variant.position,
            Some(variant.mate_ply),
            bound,
            options.node_budget,
            true,
            rules,
        );
    }
    if !validation.accepted {
        return None;
    }
    let extracted = features::extract(&variant.position, &validation.result);
    let (branching, max_correct) = validation
        .main_line
        .as_ref()
        .map(|line| (line.branching_nodes, line.max_correct))
        .unwrap_or((0, 0));
    Some((
        quality::score(QualityInput {
            position: &variant.position,
            result: &validation.result,
            features: &extracted,
            irrelevant_pieces: irrelevant,
            weights: &options.weights,
        }),
        bound,
        branching,
        max_correct,
    ))
}

/// The publishing gate at the variant's own mate length -- the bound the
/// `solve` stage and the catalogue judge it at (`solve --max-ply <matePly>`) --
/// at the run's node budget.  [`score_variant`] may have accepted it at a wider
/// bound, where rule G reads other refutations of the wrong checks.
///
/// A refusal is reported to `--emit-negatives` as a variant event with the
/// gate's reasons: a candidate negative, which `negatives::finalize` judges
/// again (at `search_node_budget`, fixed at 60,000 on the CLI) before writing it.
fn mate_length_gate(solver: &mut Solver, variant: &Variant, options: &DistillOptions) -> bool {
    let validation = exact_validate_with_rules(
        solver,
        &variant.position,
        Some(variant.mate_ply),
        variant.mate_ply,
        options.node_budget,
        true,
        options.final_rules(),
    );
    if !validation.accepted {
        report_refused(&variant.position, variant.mate_ply, validation.reasons);
    }
    validation.accepted
}

/// Reports a diagram the gate at its mate length refused to `--emit-negatives`
/// (a no-op without it): an observer only, like every other report.
fn report_refused(position: &Position, mate_ply: u32, reasons: Vec<String>) {
    if !crate::negatives::active() {
        return;
    }
    crate::negatives::record(crate::negatives::Event {
        kind: crate::negatives::EventKind::Variant,
        tool: "distill",
        position: position.clone(),
        target: Some(mate_ply),
        reasons,
        accepted: false,
        parent_sound: false,
        step: None,
        path: Vec::new(),
        chain: 0,
        parent: None,
        keeps_line: false,
    });
}

/// What the gate at the mate length says about a record published from a wider
/// bound ([`mate_length_verdict`]).
#[derive(Debug)]
pub enum MateLengthVerdict {
    /// The gate accepts the diagram and, with rule D on, no main-line-preserving
    /// step of rule D is accepted from it at that bound (it is minimal there).
    /// Carries the copy judged there: its solution, and under `--emit-record`
    /// its gate facts, are the ones at the mate length.
    Pass(Box<PipelineRecord>),
    /// The gate refuses the diagram: its reasons.
    Refused(Vec<String>),
    /// The gate accepts it, but rule D at that bound does not leave it as it is:
    /// `reasons` is [`crate::catalog_record::RULE_D`] when a piece comes off, or
    /// the reasons the reduction could not be finished for; `events` is what the
    /// reduction reported to `--emit-negatives` (`None` without it), for the
    /// caller to hand over ([`crate::negatives::release`]).
    NotMinimal {
        reasons: Vec<String>,
        events: Option<crate::negatives::Collected>,
    },
}

/// The gate at the mate length on a record published from a wider bound, and
/// rule D at that bound when rule D is on.  Every search runs on a copy.
///
/// `--emit-negatives`: what the check judges is held back and dropped on a pass,
/// so a record that passes leaves the negatives exactly as they were without the
/// check.  A refused diagram is reported as a variant event with the gate's
/// reasons; on a failure of rule D the reducer's own events come back with the
/// verdict (the diagram an accepted main-line-preserving step starts from is the
/// not-minimal one), and the caller decides whether to hand them over.
pub fn mate_length_verdict(
    solver: &mut Solver,
    record: &PipelineRecord,
    mate_ply: u32,
    options: &DistillOptions,
) -> Result<MateLengthVerdict> {
    let (verdict, held) = crate::negatives::held(|| -> Result<MateLengthVerdict> {
        let mut probe = record.clone();
        probe.solution = None;
        let gate = mate_length_options(mate_ply, options);
        let report = solve_stage_reported(&mut probe, &gate, solver)?;
        if !report.accepted {
            let reasons = probe
                .rejection
                .map(|rejection| rejection.reasons)
                .unwrap_or_default();
            return Ok(MateLengthVerdict::Refused(reasons));
        }
        if !options.reduce {
            return Ok(MateLengthVerdict::Pass(Box::new(probe)));
        }
        let outcome = reduce_accepted(
            &mut probe,
            publish_reduce_options(mate_ply, options),
            crate::reduce::TwoKindsOptions::default(),
            &gate,
            solver,
        )?
        .outcome;
        Ok(match outcome {
            ReduceAcceptedOutcome::Unchanged => MateLengthVerdict::Pass(Box::new(probe)),
            ReduceAcceptedOutcome::Reduced { .. } => MateLengthVerdict::NotMinimal {
                reasons: vec![crate::catalog_record::RULE_D.to_string()],
                events: None,
            },
            ReduceAcceptedOutcome::Fallback { reasons } => MateLengthVerdict::NotMinimal {
                reasons,
                events: None,
            },
        })
    });
    let mut verdict = verdict?;
    match &mut verdict {
        MateLengthVerdict::Pass(_) => {}
        MateLengthVerdict::Refused(reasons) => {
            if let Some(position) = record
                .normalized_sfen
                .as_deref()
                .and_then(|sfen| crate::sfen::parse_sfen(sfen).ok())
            {
                report_refused(&position, mate_ply, reasons.clone());
            }
        }
        MateLengthVerdict::NotMinimal { events, .. } => *events = held,
    }
    Ok(verdict)
}

/// The publishing gate at `mate_ply`, at the run's node budget.
fn mate_length_options(mate_ply: u32, options: &DistillOptions) -> SolveStageOptions {
    SolveStageOptions {
        max_ply: mate_ply,
        node_budget: options.node_budget,
        rules: options.final_rules(),
    }
}

/// `--emit-record`: the record the catalogue sees for `out`, a record published
/// from a wider bound that passed the gate at its mate length ([`GateFacts::view`]
/// of [`crate::catalog_record`]).  It is `out` judged at the mate length: the
/// solution, the gate facts and the reduction of `judged` (the same diagram
/// there, which rule D leaves as it is).  When the wider bound's rule D reached
/// `out` from a larger diagram, that diagram is linked as reducible only if rule D
/// at the mate length takes it to `out` too ([`mate_length_reduction_of`]); the
/// view then carries that reduction instead.
///
/// [`GateFacts::view`]: crate::catalog_record::GateFacts::view
fn catalog_view(
    solver: &mut Solver,
    out: &PipelineRecord,
    judged: PipelineRecord,
    mate_ply: u32,
    options: &DistillOptions,
) -> Result<PipelineRecord> {
    let source = out
        .gate
        .as_deref()
        .and_then(|facts| facts.reduced_from.as_deref());
    let at_mate_length = match source {
        Some(source) => mate_length_reduction_of(solver, source, out, mate_ply, options)?,
        None => None,
    };
    let judged = at_mate_length.unwrap_or(judged);
    let mut view = out.clone();
    view.solution = judged.solution;
    view.reduction = judged.reduction;
    view.gate = judged.gate;
    // the cost of the reduction the view carries (`--emit-record`'s
    // `reduce.reducedAt` / `wallMs` / `cpuMs`)
    if judged.reduce_cost.is_some() {
        view.reduce_cost = judged.reduce_cost;
    }
    Ok(view)
}

/// `--emit-record`: rule D at the mate length from `source`, the diagram the
/// wider bound's rule D took pieces off to reach `out`.  `Some` -- the reduced
/// record, with its facts and the source's judged at the mate length -- only
/// when rule D there reduces `source` to `out`'s diagram as well; otherwise the
/// source is not known to be reducible to `out` at the mate length.  On a copy,
/// with `--emit-negatives` held back.
fn mate_length_reduction_of(
    solver: &mut Solver,
    source: &PipelineRecord,
    out: &PipelineRecord,
    mate_ply: u32,
    options: &DistillOptions,
) -> Result<Option<PipelineRecord>> {
    let mut record = source.clone();
    record.solution = None;
    record.rejection = None;
    record.reduction = None;
    record.gate = None;
    // The source keeps its own history: this only judges it again.
    let provenance = record.provenance.clone();
    let gate = mate_length_options(mate_ply, options);
    let (outcome, _) = crate::negatives::held(|| -> Result<Option<ReduceAcceptedOutcome>> {
        if !solve_stage_reported(&mut record, &gate, solver)?.accepted {
            return Ok(None);
        }
        let report = reduce_accepted(
            &mut record,
            publish_reduce_options(mate_ply, options),
            crate::reduce::TwoKindsOptions::default(),
            &gate,
            solver,
        )?;
        Ok(Some(report.outcome))
    });
    let reduced = matches!(outcome?, Some(ReduceAcceptedOutcome::Reduced { .. }));
    // Spelt the same, too: the catalogue's record checks its gate facts against
    // `normalizedSfen` as text.
    if !(reduced
        && record.normalized_sfen.is_some()
        && record.normalized_sfen == out.normalized_sfen)
    {
        return Ok(None);
    }
    if let Some(from) = record
        .gate
        .as_mut()
        .and_then(|facts| facts.reduced_from.as_mut())
    {
        from.provenance = provenance;
    }
    Ok(Some(record))
}

/// Two records' diagrams are the same (compared as positions: a record written
/// before 2026-10-04 spells its hands in another order).
fn same_diagram(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => legacy_sfen_key_of(a) == legacy_sfen_key_of(b),
        _ => false,
    }
}

/// Rule D behind a published variant, at `bound`: no useless piece, not the
/// fewest pieces (owner 2026-09-19) -- one greedy attempt plus the fixpoint
/// sweep guarantees it.
fn publish_reduce_options(bound: u32, options: &DistillOptions) -> ReduceOptions {
    ReduceOptions {
        attempts: 1,
        max_ply: bound,
        node_budget: options.node_budget,
        rules: options.final_rules(),
        until_fixpoint: true,
        ..ReduceOptions::default()
    }
}

/// Solves `out` (a validated variant record) at `bound` and, with rule D on and
/// the gate accepting, reduces it at the same bound.  A rejected record keeps no
/// solution.
fn publish_at(
    solver: &mut Solver,
    out: &mut PipelineRecord,
    bound: u32,
    options: &DistillOptions,
) -> Result<Option<ReduceAcceptedOutcome>> {
    let gate = SolveStageOptions {
        // The variant's own minimum length (plus the uniqueness margin) is the
        // tightest exhaustive bound, and `stats.maxPly` records it.
        max_ply: bound,
        node_budget: options.node_budget,
        rules: options.final_rules(),
    };
    solve_stage(out, &gate, solver)?;
    // `solve` keeps the solution of a rejection by rule B alone
    // (`stages::rule_b_only`); a rejected variant is written as before, without one.
    if out.rejection.is_some() {
        out.solution = None;
    }
    if !(options.reduce && out.solution.is_some()) {
        return Ok(None);
    }
    Ok(Some(
        reduce_accepted(
            out,
            publish_reduce_options(bound, options),
            // Rule D v5: the main-line-preserving reduction alone.
            crate::reduce::TwoKindsOptions::default(),
            &gate,
            solver,
        )?
        .outcome,
    ))
}

/// A record published at its mate length whose rule D could not be finished
/// there (`Fallback`: the reducer's entry or the re-solve of its result ran out
/// of budget) is not known to be minimal, so it is written as a rejection of
/// stage `reduce` with those reasons, without a solution.  `--emit-record`
/// writes no record of it (`rejectedAfterGate`): the catalogue takes a sound
/// diagram only with its minimality known.
fn withhold_unfinished_reduction(
    out: &mut PipelineRecord,
    reduced: Option<&ReduceAcceptedOutcome>,
) {
    let Some(ReduceAcceptedOutcome::Fallback { reasons }) = reduced else {
        return;
    };
    out.rejection = Some(Rejection {
        stage: "reduce".to_string(),
        reasons: reasons.clone(),
        inert_pieces: None,
    });
    out.solution = None;
}

/// Distils one pipeline record into zero or more output records.
///
/// Records that already carry a rejection pass through untouched, exactly like
/// every other stage.  Accepted records produce either one rejection record or up
/// to `max_variants` solved-stage puzzle records.
pub fn distill_record(
    solver: &mut Solver,
    record: &PipelineRecord,
    options: &DistillOptions,
) -> Result<Vec<PipelineRecord>> {
    distill_record_judged(solver, record, options).map(|(out, _)| out)
}

/// The written variants of [`distill_record_judged`] and the folded ones,
/// each with the number of written variants judged before it.
pub type JudgedVariants = (Vec<PipelineRecord>, Vec<(usize, PipelineRecord)>);

/// [`distill_record`], and the variants it does not write because rule D
/// folded them onto a better-ranked variant's diagram -- each with the number
/// of written variants before it (the judgement order).  Those are kept only
/// while [`crate::catalog_record::capture_on`] (`--emit-record` writes a
/// record of every diagram the gate judged); the written ones are the same.
pub fn distill_record_judged(
    solver: &mut Solver,
    record: &PipelineRecord,
    options: &DistillOptions,
) -> Result<JudgedVariants> {
    let mut folded = Vec::new();
    let out = distill_record_folding(solver, record, options, &mut folded)?;
    Ok((out, folded))
}

fn distill_record_folding(
    solver: &mut Solver,
    record: &PipelineRecord,
    options: &DistillOptions,
    folded: &mut Vec<(usize, PipelineRecord)>,
) -> Result<Vec<PipelineRecord>> {
    let _phase = crate::prof::enter(crate::prof::Phase::DistillFinal);
    if !record.is_accepted() {
        return Ok(vec![record.clone()]);
    }
    let position = crate::stages::record_position(record)?;
    // `docs/legal-data.md` §3: every variant inherits its source's origin set, so
    // a source that states no origin can produce no publishable variant.  The
    // result is rejected, never defaulted to `own`; the reason and stage match
    // what `validate` writes for the same defect.
    if !record.provenance.is_root_complete() {
        return Ok(vec![rejection_record(
            record,
            options,
            vec![crate::stages::MISSING_ORIGIN_REASON.to_string()],
            &position,
            None,
            RejectionCounters::default(),
        )]);
    }
    // Derive a per-record seed so one `--seed` reproduces a whole stream and each
    // record is also reproducible on its own.
    let mut options = *options;
    options.base_seed = options.seed;
    options.seed ^=
        crate::util::seed_from_hex_prefix(&sha256_text(&record.source_id)).rotate_left(17);
    // Rule A is a per-record opt-out, exactly as in the reduce and solve stages.
    options.allow_attacker_king = options.allow_attacker_king || record.allows_attacker_king();
    match distill_position(solver, &position, &options) {
        DistillResult::NoMate { reasons } => Ok(vec![rejection_record(
            record,
            &options,
            reasons,
            &position,
            None,
            RejectionCounters::default(),
        )]),
        DistillResult::AttackerKingRequired { source_mate_ply } => Ok(vec![rejection_record(
            record,
            &options,
            vec!["attacker_king_required".to_string()],
            &position,
            Some(source_mate_ply),
            RejectionCounters::default(),
        )]),
        DistillResult::NoValidVariant {
            source_mate_ply,
            rule_rejections,
            repair,
            validations,
            search_nodes,
        } => Ok(vec![rejection_record(
            record,
            &options,
            vec!["no_strict_valid_variant".to_string()],
            &position,
            Some(source_mate_ply),
            RejectionCounters {
                rule_rejections,
                repair,
                validations,
                search_nodes,
            },
        )]),
        DistillResult::Distilled {
            source_mate_ply,
            source_board_pieces,
            variants,
            distinct,
            baseline_board_pieces,
            validations,
            search_nodes,
            rule_rejections,
            repair,
            attacker_king_removed,
        } => {
            let mut out = Vec::with_capacity(variants.len());
            for (index, (variant, quality)) in variants.iter().enumerate() {
                out.push(variant_record(
                    solver,
                    record,
                    &options,
                    VariantContext {
                        index,
                        variant,
                        quality,
                        source_mate_ply,
                        source_board_pieces,
                        distinct,
                        baseline_board_pieces,
                        validations,
                        search_nodes,
                        rule_rejections,
                        repair,
                        attacker_king_removed,
                        source_sfen: to_sfen(&position),
                    },
                )?);
            }
            if options.reduce {
                // Rule D can fold two variants of one source onto one diagram;
                // the better-ranked one (the earlier index) is kept.
                let mut seen = std::collections::HashSet::new();
                let capture = crate::catalog_record::capture_on();
                let mut kept = Vec::with_capacity(out.len());
                for record in out {
                    let keep = match (&record.solution, &record.canonical_hash) {
                        (Some(_), Some(hash)) => seen.insert(hash.clone()),
                        _ => true,
                    };
                    if keep {
                        kept.push(record);
                    } else if capture {
                        folded.push((kept.len(), record));
                    }
                }
                out = kept;
            }
            Ok(out)
        }
    }
}

struct VariantContext<'a> {
    index: usize,
    variant: &'a Variant,
    quality: &'a Quality,
    source_mate_ply: u32,
    source_board_pieces: usize,
    distinct: usize,
    baseline_board_pieces: Option<usize>,
    validations: u64,
    search_nodes: u64,
    rule_rejections: RuleRejections,
    repair: RepairStats,
    attacker_king_removed: bool,
    source_sfen: String,
}

/// The per-source counters a rejection record publishes.  Default (all zero) for
/// the two rejections decided before the variant search runs.
#[derive(Clone, Copy, Debug, Default)]
struct RejectionCounters {
    rule_rejections: RuleRejections,
    repair: RepairStats,
    validations: u64,
    search_nodes: u64,
}

fn rejection_record(
    record: &PipelineRecord,
    options: &DistillOptions,
    reasons: Vec<String>,
    position: &Position,
    source_mate_ply: Option<u32>,
    counters: RejectionCounters,
) -> PipelineRecord {
    let mut out = record.refused("distill", reasons);
    out.normalized_sfen = Some(to_sfen(&crate::canonical::normalize_attacker(position)));
    out.provenance
        .extra
        .insert("distill".to_string(), options.to_json());
    out.extra.insert(
        "distill".to_string(),
        json!({
            "rejected": true,
            "sourceMatePly": source_mate_ply,
            "sourceBoardPieces": position.board_piece_count(),
            "attempts": options.attempts,
            "seed": options.seed.to_string(),
        }),
    );
    if options.is_v2() {
        // `distill-v2` only: the rule counters of a source that produced *nothing*
        // are the only ones that say which frozen rule is holding a mate length
        // back, so a v2 run publishes them on the rejection too.  A v1 run writes
        // the five keys above and nothing else, exactly as before.
        if let Some(Value::Object(map)) = out.extra.get_mut("distill") {
            map.insert("version".to_string(), json!(options.version()));
            map.insert("validations".to_string(), json!(counters.validations));
            map.insert("searchNodes".to_string(), json!(counters.search_nodes));
            map.insert(
                "ruleRejections".to_string(),
                counters.rule_rejections.to_json(true),
            );
            map.insert("repair".to_string(), json!(counters.repair.to_json()));
        }
    }
    out
}

/// The record of variant `context.variant` of `source_record`, validated and
/// ready for the gate: what [`variant_record`] publishes.
fn validated_variant_record(
    source_record: &PipelineRecord,
    options: &DistillOptions,
    context: &VariantContext<'_>,
) -> PipelineRecord {
    let variant = context.variant;
    let sfen = to_sfen(&variant.position);
    // `docs/legal-data.md` §3: the copy carries the source's `originClass`,
    // `origins` and `rootSourceId` - a derivation with one input copies the
    // set - and this step appends itself to the lineage.  The first entry is
    // therefore the original candidate, which is what `originalCandidateId`
    // names.
    let mut out = source_record.derived(
        format!("{}-v{}", source_record.source_id, context.index),
        sfen.clone(),
        variant.mate_ply,
        "distill",
        options.version(),
    );
    out.allow_attacker_king = options.allow_attacker_king.then_some(true);
    for op in variant.ops.iter() {
        out.provenance.transformations.push(op.transformation());
    }
    let mut parameters = options.to_json();
    if let Value::Object(map) = &mut parameters {
        map.insert("variant".to_string(), json!(context.index));
        map.insert(
            "operators".to_string(),
            Value::Array(variant.ops.iter().map(|op| op.to_json()).collect()),
        );
        map.insert("sourceSfen".to_string(), json!(context.source_sfen));
        map.insert(
            "sourceHash".to_string(),
            json!(sha256_text(&legacy_sfen_key_of(&context.source_sfen))),
        );
    }
    out.provenance
        .extra
        .insert("distill".to_string(), parameters);
    validate_stage(&mut out);
    out
}

fn variant_record(
    solver: &mut Solver,
    source_record: &PipelineRecord,
    options: &DistillOptions,
    context: VariantContext<'_>,
) -> Result<PipelineRecord> {
    let variant = context.variant;
    let mut out = validated_variant_record(source_record, options, &context);
    // Published from a wider bound than the mate: the gate at the mate length
    // has the last word (module docs, "The gate at the mate length").  The
    // record stays at the wider bound only when it is accepted there and passes
    // the gate and rule D at the mate length; otherwise the variant is published
    // again at the mate length.  The ranking accepted the variant at both
    // bounds, so a refusal at the wider one is a budget's.
    let validated = (variant.solve_bound > variant.mate_ply).then(|| out.clone());
    let mut bound = variant.solve_bound;
    let mut reduced = publish_at(solver, &mut out, bound, options)?;
    let mut mate_length_gate = None;
    // `--emit-record`: the record judged at the mate length, for the catalogue's view.
    let mut judged_at_mate_length = None;
    if let Some(validated) = validated {
        let refused = match &out.rejection {
            Some(rejection) => Some((bound, rejection.reasons.clone())),
            None => match mate_length_verdict(solver, &out, variant.mate_ply, options)? {
                MateLengthVerdict::Pass(judged) => {
                    judged_at_mate_length = crate::catalog_record::capture_on().then_some(judged);
                    None
                }
                MateLengthVerdict::Refused(reasons) => Some((variant.mate_ply, reasons)),
                MateLengthVerdict::NotMinimal { reasons, events } => {
                    // Rule D runs again below, from the variant at the mate
                    // length.  When the record is the variant itself that is the
                    // same reduction with the same events, so the check's are
                    // dropped; otherwise they are the record's own.
                    let same = same_diagram(&out.normalized_sfen, &validated.normalized_sfen);
                    if let (false, Some(events)) = (same, events) {
                        crate::negatives::release(events);
                    }
                    Some((variant.mate_ply, reasons))
                }
            },
        };
        if let Some((refused_at, reasons)) = refused {
            out = validated;
            bound = variant.mate_ply;
            reduced = publish_at(solver, &mut out, bound, options)?;
            mate_length_gate = Some(json!({
                "wideBound": variant.solve_bound,
                "refusedAt": refused_at,
                "reasons": reasons,
            }));
        }
    }
    if bound == variant.mate_ply {
        withhold_unfinished_reduction(&mut out, reduced.as_ref());
    }
    let removed_from_board = context
        .source_board_pieces
        .saturating_sub(variant.position.board_piece_count());
    out.extra.insert(
        "distill".to_string(),
        json!({
            "version": options.version(),
            "variant": context.index,
            "seed": options.seed.to_string(),
            "baseSeed": options.base_seed.to_string(),
            "attempts": options.attempts,
            "beam": 0,
            "maxAttackerHand": options.max_attacker_hand,
            "allowLengthChange": options.allow_length_change,
            "sourceMatePly": context.source_mate_ply,
            "matePly": variant.mate_ply,
            "solveBound": bound,
            "lengthChanged": variant.mate_ply != context.source_mate_ply,
            "sourceBoardPieces": context.source_board_pieces,
            "finalBoardPieces": variant.position.board_piece_count(),
            "boardPiecesRemoved": removed_from_board,
            "baselineBoardPieces": context.baseline_board_pieces,
            "distinctVariants": context.distinct,
            "validations": context.validations,
            "searchNodes": context.search_nodes,
            "attackerHandExtra": variant.extra_attacker,
            "usedAttackerHandOperator": variant.uses_attacker_hand(),
            "attackerKingRemoved": context.attacker_king_removed,
            "handToDefenderOps": variant
                .ops
                .iter()
                .filter(|op| op.kind == OpKind::HandToDefender)
                .count(),
            "attackerHandAtRoot": variant
                .attacker_hand
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            "ruleRejections": context.rule_rejections.to_json(options.is_v2()),
            "operators": Value::Array(variant.ops.iter().map(|op| op.to_json()).collect()),
            "operatorSequence": Value::Array(
                variant.ops.iter().map(|op| json!(op.transformation())).collect(),
            ),
            "quality": context.quality.to_json(),
        }),
    );
    if let (Some(gate), Some(Value::Object(map))) = (mate_length_gate, out.extra.get_mut("distill"))
    {
        // Only on a record the gate at the mate length republished: the wider
        // bound it was first published at, and why that one was refused.
        map.insert("mateLengthGate".to_string(), gate);
    }
    if let (Some(outcome), Some(Value::Object(map))) = (reduced, out.extra.get_mut("distill")) {
        // `finalBoardPieces` above is the variant search's; this is what rule D
        // left, and `quality` was scored before it.
        let reduce = match outcome {
            ReduceAcceptedOutcome::Unchanged => json!({ "outcome": "unchanged" }),
            ReduceAcceptedOutcome::Reduced { removed } => json!({
                "outcome": "reduced",
                "removed": removed,
                "qualityPredatesReduction": true,
            }),
            ReduceAcceptedOutcome::Fallback { reasons } => {
                json!({ "outcome": "fallback", "reasons": reasons })
            }
        };
        map.insert("reduce".to_string(), reduce);
    }
    if options.is_v2() {
        if let Some(Value::Object(map)) = out.extra.get_mut("distill") {
            map.insert("relocations".to_string(), json!(variant.relocations));
            map.insert("repair".to_string(), json!(context.repair.to_json()));
            map.insert(
                "mainLineBranchingNodes".to_string(),
                json!(variant.branching_nodes),
            );
            map.insert("mainLineMaxCorrect".to_string(), json!(variant.max_correct));
        }
    }
    if let Some(solution) = out.solution.as_mut() {
        solution.quality = Some(context.quality.to_metadata());
    }
    if let Some(judged) = judged_at_mate_length {
        // Built from the finished record, so the view differs from it only in
        // what the mate length decides.
        let mut view = catalog_view(solver, &out, *judged, variant.mate_ply, options)?;
        if let Some(solution) = view.solution.as_mut() {
            solution.quality = Some(context.quality.to_metadata());
        }
        if let Some(facts) = out.gate.as_mut() {
            facts.view = Some(Box::new(view));
        }
    }
    Ok(out)
}

/// Builds the imported-stage record `distill` uses for a bare SFEN input line:
/// [`PipelineRecord::imported_from_sfen`] with source ids `distill-…` and the
/// transformation `distill:sfen-input`.
pub fn imported_record_from_sfen(
    position: &Position,
    ordinal: usize,
    origin: &str,
    origin_class: &str,
) -> PipelineRecord {
    PipelineRecord::imported_from_sfen(
        position,
        ordinal,
        origin,
        origin_class,
        "distill",
        DISTILL_VERSION,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::parse_sfen;

    /// Lesson 0 (5c5b) with a black pawn on 1e that takes no part (rule G).
    /// Through `PIPELINE_VERSION` 0.15.0 this test used a mate in 3 the gate
    /// accepted at bound 5 and refused at its mate length (rule G clause (d)
    /// read the refutation recorded at the gate's bound).  Since 0.16.0 rule G
    /// reads the tree of the proven mate length and every reply that refutes
    /// within the node's distance, so rule G no longer depends on the bound and
    /// that diagram (`5+B1k1/9/4+R4/8+s/9/9/9/9/9 b S 1`) is sound at both.  The
    /// republishing path is driven here with a variant refused at every bound
    /// and the quality of its sound sibling.
    const VARIANT: &str = "4k4/9/4G4/9/8P/9/9/9/4L4 b - 1";
    /// [`VARIANT`] without the pawn.
    const SOUND_SIBLING: &str = "4k4/9/4G4/9/9/9/9/9/4L4 b - 1";
    /// The diagram that differed by bound through 0.15.0.
    const WAS_WIDE_ONLY: &str = "5+B1k1/9/4+R4/8+s/9/9/9/9/9 b S 1";

    /// The safety net behind the ranking: a variant published from the wider
    /// bound that the gate at its mate length refuses is republished at the mate
    /// length -- here a rejection, because the variant itself fails there.  The
    /// ranking's own gate ([`mate_length_gate`]) keeps such a variant from ever
    /// reaching [`variant_record`], so this calls it directly.
    #[test]
    fn a_record_refused_at_its_mate_length_is_republished_there() {
        let options = DistillOptions {
            max_ply: 5,
            node_budget: 4_000_000,
            reduce: false,
            ..DistillOptions::default()
        };
        let mut solver = Solver::new();
        // The diagram that used to differ by bound: sound at both since 0.16.0.
        let was = variant_from(parse_sfen(WAS_WIDE_ONLY).unwrap(), Vec::new(), 3);
        let (_, bound, _, _) =
            score_variant(&mut solver, &was, None, &options).expect("accepted at bound 5");
        assert_eq!(bound, 5);
        assert!(mate_length_gate(&mut solver, &was, &options));
        // A variant rule G refuses at every bound, published (as if) from bound 3.
        let options = DistillOptions {
            max_ply: 3,
            ..options
        };
        let position = parse_sfen(VARIANT).unwrap();
        let mut variant = variant_from(position.clone(), Vec::new(), 1);
        assert!(score_variant(&mut solver, &variant, None, &options).is_none());
        assert!(!mate_length_gate(&mut solver, &variant, &options));
        let sibling = variant_from(parse_sfen(SOUND_SIBLING).unwrap(), Vec::new(), 1);
        let (quality, _, _, _) =
            score_variant(&mut solver, &sibling, None, &options).expect("the sibling is sound");
        variant.solve_bound = 3;
        let source = imported_record_from_sfen(
            &position,
            0,
            "own",
            crate::jsonl::ORIGIN_CLASS_OWN_GENERATED,
        );
        let out = variant_record(
            &mut solver,
            &source,
            &options,
            VariantContext {
                index: 0,
                variant: &variant,
                quality: &quality,
                source_mate_ply: 1,
                source_board_pieces: position.board_piece_count(),
                distinct: 1,
                baseline_board_pieces: None,
                validations: 0,
                search_nodes: 0,
                rule_rejections: RuleRejections::default(),
                repair: RepairStats::default(),
                attacker_king_removed: false,
                source_sfen: VARIANT.to_string(),
            },
        )
        .unwrap();
        let rejection = out.rejection.as_ref().expect("refused at the mate length");
        assert_eq!(rejection.reasons, ["inert_attacker_piece"]);
        assert!(out.solution.is_none());
        let distill = &out.extra["distill"];
        assert_eq!(distill["solveBound"], 1);
        assert_eq!(distill["mateLengthGate"]["wideBound"], 3);
        // The wide gate itself refuses it now (rule G no longer depends on the bound).
        assert_eq!(distill["mateLengthGate"]["refusedAt"], 3);
        assert_eq!(
            distill["mateLengthGate"]["reasons"],
            json!(["inert_attacker_piece"])
        );
    }

    /// A 5-ply row of the yane7 run (`cat-66b8e543…`) that rule D leaves alone
    /// at bound 7 and reduces at its mate length
    /// (`docs/experiments/distill-rule-g/README.md` §6).
    const NOT_MINIMAL_AT_MATE_LENGTH: &str = "9/1+R7/6p1k/8p/9/7N1/9/9/9 b BGrb3g4s3n4l16p 1";

    /// One negatives event, for comparing two runs.
    fn signature(event: &crate::negatives::Event) -> String {
        format!(
            "{}|{}|{}|{:?}|{:?}|{:?}|{}|{}|{:?}",
            event.kind.label(),
            to_sfen(&event.position),
            event.accepted,
            event.reasons,
            event.target,
            event.parent,
            event.keeps_line,
            event.chain,
            event
                .step
                .as_ref()
                .map(|piece| (piece.square, piece.piece.clone())),
        )
    }

    /// A record rule D reduces at the mate length but not at the wider bound is
    /// republished at the mate length, and `--emit-negatives` sees each
    /// reduction once: the wider one's events and the republished one's, not the
    /// check's copy of the latter (the same reduction from the same diagram).
    #[test]
    fn a_record_not_minimal_at_its_mate_length_is_reduced_there_once() {
        let options = DistillOptions {
            max_ply: 7,
            node_budget: 600_000,
            ..DistillOptions::default()
        };
        assert!(options.reduce);
        let position = parse_sfen(NOT_MINIMAL_AT_MATE_LENGTH).unwrap();
        let mut variant = variant_from(position.clone(), Vec::new(), 5);
        let mut solver = Solver::new();
        let (quality, bound, _, _) =
            score_variant(&mut solver, &variant, None, &options).expect("accepted at bound 7");
        assert_eq!(bound, 7);
        assert!(mate_length_gate(&mut solver, &variant, &options));
        variant.solve_bound = bound;
        let source = imported_record_from_sfen(
            &position,
            0,
            "own",
            crate::jsonl::ORIGIN_CLASS_OWN_GENERATED,
        );
        let context = || VariantContext {
            index: 0,
            variant: &variant,
            quality: &quality,
            source_mate_ply: 5,
            source_board_pieces: position.board_piece_count(),
            distinct: 1,
            baseline_board_pieces: None,
            validations: 0,
            search_nodes: 0,
            rule_rejections: RuleRejections::default(),
            repair: RepairStats::default(),
            attacker_king_removed: false,
            source_sfen: NOT_MINIMAL_AT_MATE_LENGTH.to_string(),
        };
        // The check at the mate length on the variant: rule D takes a piece off,
        // and its events come back with the verdict.
        let validated = validated_variant_record(&source, &options, &context());
        crate::negatives::begin();
        let verdict = mate_length_verdict(&mut solver, &validated, 5, &options).unwrap();
        assert!(crate::negatives::take().events.is_empty());
        let MateLengthVerdict::NotMinimal {
            reasons,
            events: Some(check),
        } = verdict
        else {
            panic!("{verdict:?}");
        };
        assert_eq!(reasons, [crate::catalog_record::RULE_D]);
        let check: Vec<String> = check.events.iter().map(signature).collect();
        assert!(!check.is_empty());

        crate::negatives::begin();
        let out = variant_record(&mut solver, &source, &options, context()).unwrap();
        let events: Vec<String> = crate::negatives::take()
            .events
            .iter()
            .map(signature)
            .collect();
        assert!(out.is_accepted());
        let distill = &out.extra["distill"];
        assert_eq!(distill["solveBound"], 5);
        assert_eq!(distill["mateLengthGate"]["wideBound"], 7);
        assert_eq!(distill["mateLengthGate"]["refusedAt"], 5);
        assert_eq!(
            distill["mateLengthGate"]["reasons"],
            json!([crate::catalog_record::RULE_D])
        );
        assert_eq!(distill["reduce"]["outcome"], "reduced");
        // The wider reduction's events, then the republication's -- the same
        // reduction as the check's, reported once.  On this budget a check of
        // the wider reduction (bound 7) runs out, and since the early stop
        // (owner 2026-10-25) a stopped reduction hands over no event: only the
        // republication's remain, and the check's copy is not among them.
        // (The budget was 4,000,000 until the prober's leaf search of
        // 2026-10-26 made the wider check finish within it; at 600,000 it runs
        // out again, while 300,000 is too small to accept the source at all.)
        assert!(events.ends_with(&check));
        let before = &events[..events.len() - check.len()];
        assert!(before.is_empty(), "the wider reduction stopped");
        assert_eq!(events, check, "the check's events were reported");
    }

    /// `--emit-record` links the diagram the wider bound reduced from only when
    /// rule D at the mate length makes the same reduction.
    #[test]
    fn the_catalogue_links_a_reduction_only_if_the_mate_length_makes_it() {
        // Mate in 3; the defender pawn on 3d comes off with the main line kept
        // (`tests/reduce_two_kinds.rs`, `SECOND_ROUND`), leaving `MINIMAL`.
        const REDUCIBLE: &str = "9/8+R/4+Bgk2/6p2/5S3/9/9/9/9 b rb3g3s4n4l17p 1";
        const MINIMAL: &str = "9/8+R/4+Bgk2/9/5S3/9/9/9/9 b rb3g3s4n4l18p 1";
        let options = DistillOptions {
            node_budget: 1_000_000,
            ..DistillOptions::default()
        };
        let mut solver = Solver::new();
        let mut solved = |sfen: &str| {
            let position = parse_sfen(sfen).unwrap();
            let mut record = imported_record_from_sfen(
                &position,
                0,
                "own",
                crate::jsonl::ORIGIN_CLASS_OWN_GENERATED,
            );
            record.target_mate_ply = Some(3);
            validate_stage(&mut record);
            let gate = SolveStageOptions {
                max_ply: 5,
                node_budget: 1_000_000,
                rules: TsumeRules::puzzle(),
            };
            solve_stage(&mut record, &gate, &mut solver).unwrap();
            assert!(record.is_accepted());
            record
        };
        let reducible = solved(REDUCIBLE);
        let minimal = solved(MINIMAL);
        let mut solver = Solver::new();
        let reduced = mate_length_reduction_of(&mut solver, &reducible, &minimal, 3, &options)
            .unwrap()
            .expect("rule D at 3 takes the pawn off");
        let reduction = reduced.reduction.as_ref().unwrap();
        assert_eq!(reduction.removed.len(), 1);
        assert_eq!(reduction.removed[0].piece, "wP");
        assert_eq!(reduced.solution.as_ref().unwrap().stats.max_ply, 3);
        // The source's history, then rule D's and its re-solve, once.
        let history = &reducible.provenance.transformations;
        let transformations = &reduced.provenance.transformations;
        assert_eq!(&transformations[..history.len()], &history[..]);
        assert_eq!(
            transformations.len(),
            history.len() + 4,
            "{transformations:?}"
        );
        // Nothing comes off a minimal diagram, and a diagram rule D takes
        // elsewhere is not linked to this one.
        assert!(
            mate_length_reduction_of(&mut solver, &minimal, &minimal, 3, &options)
                .unwrap()
                .is_none()
        );
        assert!(
            mate_length_reduction_of(&mut solver, &reducible, &reducible, 3, &options)
                .unwrap()
                .is_none()
        );
    }

    /// Rule D unfinished at the mate length: the record is withheld, the other
    /// outcomes leave it as it is.
    #[test]
    fn an_unfinished_reduction_at_the_mate_length_withholds_the_record() {
        let position = parse_sfen("9/8+R/4+Bgk2/9/5S3/9/9/9/9 b rb3g3s4n4l18p 1").unwrap();
        let mut record = imported_record_from_sfen(
            &position,
            0,
            "own",
            crate::jsonl::ORIGIN_CLASS_OWN_GENERATED,
        );
        validate_stage(&mut record);
        let gate = SolveStageOptions {
            max_ply: 3,
            node_budget: 1_000_000,
            rules: TsumeRules::puzzle(),
        };
        solve_stage(&mut record, &gate, &mut Solver::new()).unwrap();
        assert!(record.is_accepted() && record.solution.is_some());
        for outcome in [
            None,
            Some(ReduceAcceptedOutcome::Unchanged),
            Some(ReduceAcceptedOutcome::Reduced { removed: 1 }),
        ] {
            let mut out = record.clone();
            withhold_unfinished_reduction(&mut out, outcome.as_ref());
            assert!(out.is_accepted() && out.solution.is_some());
        }
        let mut out = record.clone();
        withhold_unfinished_reduction(
            &mut out,
            Some(&ReduceAcceptedOutcome::Fallback {
                reasons: vec!["solver_budget_exhausted".to_string()],
            }),
        );
        let rejection = out.rejection.as_ref().expect("withheld");
        assert_eq!(rejection.stage, "reduce");
        assert_eq!(rejection.reasons, ["solver_budget_exhausted"]);
        assert!(out.solution.is_none());
        // `--emit-record` writes the accepted record, and nothing of the withheld one.
        let facts = |record: &PipelineRecord| {
            Some(Box::new(crate::catalog_record::GateFacts {
                sfen: record.normalized_sfen.clone().unwrap(),
                mate_ply: Some(3),
                had_solution: true,
                ..Default::default()
            }))
        };
        let ctx = crate::catalog_record::RecordContext {
            solver_id: "S-0000000-unit".to_string(),
            produced_by: "unit:distill".to_string(),
            route: "distill".to_string(),
            generation: 0,
            reduce_version: Some("D-v5/0.13.0/1000000".to_string()),
            certify: true,
        };
        let mut accepted = record.clone();
        accepted.gate = facts(&accepted);
        assert!(crate::catalog_record::generated(&accepted, &ctx)
            .iter()
            .all(Result::is_ok));
        out.gate = facts(&out);
        assert_eq!(
            crate::catalog_record::generated(&out, &ctx),
            vec![Err("rejectedAfterGate")]
        );
    }
}
