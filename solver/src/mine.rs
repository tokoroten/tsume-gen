//! Mate mining: random position -> AI play -> mate detection -> candidate
//! (spec 9.1.1 families A/B/C, spec 9.1.5, spec 9.1.6, spec 9.1.12).
//!
//! ```text
//! position source            move policy                mate miner
//! ----------------           -----------                ----------
//! constrained random    ->   builtin alpha-beta    ->    bounded AND/OR search
//! standard initial           random                      (exact, never heuristic)
//! ```
//!
//! One *game* is the unit of work: a start position plus a bounded stochastic
//! self-play walk, with a bounded exact mate search after every eligible ply.  A
//! game's identity is `(base seed, game index)` and nothing else, so a candidate
//! is reproducible from its provenance regardless of how many threads produced it
//! (spec 9.1.12: each worker draws from a disjoint set of game indices, and the
//! seed of a game never depends on which worker ran it).
//!
//! Emitted records are imported-stage `PipelineRecord`s, ready for
//! [`crate::distill`] or for the existing `validate | reduce | solve` stages.
//! Correctness is never asserted by the generator: the exact solver downstream
//! decides, exactly as spec 9.1 requires.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::dfpn::{ProbeEngine, ProbeOptions, Prober, Verdict};
use crate::engine::{BuiltinEngine, EngineConfig, SamplingConfig};
use crate::exact::{
    dfpn_disproof_is_conclusive, dfpn_useless_mode, exact_validate_with_rules,
    minimum_mate_validate_with_rules, TsumeRules,
};
use crate::generate::{
    generate_position, Distribution, GeneratorConfig, MovePolicy, RandomLegalPolicy,
};
use crate::jsonl::{write_json_line, PipelineRecord, Provenance};
use crate::movegen::{gives_check, legal_moves};
use crate::piece::{opponent, BLACK};
use crate::position::{Move, Position, Usi, NO_SQUARE};
use crate::rng::{Rng, SplitMix64};
use crate::sfen::{legacy_sfen_key_of, parse_sfen, to_legacy_sfen_key, to_sfen};
use crate::solver::{Outcome, Solver};
use crate::util::{sha256_text, stable_json};

/// Recorded in `provenance.generatorVersion` for mined candidates.
///
/// `mate-mining-v2` (2026-09-14) changed what the same `--seed` produces, so a
/// `v1` run is reproducible only at its recorded commit (`docs/generation.md`
/// §5.1): the per-game seed no longer comes from `base + index`, which made
/// consecutive base seeds walk the same games one apart
/// ([`crate::rng::derive_stream_seed`]), and `provenance.generatorDistribution`
/// now names the distribution a candidate was actually drawn from rather than the
/// `--distribution` option.
pub const MINE_GENERATOR_VERSION: &str = "mate-mining-v2";
/// Version suffix of the `mine:<family>` transformation entry, bumped with
/// [`MINE_GENERATOR_VERSION`] whenever the same seed stops producing the same
/// candidates.
pub const MINE_TRANSFORMATION_VERSION: &str = "v2";
/// The standard shogi initial position (family A).
pub const STANDARD_INITIAL_SFEN: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Candidate-generation family (spec 9.1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// B: constrained random position, mate search immediately.
    RandomDirect,
    /// C: constrained random position, short stochastic self-play, then mining.
    RandomSelfplay,
    /// A: standard initial position, stochastic self-play, then mining.
    StandardSelfplay,
}

impl Family {
    /// The `provenance.sourceType` value, which is also the CLI spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Family::RandomDirect => "random_direct",
            Family::RandomSelfplay => "random_selfplay",
            Family::StandardSelfplay => "standard_selfplay",
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "random_direct" => Some(Family::RandomDirect),
            "random_selfplay" => Some(Family::RandomSelfplay),
            "standard_selfplay" => Some(Family::StandardSelfplay),
            _ => None,
        }
    }
    /// `true` when the family walks the position with a move policy.
    pub const fn plays(self) -> bool {
        !matches!(self, Family::RandomDirect)
    }
    /// Historical reachability holds by construction only for family A.
    pub const fn historically_reachable(self) -> bool {
        matches!(self, Family::StandardSelfplay)
    }
}

/// `RandomPositionGeneratorConfig.sideToMovePolicy` (spec 9.1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideToMovePolicy {
    /// Coin flip per candidate.
    Random,
    /// Always mine for black, which is what the strict convention normalises to.
    AttackerBiased,
    /// Alternate by game index.
    Balanced,
}

impl SideToMovePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            SideToMovePolicy::Random => "random",
            SideToMovePolicy::AttackerBiased => "attacker-biased",
            SideToMovePolicy::Balanced => "balanced",
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "random" => Some(SideToMovePolicy::Random),
            "attacker-biased" => Some(SideToMovePolicy::AttackerBiased),
            "balanced" => Some(SideToMovePolicy::Balanced),
            _ => None,
        }
    }
}

/// The full spec 9.1.2 generator configuration.
///
/// [`GeneratorConfig`] carries the fields the existing constrained-random source
/// understands; the two extra spec fields live here so `--config` can accept the
/// documented JSON verbatim and so both end up in `generatorConfigHash`.
#[derive(Clone, Copy, Debug)]
pub struct MineGeneratorConfig {
    pub generator: GeneratorConfig,
    /// `attackerKing` came from `--config`, so the per-family default must not
    /// override it.
    pub attacker_king_set: bool,
    /// The constrained source always emits structurally legal diagrams, so this
    /// is recorded for provenance and comparison rather than acted on.
    pub require_strict_legality: bool,
    pub side_to_move_policy: SideToMovePolicy,
}

impl Default for MineGeneratorConfig {
    fn default() -> Self {
        Self {
            generator: GeneratorConfig::default(),
            attacker_king_set: false,
            require_strict_legality: true,
            side_to_move_policy: SideToMovePolicy::AttackerBiased,
        }
    }
}

impl MineGeneratorConfig {
    /// Overlays a `RandomPositionGeneratorConfig` JSON object (camelCase, spec
    /// 9.1.2).  Unknown keys are rejected so a typo cannot silently do nothing.
    pub fn overlay_json(&mut self, value: &Value) -> Result<()> {
        let object = value
            .as_object()
            .ok_or_else(|| anyhow!("--config must be a JSON object"))?;
        for (key, item) in object.iter() {
            match key.as_str() {
                "minBoardPieces" => self.generator.min_board_pieces = usize_field(key, item)?,
                "maxBoardPieces" => self.generator.max_board_pieces = usize_field(key, item)?,
                "maxMaterialImbalance" => {
                    self.generator.max_material_imbalance = usize_field(key, item)? as i32
                }
                "allowInitialCheck" => self.generator.allow_initial_check = bool_field(key, item)?,
                "requireStrictLegality" => self.require_strict_legality = bool_field(key, item)?,
                "kingZoneRadius" => {
                    self.generator.king_zone_radius = usize_field(key, item)? as i32
                }
                "kingZoneOccupancyRange" => {
                    let pair = item
                        .as_array()
                        .filter(|pair| pair.len() == 2)
                        .ok_or_else(|| anyhow!("{key} must be a two-element array"))?;
                    self.generator.king_zone_occupancy =
                        (usize_field(key, &pair[0])?, usize_field(key, &pair[1])?);
                }
                "handPieceRate" => self.generator.hand_piece_rate = float_field(key, item)?,
                "promotionRate" => self.generator.promotion_rate = float_field(key, item)?,
                "attackerKing" => {
                    self.generator.attacker_king = bool_field(key, item)?;
                    self.attacker_king_set = true;
                }
                "sideToMovePolicy" => {
                    let text = item
                        .as_str()
                        .ok_or_else(|| anyhow!("{key} must be a string"))?;
                    self.side_to_move_policy = SideToMovePolicy::parse(text)
                        .ok_or_else(|| anyhow!("unknown sideToMovePolicy: {text}"))?;
                }
                // The seed is a CLI argument; accepting it here keeps documented
                // config files usable without editing.
                "seed" => {}
                other => bail!("unknown --config key: {other}"),
            }
        }
        if self.generator.min_board_pieces > self.generator.max_board_pieces {
            bail!("minBoardPieces must not exceed maxBoardPieces");
        }
        Ok(())
    }

    /// The JSON hashed into `provenance.generatorConfigHash`.
    pub fn to_json(&self) -> Value {
        let generator = &self.generator;
        json!({
            "minBoardPieces": generator.min_board_pieces,
            "maxBoardPieces": generator.max_board_pieces,
            "maxMaterialImbalance": generator.max_material_imbalance,
            "allowInitialCheck": generator.allow_initial_check,
            "requireStrictLegality": self.require_strict_legality,
            "kingZoneRadius": generator.king_zone_radius,
            "kingZoneOccupancyRange": [generator.king_zone_occupancy.0, generator.king_zone_occupancy.1],
            "handPieceRate": generator.hand_piece_rate,
            "promotionRate": generator.promotion_rate,
            "attackerKing": generator.attacker_king,
            "sideToMovePolicy": self.side_to_move_policy.as_str(),
        })
    }

    pub fn config_hash(&self) -> String {
        sha256_text(&stable_json(&self.to_json()))
    }

    /// Rule E default for a family: a self-play family needs both kings for the
    /// game to be playable, a directly-mined family does not want one at all.
    ///
    /// An explicit `attackerKing` in `--config` always wins.
    pub fn apply_family_default(&mut self, plays: bool) {
        if !self.attacker_king_set {
            self.generator.attacker_king = plays;
        }
    }
}

fn usize_field(key: &str, value: &Value) -> Result<usize> {
    value
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| anyhow!("{key} must be a non-negative integer"))
}

fn bool_field(key: &str, value: &Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| anyhow!("{key} must be a boolean"))
}

fn float_field(key: &str, value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| anyhow!("{key} must be a number"))
}

/// Which move policy the mining loop drives.
#[derive(Clone, Debug)]
pub enum PolicySpec {
    /// Uniform over legal moves; cheap, used by tests and as a diversity baseline.
    Random,
    Builtin(EngineConfig),
}

impl PolicySpec {
    /// Parses `random` or `builtin`.
    ///
    /// `usi:<command line>` (an external USI engine, one process per worker
    /// thread) was removed on 2026-10-27 (owner,
    /// `docs/experiments/solver-cli-audit/README.md` §8): the solver does not
    /// drive engines.  Positions from self-play between real engines come from a
    /// separate generator outside the solver that writes the games' final
    /// positions, fed to `distill`.  Old records' `mine.policy` values stay as
    /// they are.
    pub fn parse(text: &str, engine: EngineConfig) -> Result<Self> {
        if text == "builtin" {
            return Ok(PolicySpec::Builtin(engine));
        }
        if text == "random" {
            return Ok(PolicySpec::Random);
        }
        if text.starts_with("usi:") {
            bail!(
                "--policy usi:<command> was removed on 2026-10-27: the solver does not drive \
                 engines; write self-play positions with a separate generator and feed them \
                 to distill"
            );
        }
        bail!("--policy expects builtin or random")
    }

    /// Builds one policy instance.  Called once per worker thread.
    pub fn build(&self) -> Result<Policy> {
        Ok(match self {
            PolicySpec::Random => Policy::Random(RandomLegalPolicy),
            PolicySpec::Builtin(config) => Policy::Builtin(BuiltinEngine::new(*config)),
        })
    }
}

/// A live move policy.
pub enum Policy {
    Random(RandomLegalPolicy),
    Builtin(BuiltinEngine),
}

impl Policy {
    /// Picks one move, or `None` when the side to move has none.
    ///
    /// Unlike [`MovePolicy::select`] this reports transport failures, so the CLI
    /// can stop instead of silently ending every game.
    pub fn select(&mut self, position: &Position, rng: &mut dyn Rng) -> Result<Option<Move>> {
        match self {
            Policy::Random(policy) => Ok(policy.select(position, rng)),
            Policy::Builtin(engine) => Ok(engine.select(position, rng)),
        }
    }

    /// Engine identity for provenance.
    pub fn meta(&self) -> PolicyMeta {
        match self {
            Policy::Random(_) => PolicyMeta {
                kind: "random",
                name: "uniform-random-legal".to_string(),
                version: crate::jsonl::PIPELINE_VERSION.to_string(),
                options_hash: sha256_text("uniform-random-legal"),
                options: json!({}),
            },
            Policy::Builtin(engine) => PolicyMeta {
                kind: "builtin",
                name: crate::engine::BUILTIN_ENGINE_NAME.to_string(),
                version: crate::engine::BUILTIN_ENGINE_VERSION.to_string(),
                options_hash: engine.config.options_hash(),
                options: engine.config.to_json(),
            },
        }
    }
}

/// Engine identity recorded with every candidate.
#[derive(Clone, Debug)]
pub struct PolicyMeta {
    pub kind: &'static str,
    pub name: String,
    pub version: String,
    pub options_hash: String,
    pub options: Value,
}

/// Mining configuration.
#[derive(Clone, Copy, Debug)]
pub struct MineOptions {
    pub family: Family,
    /// Candidate records to emit before stopping.
    pub count: usize,
    pub seed: u64,
    /// Inclusive self-play ply range; one value is drawn per game.
    pub plies: (u32, u32),
    /// Mate-search ply bound.
    pub max_ply: u32,
    /// Search every `check_every`-th eligible ply.
    pub check_every: u32,
    /// First ply that is eligible for mining.
    pub check_from: u32,
    pub node_budget: u64,
    pub min_board_pieces: usize,
    pub max_board_pieces: usize,
    pub max_games: Option<u64>,
    pub time_limit: Option<f64>,
    /// How many `retrograde_from_mate` candidates to emit when a game is actually
    /// checkmated (positions 1, 3, 5, ... plies before the mate).
    pub retrograde: usize,
    /// Strip far pieces before emitting (`--prestrip`, 2026-10-01).  Off by
    /// default.  On a found mate of length `k`, board pieces are tried for
    /// removal one at a time, farthest from the defender king first (ties by
    /// square), and a removal is kept when the position still mates in exactly
    /// `k` -- a plain mate search, far cheaper than `distill`'s full gate on every
    /// step.  Meant for crowded `uniform` boards, where `distill` otherwise spends
    /// its node budget deleting a dozen irrelevant pieces.  `distill` still
    /// decides; the record keeps the unstripped diagram in `mine.prestrip`.
    pub prestrip: bool,
    /// Emit only candidates that already pass the full strict gate.  Off by
    /// default: `distill` is the component that decides.
    pub require_strict_valid: bool,
    /// Which engine detects a mate ([`crate::dfpn`]).  [`ProbeEngine::Exact`] is the
    /// default and reproduces every earlier run byte for byte; [`ProbeEngine::Dfpn`]
    /// puts the df-pn prober in front of the exact gate, which still decides
    /// `strictValid`, so a dense board the exact solver cannot finish is at least
    /// reported as a mate of a known length.
    pub probe: ProbeEngine,
    /// Node budget of one df-pn probe.  Ignored with [`ProbeEngine::Exact`].
    pub dfpn_node_budget: u64,
    pub generator: MineGeneratorConfig,
    pub distribution: Option<Distribution>,
}

impl Default for MineOptions {
    fn default() -> Self {
        Self {
            family: Family::RandomSelfplay,
            count: 100,
            seed: 1,
            plies: (5, 40),
            max_ply: 7,
            check_every: 1,
            check_from: 1,
            node_budget: 200_000,
            min_board_pieces: 3,
            max_board_pieces: 30,
            max_games: None,
            time_limit: None,
            retrograde: 3,
            prestrip: false,
            require_strict_valid: false,
            probe: ProbeEngine::Exact,
            dfpn_node_budget: 1_000_000,
            generator: MineGeneratorConfig::default(),
            distribution: None,
        }
    }
}

impl MineOptions {
    /// The mining parameters recorded in every candidate.
    /// The df-pn keys appear **only** under `--probe dfpn`, so a default run keeps
    /// producing the very bytes it produced before the prober existed.
    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "family": self.family.as_str(),
            "plies": [self.plies.0, self.plies.1],
            "maxPly": self.max_ply,
            "checkEvery": self.check_every,
            "checkFrom": self.check_from,
            "nodeBudget": self.node_budget,
            "minBoardPieces": self.min_board_pieces,
            "maxBoardPieces": self.max_board_pieces,
            "retrograde": self.retrograde,
            "requireStrictValid": self.require_strict_valid,
            "distribution": self.distribution.map(|d| d.as_str()),
        });
        if self.prestrip {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("prestrip".to_string(), json!(true));
        }
        if self.probe.is_dfpn() {
            let map = value.as_object_mut().expect("a JSON object");
            map.insert("probe".to_string(), json!(self.probe.as_str()));
            map.insert("dfpnNodeBudget".to_string(), json!(self.dfpn_node_budget));
        }
        value
    }
}

/// Counters collected over a whole run, for the generator comparison report.
#[derive(Clone, Copy, Debug, Default)]
pub struct MineStats {
    pub games: u64,
    pub generator_failures: u64,
    pub plies_played: u64,
    pub positions_considered: u64,
    pub cheap_filtered: u64,
    pub mate_searches: u64,
    pub search_nodes: u64,
    pub mates_found: u64,
    pub strict_valid: u64,
    pub unique_first_move: u64,
    pub emitted: u64,
    pub retrograde_emitted: u64,
    pub checkmated_games: u64,
    /// Always 0 since the USI engine path went (2026-10-27); kept so the
    /// stats file keeps its keys.
    pub illegal_engine_moves: u64,
}

impl MineStats {
    fn merge(&mut self, other: &MineStats) {
        self.games += other.games;
        self.generator_failures += other.generator_failures;
        self.plies_played += other.plies_played;
        self.positions_considered += other.positions_considered;
        self.cheap_filtered += other.cheap_filtered;
        self.mate_searches += other.mate_searches;
        self.search_nodes += other.search_nodes;
        self.mates_found += other.mates_found;
        self.strict_valid += other.strict_valid;
        self.unique_first_move += other.unique_first_move;
        self.emitted += other.emitted;
        self.retrograde_emitted += other.retrograde_emitted;
        self.checkmated_games += other.checkmated_games;
        self.illegal_engine_moves += other.illegal_engine_moves;
    }

    /// The JSON `--stats-out` writes.
    pub fn to_json(&self) -> Value {
        json!({
            "games": self.games,
            "generatorFailures": self.generator_failures,
            "pliesPlayed": self.plies_played,
            "positionsConsidered": self.positions_considered,
            "cheapFiltered": self.cheap_filtered,
            "mateSearches": self.mate_searches,
            "searchNodes": self.search_nodes,
            "matesFound": self.mates_found,
            "strictValid": self.strict_valid,
            "uniqueFirstMove": self.unique_first_move,
            "emitted": self.emitted,
            "retrogradeEmitted": self.retrograde_emitted,
            "checkmatedGames": self.checkmated_games,
            "illegalEngineMoves": self.illegal_engine_moves,
        })
    }
}

/// Deterministic per-game seed.  Depends only on the base seed and the game
/// index, never on the worker that ran it.
///
/// [`crate::rng::derive_stream_seed`] owns the construction and documents why it
/// is not `SplitMix64::new(base + index)`: that shape made the games of
/// `--seed 20260915` the games of `--seed 20260914` shifted by one, so a
/// production loop that stepped the seed re-mined the same candidates every batch.
pub fn game_seed(base: u64, index: u64) -> u64 {
    crate::rng::derive_stream_seed(base, index)
}

/// Cheap pre-filters (spec 9.1.6).  Never a correctness check: they only avoid
/// spending an exact search on a position that cannot become a tsume.
fn cheap_filters(position: &Position, options: &MineOptions) -> bool {
    let pieces = position.board_piece_count();
    if pieces < options.min_board_pieces || pieces > options.max_board_pieces {
        return false;
    }
    let defender = opponent(position.side);
    if position.king_square(defender) == NO_SQUARE {
        return false;
    }
    // A strict tsume may not start with the defender already in check.
    if position.is_in_check(defender) {
        return false;
    }
    let moves = legal_moves(position);
    if moves.is_empty() {
        return false;
    }
    moves.iter().any(|&mv| gives_check(position, mv))
}

/// What one mate search reports back to the loop.
struct MateHit {
    /// The mate distance (rule F applied, as the exact gate measures it); `0` when
    /// there is no mate -- or when there is one whose distance nobody settled, see
    /// [`MateHit::proving_bound`].
    mate_ply: u32,
    /// `--probe dfpn` only: the smallest bound the prober proved a mate at, set
    /// **only** when the exact solve at that bound ran out of budget, so the mate
    /// exists but its distance is unknown.  Under the classical reading of rule F
    /// that bound is not the distance (F1 can make the distance shorter), so it is
    /// recorded as `mine.provingBound` and never as `matePly`, and the
    /// distance-preserving prestrip does not run.
    proving_bound: Option<u32>,
    first_move: Option<Usi>,
    unique_first_move: bool,
    strict_valid: bool,
    nodes: u64,
    reasons: Vec<String>,
}

impl MateHit {
    /// A forced mate exists: its distance is known, or the prober proved one.
    fn mate_found(&self) -> bool {
        self.mate_ply > 0 || self.proving_bound.is_some()
    }
}

/// One bounded exact mate search (spec 9.1.5), by iterative deepening so a short
/// mate costs a short search.
///
/// Acceptance uses [`TsumeRules::candidate`], not the full puzzle rules: a position
/// lifted out of a self-played game still has the attacker's king on the board
/// (rule A) and whatever the attacker captured in hand (rule C), and both are
/// `distill`'s job to fix -- rule A with its mandatory pre-pass, rule C with the
/// hand-transfer operator.  Judging a raw candidate by the publishing rules would
/// report a hit rate of nearly zero for every self-play family and say nothing
/// about how many puzzles that family can actually yield.  `mine.strictValid`
/// therefore means "forced, minimal, uniquely-started and free of yozume": rule B
/// applies to a mined candidate exactly as it does to a published one, at every
/// mate length since `PIPELINE_VERSION` 0.12.0 (until then from 9 plies up only
/// under `--strict-rule-b`, P8-016, which became the default and was removed).
fn mate_search(
    solver: &mut Solver,
    prober: &mut Prober,
    position: &Position,
    options: &MineOptions,
) -> Option<MateHit> {
    let _phase = crate::prof::enter(crate::prof::Phase::MineMateSearch);
    let rules = TsumeRules::candidate().allowing_attacker_king(true);
    // `--probe dfpn`: find the minimal **real-ply** length with the prober and ask
    // the exact solver once, at that bound, for the verdict -- instead of paying the
    // exact solver for every refuted shorter bound on the way up.  The prober never
    // decides `strictValid`; when the exact search at that bound runs out of budget
    // the candidate is still reported with the length the prober proved, flagged
    // not strict-valid, which is the whole point on a board the exact solver cannot
    // finish.
    let mut probe_nodes = 0;
    // The prober reads rule F as far as it can (`dfpn_useless_mode`: F2 under the
    // classical reading).  Its proofs are sound for `rules` and its smallest
    // proving bound is an upper bound on the rule-F distance; the exact solve at
    // it measures the distance, and only a candidate whose exact solve ran out of
    // budget reports the bound -- as in `distill::source_probe`.  Under the
    // classical reading its disproofs are not conclusive (`PIPELINE_VERSION`
    // 0.11.0: the exact solver's credit search can still find a mate), so a
    // disproof falls through to the exact iterative deepening below.
    let mode = dfpn_useless_mode(rules);
    if options.probe.is_dfpn() {
        let probe = prober.minimal_mate_ply(
            position,
            &ProbeOptions::default()
                .max_ply(options.max_ply)
                .node_budget(options.dfpn_node_budget)
                .useless(mode),
        );
        probe_nodes = probe.nodes;
        match probe.verdict {
            Verdict::Disproven if dfpn_disproof_is_conclusive(rules) => {
                return Some(MateHit {
                    mate_ply: 0,
                    first_move: None,
                    unique_first_move: false,
                    strict_valid: false,
                    nodes: probe_nodes,
                    reasons: vec!["no_forced_mate_within_bound".to_string()],
                    proving_bound: None,
                });
            }
            Verdict::Proven => {
                let bound = probe.mate_ply.expect("a proven probe carries its length");
                let validation = exact_validate_with_rules(
                    solver,
                    position,
                    None,
                    bound,
                    options.node_budget,
                    false,
                    rules,
                );
                let nodes = probe_nodes + validation.result.stats.nodes;
                if validation.result.outcome != Outcome::Mate {
                    // The exact solve could not finish.  Under `keep` the proving
                    // bound *is* the distance; under rule F (the candidate rules
                    // read it classically) it is only an upper bound on it, so the
                    // distance stays unknown (Codex review 2026-10-07).
                    let (mate_ply, proving_bound) =
                        if rules.useless_mode() == crate::solver::UselessInterposition::Keep {
                            (bound, None)
                        } else {
                            (0, Some(bound))
                        };
                    return Some(MateHit {
                        mate_ply,
                        proving_bound,
                        first_move: None,
                        unique_first_move: false,
                        strict_valid: false,
                        nodes,
                        reasons: validation.reasons,
                    });
                }
                return Some(MateHit {
                    proving_bound: None,
                    mate_ply: validation.result.distance.unwrap_or(bound),
                    first_move: validation.first_move,
                    unique_first_move: validation.unique_first_move,
                    strict_valid: validation.accepted,
                    nodes,
                    reasons: validation.reasons,
                });
            }
            Verdict::Disproven | Verdict::Unknown => {}
        }
    }
    let (_, validation) = minimum_mate_validate_with_rules(
        solver,
        position,
        options.max_ply,
        options.node_budget,
        false,
        rules,
    );
    let nodes = probe_nodes + validation.result.stats.nodes;
    if validation.result.outcome != Outcome::Mate {
        return Some(MateHit {
            mate_ply: 0,
            proving_bound: None,
            first_move: None,
            unique_first_move: false,
            strict_valid: false,
            nodes,
            reasons: validation.reasons,
        });
    }
    Some(MateHit {
        mate_ply: validation.result.distance.unwrap_or(0),
        proving_bound: None,
        first_move: validation.first_move,
        unique_first_move: validation.unique_first_move,
        strict_valid: validation.accepted,
        nodes,
        reasons: validation.reasons,
    })
}

/// Recorded in `mine.prestrip.version` and in the `mine:prestrip-<v>` transformation.
pub const PRESTRIP_VERSION: &str = "v1";

/// `--prestrip`: greedily removes board pieces, farthest from the defender king
/// first, while the position keeps a mate of exactly `hit.mate_ply` (under the
/// strict material convention, so a removed piece is the defender's).  Returns
/// the stripped position, its own [`MateHit`] (whose `nodes` include every trial
/// search) and the number of pieces removed, or `None` when nothing could go.
fn prestrip(
    solver: &mut Solver,
    prober: &mut Prober,
    position: &Position,
    hit: &MateHit,
    options: &MineOptions,
) -> Option<(Position, MateHit, usize)> {
    let _phase = crate::prof::enter(crate::prof::Phase::MinePrestrip);
    if hit.mate_ply == 0 {
        return None;
    }
    let rules = TsumeRules::candidate().allowing_attacker_king(true);
    let mut nodes = 0;
    let (stripped, removed) =
        crate::prestrip::strip(position, crate::prestrip::Order::Distance, |trial| {
            if !cheap_filters(trial, options) {
                return crate::prestrip::Keep::No;
            }
            let (_, validation) = minimum_mate_validate_with_rules(
                solver,
                trial,
                hit.mate_ply,
                options.node_budget,
                false,
                rules,
            );
            nodes += validation.result.stats.nodes;
            if validation.result.outcome == Outcome::Mate
                && validation.result.distance == Some(hit.mate_ply)
            {
                crate::prestrip::Keep::Yes
            } else {
                crate::prestrip::Keep::No
            }
        });
    if removed.is_empty() {
        return None;
    }
    let mut stripped_hit = mate_search(solver, prober, &stripped, options)?;
    stripped_hit.nodes += nodes;
    Some((stripped, stripped_hit, removed.len()))
}

/// Everything needed to write one candidate record.
struct Candidate<'a> {
    position: &'a Position,
    start_sfen: &'a str,
    moves: &'a [String],
    game_index: u64,
    game_seed: u64,
    /// The distribution the start position was actually drawn from; `None` for
    /// family A, whose start position is the standard initial one.
    distribution: Option<Distribution>,
    ply: u32,
    hit: &'a MateHit,
    retrograde_distance: Option<usize>,
    /// `--prestrip`: the diagram before stripping and how many pieces went.
    unstripped: Option<(&'a Position, usize)>,
}

fn candidate_record(
    candidate: &Candidate<'_>,
    options: &MineOptions,
    meta: &PolicyMeta,
) -> PipelineRecord {
    let _phase = crate::prof::enter(crate::prof::Phase::MineRecord);
    let sfen = to_sfen(candidate.position);
    // Ids hash the pre-2026-10-04 spelling so a candidate keeps its id (`to_legacy_sfen_key`).
    let position_hash = sha256_text(&to_legacy_sfen_key(candidate.position));
    let family = options.family.as_str();
    let tag = if candidate.retrograde_distance.is_some() {
        "retro"
    } else {
        "mine"
    };
    let source_id = format!(
        "{tag}-{family}-{}-{}-{}",
        candidate.game_index,
        candidate.ply,
        &position_hash[..12]
    );
    let mut provenance = Provenance::generated(
        family,
        MINE_GENERATOR_VERSION,
        options.generator.config_hash(),
    );
    // `generatorDistribution` names the distribution this candidate was actually
    // drawn from, which is what a diversity report groups by.  Under the default
    // weighted mix that is one of `king-zone` / `slider-pressure` / `mate-shell`
    // per candidate; recording the *option* here instead (as `mate-mining-v1`
    // did) made every record of a weighted-mix run read `weighted-mix` and hid
    // the mix.  The option itself is kept beside it in
    // `generatorDistributionOption`.
    let distribution_option = match options.family {
        Family::StandardSelfplay => "standard-initial".to_string(),
        _ => options
            .distribution
            .map(|distribution| distribution.as_str().to_string())
            .unwrap_or_else(|| "weighted-mix".to_string()),
    };
    provenance.generator_distribution = Some(match candidate.distribution {
        Some(distribution) => distribution.as_str().to_string(),
        None => distribution_option.clone(),
    });
    provenance.generator_distribution_option = Some(distribution_option);
    provenance.proposal_index = Some(candidate.game_index);
    provenance.proposal_fingerprint = Some(sha256_text(&legacy_sfen_key_of(candidate.start_sfen)));
    provenance.source_position_hash = Some(position_hash);
    provenance.rng_seed = Some(candidate.game_seed.to_string());
    provenance.self_play_plies = Some(candidate.ply);
    provenance.historical_reachability_checked = options.family.historically_reachable();
    if candidate.hit.mate_ply > 0 {
        provenance.discovered_mate_ply = Some(candidate.hit.mate_ply);
    }
    if options.family.plays() {
        provenance.engine_name = Some(meta.name.clone());
        provenance.engine_version = Some(meta.version.clone());
        provenance.engine_options_hash = Some(meta.options_hash.clone());
    }
    // A mined candidate is project work (`docs/legal-data.md` §3): own origin,
    // its own id as the chain root, and no derivation steps yet.
    provenance.set_root_origin(
        crate::jsonl::ORIGIN_CLASS_OWN_GENERATED,
        crate::jsonl::OWN_ORIGIN,
        &source_id,
    );
    provenance.transformations = vec![format!("mine:{family}-{MINE_TRANSFORMATION_VERSION}")];
    if options.family.plays() {
        provenance
            .transformations
            .push(format!("mine:stochastic-selfplay-{}", candidate.ply));
    }
    provenance
        .transformations
        .push(format!("mine:bounded-mate-search-{}", options.max_ply));
    if candidate.retrograde_distance.is_some() {
        provenance
            .transformations
            .push("mine:retrograde-from-mate".to_string());
    }
    if candidate.unstripped.is_some() {
        provenance
            .transformations
            .push(format!("mine:prestrip-{PRESTRIP_VERSION}"));
    }
    let mut extra = serde_json::Map::new();
    extra.insert(
        "mine".to_string(),
        json!({
            "version": MINE_GENERATOR_VERSION,
            "family": family,
            "distribution": provenance.generator_distribution,
            "gameIndex": candidate.game_index,
            "gameSeed": candidate.game_seed.to_string(),
            "baseSeed": options.seed.to_string(),
            "startSfen": candidate.start_sfen,
            "plyIndex": candidate.ply,
            "moves": candidate.moves,
            // `null` when a mate was proven but its distance is unknown; the
            // proving bound then rides in `provingBound` (absent otherwise).
            "matePly": if candidate.hit.proving_bound.is_some() {
                Value::Null
            } else {
                json!(candidate.hit.mate_ply)
            },
            "firstMove": candidate.hit.first_move.map(|usi| usi.to_string()),
            "uniqueFirstMove": candidate.hit.unique_first_move,
            "strictValid": candidate.hit.strict_valid,
            // The attacker is the side to move: a mined candidate has not been
            // rotated to attacker-black yet, so a self-played position is just as
            // likely to have white on the move.
            "attackerKingOnBoard": candidate
                .position
                .king_square(candidate.position.side)
                != crate::position::NO_SQUARE,
            "attackerHand": candidate.position.hands[candidate.position.side as usize]
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            "rejectReasons": candidate.hit.reasons,
            "searchNodes": candidate.hit.nodes,
            "retrogradeFromMate": candidate.retrograde_distance.is_some(),
            "retrogradePlies": candidate.retrograde_distance,
            "options": options.to_json(),
            "policy": {
                "kind": meta.kind,
                "name": meta.name,
                "version": meta.version,
                "optionsHash": meta.options_hash,
                "options": meta.options,
            },
            "generatorConfig": options.generator.to_json(),
        }),
    );
    // Only on a proven mate of unknown distance, so every other record keeps its
    // bytes.
    if let (Some(bound), Some(Value::Object(mine))) =
        (candidate.hit.proving_bound, extra.get_mut("mine"))
    {
        mine.insert("provingBound".to_string(), json!(bound));
    }
    if let Some((original, removed)) = candidate.unstripped {
        if let Some(Value::Object(mine)) = extra.get_mut("mine") {
            mine.insert(
                "prestrip".to_string(),
                json!({
                    "version": PRESTRIP_VERSION,
                    "removed": removed,
                    "sourceSfen": to_sfen(original),
                }),
            );
        }
    }
    PipelineRecord::new_imported(source_id, sfen, provenance, extra)
}

/// The start position of one game and the distribution it was drawn from (`None`
/// for the standard initial position), or `None` when the constrained generator
/// could not satisfy its constraints for this seed.
fn start_position(
    index: u64,
    seed: u64,
    options: &MineOptions,
) -> Option<(Position, Option<Distribution>)> {
    if options.family == Family::StandardSelfplay {
        return parse_sfen(STANDARD_INITIAL_SFEN)
            .ok()
            .map(|position| (position, None));
    }
    let (mut position, drawn) =
        generate_position(seed, &options.generator.generator, options.distribution)?;
    let flip = match options.generator.side_to_move_policy {
        SideToMovePolicy::AttackerBiased => false,
        SideToMovePolicy::Balanced => index % 2 == 1,
        SideToMovePolicy::Random => {
            SplitMix64::new(seed ^ 0x5851_F42D_4C95_7F2D).next_u64() & 1 == 1
        }
    };
    if flip {
        position.side = crate::piece::WHITE;
        position.refresh();
        if position.is_in_check(crate::piece::WHITE) {
            // Flipping must not hand the mover an already-checked king.
            position.side = BLACK;
            position.refresh();
        }
    }
    Some((position, Some(drawn)))
}

/// Result of playing one game.
#[derive(Default)]
struct GameOutput {
    records: Vec<PipelineRecord>,
    stats: MineStats,
}

/// Plays and mines one game.  Pure in `(options, index)` apart from the policy's
/// own state, which is itself a pure function of the position.
fn run_game(
    solver: &mut Solver,
    prober: &mut Prober,
    policy: &mut Policy,
    meta: &PolicyMeta,
    index: u64,
    options: &MineOptions,
    deadline: &dyn Fn() -> bool,
) -> Result<GameOutput> {
    let _phase = crate::prof::enter(crate::prof::Phase::MineGenerate);
    let mut out = GameOutput::default();
    out.stats.games = 1;
    let seed = game_seed(options.seed, index);
    let Some((start, distribution)) = start_position(index, seed, options) else {
        out.stats.generator_failures = 1;
        return Ok(out);
    };
    let start_sfen = to_sfen(&start);
    let mut rng = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let span = options.plies.1.saturating_sub(options.plies.0) + 1;
    let plies = if options.family.plays() {
        options.plies.0 + (rng.next_u64() % span as u64) as u32
    } else {
        0
    };
    let mut position = start.clone();
    let mut moves: Vec<String> = Vec::new();
    let mut history: Vec<Position> = vec![start.clone()];
    let mut checkmated = false;
    for ply in 0..=plies {
        let eligible = ply >= options.check_from
            && (ply.saturating_sub(options.check_from)) % options.check_every.max(1) == 0;
        if eligible {
            out.stats.positions_considered += 1;
            if !cheap_filters(&position, options) {
                out.stats.cheap_filtered += 1;
            } else {
                out.stats.mate_searches += 1;
                if let Some(hit) = mate_search(solver, prober, &position, options) {
                    out.stats.search_nodes += hit.nodes;
                    if hit.mate_found() {
                        out.stats.mates_found += 1;
                        if hit.strict_valid {
                            out.stats.strict_valid += 1;
                        }
                        if hit.unique_first_move {
                            out.stats.unique_first_move += 1;
                        }
                        let stripped = if options.prestrip {
                            prestrip(solver, prober, &position, &hit, options)
                        } else {
                            None
                        };
                        let (emit_position, emit_hit, unstripped) = match &stripped {
                            Some((stripped, stripped_hit, removed)) => {
                                out.stats.search_nodes += stripped_hit.nodes;
                                (stripped, stripped_hit, Some((&position, *removed)))
                            }
                            None => (&position, &hit, None),
                        };
                        if !options.require_strict_valid || emit_hit.strict_valid {
                            out.records.push(candidate_record(
                                &Candidate {
                                    position: emit_position,
                                    start_sfen: &start_sfen,
                                    moves: &moves,
                                    game_index: index,
                                    game_seed: seed,
                                    distribution,
                                    ply,
                                    hit: emit_hit,
                                    retrograde_distance: None,
                                    unstripped,
                                },
                                options,
                                meta,
                            ));
                            out.stats.emitted += 1;
                        }
                    }
                }
            }
        }
        if ply == plies || deadline() {
            break;
        }
        let Some(mv) = policy.select(&position, &mut rng)? else {
            // No legal move: checkmate when in check, and a loss either way.
            checkmated = position.is_in_check(position.side);
            break;
        };
        moves.push(mv.to_usi().to_string());
        position.make(mv);
        history.push(position.clone());
        out.stats.plies_played += 1;
    }
    if checkmated && options.retrograde > 0 {
        out.stats.checkmated_games += 1;
        let mated = history.len() - 1;
        for step in 0..options.retrograde {
            let back = 1 + step * 2;
            if back > mated {
                break;
            }
            let candidate_position = &history[mated - back];
            if !cheap_filters(candidate_position, options) {
                out.stats.cheap_filtered += 1;
                continue;
            }
            out.stats.mate_searches += 1;
            let Some(hit) = mate_search(solver, prober, candidate_position, options) else {
                continue;
            };
            out.stats.search_nodes += hit.nodes;
            if hit.mate_found() {
                out.stats.mates_found += 1;
                if hit.strict_valid {
                    out.stats.strict_valid += 1;
                }
                if hit.unique_first_move {
                    out.stats.unique_first_move += 1;
                }
            }
            let stripped = if options.prestrip && hit.mate_ply > 0 {
                prestrip(solver, prober, candidate_position, &hit, options)
            } else {
                None
            };
            let (emit_position, emit_hit, unstripped) = match &stripped {
                Some((stripped, stripped_hit, removed)) => {
                    out.stats.search_nodes += stripped_hit.nodes;
                    (stripped, stripped_hit, Some((candidate_position, *removed)))
                }
                None => (candidate_position, &hit, None),
            };
            out.records.push(candidate_record(
                &Candidate {
                    position: emit_position,
                    start_sfen: &start_sfen,
                    moves: &moves[..(mated - back).min(moves.len())],
                    game_index: index,
                    game_seed: seed,
                    distribution,
                    ply: (mated - back) as u32,
                    hit: emit_hit,
                    retrograde_distance: Some(back),
                    unstripped,
                },
                options,
                meta,
            ));
            out.stats.emitted += 1;
            out.stats.retrograde_emitted += 1;
        }
    }
    Ok(out)
}

/// Result of a whole mining run.
#[derive(Clone, Copy, Debug)]
pub struct MineRun {
    pub stats: MineStats,
    pub wall_seconds: f64,
    pub threads: usize,
}

/// Runs the mining loop and writes JSONL candidates to `out`.
///
/// Workers pull game indices from one shared counter, so the set of indices each
/// worker handles is disjoint and the seed of every game depends only on the base
/// seed and the index (spec 9.1.12).  Finished games are written in game-index
/// order through a reorder buffer, and `count` cuts that ordered stream, so the
/// output -- and the run counters, which merge exactly the games that reached the
/// output -- are the same bytes for every thread count.  (Before 2026-10-01 records
/// were written in completion order, so only `threads = 1` was reproducible.)
/// A `time_limit` stop is the one exception: where it falls depends on the clock.
pub fn run(
    options: &MineOptions,
    spec: &PolicySpec,
    threads: usize,
    out: &mut dyn Write,
    progress: bool,
) -> Result<MineRun> {
    let threads = threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("building the mining worker pool")?;
    let next_game = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let (tx, rx) = std::sync::mpsc::channel::<(u64, Result<(Vec<String>, MineStats)>)>();
    let started = Instant::now();
    let time_limit = options.time_limit;
    let expired = || match time_limit {
        Some(limit) => started.elapsed().as_secs_f64() >= limit,
        None => false,
    };
    let mut stats = MineStats::default();
    pool.in_place_scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let next_game = &next_game;
            let stop = &stop;
            let failure = &failure;
            scope.spawn(move |_| {
                let mut solver = Solver::new();
                let mut prober = Prober::new();
                let mut policy = match spec.build() {
                    Ok(policy) => policy,
                    Err(error) => {
                        *failure.lock().expect("policy failure lock") = Some(error);
                        stop.store(true, Ordering::Relaxed);
                        return;
                    }
                };
                let meta = policy.meta();
                loop {
                    if stop.load(Ordering::Relaxed) || expired() {
                        break;
                    }
                    let index = next_game.fetch_add(1, Ordering::Relaxed);
                    if options.max_games.is_some_and(|max| index >= max) {
                        break;
                    }
                    let deadline = || stop.load(Ordering::Relaxed) || expired();
                    let game = run_game(
                        &mut solver,
                        &mut prober,
                        &mut policy,
                        &meta,
                        index,
                        options,
                        &deadline,
                    )
                    .and_then(|game| {
                        let lines = game
                            .records
                            .iter()
                            .map(write_json_line)
                            .collect::<Result<Vec<_>>>()?;
                        Ok((lines, game.stats))
                    });
                    let failed = game.is_err();
                    if tx.send((index, game)).is_err() || failed {
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            });
        }
        drop(tx);
        // Reorder buffer: game `next` is written only after every game before it.
        let mut pending: std::collections::BTreeMap<u64, (Vec<String>, MineStats)> =
            std::collections::BTreeMap::new();
        let mut next = 0u64;
        let mut written = 0usize;
        let mut done = options.count == 0;
        let mut last_report = Instant::now();
        for (index, game) in rx {
            if done {
                continue;
            }
            match game {
                Ok(game) => {
                    pending.insert(index, game);
                }
                Err(error) => {
                    *failure.lock().expect("game failure lock") = Some(error);
                    stop.store(true, Ordering::Relaxed);
                    done = true;
                    continue;
                }
            }
            while let Some((lines, game_stats)) = pending.remove(&next) {
                next += 1;
                stats.merge(&game_stats);
                for line in lines {
                    if written >= options.count {
                        break;
                    }
                    if let Err(error) = writeln!(out, "{line}") {
                        *failure.lock().expect("write failure lock") = Some(error.into());
                        stop.store(true, Ordering::Relaxed);
                        done = true;
                        break;
                    }
                    written += 1;
                }
                if written >= options.count {
                    stop.store(true, Ordering::Relaxed);
                    done = true;
                }
                if done {
                    break;
                }
            }
            if progress && last_report.elapsed().as_secs_f64() >= 2.0 {
                let elapsed = started.elapsed().as_secs_f64().max(1e-9);
                eprintln!(
                    "tsume-solver mine: {written}/{} candidates, {:.1}/s, {} games",
                    options.count,
                    written as f64 / elapsed,
                    next_game.load(Ordering::Relaxed)
                );
                last_report = Instant::now();
            }
        }
        // Games past a gap left by a deadline stop never reach the output.
    });
    if let Some(error) = failure.into_inner().expect("failure lock") {
        return Err(error);
    }
    out.flush()?;
    stats.emitted = stats.emitted.min(options.count as u64);
    Ok(MineRun {
        stats,
        wall_seconds: started.elapsed().as_secs_f64(),
        threads,
    })
}

/// Convenience wrapper used by tests: collects the JSONL output into a string.
pub fn run_to_string(
    options: &MineOptions,
    spec: &PolicySpec,
    threads: usize,
) -> Result<(String, MineRun)> {
    let mut buffer: Vec<u8> = Vec::new();
    let run = run(options, spec, threads, &mut buffer, false)?;
    Ok((String::from_utf8(buffer)?, run))
}

/// Default sampling configuration, exposed so the CLI and tests agree.
pub fn default_sampling() -> SamplingConfig {
    SamplingConfig::default()
}
