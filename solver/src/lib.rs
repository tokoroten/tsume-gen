//! # tsume-solver
//!
//! The offline strict-tsume rules engine, exact solver and JSONL pipeline stages
//! for TSUME (spec 9, 10, 11).  It reproduces the semantics of the TypeScript
//! reference implementation (`packages/core-rules`, `packages/solver`,
//! `tools/puzzle-pipeline`) exactly, and much faster, so the production pipeline
//! can process millions of candidates and re-solve each one dozens of times
//! during reduction.
//!
//! ## Parity contract
//!
//! * [`movegen::legal_moves`] returns the same set as `generateLegalMoves`
//!   (compare as sorted USI).
//! * [`strict_material::normalize`] equals `withStrictDefenderMaterial`.
//! * [`solver::Solver::solve`] equals `solveTsume`, including
//!   `stats.nodes`, `stats.cacheHits` and the exported graph structure.
//! * [`stages`] read and write the same `PipelineRecord` documents as the
//!   TypeScript `validate`, `reduce` and `solve` stages, including the
//!   bit-for-bit reducer (`SplitMix64`, seed derivation, shuffle, acceptance
//!   test, tie-break).
//!
//! ## Layout
//!
//! | module | role |
//! | --- | --- |
//! | [`piece`] | piece codes, direction and attack tables |
//! | [`position`] | mailbox board, hands, incremental Zobrist key, make/unmake |
//! | [`sfen`] | SFEN parse/serialise, USI squares |
//! | [`movegen`] | pseudo-moves, legality, `uchi-fuzume`, perft |
//! | [`validate`] | structural diagram validation |
//! | [`strict_material`] | strict-tsume defender remainder |
//! | [`solver`] | bounded AND/OR search, transposition table, graph |
//! | [`exact`] | publishable-puzzle gate and runtime graph pruning |
//! | [`dfpn`] | df-pn mate **prober** (never a verdict; see the module docs) |
//! | [`pardfpn`] | the lock-free tables of the parallel classical prober (`--prober-threads`) |
//! | [`rule_b_probe`] | df-pn rule B pre-filter: reject-only, never a verdict |
//! | [`inert`] | rule G: board pieces the solution never uses (無駄駒) |
//! | `keytable` | exact open-addressing map of the classical prober's tables |
//! | [`features`] | difficulty features |
//! | [`canonical`] | attacker orientation, mirror, runtime encoding, exact hash |
//! | [`reduce`] | seeded greedy piece reduction |
//! | [`stages`] | JSONL pipeline stages |
//! | [`jsonl`] | `PipelineRecord` serde model |
//! | [`catalog_record`] | `--emit-record`: catalog records from the gate's facts |
//! | [`generate`] | candidate-generation extension points |
//! | [`rng`] | `SplitMix64` parity RNG |
//! | [`unmove`] | exact 2-ply predecessor enumeration (the reverse method) |
//! | [`extend`] | the model-free reverse method: enumerate, then gate every candidate |
//! | [`schedule`] | order-preserving bounded worker pool for the JSONL stages |
//! | [`cpu`] | measured process CPU time for the `--stats-out` counters |
//!
//! ## Extension points
//!
//! [`generate`] holds the constrained-random position source and the
//! [`generate::MovePolicy`] interface; mining and distillation are built on
//! [`solver::Solver`] and [`reduce::reduce`].

pub mod bitboard;
pub mod canonical;
pub mod catalog_record;
pub mod cdfpn;
pub mod clock;
pub mod construct;
pub mod cpu;
pub mod dfpn;
pub mod diagnostics;
pub mod distill;
pub mod engine;
pub mod exact;
pub mod extend;
pub mod features;
pub mod generate;
pub mod inert;
pub mod jsonl;
mod keytable;
pub mod mine;
pub mod movegen;
pub mod negatives;
pub mod pardfpn;
pub mod piece;
pub mod position;
pub mod posprof;
pub mod prestrip;
pub mod prof;
pub mod quality;
pub mod record_clock;
pub mod reduce;
pub mod rng;
pub mod rule_b_probe;
pub mod schedule;
pub mod sfen;
pub mod solver;
pub mod stages;
pub mod stored_paths;
pub mod strict_material;
pub mod unmove;
pub mod util;
pub mod validate;
pub mod zobrist;

pub use movegen::{gives_check, legal_moves, legal_moves_sorted, perft};
pub use piece::{opponent, BLACK, WHITE};
pub use position::{Move, Position, Usi};
pub use sfen::{parse_sfen, to_sfen, SfenError};
pub use solver::{
    Outcome, SearchLimits, SearchRules, SolveOptions, SolveResult, SolveStats, Solver,
};

/// `true` when `color` is in check.
pub fn is_in_check(position: &Position, color: u8) -> bool {
    position.is_in_check(color)
}
