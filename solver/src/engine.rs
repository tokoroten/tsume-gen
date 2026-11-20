//! The built-in stochastic playing engine used by `mine` (spec 9.1.4).
//!
//! This is deliberately a *small but real* engine, not a placeholder: alpha-beta
//! with iterative deepening, a node cap, capture-ordered moves, a quiescence
//! search over captures, and a material + king-safety/mobility evaluation.  Its
//! job is to pull a random diagram toward a coherent attack/defence structure
//! (generation family C), not to play strong shogi.
//!
//! Move *selection* is stochastic on purpose (spec 9.1.4): deterministic
//! best-move self-play collapses candidate diversity.  [`sample_top_k`] samples
//! from the top-K root moves that lie within an evaluation gap of the best, using
//! the example rank distribution of the spec (~50 / 25 / 15 / 10 %) scaled by a
//! temperature-weighted evaluation factor.  (It also drove the external USI
//! adapter, removed on 2026-10-27.)

use serde_json::json;

use crate::generate::MovePolicy;
use crate::movegen::{legal_moves, MoveBuf};
use crate::piece::*;
use crate::position::{Move, Position, NO_SQUARE};
use crate::rng::Rng;
use crate::util::{sha256_text, stable_json};

/// Engine identity recorded in `provenance.engineName`.
pub const BUILTIN_ENGINE_NAME: &str = "tsume-builtin-ab";
/// Engine identity recorded in `provenance.engineVersion`.
pub const BUILTIN_ENGINE_VERSION: &str = "0.1.0";

/// Score of a mate at ply 0.  Far outside any material evaluation.
pub const MATE_VALUE: i32 = 100_000;
/// Infinity for the alpha-beta window.
const INF: i32 = 1_000_000;
/// Hard cap on quiescence depth, so a capture chain cannot run away.
const MAX_QUIESCENCE_PLY: i32 = 24;

/// Centipawn value of an unpromoted piece, indexed by piece type.
const VALUE: [i32; 8] = [90, 315, 405, 495, 540, 855, 990, 0];
/// Centipawn value of a promoted piece, indexed by base piece type.
const PROMOTED_VALUE: [i32; 8] = [540, 540, 540, 540, 540, 945, 1395, 0];
/// A piece in hand is worth slightly more than the same piece on the board
/// because it can be dropped anywhere.
const HAND_BONUS: i32 = 15;
/// Per own piece adjacent to the own king.
const SHELTER: i32 = 10;
/// Per king-neighbour square that is not blocked by an own piece.
const KING_MOBILITY: i32 = 8;
/// Per enemy piece inside the own king zone (Chebyshev radius 2).
const ZONE_PRESSURE: i32 = 16;

/// The spec 9.1.4 example distribution over accepted candidates.
const RANK_WEIGHTS: [f64; 4] = [0.50, 0.25, 0.15, 0.10];

/// A root move with its search score, in centipawns from the mover's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

/// Stochastic top-K sampling parameters (spec 9.1.4).
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    /// Number of candidates considered, best first.
    pub top_k: usize,
    /// Maximum centipawn gap from the best move for a candidate to be accepted.
    pub eval_gap: i32,
    /// `0` picks the best accepted candidate; larger values flatten toward the
    /// rank distribution.
    pub temperature: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            eval_gap: 120,
            temperature: 1.0,
        }
    }
}

/// Built-in engine configuration.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// Iterative-deepening target depth in plies.
    pub depth: u32,
    /// Node cap per decision.  Search stops and the last completed iteration wins.
    pub node_cap: u64,
    pub sampling: SamplingConfig,
    /// Quiescence search over captures.
    pub quiescence: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            depth: 4,
            node_cap: 30_000,
            sampling: SamplingConfig::default(),
            quiescence: true,
        }
    }
}

impl EngineConfig {
    /// The JSON recorded in provenance and hashed into `engineOptionsHash`.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "depth": self.depth,
            "nodeCap": self.node_cap,
            "topK": self.sampling.top_k,
            "evalGap": self.sampling.eval_gap,
            "temperature": self.sampling.temperature,
            // `mine --attacker-bias` (a centipawn bonus for checks and captures
            // at the root) went on 2026-10-27; the key keeps its only value so
            // `engineOptionsHash` does not change.
            "attackerBias": 0,
            "quiescence": self.quiescence,
        })
    }

    /// `engineOptionsHash`: SHA-256 over the stable JSON of [`Self::to_json`].
    pub fn options_hash(&self) -> String {
        sha256_text(&stable_json(&self.to_json()))
    }
}

/// Static evaluation from the point of view of the side to move.
///
/// Material dominates; the positional terms are king shelter, king mobility
/// (unblocked neighbour squares) and enemy pressure inside each king zone, plus a
/// tiny advancement bonus so idle pieces still prefer to move forward.  Every
/// term is a plain board scan, which keeps the leaf cost low enough for the
/// default 30 k node cap per decision.
pub fn evaluate(position: &Position) -> i32 {
    let mut score = [0i32; 2];
    for (index, &c) in position.board.iter().enumerate() {
        if c == EMPTY {
            continue;
        }
        let color = color_of(c) as usize;
        let t = ptype(c) as usize;
        if t == PT_K as usize {
            continue;
        }
        score[color] += if is_promoted(c) {
            PROMOTED_VALUE[t]
        } else {
            VALUE[t]
        };
        let rank = rank_of(index) as i32;
        score[color] += if color == BLACK as usize {
            8 - rank
        } else {
            rank
        };
    }
    for (color, hand) in position.hands.iter().enumerate() {
        for (t, &count) in hand.iter().enumerate() {
            score[color] += count as i32 * (VALUE[t] + HAND_BONUS);
        }
    }
    let tables = &TABLES;
    for color in [BLACK, WHITE] {
        let king = position.king_square(color);
        if king == NO_SQUARE {
            continue;
        }
        let index = color as usize;
        let king = king as usize;
        for &(_, target) in &tables.neighbours[king][..tables.neighbours_len[king] as usize] {
            let c = position.board[target as usize];
            if c != EMPTY && color_of(c) == color {
                score[index] += SHELTER;
            } else {
                score[index] += KING_MOBILITY;
            }
        }
        let (kf, kr) = (file_of(king) as i32, rank_of(king) as i32);
        for (square, &c) in position.board.iter().enumerate() {
            if c == EMPTY || color_of(c) == color {
                continue;
            }
            let distance = (file_of(square) as i32 - kf)
                .abs()
                .max((rank_of(square) as i32 - kr).abs());
            if distance <= 2 {
                score[index] -= ZONE_PRESSURE;
            }
        }
    }
    let me = position.side as usize;
    score[me] - score[1 - me]
}

/// Alpha-beta engine.  One instance per worker thread: it owns a move stack and a
/// node counter and holds no cross-position state, so every decision depends only
/// on the position and the configuration.
pub struct BuiltinEngine {
    pub config: EngineConfig,
    buf: MoveBuf,
    nodes: u64,
    /// Nodes visited by the most recent [`Self::root_scores`] call.
    pub last_nodes: u64,
}

impl Default for BuiltinEngine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

impl BuiltinEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            buf: MoveBuf::new(),
            nodes: 0,
            last_nodes: 0,
        }
    }

    /// Every legal root move with its score, best first.  Ties break on USI text,
    /// so the list is deterministic for a given position and configuration.
    pub fn root_scores(&mut self, position: &Position) -> Vec<ScoredMove> {
        self.nodes = 0;
        let moves = legal_moves(position);
        if moves.is_empty() {
            self.last_nodes = 0;
            return Vec::new();
        }
        let mut work = position.clone();
        let mut best: Vec<ScoredMove> = moves
            .iter()
            .map(|&mv| ScoredMove { mv, score: 0 })
            .collect();
        best.sort_unstable_by_key(|sm| sm.mv.to_usi());
        for depth in 1..=self.config.depth as i32 {
            let mut round: Vec<ScoredMove> = Vec::with_capacity(best.len());
            for sm in best.iter() {
                if self.nodes >= self.config.node_cap {
                    break;
                }
                let undo = work.make(sm.mv);
                let score = -self.negamax(&mut work, depth - 1, 1, -INF, INF);
                work.unmake(sm.mv, undo);
                round.push(ScoredMove { mv: sm.mv, score });
            }
            if round.len() < best.len() {
                // The node cap hit mid-iteration; keep the last complete one.
                break;
            }
            round.sort_by(|a, b| {
                b.score
                    .cmp(&a.score)
                    .then_with(|| a.mv.to_usi().cmp(&b.mv.to_usi()))
            });
            best = round;
            if self.nodes >= self.config.node_cap {
                break;
            }
        }
        self.last_nodes = self.nodes;
        best
    }

    fn negamax(&mut self, pos: &mut Position, depth: i32, ply: i32, alpha: i32, beta: i32) -> i32 {
        if depth <= 0 {
            return if self.config.quiescence {
                self.quiesce(pos, ply, alpha, beta)
            } else {
                self.nodes += 1;
                evaluate(pos)
            };
        }
        self.nodes += 1;
        if self.nodes >= self.config.node_cap {
            return evaluate(pos);
        }
        let range = self.buf.gen_legal(pos, true);
        if range.is_empty() {
            self.buf.buf.truncate(range.start);
            // No legal move in shogi loses, in check or not.
            return -MATE_VALUE + ply;
        }
        self.order(pos, range.clone());
        let mut alpha = alpha;
        let mut best = -INF;
        for index in range.clone() {
            let mv = self.buf.buf[index];
            let undo = pos.make(mv);
            let score = -self.negamax(pos, depth - 1, ply + 1, -beta, -alpha);
            pos.unmake(mv, undo);
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        self.buf.buf.truncate(range.start);
        best
    }

    fn quiesce(&mut self, pos: &mut Position, ply: i32, alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        let in_check = pos.check_info(pos.side).in_check();
        let stand = evaluate(pos);
        let mut alpha = alpha;
        let exhausted = self.nodes >= self.config.node_cap || ply >= MAX_QUIESCENCE_PLY;
        if in_check {
            if exhausted {
                return stand;
            }
        } else {
            if stand >= beta || exhausted {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
        }
        let range = self.buf.gen_legal(pos, true);
        if range.is_empty() {
            self.buf.buf.truncate(range.start);
            return -MATE_VALUE + ply;
        }
        let hot = if in_check {
            range.clone()
        } else {
            // Captures only: everything else is assumed quiet enough.
            let mut write = range.start;
            for read in range.clone() {
                let mv = self.buf.buf[read];
                if !mv.is_drop() && pos.board[mv.to as usize] != EMPTY {
                    self.buf.buf[write] = mv;
                    write += 1;
                }
            }
            self.buf.buf.truncate(write);
            range.start..write
        };
        if hot.is_empty() {
            self.buf.buf.truncate(range.start);
            return stand;
        }
        self.order(pos, hot.clone());
        let mut best = if in_check { -INF } else { stand };
        for index in hot.clone() {
            let mv = self.buf.buf[index];
            let undo = pos.make(mv);
            let score = -self.quiesce(pos, ply + 1, -beta, -alpha);
            pos.unmake(mv, undo);
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        self.buf.buf.truncate(range.start);
        best
    }

    /// Captures first, by victim value, then promotions, then the rest; USI text
    /// breaks every tie so the order never depends on generation accidents.
    fn order(&mut self, pos: &Position, range: std::ops::Range<usize>) {
        self.buf.buf[range].sort_unstable_by_key(|mv| {
            let victim = if mv.is_drop() {
                EMPTY
            } else {
                pos.board[mv.to as usize]
            };
            let mut gain = 0i32;
            if victim != EMPTY {
                let t = ptype(victim) as usize;
                gain += 1_000
                    + if is_promoted(victim) {
                        PROMOTED_VALUE[t]
                    } else {
                        VALUE[t]
                    };
            }
            if mv.promote {
                gain += 300;
            }
            (-gain, mv.to_usi())
        });
    }
}

/// Samples one move from `scores` (best first) per spec 9.1.4.
///
/// Candidates are the first `top_k` moves within `eval_gap` centipawns of the
/// best.  Candidate `i` gets weight `RANK_WEIGHTS[i] * exp(-deficit / (eval_gap *
/// temperature))`, so `temperature = 0` always plays the best move and a large
/// temperature approaches the flat 50/25/15/10 rank distribution.
pub fn sample_top_k(
    scores: &[ScoredMove],
    config: &SamplingConfig,
    rng: &mut dyn Rng,
) -> Option<Move> {
    let best = scores.first()?.score;
    let gap = config.eval_gap.max(0);
    let pool: Vec<&ScoredMove> = scores
        .iter()
        .filter(|entry| best.saturating_sub(entry.score) <= gap)
        .take(config.top_k.max(1))
        .collect();
    let chosen = pool.first().copied()?;
    if pool.len() == 1 || config.temperature <= 0.0 {
        return Some(chosen.mv);
    }
    let scale = (gap.max(1) as f64) * config.temperature;
    let weights: Vec<f64> = pool
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let rank = RANK_WEIGHTS
                .get(index)
                .copied()
                .unwrap_or(RANK_WEIGHTS[3] / (index as f64 - 2.0));
            let deficit = best.saturating_sub(entry.score) as f64;
            rank * (-deficit / scale).exp()
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Some(chosen.mv);
    }
    let mut target = rng.float() * total;
    for (entry, weight) in pool.iter().zip(weights.iter()) {
        target -= weight;
        if target <= 0.0 {
            return Some(entry.mv);
        }
    }
    Some(pool[pool.len() - 1].mv)
}

impl MovePolicy for BuiltinEngine {
    fn select(&mut self, position: &Position, rng: &mut dyn Rng) -> Option<Move> {
        let scores = self.root_scores(position);
        sample_top_k(&scores, &self.config.sampling, rng)
    }
}
