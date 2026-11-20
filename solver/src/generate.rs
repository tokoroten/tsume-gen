//! Candidate generation extension points (spec 9.1.14).
//!
//! A constrained-random position source ([`ConstrainedRandomSource`]) and the
//! [`MovePolicy`] interface with a uniform implementation.  Self-play mining
//! (`mine`), engine policies and distillation live in their own modules.  (The
//! `PositionSource` / `MateMiner` traits and `ExactMateMiner`, never used by
//! those loops, were removed on 2026-10-27.)
//!
//! [`ConstrainedRandomSource`] follows the distributions of
//! `tools/puzzle-pipeline/src/generator.ts` (`king-zone`, `slider-pressure`,
//! `mate-shell`) closely enough to reproduce the same candidate stream for a
//! given seed, which is what makes the Rust and TypeScript benchmarks
//! comparable.  Bit parity is a convenience here, not a contract.

use crate::movegen::legal_moves;
use crate::piece::*;
use crate::position::{Move, Position};
use crate::rng::{Rng, RngExt, SplitMix64};

/// Chooses a move during self-play or candidate walking.
pub trait MovePolicy {
    /// Returns `None` when the side to move has no legal move.
    fn select(&mut self, position: &Position, rng: &mut dyn Rng) -> Option<Move>;
}

/// `RandomPositionGeneratorConfig`.
#[derive(Clone, Copy, Debug)]
pub struct GeneratorConfig {
    pub min_board_pieces: usize,
    pub max_board_pieces: usize,
    pub max_material_imbalance: i32,
    pub allow_initial_check: bool,
    pub king_zone_radius: i32,
    pub king_zone_occupancy: (usize, usize),
    pub hand_piece_rate: f64,
    pub promotion_rate: f64,
    /// Rule E: place an attacker king.  **Off by default**, because rule A says a
    /// published puzzle has none, so generating one only to delete it again wastes
    /// a square and distorts the king-zone occupancy constraint.  Self-play
    /// families turn it back on, since a game needs both kings to be playable.
    pub attacker_king: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_board_pieces: 5,
            max_board_pieces: 14,
            max_material_imbalance: 8,
            allow_initial_check: false,
            king_zone_radius: 2,
            king_zone_occupancy: (2, 8),
            hand_piece_rate: 0.42,
            promotion_rate: 0.14,
            attacker_king: false,
        }
    }
}

/// The generator distributions of `generator.ts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    KingZone,
    SliderPressure,
    MateShell,
    /// Unweighted scatter (owner 2026-10-01): kings, pieces, squares and colours
    /// all uniform, many pieces, and `distill` removes what the mate does not use.
    /// Never part of the default weighted mix; ask for it with `--distribution`.
    Uniform,
}

impl Distribution {
    pub const fn as_str(self) -> &'static str {
        match self {
            Distribution::KingZone => "king-zone",
            Distribution::SliderPressure => "slider-pressure",
            Distribution::MateShell => "mate-shell",
            Distribution::Uniform => "uniform",
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "king-zone" => Some(Distribution::KingZone),
            "slider-pressure" => Some(Distribution::SliderPressure),
            "mate-shell" => Some(Distribution::MateShell),
            "uniform" => Some(Distribution::Uniform),
            _ => None,
        }
    }
}

const INVENTORY: [usize; 7] = [18, 4, 4, 4, 4, 2, 2];
const WEIGHTED_DISTRIBUTIONS: [Distribution; 4] = [
    Distribution::KingZone,
    Distribution::KingZone,
    Distribution::SliderPressure,
    Distribution::MateShell,
];

fn chebyshev(a: usize, b: usize) -> i32 {
    let (af, ar) = (file_of(a) as i32, rank_of(a) as i32);
    let (bf, br) = (file_of(b) as i32, rank_of(b) as i32);
    (af - bf).abs().max((ar - br).abs())
}

fn random_empty(
    position: &Position,
    rng: &mut dyn Rng,
    predicate: impl Fn(usize) -> bool,
) -> Option<usize> {
    let choices: Vec<usize> = (0..81)
        .filter(|&index| position.board[index] == EMPTY && predicate(index))
        .collect();
    if choices.is_empty() {
        None
    } else {
        Some(*rng.pick(&choices))
    }
}

fn dead_square(t: u8, color: u8, rank: usize) -> bool {
    match t {
        PT_P | PT_L => rank == if color == BLACK { 0 } else { 8 },
        PT_N => {
            if color == BLACK {
                rank <= 1
            } else {
                rank >= 7
            }
        }
        _ => false,
    }
}

fn has_pawn_in_file(position: &Position, color: u8, file: usize) -> bool {
    (0..9).any(|rank| {
        let c = position.board[sq(file, rank)];
        c != EMPTY && color_of(c) == color && ptype(c) == PT_P && !is_promoted(c)
    })
}

fn can_place(position: &Position, index: usize, t: u8, color: u8, promoted: bool) -> bool {
    position.board[index] == EMPTY
        && (promoted || !dead_square(t, color, rank_of(index)))
        && !(t == PT_P && !promoted && has_pawn_in_file(position, color, file_of(index)))
}

/// Places the defender king on rank a-c and, when `attacker_king` is set, an
/// attacker king on rank f-i that is not adjacent to it.
fn base_position(rng: &mut dyn Rng, attacker_king: bool) -> (Position, usize) {
    let mut position = Position::empty();
    let defender_rank = rng.int(3);
    let defender_file = rng.int(9);
    let defender_king = defender_rank * 9 + defender_file;
    position.board[defender_king] = code(WHITE, PT_K, false);
    if attacker_king {
        let square = random_empty(&position, rng, |index| {
            rank_of(index) >= 5 && chebyshev(index, defender_king) > 1
        })
        .expect("attacker king square");
        position.board[square] = code(BLACK, PT_K, false);
    }
    position.refresh();
    (position, defender_king)
}

fn add_random_material(
    position: &mut Position,
    defender_king: usize,
    rng: &mut dyn Rng,
    count: usize,
    config: &GeneratorConfig,
) {
    let mut used = [0usize; 7];
    for &c in position.board.iter() {
        if c != EMPTY && ptype(c) != PT_K {
            used[ptype(c) as usize] += 1;
        }
    }
    for _ in 0..count {
        let available: Vec<u8> = HAND_PIECES
            .iter()
            .copied()
            .filter(|&t| used[t as usize] < INVENTORY[t as usize])
            .collect();
        if available.is_empty() {
            break;
        }
        let t = *rng.pick(&available);
        let color = if rng.chance(0.62) { BLACK } else { WHITE };
        let promoted = promotable_type(t) && rng.chance(config.promotion_rate);
        let near_king = rng.chance(0.72);
        let radius = config.king_zone_radius + 1;
        let square = random_empty(position, rng, |index| {
            (chebyshev(index, defender_king) <= radius) == near_king
                && can_place(position, index, t, color, promoted)
        })
        .or_else(|| {
            random_empty(position, rng, |index| {
                can_place(position, index, t, color, promoted)
            })
        });
        let Some(square) = square else { continue };
        position.board[square] = code(color, t, promoted);
        used[t as usize] += 1;
    }
    position.refresh();
    for &t in HAND_PIECES.iter() {
        let remaining = INVENTORY[t as usize] - used[t as usize];
        if remaining > 0 && rng.chance(config.hand_piece_rate) {
            let count = remaining.min(1 + rng.int(remaining.min(2)));
            position.set_hand(BLACK, t, count);
        }
    }
}

/// One hand-crafted mate shell.
///
/// Every mating square is a contact or knight check on purpose: under the strict
/// convention the defender holds nearly a full set, so a *distant* check is always
/// blockable and could never be a mate in one.  None of the shells places an
/// attacker king (rule A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MateShell {
    /// 頭金: the gold steps onto the king's head, guarded down the file by a lance.
    HeadGoldLance,
    /// The same idea with a rook on the file instead of a lance.
    HeadGoldRookFile,
    /// A guarded gold drop, the rook covering the mating square along rank b.
    RookRankGoldDrop,
    /// A gold drop guarded diagonally by a bishop.
    BishopGoldDrop,
    /// A gold drop in the corner, guarded down the file by a rook.
    CornerRookGoldDrop,
    /// 腹金: a gold beside a king sealed in by its own pawn.
    BellyGoldSilver,
    /// The only legal check is a knight drop, against a king boxed in by its own
    /// gold, knight and pawn.
    KnightBoxCorner,
}

/// The shells in array order; the index drawn by [`mate_shell`] indexes this.
const MATE_SHELLS: [MateShell; 7] = [
    MateShell::HeadGoldLance,
    MateShell::HeadGoldRookFile,
    MateShell::RookRankGoldDrop,
    MateShell::BishopGoldDrop,
    MateShell::CornerRookGoldDrop,
    MateShell::BellyGoldSilver,
    MateShell::KnightBoxCorner,
];

/// Lays out one shell, consuming exactly one draw from `rng`.
///
/// Coordinates are `(rank, file)` with `rank * 9 + file` the board index, i.e.
/// `(0, 0)` is 9a and `(0, 8)` is 1a.
fn build_shell(shell: MateShell, rng: &mut dyn Rng) -> Position {
    let mut position = Position::empty();
    position.side = BLACK;
    let mut place = |rank: usize, file: usize, color: u8, t: u8| {
        position.board[rank * 9 + file] = code(color, t, false);
    };
    let mut hand: Option<(u8, usize)> = None;
    match shell {
        MateShell::HeadGoldLance => {
            let file = rng.int(9);
            place(0, file, WHITE, PT_K);
            place(2, file, BLACK, PT_G);
            place(8, file, BLACK, PT_L);
        }
        MateShell::HeadGoldRookFile => {
            let file = rng.int(9);
            place(0, file, WHITE, PT_K);
            place(2, file, BLACK, PT_G);
            place(5, file, BLACK, PT_R);
        }
        MateShell::RookRankGoldDrop => {
            let file = 1 + rng.int(7);
            let rook = if file >= 4 { file - 3 } else { file + 3 };
            place(0, file, WHITE, PT_K);
            place(1, rook, BLACK, PT_R);
            hand = Some((PT_G, 1));
        }
        MateShell::BishopGoldDrop => {
            let file = rng.int(9);
            let bishop = if file >= 4 { file - 2 } else { file + 2 };
            place(0, file, WHITE, PT_K);
            place(3, bishop, BLACK, PT_B);
            hand = Some((PT_G, 1));
        }
        MateShell::CornerRookGoldDrop => {
            let edge = if rng.chance(0.5) { 8 } else { 0 };
            let inner = if edge == 8 { 7 } else { 1 };
            place(0, edge, WHITE, PT_K);
            place(4, inner, BLACK, PT_R);
            hand = Some((PT_G, 1));
        }
        MateShell::BellyGoldSilver => {
            let edge = if rng.chance(0.5) { 8 } else { 0 };
            let inner = if edge == 8 { 7 } else { 1 };
            place(0, edge, WHITE, PT_K);
            place(1, edge, WHITE, PT_P);
            place(2, inner, BLACK, PT_S);
            hand = Some((PT_G, 1));
        }
        MateShell::KnightBoxCorner => {
            let edge = if rng.chance(0.5) { 8 } else { 0 };
            let inner = if edge == 8 { 7 } else { 1 };
            place(0, edge, WHITE, PT_K);
            place(0, inner, WHITE, PT_G);
            place(1, inner, WHITE, PT_N);
            place(1, edge, WHITE, PT_P);
            hand = Some((PT_N, 1));
        }
    }
    position.refresh();
    if let Some((piece, count)) = hand {
        position.set_hand(BLACK, piece, count);
    }
    position
}

/// The `mate-shell` distribution: seven hand-crafted shells, one drawn per seed.
///
/// The family index is drawn **once** from a stream seeded with `seed` alone, so it
/// does not depend on which distribution the weighted outer pick chose.  A shell
/// whose layout fails structural validation steps to `(first + offset) % 7`; each
/// layout consumes its own draw from the shared stream, so a skip advances it.
/// `None` after seven failures, which no shell has ever produced.
fn mate_shell(seed: u64) -> Option<Position> {
    let mut rng = SplitMix64::new(seed);
    let first = rng.int(MATE_SHELLS.len());
    for offset in 0..MATE_SHELLS.len() {
        let position = build_shell(MATE_SHELLS[(first + offset) % MATE_SHELLS.len()], &mut rng);
        if crate::validate::validate_position(
            &position,
            crate::validate::ValidateOptions::pipeline(),
        )
        .is_empty()
        {
            return Some(position);
        }
    }
    None
}

/// Generates one candidate position.  `None` when the constraints could not be
/// satisfied in 128 attempts, or when no mate shell could be laid out (the
/// TypeScript version throws in both cases).
/// Most attacker hand pieces the `uniform` distribution draws: `distill` keeps at
/// most three (`DistillOptions::max_attacker_hand`), so a fourth only costs a hand transfer.
const UNIFORM_MAX_HAND: usize = 3;

/// The `uniform` distribution.
///
/// * the defender king on any of the 81 squares; with `attacker_king`, the
///   attacker king on any square not adjacent to it;
/// * `minBoardPieces..=maxBoardPieces` board pieces (kings included), each one
///   drawn uniformly from the physical pieces still unused (so pawns are common
///   because there are 18 of them), coloured by a fair coin, promoted with
///   `promotionRate`, and put on a uniformly chosen legal empty square;
/// * `0..=3` attacker hand pieces, drawn uniformly from what is left;
/// * no king-zone or material-balance constraint: only legality, and neither king
///   in check unless `allowInitialCheck`.
fn uniform_position(rng: &mut dyn Rng, config: &GeneratorConfig) -> Option<Position> {
    for _ in 0..128 {
        let mut position = Position::empty();
        let defender_king = rng.int(81);
        position.board[defender_king] = code(WHITE, PT_K, false);
        let mut kings = 1;
        if config.attacker_king {
            let square = random_empty(&position, rng, |index| chebyshev(index, defender_king) > 1)?;
            position.board[square] = code(BLACK, PT_K, false);
            kings += 1;
        }
        let mut pool: Vec<u8> = HAND_PIECES
            .iter()
            .flat_map(|&t| std::iter::repeat_n(t, INVENTORY[t as usize]))
            .collect();
        let span = config.max_board_pieces - config.min_board_pieces + 1;
        let target = (config.min_board_pieces + rng.int(span)).saturating_sub(kings);
        let mut placed = 0;
        let mut unplaced: Vec<u8> = Vec::new();
        while placed < target && !pool.is_empty() {
            let t = pool.swap_remove(rng.int(pool.len()));
            let color = if rng.chance(0.5) { BLACK } else { WHITE };
            let promoted = promotable_type(t) && rng.chance(config.promotion_rate);
            match random_empty(&position, rng, |index| {
                can_place(&position, index, t, color, promoted)
            }) {
                Some(square) => {
                    position.board[square] = code(color, t, promoted);
                    placed += 1;
                }
                None => unplaced.push(t),
            }
        }
        pool.extend(unplaced);
        for _ in 0..rng.int(UNIFORM_MAX_HAND + 1) {
            if pool.is_empty() {
                break;
            }
            let t = pool.swap_remove(rng.int(pool.len()));
            let count = position.hands[BLACK as usize][t as usize] as usize + 1;
            position.set_hand(BLACK, t, count);
        }
        position.refresh();
        if !crate::validate::validate_position(
            &position,
            crate::validate::ValidateOptions::pipeline(),
        )
        .is_empty()
        {
            continue;
        }
        if config.attacker_king && position.is_in_check(BLACK) {
            continue;
        }
        if !config.allow_initial_check && position.is_in_check(WHITE) {
            continue;
        }
        return Some(position);
    }
    None
}

pub fn generate_position(
    seed: u64,
    config: &GeneratorConfig,
    distribution: Option<Distribution>,
) -> Option<(Position, Distribution)> {
    let mut rng = SplitMix64::new(seed);
    let selected = distribution.unwrap_or_else(|| *rng.pick(&WEIGHTED_DISTRIBUTIONS));
    if selected == Distribution::MateShell {
        return mate_shell(seed).map(|position| (position, selected));
    }
    if selected == Distribution::Uniform {
        return uniform_position(&mut rng, config).map(|position| (position, selected));
    }
    for _ in 0..128 {
        let (mut position, defender_king) = base_position(&mut rng, config.attacker_king);
        if selected == Distribution::SliderPressure {
            let rank = 8usize.min(rank_of(defender_king) + 2 + rng.int(2));
            let index = rank * 9 + file_of(defender_king);
            if position.board[index] == EMPTY {
                let t = if rng.chance(0.5) { PT_R } else { PT_L };
                position.board[index] = code(BLACK, t, false);
                position.refresh();
            }
        }
        let span = config.max_board_pieces - config.min_board_pieces + 1;
        let target = config.min_board_pieces + rng.int(span);
        let present = position.board_piece_count();
        add_random_material(
            &mut position,
            defender_king,
            &mut rng,
            target.saturating_sub(present),
            config,
        );
        let zone_count = (0..81)
            .filter(|&index| {
                position.board[index] != EMPTY
                    && chebyshev(index, defender_king) <= config.king_zone_radius
            })
            .count();
        if !crate::validate::validate_position(
            &position,
            crate::validate::ValidateOptions::pipeline(),
        )
        .is_empty()
        {
            continue;
        }
        // Rule E: with no attacker king there is no attacker-in-check test to run.
        if config.attacker_king && position.is_in_check(BLACK) {
            continue;
        }
        if !config.allow_initial_check && position.is_in_check(WHITE) {
            continue;
        }
        if zone_count < config.king_zone_occupancy.0 || zone_count > config.king_zone_occupancy.1 {
            continue;
        }
        let attackers = position
            .board
            .iter()
            .filter(|&&c| c != EMPTY && color_of(c) == BLACK)
            .count() as i32;
        let defenders = position
            .board
            .iter()
            .filter(|&&c| c != EMPTY && color_of(c) == WHITE)
            .count() as i32;
        if (attackers - defenders).abs() > config.max_material_imbalance {
            continue;
        }
        return Some((position, selected));
    }
    None
}

/// Constrained-random candidate source.
#[derive(Clone, Debug, Default)]
pub struct ConstrainedRandomSource {
    pub config: GeneratorConfig,
    pub distribution: Option<Distribution>,
}

impl ConstrainedRandomSource {
    pub fn new(config: GeneratorConfig, distribution: Option<Distribution>) -> Self {
        Self {
            config,
            distribution,
        }
    }

    /// Fallible form: `None` when the constraints could not be satisfied.
    pub fn try_next(&self, seed: u64) -> Option<(Position, Distribution)> {
        generate_position(seed, &self.config, self.distribution)
    }
}

/// Uniform choice over the legal moves.
#[derive(Clone, Copy, Debug, Default)]
pub struct RandomLegalPolicy;

impl MovePolicy for RandomLegalPolicy {
    fn select(&mut self, position: &Position, rng: &mut dyn Rng) -> Option<Move> {
        let moves = legal_moves(position);
        if moves.is_empty() {
            None
        } else {
            Some(*rng.pick(&moves))
        }
    }
}
