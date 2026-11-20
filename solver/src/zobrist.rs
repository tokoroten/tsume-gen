//! Deterministic 128-bit Zobrist keys for transposition lookups.
//!
//! The TypeScript solver keys its cache by the serialised SFEN string
//! (`role|remaining|board side hands`).  Building that string costs more than
//! the node it protects, so the Rust solver keys by an incrementally maintained
//! 128-bit key over the same information: every board square/piece pair, every
//! hand (colour, type, count) triple, and the side to move.  Two distinct
//! positions therefore collide only if a 128-bit key collides, which is far
//! below the probability of a hardware fault; the search result is still exact.
//!
//! Keys are generated from a fixed SplitMix64 stream so a given build always
//! produces the same keys, which keeps the solver's own output reproducible.

/// Highest hand count that can be hashed (and therefore stored) per piece type.
pub const MAX_HAND_COUNT: usize = 255;

pub struct Keys {
    /// `[piece code][square]`
    pub piece: [[u128; 81]; 64],
    /// `[colour][hand piece][count]`
    pub hand: [[[u128; MAX_HAND_COUNT + 1]; 7]; 2],
    /// XORed in while white is to move.
    pub white_to_move: u128,
}

/// SplitMix64's step ([`crate::rng::SplitMix64`]), as a `const fn`.
const fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const fn next_key(state: &mut u64) -> u128 {
    let high = split_mix(state);
    let low = split_mix(state);
    ((high as u128) << 64) | low as u128
}

/// The keys, at compile time (2026-10-27: a plain `static`, read without the
/// lazy initialisation check of a `LazyLock`); the same stream as before.
const fn build() -> Keys {
    let mut state = 0x7375_6d65_5f7a_6f62u64;
    let mut keys = Keys {
        piece: [[0; 81]; 64],
        hand: [[[0; MAX_HAND_COUNT + 1]; 7]; 2],
        white_to_move: 0,
    };
    let mut c = 0;
    while c < 64 {
        let mut square = 0;
        while square < 81 {
            keys.piece[c][square] = next_key(&mut state);
            square += 1;
        }
        c += 1;
    }
    let mut color = 0;
    while color < 2 {
        let mut piece = 0;
        while piece < 7 {
            let mut count = 0;
            while count <= MAX_HAND_COUNT {
                keys.hand[color][piece][count] = next_key(&mut state);
                count += 1;
            }
            piece += 1;
        }
        color += 1;
    }
    keys.white_to_move = next_key(&mut state);
    keys
}

pub static KEYS: Keys = build();
