//! Seeded randomness.
//!
//! [`SplitMix64`] is a bit-exact port of `SplitMix64` in
//! `tools/puzzle-pipeline/src/util.ts`, including `int`, `float`, `bool`,
//! `pick`, and the descending Fisher-Yates `shuffle`, so seeded pipeline
//! decisions (reducer orders, generator draws) reproduce the TypeScript result
//! exactly.

/// A 64-bit random source.  Object safe so pipeline code can take
/// `&mut dyn Rng` and still be driven by any generator.
pub trait Rng {
    fn next_u64(&mut self) -> u64;

    /// `SplitMix64.float()`: the top 53 bits scaled into `[0, 1)`.
    fn float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// `SplitMix64.int(maxExclusive)`: unbiased-by-construction-in-TS modulo.
    ///
    /// # Panics
    /// Panics when `max_exclusive` is zero, mirroring the TypeScript `RangeError`.
    fn int(&mut self, max_exclusive: usize) -> usize {
        assert!(max_exclusive > 0, "maxExclusive");
        (self.next_u64() % max_exclusive as u64) as usize
    }

    /// `SplitMix64.bool(probability)`.
    fn chance(&mut self, probability: f64) -> bool {
        self.float() < probability
    }
}

/// Generic helpers kept out of [`Rng`] so that trait stays object safe.
pub trait RngExt: Rng {
    fn pick<'a, T>(&mut self, values: &'a [T]) -> &'a T {
        assert!(!values.is_empty(), "Cannot pick from empty list");
        let index = self.int(values.len());
        &values[index]
    }

    /// `SplitMix64.shuffle`: descending Fisher-Yates with `int(i + 1)`.
    fn shuffle<T: Clone>(&mut self, values: &[T]) -> Vec<T> {
        let mut result = values.to_vec();
        let mut i = result.len();
        while i > 1 {
            i -= 1;
            let j = self.int(i + 1);
            result.swap(i, j);
        }
        result
    }
}

impl<T: Rng + ?Sized> RngExt for T {}

/// Bit-exact port of the pipeline's `SplitMix64`.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

/// The golden-ratio odd constant SplitMix64 itself steps its state with; reused
/// here only as an odd multiplier, which is what makes `index` injective.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// Seed of item `index` in the independent stream named by `base`.
///
/// **Do not write `SplitMix64::new(base + index)`.**  That shape — which this
/// repository used until 2026-09-14 — makes consecutive bases produce *shifted
/// copies* of one another: item `i` of base `b + 1` is item `i + 1` of base `b`.
/// Two `mine` runs seeded one day apart therefore re-walk almost the same games
/// (measured 2026-09-14: 3 999 of 4 000 games identical between `--seed 20260914`
/// and `--seed 20260915`, re-finding the same 495 five-ply candidates in every
/// batch of a production loop).
///
/// The construction is
///
/// ```text
/// key  = SplitMix64::new(base).next_u64()                       // per-run key
/// seed = SplitMix64::new(key ^ index.wrapping_mul(GAMMA)).next_u64()
/// ```
///
/// with `GAMMA = 0x9E37_79B9_7F4A_7C15`.  `base` goes through a full SplitMix64
/// finaliser first, so `base` and `base + 1` give unrelated keys and no offset
/// relates the two streams.  Multiplying `index` by an odd constant is a bijection
/// on `u64` and xor with the key is another, so the *inputs* of one stream are all
/// distinct, and the second finaliser — also a bijection — keeps its outputs
/// distinct too: a stream never repeats a seed.
///
/// One call site deliberately keeps the old shape and says so where it does:
/// [`crate::reduce::SeededAttempts`], whose schedule is bit-exact TypeScript
/// parity (`tools/puzzle-pipeline/src/stages.ts`).  (The `bench` position
/// sampler, which drew the TypeScript benchmark's positions, went with `bench`
/// on 2026-10-27.)  Everything else - `mine`'s game seeds first of all - derives its
/// per-item seeds here.
pub fn derive_stream_seed(base: u64, index: u64) -> u64 {
    let key = SplitMix64::new(base).next_u64();
    SplitMix64::new(key ^ index.wrapping_mul(GOLDEN_GAMMA)).next_u64()
}

impl Rng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values produced by `new SplitMix64(1n)` in Node 22 using
    // tools/puzzle-pipeline/src/util.ts.
    #[test]
    fn matches_typescript_stream() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(rng.next_u64(), 10_451_216_379_200_822_465);
        assert_eq!(rng.next_u64(), 13_757_245_211_066_428_519);
        assert_eq!(rng.next_u64(), 17_911_839_290_282_890_590);
        assert_eq!(rng.next_u64(), 8_196_980_753_821_780_235);
    }

    /// Defect fixed 2026-09-14: with `SplitMix64::new(base + index)` the streams of
    /// two consecutive bases were the same stream shifted by one, so runs seeded on
    /// consecutive days re-explored the same candidates.  Neighbouring bases must
    /// now share no seed at all.
    #[test]
    fn derived_streams_of_neighbouring_bases_are_disjoint() {
        const LEN: u64 = 10_000;
        let stream = |base: u64| -> std::collections::HashSet<u64> {
            (0..LEN)
                .map(|index| derive_stream_seed(base, index))
                .collect()
        };
        let base = 20_260_914u64;
        let first = stream(base);
        // A stream never repeats a seed, so the set is as long as the stream.
        assert_eq!(first.len() as u64, LEN);
        for other in [base + 1, base + 1_000] {
            let second = stream(other);
            assert_eq!(second.len() as u64, LEN);
            assert_eq!(
                first.intersection(&second).count(),
                0,
                "streams of {base} and {other} share a seed"
            );
        }
    }

    #[test]
    fn shuffle_is_descending_fisher_yates() {
        let mut rng = SplitMix64::new(7);
        // `new SplitMix64(7n).shuffle([0..7])` in Node 22.
        let out = rng.shuffle(&[0u32, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(out, vec![1, 4, 5, 2, 6, 0, 3, 7]);
    }
}
