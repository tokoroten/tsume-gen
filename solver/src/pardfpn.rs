//! The lock-free tables of the parallel classical df-pn prober
//! (`--prober-threads`, [`crate::cdfpn`]; `docs/experiments/par-dfpn/README.md`).
//!
//! [`SeqTable`] maps random 128-bit keys to one 64-bit word of data plus 24
//! bits of caller metadata, for any number of threads at once, and no thread
//! ever waits for another:
//!
//! * a slot is four words `[w, k0, k1, d]` (32 bytes); two slots make one
//!   64-byte bucket, and a key's bucket is the low bits of `key[1]`;
//! * `w` holds a **sequence number** in its low 40 bits (even when the slot is
//!   stable, odd while a writer holds it; it repeats only after `2^39` writes
//!   to one slot, which at the rate the hottest node is rewritten is years,
//!   so no reader is ever suspended across a repeat), the caller's 8 bits of
//!   metadata, the low byte of the table's epoch, and a **visitor count** in
//!   the top 8 bits (the virtual proof numbers of the parallel search: how
//!   many threads are searching under the node right now);
//! * a writer takes a slot by turning its even sequence odd with one
//!   compare-and-swap, writes `k0`, `k1` and `d`, and publishes the next even
//!   sequence with the new metadata.  A writer that finds the slot held, or
//!   loses the compare-and-swap, tries again a bounded number of times and
//!   then **drops its write** (lossy: every value written here is one the
//!   search can recompute, so losing one costs work, never a verdict);
//! * a reader loads `w`, then the three words, then (after an acquire fence)
//!   `w` again, and accepts the slot only when the sequence was even and
//!   unchanged and the key is the one it asked for (the seqlock reader of
//!   Boehm 2012 with atomic data words; the second load may differ from the
//!   first in the visitor bits only, which no write under the lock touches);
//! * a visitor increments or decrements the count with one compare-and-swap
//!   on `w` after reading the key under that same `w`: the swap succeeds only
//!   when `w` is still what it was, so no writer took the slot in between and
//!   the key it compared is the key it counted.  A visitor that finds the slot
//!   held, or the key gone, does nothing (its matching `leave` then also does
//!   nothing: the count was reset to zero when the key was replaced).
//!
//! An empty slot has `d == 0`; every caller's `d` is nonzero.  A full bucket
//! replaces the slot of the lowest *priority* (the caller's order, with a
//! slot that has visitors never chosen before one without), so a table
//! holds at most `2^bits` entries and never grows.  Clearing the table
//! ([`SeqTable::clear`]) starts a new **epoch**: every key is salted with
//! the epoch before it is looked up or stored, so the entries of an earlier
//! epoch match nothing (a 128-bit salt, as improbable as any other key
//! collision) and are the first to be replaced; no memory is written, except
//! that every 256th epoch (when the epoch byte the slots keep would repeat)
//! zeroes the whole table.
//!
//! The same table, with different packings of `d` and the metadata, serves
//! both the parallel prober's transposition table (proof and disproof numbers,
//! purity, work) and its fact table (per position the bounds proven mate or
//! not mate).

use std::sync::atomic::{fence, AtomicBool, AtomicI64, AtomicU64, Ordering};

/// [`SeqTable::put_with`]: no visit, count one, or undo one.
const VISIT_NONE: u8 = 0;
const VISIT_ENTER: u8 = 1;
const VISIT_LEAVE: u8 = 2;

/// Slots per bucket: one 64-byte line (2026-10-28: four slots on two
/// lines before; one line is 4-8 % faster at P=4-12,
/// `docs/experiments/solver-scaling/README.md` §15).
const WAYS: usize = 2;
/// Live entries, as a percentage of the slots, at which a sweep is due
/// ([`SeqTable::needs_sweep`]); the tests' `SeqTable::with_sweep_at` overrides it.
/// 90 since 2026-10-28 (30 before: with one-line buckets a full bucket
/// replaces its cheapest slot, and fewer sweeps are 1.03x at P=4-8,
/// `docs/experiments/solver-scaling/README.md` §15).
pub const SWEEP_AT_PERCENT: usize = 90;
/// Priorities the sweep tells apart (larger ones count as the largest).
const SWEEP_CLASSES: usize = 512;
/// Words per slot.
const WORDS: usize = 4;
/// Bytes per slot.
pub const SLOT_BYTES: usize = WORDS * 8;

const SEQ_MASK: u64 = (1 << 40) - 1;
const META_SHIFT: u32 = 40;
const META_MASK: u64 = 0xFF << META_SHIFT;
const EPOCH_SHIFT: u32 = 48;
const EPOCH_MASK: u64 = 0xFF << EPOCH_SHIFT;
const VISITOR_SHIFT: u32 = 56;
const VISITOR_ONE: u64 = 1 << VISITOR_SHIFT;
const VISITOR_MASK: u64 = 0xFF << VISITOR_SHIFT;
/// The largest visitor count a slot holds; a visit past it is not counted.
const VISITOR_MAX: u64 = 0xFF;

/// How often a reader or writer retries a slot that is held or changing
/// before it gives up on it (a held slot is let go within a few dozen
/// instructions, so these are spins, not waits).
const SPINS: usize = 64;
/// How often a writer retries the whole bucket before it drops its write.
const PUT_ATTEMPTS: usize = 8;

// [`SeqTable::new`] reinterprets a boxed `[u64]` as `[AtomicU64]`.
const _: () = assert!(
    std::mem::size_of::<AtomicU64>() == std::mem::size_of::<u64>()
        && std::mem::align_of::<AtomicU64>() == std::mem::align_of::<u64>()
);

/// What a lookup found: the data word, the caller's 8 bits and the visitors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub d: u64,
    pub meta: u32,
    pub visitors: u8,
}

/// What a write did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Put {
    /// Nothing: a full bucket whose every slot is being visited, or
    /// contention for every attempt.
    Dropped,
    /// The entry is in the table; with `visit`, the visit was not counted
    /// (the count is saturated), so no [`SeqTable::leave`] is due.
    Written,
    /// The entry is in the table and the visit was counted, from the count
    /// given (the visitors the slot already had for this key).
    Visited(u8),
}

impl Put {
    /// Whether the entry is in the table.
    #[inline]
    pub fn written(self) -> bool {
        self != Put::Dropped
    }

    /// Whether a visit was counted (a [`SeqTable::leave`] is due).
    #[inline]
    pub fn visited(self) -> bool {
        matches!(self, Put::Visited(_))
    }
}

/// Shards of the fill count (one cache line each), so that writers on many
/// threads do not all bump one word.
const FILL_SHARDS: usize = 16;

/// One shard: the fill count and, when counting is on
/// ([`SeqTable::set_counting`]), the measurement counters of
/// [`Counter`] (one cache line in all).
#[repr(align(64))]
struct Shard {
    fill: AtomicI64,
    counts: [AtomicU64; COUNTERS],
}

/// Measurement counters of a table (`TSUME_PDFPN_PROF`; off by default, when
/// nothing is counted and no shared line is written for them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Counter {
    /// Writes dropped because every slot of the bucket had visitors.
    DropVisited,
    /// Writes dropped after every attempt found the slots held or changing.
    DropContended,
    /// Writes that replaced a live entry of the current epoch.
    Evict,
    /// Attempts of a write beyond the first (a slot held, lost, or changed).
    PutRetry,
    /// Spins of a read beyond the first (a writer held the slot, or the
    /// sequence moved between the two loads).
    ReadRetry,
    /// Reads that gave up after every spin.
    ReadFail,
    /// Visits or leaves that found the slot held for every spin.
    AdjustBusy,
}
const COUNTERS: usize = 7;

/// What a visit did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visit {
    /// The count went up (from the count given); [`SeqTable::leave`] later
    /// brings it down.
    Counted(u8),
    /// The key is not in the table, or its slot was held or contended for
    /// every spin: nothing counted.
    Absent,
}

/// What a leave did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leave {
    /// The count went down, or the key (and its count) is gone.
    Done,
    /// The slot was held or contended for every spin: the count is still up.
    /// The caller tries again later (a held slot is let go within a few dozen
    /// instructions); until then the node reads as one visitor busier.
    Busy,
}

/// What [`SeqTable::adjust`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Adjust {
    /// Done; the count before.
    Done(u8),
    Absent,
    Busy,
}

/// One cache line of the table: a bucket.
#[repr(C, align(64))]
struct Line([AtomicU64; WAYS * WORDS]);

const _: () = assert!(std::mem::size_of::<Line>() == 64);

pub struct SeqTable {
    /// The buckets, each on its own cache line.
    lines: Box<[Line]>,
    /// Bucket count minus one.
    mask: usize,
    /// Bumped by [`SeqTable::clear`]: every key is salted with it, so the
    /// entries of earlier epochs match nothing and are evicted first (their
    /// low byte is kept in each slot), and clearing never touches the
    /// memory.
    epoch: AtomicU64,
    /// The two salts of [`SeqTable::salted`] for the current epoch, set with
    /// it (2026-10-27: computed on every lookup before).
    salt: [AtomicU64; 2],
    /// Live entries (of the current epoch): up by one for every slot a
    /// writer fills, down by one for every slot a sweep empties, each on the
    /// shard of the slot's key; exact once the writers are quiet.
    fill: Box<[Shard]>,
    /// Whether a sweep is in progress (one at a time).
    sweeping: AtomicBool,
    /// Live entries at which a sweep is due.
    sweep_at: usize,
    /// Whether the [`Counter`]s are kept (measurement only).
    counting: AtomicBool,
    /// Sweeps run, and slots they emptied (measurement).
    sweeps: AtomicU64,
    swept: AtomicU64,
}

#[inline]
fn meta_of(w: u64) -> u32 {
    ((w & META_MASK) >> META_SHIFT) as u32
}

#[inline]
fn epoch_of(w: u64) -> u8 {
    ((w & EPOCH_MASK) >> EPOCH_SHIFT) as u8
}

/// The salts of `epoch`'s keys.
const fn salts(epoch: u64) -> [u64; 2] {
    let salt = mix64(epoch);
    [salt, mix64(salt ^ 0x5bd1_e995_1234_5678)]
}

#[inline]
const fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn visitors_of(w: u64) -> u8 {
    (w >> VISITOR_SHIFT) as u8
}

impl SeqTable {
    /// A table of `2^bits` slots (`bits >= 1`), all empty, taking
    /// `2^bits * 32` bytes.  The memory is zero-filled by the allocator and
    /// touched only as slots are used.
    pub fn new(bits: u32) -> Self {
        Self::build(bits, SWEEP_AT_PERCENT)
    }

    /// [`SeqTable::new`] with a sweep due at `percent` of the slots live
    /// (tests only).
    #[cfg(test)]
    pub(crate) fn with_sweep_at(bits: u32, percent: usize) -> Self {
        Self::build(bits, percent)
    }

    fn build(bits: u32, percent: usize) -> Self {
        let slots = 1usize << bits.clamp(WAYS.trailing_zeros(), 40);
        let buckets = slots / WAYS;
        let layout =
            std::alloc::Layout::array::<Line>(buckets).expect("a table that fits in memory");
        // SAFETY: the layout is `buckets` lines (nonzero size); zeroed memory
        // is a valid `Line` (`AtomicU64` has the bit validity of `u64`), and
        // the box frees it with the same layout.  Zeroed by the allocator,
        // so the pages are touched only as slots are used.
        let lines: Box<[Line]> = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                ptr.cast::<Line>(),
                buckets,
            ))
        };
        SeqTable {
            lines,
            mask: buckets - 1,
            epoch: AtomicU64::new(1),
            salt: salts(1).map(AtomicU64::new),
            fill: (0..FILL_SHARDS)
                .map(|_| Shard {
                    fill: AtomicI64::new(0),
                    counts: std::array::from_fn(|_| AtomicU64::new(0)),
                })
                .collect(),
            sweeping: AtomicBool::new(false),
            sweep_at: (slots * percent.clamp(1, 100) / 100).max(1),
            counting: AtomicBool::new(false),
            sweeps: AtomicU64::new(0),
            swept: AtomicU64::new(0),
        }
    }

    /// Turns the measurement counters on or off.
    pub fn set_counting(&self, on: bool) {
        self.counting.store(on, Ordering::Relaxed);
    }

    /// Adds one to `counter` on the shard of `salt` (when counting).
    #[inline]
    fn count(&self, counter: Counter, salt: usize) {
        if self.counting.load(Ordering::Relaxed) {
            self.fill[salt % FILL_SHARDS].counts[counter as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The measurement counters, summed over the shards, then the sweeps
    /// and the slots they emptied.
    pub fn counters(&self) -> ([u64; COUNTERS], u64, u64) {
        let mut out = [0u64; COUNTERS];
        for shard in self.fill.iter() {
            for (sum, counter) in out.iter_mut().zip(shard.counts.iter()) {
                *sum += counter.load(Ordering::Relaxed);
            }
        }
        (
            out,
            self.sweeps.load(Ordering::Relaxed),
            self.swept.load(Ordering::Relaxed),
        )
    }

    /// The names of [`SeqTable::counters`]'s entries, in order.
    pub const COUNTER_NAMES: [&'static str; COUNTERS] = [
        "dropVisited",
        "dropContended",
        "evict",
        "putRetry",
        "readRetry",
        "readFail",
        "adjustBusy",
    ];

    /// Live entries, approximately ([`SeqTable::fill`]).
    #[inline]
    pub fn fill(&self) -> usize {
        self.fill
            .iter()
            .map(|shard| shard.fill.load(Ordering::Relaxed))
            .sum::<i64>()
            .max(0) as usize
    }

    /// Whether the table is full enough for a sweep (`SWEEP_AT_PERCENT` of
    /// the slots live: beyond that, buckets fill up and writes start to be
    /// dropped or to evict).
    #[inline]
    pub fn needs_sweep(&self) -> bool {
        self.fill() >= self.sweep_at
    }

    /// The one-thread prober's garbage collection, on a live table: drops
    /// the entries of the lowest `priority` values that together hold at
    /// least half of the live entries, except the slots with visitors (a
    /// thread is under them) and the slots held by a writer at that moment.
    /// Other threads keep reading and writing meanwhile (an entry they lose
    /// is work, never a verdict).  One sweep at a time: `false` when another
    /// thread is sweeping.
    pub fn sweep<P: Fn(u64, u32) -> u64>(&self, priority: P) -> bool {
        if self
            .sweeping
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        let epoch = self.epoch.load(Ordering::Relaxed) as u8;
        let slots = self.slots();
        // What the live entries are worth.
        let mut counts = [0u64; SWEEP_CLASSES];
        let mut live = 0u64;
        for slot in 0..slots {
            let w = self.words()[slot * WORDS].load(Ordering::Relaxed);
            let d = self.words()[slot * WORDS + 3].load(Ordering::Relaxed);
            if d == 0 || epoch_of(w) != epoch {
                continue;
            }
            live += 1;
            counts[(priority(d, meta_of(w)) as usize).min(SWEEP_CLASSES - 1)] += 1;
        }
        // The lowest values holding at least half of them go.
        let mut limit = 0usize;
        let mut dropped = 0u64;
        for (value, &count) in counts.iter().enumerate() {
            dropped += count;
            limit = value;
            if dropped * 2 >= live {
                break;
            }
        }
        let mut removed = 0u64;
        for slot in 0..slots {
            let base = slot * WORDS;
            let w = self.words()[base].load(Ordering::Relaxed);
            if w & 1 == 1 || visitors_of(w) > 0 || epoch_of(w) != epoch {
                continue;
            }
            let d = self.words()[base + 3].load(Ordering::Relaxed);
            if d == 0 || (priority(d, meta_of(w)) as usize).min(SWEEP_CLASSES - 1) > limit {
                continue;
            }
            // Take the slot (a visit or a write since the read fails this).
            if self.words()[base]
                .compare_exchange(w, w | 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            fence(Ordering::Release);
            let k0 = self.words()[base + 1].load(Ordering::Relaxed);
            self.words()[base + 3].store(0, Ordering::Relaxed);
            self.words()[base].store((w + 2) & SEQ_MASK, Ordering::Release);
            // Counted off the shard the writer counted it on (the slot's
            // key), so the writers' concurrent counts are kept.
            self.fill[(k0 as usize) % FILL_SHARDS]
                .fill
                .fetch_sub(1, Ordering::Relaxed);
            removed += 1;
        }
        if self.counting.load(Ordering::Relaxed) {
            self.sweeps.fetch_add(1, Ordering::Relaxed);
            self.swept.fetch_add(removed, Ordering::Relaxed);
        }
        self.sweeping.store(false, Ordering::Release);
        true
    }

    /// `key` salted with the current epoch (the key the slots hold).
    #[inline]
    fn salted(&self, key: &[u64; 2]) -> ([u64; 2], u64) {
        let epoch = self.epoch.load(Ordering::Relaxed);
        (
            [
                key[0] ^ self.salt[0].load(Ordering::Relaxed),
                key[1] ^ self.salt[1].load(Ordering::Relaxed),
            ],
            epoch,
        )
    }

    /// Slots.
    pub fn slots(&self) -> usize {
        self.words().len() / WORDS
    }

    /// Bytes of the table.
    pub fn bytes(&self) -> usize {
        self.words().len() * 8
    }

    /// Empties the table: a new epoch, whose salted keys match no entry
    /// written before; every 256th epoch zeroes the slots too, so the epoch
    /// byte a slot keeps always tells an entry of the current epoch from an
    /// older one.  Only when no other thread uses the table (the parallel
    /// prober's `reset`, between questions).
    pub fn clear(&self) {
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        for (slot, salt) in self.salt.iter().zip(salts(epoch)) {
            slot.store(salt, Ordering::Relaxed);
        }
        if epoch & 0xFF == 0 {
            for word in self.words().iter() {
                word.store(0, Ordering::Relaxed);
            }
        }
        for shard in self.fill.iter() {
            shard.fill.store(0, Ordering::Relaxed);
        }
    }

    /// Whether some bucket holds one key twice (tests only; a full scan).
    #[doc(hidden)]
    pub fn has_duplicate(&self) -> bool {
        (0..=self.mask).any(|bucket| {
            let base = bucket * WAYS * WORDS;
            let keys: Vec<[u64; 3]> = (0..WAYS)
                .map(|way| {
                    let slot = base + way * WORDS;
                    [
                        self.words()[slot + 1].load(Ordering::Relaxed),
                        self.words()[slot + 2].load(Ordering::Relaxed),
                        self.words()[slot + 3].load(Ordering::Relaxed),
                    ]
                })
                .filter(|slot| slot[2] != 0)
                .collect();
            (0..keys.len()).any(|i| (i + 1..keys.len()).any(|j| keys[i][..2] == keys[j][..2]))
        })
    }

    /// Slots holding an entry of the current epoch (measurement only; a
    /// full scan; exact, since the slots of older epochs are zeroed before
    /// their epoch byte could repeat).
    pub fn occupied(&self) -> usize {
        let epoch = self.epoch.load(Ordering::Relaxed) as u8;
        (0..self.slots())
            .filter(|&slot| {
                self.words()[slot * WORDS + 3].load(Ordering::Relaxed) != 0
                    && epoch_of(self.words()[slot * WORDS].load(Ordering::Relaxed)) == epoch
            })
            .count()
    }

    /// The table as one run of words (`[w, k0, k1, d]` per slot).
    #[inline]
    fn words(&self) -> &[AtomicU64] {
        // SAFETY: `Line` is `repr(C)` around `WAYS * WORDS` `AtomicU64`s with
        // no padding (64 bytes, asserted), so the lines are that many words
        // each, contiguous, for as long as `self` lives.
        unsafe {
            std::slice::from_raw_parts(
                self.lines.as_ptr().cast::<AtomicU64>(),
                self.lines.len() * WAYS * WORDS,
            )
        }
    }

    /// The first word of the bucket of a salted key.
    #[inline]
    fn bucket_base(&self, key: &[u64; 2]) -> usize {
        ((key[1] as usize) & self.mask) * WAYS * WORDS
    }

    /// A consistent view `[w, k0, k1, d]` of the slot at `base`, with `w`
    /// even; `None` when a writer held it for every spin.
    #[inline]
    fn read_slot(&self, base: usize) -> Option<[u64; 4]> {
        let (w, k0, k1, d) = (
            &self.words()[base],
            &self.words()[base + 1],
            &self.words()[base + 2],
            &self.words()[base + 3],
        );
        for spin in 0..SPINS {
            if spin > 0 {
                self.count(Counter::ReadRetry, base / WORDS);
            }
            let w1 = w.load(Ordering::Acquire);
            if w1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let a = k0.load(Ordering::Relaxed);
            let b = k1.load(Ordering::Relaxed);
            let c = d.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            let w2 = w.load(Ordering::Relaxed);
            if (w1 ^ w2) & !VISITOR_MASK == 0 {
                // Only the visitor bits may have moved: the data is the data
                // published with this sequence number.
                return Some([w2, a, b, c]);
            }
        }
        self.count(Counter::ReadFail, base / WORDS);
        None
    }

    /// Asks the CPU to bring `key`'s bucket (one cache line) into cache.
    #[inline]
    pub fn prefetch(&self, key: &[u64; 2]) {
        let (key, _) = self.salted(key);
        let base = self.bucket_base(&key);
        let line = self.words().as_ptr().wrapping_add(base).cast::<u8>();
        crate::keytable::prefetch(line);
    }

    /// The entry of `key`, if the table holds it.
    #[inline]
    pub fn get(&self, key: &[u64; 2]) -> Option<Hit> {
        let (key, _) = self.salted(key);
        let key = &key;
        let base = self.bucket_base(key);
        for way in 0..WAYS {
            let slot = base + way * WORDS;
            // A slot whose first key word differs holds another key (or is
            // being given one): skipped without the consistent read, which
            // was a tenth of the work per node of a parallel search
            // (2026-10-27).
            if self.words()[slot + 1].load(Ordering::Relaxed) != key[0] {
                continue;
            }
            if let Some([w, k0, k1, d]) = self.read_slot(slot) {
                if d != 0 && k0 == key[0] && k1 == key[1] {
                    return Some(Hit {
                        d,
                        meta: meta_of(w),
                        visitors: visitors_of(w),
                    });
                }
            }
        }
        None
    }

    /// Writes `key`'s entry: `merge` is given the entry the slot holds for
    /// this key (`None` when the key is absent) and returns the data word
    /// (nonzero) and metadata to store; a full bucket evicts the slot of the
    /// lowest `priority` (its data and metadata), except a slot with
    /// visitors (when every slot has visitors the write is dropped; a sweep,
    /// not a write, is what makes room: [`SeqTable::sweep`]).
    /// With `visit`, the published entry counts one more visitor (a visit
    /// that [`SeqTable::leave`] undoes), so a node is counted from the moment
    /// it is created ([`Put::Visited`]; [`Put::Written`] when the count is
    /// saturated).  [`Put::Dropped`] after bounded retries: nothing written,
    /// nothing counted.
    pub fn put<M, P>(&self, key: &[u64; 2], merge: M, priority: P, visit: bool) -> Put
    where
        M: Fn(Option<(u64, u32)>) -> (u64, u32),
        P: Fn(u64, u32) -> u64,
    {
        self.put_with(
            key,
            merge,
            priority,
            if visit { VISIT_ENTER } else { VISIT_NONE },
        )
    }

    /// [`SeqTable::put`] (without a visit) that also undoes one counted
    /// visit of `key` in the same write, as [`SeqTable::leave`] would
    /// (2026-10-27: one compare-and-swap per expanded node instead of two).
    /// `true` when the visit is undone (also when the key was not in the
    /// table: its count went with it); after [`Put::Dropped`], nothing was
    /// done and the caller leaves as before.
    pub fn put_leaving<M, P>(&self, key: &[u64; 2], merge: M, priority: P) -> Put
    where
        M: Fn(Option<(u64, u32)>) -> (u64, u32),
        P: Fn(u64, u32) -> u64,
    {
        self.put_with(key, merge, priority, VISIT_LEAVE)
    }

    fn put_with<M, P>(&self, key: &[u64; 2], merge: M, priority: P, mode: u8) -> Put
    where
        M: Fn(Option<(u64, u32)>) -> (u64, u32),
        P: Fn(u64, u32) -> u64,
    {
        let visit = mode == VISIT_ENTER;
        let (key, epoch) = self.salted(key);
        let key = &key;
        let epoch_byte = epoch as u8;
        let base = self.bucket_base(key);
        'attempt: for attempt in 0..PUT_ATTEMPTS {
            if attempt > 0 {
                self.count(Counter::PutRetry, key[0] as usize);
            }
            // Where the key sits, else the least valuable slot.
            let mut target = None;
            // The least valuable slot, with the `w` it was ranked by.
            let mut victim: Option<(u64, usize, u64)> = None;
            for way in 0..WAYS {
                let slot = base + way * WORDS;
                let Some([w, k0, k1, d]) = self.read_slot(slot) else {
                    continue;
                };
                if d != 0 && k0 == key[0] && k1 == key[1] {
                    target = Some(slot);
                    break;
                }
                let rank = if d == 0 || epoch_of(w) != epoch_byte {
                    // Empty, or left over from an earlier epoch.
                    0
                } else {
                    let rank = priority(d, meta_of(w)).min(u64::MAX >> 2) + 1;
                    if visitors_of(w) > 0 {
                        rank | 1 << 62
                    } else {
                        rank
                    }
                };
                if victim.is_none_or(|(best, _, _)| rank < best) {
                    victim = Some((rank, slot, w));
                }
            }
            let (slot, ranked_w, fills) = match (target, victim) {
                (Some(slot), _) => (slot, None, false),
                (None, Some((rank, slot, w))) => {
                    // A full bucket replaces its least valuable slot, but
                    // never a visited one.
                    if rank >= 1 << 62 {
                        self.count(Counter::DropVisited, key[0] as usize);
                        return Put::Dropped;
                    }
                    (slot, Some(w), rank == 0)
                }
                // Both slots held by writers right now.
                (None, None) => {
                    std::hint::spin_loop();
                    continue;
                }
            };
            // Take the slot.
            let w = self.words()[slot].load(Ordering::Relaxed);
            if w & 1 == 1
                || self.words()[slot]
                    .compare_exchange(w, w | 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_err()
            {
                std::hint::spin_loop();
                continue;
            }
            // The data stores below are ordered after the odd sequence
            // number: a reader that sees any of them and then fences
            // (`read_slot`) sees the odd `w` too, so its view is rejected.
            fence(Ordering::Release);
            // Held: the slot's words are stable and ours.
            let k0 = self.words()[slot + 1].load(Ordering::Relaxed);
            let k1 = self.words()[slot + 2].load(Ordering::Relaxed);
            let d = self.words()[slot + 3].load(Ordering::Relaxed);
            let same = d != 0 && k0 == key[0] && k1 == key[1];
            if !same {
                // Not an update: make sure the key is not in the other slot
                // (a duplicate would count visitors on one copy and leave
                // them on the other), which needs a definite read of it --
                // a slot held by another writer right now may be taking this
                // very key -- and that we are not about to evict a slot that
                // changed since it was ranked (its entry, or its visitors).
                // Otherwise let go unchanged (the same even sequence: nothing
                // was written) and start over.  The fence orders the read of
                // the other slot after the take of this one for every writer
                // (two taking both slots of a bucket for one key cannot both
                // read the other as free), so two copies are never made.
                fence(Ordering::SeqCst);
                let changed = target.is_some()
                    || ranked_w.is_some_and(|ranked| ranked != w)
                    || (0..WAYS).any(|way| {
                        let other = base + way * WORDS;
                        other != slot
                            && self
                                .read_slot(other)
                                .is_none_or(|[_, k0, k1, d]| d != 0 && k0 == key[0] && k1 == key[1])
                    });
                if changed {
                    self.words()[slot].store(w, Ordering::Release);
                    std::hint::spin_loop();
                    continue 'attempt;
                }
            }
            let (nd, nmeta) = merge(same.then_some((d, meta_of(w))));
            debug_assert!(nd != 0, "a zero data word marks an empty slot");
            if !same && !fills {
                self.count(Counter::Evict, key[0] as usize);
            }
            self.words()[slot + 1].store(key[0], Ordering::Relaxed);
            self.words()[slot + 2].store(key[1], Ordering::Relaxed);
            self.words()[slot + 3].store(nd, Ordering::Relaxed);
            let mut visitors = if same { w & VISITOR_MASK } else { 0 };
            if mode == VISIT_LEAVE && visitors != 0 {
                visitors -= VISITOR_ONE;
            }
            let counted = visit && visitors >> VISITOR_SHIFT < VISITOR_MAX;
            if counted {
                visitors += VISITOR_ONE;
            }
            let nw = ((w + 2) & SEQ_MASK)
                | (u64::from(nmeta) << META_SHIFT) & META_MASK
                | u64::from(epoch_byte) << EPOCH_SHIFT
                | visitors;
            self.words()[slot].store(nw, Ordering::Release);
            if fills {
                self.fill[(key[0] as usize) % FILL_SHARDS]
                    .fill
                    .fetch_add(1, Ordering::Relaxed);
            }
            return if counted {
                Put::Visited((visitors >> VISITOR_SHIFT) as u8 - 1)
            } else {
                Put::Written
            };
        }
        self.count(Counter::DropContended, key[0] as usize);
        Put::Dropped
    }

    /// Counts one more thread under `key` ([`Visit::Counted`]), if the table
    /// holds it.
    #[inline]
    pub fn visit(&self, key: &[u64; 2]) -> Visit {
        match self.adjust(key, true) {
            Adjust::Done(before) => Visit::Counted(before),
            Adjust::Absent | Adjust::Busy => Visit::Absent,
        }
    }

    /// Undoes one [`Visit::Counted`] visit of `key`.  [`Leave::Done`] also
    /// when the key was replaced meanwhile (its count went with it, and a
    /// count of zero is never lowered).
    #[inline]
    pub fn leave(&self, key: &[u64; 2]) -> Leave {
        match self.adjust(key, false) {
            Adjust::Done(_) | Adjust::Absent => Leave::Done,
            Adjust::Busy => Leave::Busy,
        }
    }

    /// Adds (`up`) or removes one visitor of `key`.
    fn adjust(&self, key: &[u64; 2], up: bool) -> Adjust {
        let (key, _) = self.salted(key);
        let key = &key;
        let base = self.bucket_base(key);
        for way in 0..WAYS {
            let slot = base + way * WORDS;
            if self.words()[slot + 1].load(Ordering::Relaxed) != key[0] {
                continue;
            }
            let w = &self.words()[slot];
            let mut spins = 0usize;
            loop {
                if spins >= SPINS {
                    self.count(Counter::AdjustBusy, key[0] as usize);
                    return Adjust::Busy;
                }
                spins += 1;
                let w1 = w.load(Ordering::Acquire);
                if w1 & 1 == 1 {
                    // Held by a writer (for a few dozen instructions).
                    std::hint::spin_loop();
                    continue;
                }
                let k0 = self.words()[slot + 1].load(Ordering::Relaxed);
                let k1 = self.words()[slot + 2].load(Ordering::Relaxed);
                let d = self.words()[slot + 3].load(Ordering::Relaxed);
                fence(Ordering::Acquire);
                if d == 0 || k0 != key[0] || k1 != key[1] {
                    if w.load(Ordering::Relaxed) != w1 {
                        // Possibly torn: look again.
                        continue;
                    }
                    // Stable and not this key: try the other way.
                    break;
                }
                let count = w1 >> VISITOR_SHIFT;
                let w2 = if up {
                    if count >= VISITOR_MAX {
                        return Adjust::Absent;
                    }
                    w1 + VISITOR_ONE
                } else {
                    if count == 0 {
                        return Adjust::Absent;
                    }
                    w1 - VISITOR_ONE
                };
                // Succeeds only when nothing (no writer, no other visitor)
                // changed the slot since `w1`: the key we compared is the key
                // we count.
                if w.compare_exchange_weak(w1, w2, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    return Adjust::Done(count as u8);
                }
            }
        }
        Adjust::Absent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    impl SeqTable {
        /// [`SeqTable::put`] as a boolean (the tests' asserts).
        fn put_written<M, P>(&self, key: &[u64; 2], merge: M, priority: P, visit: bool) -> bool
        where
            M: Fn(Option<(u64, u32)>) -> (u64, u32),
            P: Fn(u64, u32) -> u64,
        {
            self.put(key, merge, priority, visit).written()
        }
    }

    /// At 255 visitors a visit is not counted, by `visit` or by a `put`
    /// with `visit`, and the leaves that undo the counted visits bring the
    /// count back to zero exactly.
    #[test]
    fn a_saturated_visitor_count_is_never_overdrawn() {
        let table = SeqTable::new(4);
        let k = key(3);
        assert_eq!(
            table.put(&k, |_| (1, 0), |_, m| m as u64, true),
            Put::Visited(0)
        );
        for _ in 1..255 {
            assert!(matches!(table.visit(&k), Visit::Counted(_)));
        }
        assert_eq!(table.get(&k).unwrap().visitors, 255);
        assert_eq!(table.visit(&k), Visit::Absent);
        assert_eq!(
            table.put(&k, |_| (2, 0), |_, m| m as u64, true),
            Put::Written
        );
        assert_eq!(table.get(&k).unwrap().visitors, 255);
        for _ in 0..255 {
            assert_eq!(table.leave(&k), Leave::Done);
        }
        assert_eq!(table.get(&k).unwrap().visitors, 0);
        assert_eq!(table.leave(&k), Leave::Done);
        assert_eq!(table.get(&k).unwrap().visitors, 0);
    }

    /// A visit reports the count it raised from, and a dropped write (a
    /// bucket full of visited slots, which are never replaced) is counted
    /// when counting is on, and only then.
    #[test]
    fn visits_report_the_count_before_and_drops_are_counted() {
        let table = SeqTable::new(1);
        let keys: Vec<[u64; 2]> = (1..=3).map(key).collect();
        for k in &keys[..2] {
            assert_eq!(
                table.put(k, |_| (1, 0), |_, m| m as u64, true),
                Put::Visited(0)
            );
        }
        assert_eq!(table.visit(&keys[0]), Visit::Counted(1));
        assert_eq!(table.visit(&keys[0]), Visit::Counted(2));
        assert_eq!(table.get(&keys[0]).unwrap().visitors, 3);
        // A write with a visit on a visited key reports the count before.
        assert_eq!(
            table.put(&keys[0], |old| old.unwrap(), |_, m| m as u64, true),
            Put::Visited(3)
        );
        assert_eq!(table.leave(&keys[0]), Leave::Done);
        // Every slot is visited: a third key is dropped, uncounted.
        assert_eq!(
            table.put(&keys[2], |_| (9, 0), |_, m| m as u64, false),
            Put::Dropped
        );
        assert_eq!(table.counters().0[Counter::DropVisited as usize], 0);
        table.set_counting(true);
        assert_eq!(
            table.put(&keys[2], |_| (9, 0), |_, m| m as u64, false),
            Put::Dropped
        );
        assert_eq!(table.counters().0[Counter::DropVisited as usize], 1);
        assert_eq!(table.counters().0[Counter::Evict as usize], 0);
        for _ in 0..3 {
            assert_eq!(table.leave(&keys[0]), Leave::Done);
        }
        // Now key 1 (unvisited, priority 0) is replaced: an eviction.
        assert!(table
            .put(&keys[2], |_| (9, 0), |_, m| m as u64, false)
            .written());
        assert_eq!(table.get(&keys[0]), None);
        assert_eq!(table.counters().0[Counter::Evict as usize], 1);
        // Sweeps are counted only while counting is on.
        table.set_counting(false);
        assert!(table.sweep(|_, m| m as u64));
        assert_eq!(table.counters().1, 0);
        table.set_counting(true);
        assert!(table.sweep(|_, m| m as u64));
        assert_eq!(table.counters().1, 1);
    }

    fn key(i: u64) -> [u64; 2] {
        let a = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678;
        let b = i.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(17) ^ 0xABCD;
        [a, b | 1]
    }

    #[test]
    fn a_leaving_write_stores_and_undoes_one_visit() {
        let table = SeqTable::new(4);
        let merge = |old: Option<(u64, u32)>| (old.map_or(7, |(d, _)| d + 1), 3);
        // Absent: written, nothing to undo.
        assert_eq!(
            table.put_leaving(&key(1), merge, |_, m| m as u64),
            Put::Written
        );
        assert_eq!(table.get(&key(1)).unwrap().visitors, 0);
        assert_eq!(table.visit(&key(1)), Visit::Counted(0));
        assert_eq!(table.visit(&key(1)), Visit::Counted(1));
        assert_eq!(
            table.put_leaving(&key(1), merge, |_, m| m as u64),
            Put::Written
        );
        assert_eq!(
            table.get(&key(1)),
            Some(Hit {
                d: 8,
                meta: 3,
                visitors: 1
            })
        );
        assert_eq!(
            table.put_leaving(&key(1), merge, |_, m| m as u64),
            Put::Written
        );
        assert_eq!(table.get(&key(1)).unwrap().visitors, 0);
        // A count of zero is never lowered.
        assert_eq!(
            table.put_leaving(&key(1), merge, |_, m| m as u64),
            Put::Written
        );
        assert_eq!(table.get(&key(1)).unwrap().visitors, 0);
        assert_eq!(table.get(&key(1)).unwrap().d, 10);
    }

    #[test]
    fn put_get_visit_leave_single_thread() {
        let table = SeqTable::new(4);
        assert_eq!(table.slots(), 16);
        assert_eq!(table.get(&key(1)), None);
        assert!(table.put_written(
            &key(1),
            |old| (old.map_or(7, |(d, _)| d + 1), 5),
            |_, m| m as u64,
            false
        ));
        assert_eq!(
            table.get(&key(1)),
            Some(Hit {
                d: 7,
                meta: 5,
                visitors: 0
            })
        );
        assert!(table.put_written(
            &key(1),
            |old| (old.map_or(7, |(d, _)| d + 1), 6),
            |_, m| m as u64,
            true
        ));
        assert_eq!(
            table.get(&key(1)),
            Some(Hit {
                d: 8,
                meta: 6,
                visitors: 1
            })
        );
        assert_eq!(table.visit(&key(1)), Visit::Counted(1));
        assert_eq!(table.get(&key(1)).unwrap().visitors, 2);
        assert_eq!(table.leave(&key(1)), Leave::Done);
        assert_eq!(table.leave(&key(1)), Leave::Done);
        assert_eq!(table.get(&key(1)).unwrap().visitors, 0);
        assert_eq!(table.leave(&key(1)), Leave::Done);
        assert_eq!(table.get(&key(1)).unwrap().visitors, 0);
        assert_eq!(table.visit(&key(2)), Visit::Absent);
        table.clear();
        assert_eq!(table.get(&key(1)), None);
    }

    /// 256 clears bring the epoch byte round: the slots are zeroed then,
    /// so an entry 256 epochs old is neither found nor counted.
    #[test]
    fn the_epoch_byte_never_repeats_on_a_live_entry() {
        let table = SeqTable::new(4);
        assert!(table.put_written(&key(7), |_| (9, 3), |_, m| m as u64, false));
        assert_eq!(table.occupied(), 1);
        for round in 1..=256 {
            table.clear();
            assert_eq!(table.get(&key(7)), None, "round {round}");
            assert_eq!(table.occupied(), 0, "round {round}");
        }
        assert!(table.put_written(&key(7), |_| (9, 3), |_, m| m as u64, false));
        assert_eq!(table.occupied(), 1);
    }

    #[test]
    fn a_full_bucket_replaces_its_least_valuable_slot_and_never_a_visited_one() {
        assert_eq!(WAYS, 2, "the test is written for two-slot buckets");
        // The smallest table is one bucket: both slots.
        let table = SeqTable::new(1);
        assert_eq!(table.slots(), 2);
        let keys: Vec<[u64; 2]> = (1..=5).map(key).collect();
        assert!(table.put_written(&keys[0], |_| (1, 1), |_, m| m as u64, true));
        assert!(table.put_written(&keys[1], |_| (2, 9), |_, m| m as u64, false));
        assert_eq!(table.fill(), 2);
        // Full: a cheap entry (priority 2) replaces the unvisited slot (key
        // 2, priority 9), never key 1 (priority 1, one visitor).
        assert!(table.put_written(&keys[2], |_| (3, 2), |_, m| m as u64, false));
        assert_eq!(table.get(&keys[0]).map(|h| h.d), Some(1));
        assert_eq!(table.get(&keys[1]), None);
        assert_eq!(table.get(&keys[2]).map(|h| h.d), Some(3));
        assert_eq!(table.fill(), 2);
        // No slot is kept for its value alone: key 4 (priority 3) takes the
        // slot of key 3.
        assert!(table.put_written(&keys[3], |_| (4, 3), |_, m| m as u64, false));
        assert_eq!(table.get(&keys[2]), None);
        // Once key 1 is left, it (priority 1, below key 4's 3) is the victim.
        assert_eq!(table.leave(&keys[0]), Leave::Done);
        assert!(table.put_written(&keys[4], |_| (5, 8), |_, m| m as u64, false));
        assert_eq!(table.get(&keys[0]), None);
        assert_eq!(table.get(&keys[3]).map(|h| h.d), Some(4));
        assert_eq!(table.get(&keys[4]).map(|h| h.d), Some(5));
    }

    /// Writers fill a small table while another thread sweeps it over and
    /// over; once all are done, the fill count is the occupied slots exactly
    /// (a sweep subtracts on the shard the writer added on).
    #[test]
    fn the_fill_count_survives_sweeps_under_concurrent_writes() {
        // A low sweep point, so that the sweeper runs while they write.
        let table = Arc::new(SeqTable::with_sweep_at(10, 30));
        // The sweeper runs until every writer is done (and at least 2,000
        // rounds), not a fixed number of rounds: on a slow build (coverage
        // instrumentation, 2026-11-05) a fixed count could end before the
        // writers filled the table, and `sweeps > 0` failed.
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writers: Vec<_> = (0..4u64)
            .map(|t| {
                let table = Arc::clone(&table);
                std::thread::spawn(move || {
                    for i in 0..40_000u64 {
                        let k = key(t << 40 | i);
                        let meta = (i % 7) as u32;
                        table.put(&k, |_| (i + 1, meta), |_, m| m as u64, false);
                    }
                })
            })
            .collect();
        let sweeper = {
            let table = Arc::clone(&table);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut sweeps = 0u32;
                let mut rounds = 0u32;
                while rounds < 2_000 || !done.load(std::sync::atomic::Ordering::Acquire) {
                    if table.needs_sweep() && table.sweep(|_, m| m as u64) {
                        sweeps += 1;
                    }
                    rounds += 1;
                    std::thread::yield_now();
                }
                sweeps
            })
        };
        for writer in writers {
            writer.join().unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Release);
        let sweeps = sweeper.join().unwrap();
        assert!(sweeps > 0, "the sweeper ran");
        assert_eq!(table.fill(), table.occupied());
        while table.needs_sweep() {
            assert!(table.sweep(|_, m| m as u64));
            assert_eq!(table.fill(), table.occupied());
        }
    }

    /// A sweep drops the cheapest half of the live entries, keeps the rest
    /// and every visited entry, and the fill count follows.
    #[test]
    fn a_sweep_drops_the_cheap_half_and_keeps_visited_entries() {
        let table = SeqTable::with_sweep_at(6, 50);
        assert_eq!(table.slots(), 64);
        // 48 keys, at most two per bucket (the salt keeps the low bits
        // distinct).
        let keys: Vec<[u64; 2]> = (0..48u64)
            .map(|i| [mix64(i + 1000), (i % 32) | i << 32])
            .collect();
        for (i, key) in keys.iter().enumerate() {
            assert!(table.put_written(key, |_| (i as u64 + 1, i as u32), |_, m| m as u64, false));
        }
        assert_eq!(table.fill(), 48);
        assert!(table.needs_sweep());
        assert_eq!(table.visit(&keys[2]), Visit::Counted(0));
        assert!(table.sweep(|_, m| m as u64));
        assert!(!table.sweeping.load(Ordering::Relaxed));
        // Priorities 0..=23 are half of 48: gone, except the visited one.
        for (i, key) in keys.iter().enumerate() {
            let expected = i >= 24 || i == 2;
            assert_eq!(table.get(key).is_some(), expected, "key {i}");
        }
        assert_eq!(table.fill(), 25);
        assert_eq!(table.occupied(), 25);
        assert!(!table.needs_sweep());
        assert_eq!(table.leave(&keys[2]), Leave::Done);
        // The freed slots take new entries again.
        assert!(table.put_written(&keys[5], |_| (6, 5), |_, m| m as u64, false));
        assert_eq!(table.fill(), 26);
    }

    /// Many threads write, read, visit and leave a small table: every read
    /// returns a word that was written for that key, never a torn or mixed
    /// one, and the visitor count returns to zero.
    #[test]
    fn contended_reads_are_never_torn() {
        let table = Arc::new(SeqTable::new(6));
        let threads = 8;
        let rounds = 20_000u64;
        let keys = 24u64;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let table = Arc::clone(&table);
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    for round in 0..rounds {
                        let k = (round * 7 + t) % keys;
                        let key = key(k);
                        // Data word: the key index in the high bits, a version
                        // in the low; metadata: the same version's low bits.
                        let version = round * threads + t;
                        let d = (k << 40) | (version & 0xFF) | 1 << 32;
                        let meta = (version & 0xFF) as u32;
                        table.put(&key, |_| (d, meta), |_, m| m as u64, false);
                        let counted = matches!(table.visit(&key), Visit::Counted(_));
                        if let Some(hit) = table.get(&key) {
                            assert_eq!(hit.d >> 40, k, "torn data word");
                            assert_eq!(
                                (hit.d & 0xFF) as u32,
                                hit.meta,
                                "data and meta from different writes"
                            );
                            seen += 1;
                        }
                        if counted {
                            while table.leave(&key) == Leave::Busy {
                                std::thread::yield_now();
                            }
                        }
                    }
                    seen
                })
            })
            .collect();
        let seen: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(seen > 0);
        assert!(!table.has_duplicate(), "a key in both slots of a bucket");
        for k in 0..keys {
            if let Some(hit) = table.get(&key(k)) {
                assert_eq!(hit.visitors, 0, "key {k}: visitors left behind");
                assert_eq!(hit.d >> 40, k);
            }
        }
    }

    /// Every thread creates the same new keys at the same time (as threads
    /// expanding one node at once do), counted in: one copy per key, and
    /// the counts come back to zero.
    #[test]
    fn simultaneous_inserts_make_one_copy() {
        for round in 0..200u64 {
            let table = Arc::new(SeqTable::new(3));
            let threads = 8;
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let table = Arc::clone(&table);
                    std::thread::spawn(move || {
                        let mut counted = Vec::new();
                        for k in 0..6u64 {
                            let key = key(round * 100 + k);
                            let fresh = (t + 1) << 8 | (k + 1);
                            if table
                                .put(&key, |old| old.unwrap_or((fresh, 0)), |_, m| m as u64, true)
                                .visited()
                            {
                                counted.push(key);
                            }
                        }
                        for key in counted {
                            while table.leave(&key) == Leave::Busy {
                                std::thread::yield_now();
                            }
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
            assert!(!table.has_duplicate(), "round {round}: a duplicate key");
            for k in 0..6u64 {
                if let Some(hit) = table.get(&key(round * 100 + k)) {
                    assert_eq!(hit.visitors, 0, "round {round} key {k}: visitors left");
                    assert_eq!(
                        hit.d & 0xFF,
                        k + 1,
                        "round {round} key {k}: another key's word"
                    );
                }
            }
        }
    }
}
