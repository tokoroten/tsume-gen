//! An exact map from random 128-bit keys to small `Copy` values, for the hot
//! tables of the classical df-pn prober ([`crate::cdfpn`]).
//!
//! It holds exactly the entries a `HashMap<[u64; 2], V>` would hold after the
//! same inserts, updates, `retain`s and `clear`s: keys are compared in full, no
//! entry is ever dropped or replaced except by those calls.  Only where an entry
//! sits differs, and no caller depends on that ([`KeyTable::values`] is only
//! counted).
//!
//! Why not the `HashMap`: the keys are already uniformly random (Zobrist
//! folds), so the slot index is the low bits of `key[0]` and the 7-bit tag
//! the top bits of `key[1]` with no hashing at all; and the probe position of a
//! key is known before it is looked up, so the caller can prefetch it
//! ([`KeyTable::prefetch`]) while it still generates the other children.
//!
//! Layout (the `HashMap`'s, without its SIMD): a tag byte per slot (`0` empty,
//! `0x80 | 7 bits` full), mirrored past the end so a group of eight tags can be
//! read at any slot, and a separate array of `(key, value)` slots.  Linear
//! probing, eight tags at a time; at most 7/8 full, doubling when an insert would
//! pass that (the `HashMap`'s load, so the memory is the same).  Nothing is ever
//! deleted except by [`KeyTable::retain`] (which re-places the survivors in
//! place) and [`KeyTable::clear`], so a lookup stops at the first group with an
//! empty tag.

/// Tags read at once.
const GROUP: usize = 8;
const LSB: u64 = 0x0101_0101_0101_0101;
const MSB: u64 = 0x8080_8080_8080_8080;

#[derive(Clone, Copy)]
struct Slot<V: Copy> {
    key: [u64; 2],
    value: V,
}

pub(crate) struct KeyTable<V: Copy + Default> {
    /// `capacity + GROUP` tags; `tags[capacity + i]` mirrors `tags[i]`.
    tags: Vec<u8>,
    slots: Vec<Slot<V>>,
    /// `capacity - 1` (`capacity` a power of two), `0` before the first insert.
    mask: usize,
    len: usize,
}

impl<V: Copy + Default> Default for KeyTable<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn home(key: &[u64; 2], mask: usize) -> usize {
    (key[0] as usize) & mask
}

#[inline]
fn tag(key: &[u64; 2]) -> u8 {
    0x80 | (key[1] >> 57) as u8
}

/// The bytes of `group` equal to `byte`, as the high bit of each byte (a byte
/// above a true match may be reported too; the key comparison sorts it out).
#[inline]
fn matches(group: u64, byte: u8) -> u64 {
    let x = group ^ (LSB * u64::from(byte));
    x.wrapping_sub(LSB) & !x & MSB
}

/// The empty bytes of `group` (exact: a full tag has its high bit set).
#[inline]
fn empties(group: u64) -> u64 {
    !group & MSB
}

impl<V: Copy + Default> KeyTable<V> {
    pub(crate) fn new() -> Self {
        KeyTable {
            tags: Vec::new(),
            slots: Vec::new(),
            mask: 0,
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// The most entries before the table doubles.
    #[inline]
    fn limit(&self) -> usize {
        self.capacity() / 8 * 7
    }

    /// Eight tags from slot `pos` on (little-endian: byte `i` is slot `pos + i`).
    #[inline]
    fn group(&self, pos: usize) -> u64 {
        u64::from_le_bytes(
            self.tags[pos..pos + GROUP]
                .try_into()
                .expect("a group of tags"),
        )
    }

    #[inline]
    fn set_tag(&mut self, index: usize, value: u8) {
        self.tags[index] = value;
        if index < GROUP {
            let capacity = self.capacity();
            self.tags[capacity + index] = value;
        }
    }

    /// Where `key` sits, if it does.
    #[inline]
    fn find(&self, key: &[u64; 2]) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let want = tag(key);
        let mut pos = home(key, self.mask);
        loop {
            let group = self.group(pos);
            let mut hits = matches(group, want);
            while hits != 0 {
                let index = (pos + (hits.trailing_zeros() / 8) as usize) & self.mask;
                if self.slots[index].key == *key {
                    return Some(index);
                }
                hits &= hits - 1;
            }
            if empties(group) != 0 {
                return None;
            }
            pos = (pos + GROUP) & self.mask;
        }
    }

    /// The first empty slot from `key`'s home on.  The table is never full.
    #[inline]
    fn find_empty(&self, key: &[u64; 2]) -> usize {
        let mut pos = home(key, self.mask);
        loop {
            let free = empties(self.group(pos));
            if free != 0 {
                return (pos + (free.trailing_zeros() / 8) as usize) & self.mask;
            }
            pos = (pos + GROUP) & self.mask;
        }
    }

    #[inline]
    pub(crate) fn get(&self, key: &[u64; 2]) -> Option<&V> {
        self.find(key).map(|index| &self.slots[index].value)
    }

    #[inline]
    pub(crate) fn contains_key(&self, key: &[u64; 2]) -> bool {
        self.find(key).is_some()
    }

    /// The value of `key`, inserting `value` first when it is absent.
    #[inline]
    pub(crate) fn get_or_insert(&mut self, key: [u64; 2], value: V) -> &mut V {
        if let Some(index) = self.find(&key) {
            return &mut self.slots[index].value;
        }
        if self.len + 1 > self.limit() {
            self.grow();
        }
        let index = self.find_empty(&key);
        self.set_tag(index, tag(&key));
        self.slots[index] = Slot { key, value };
        self.len += 1;
        &mut self.slots[index].value
    }

    /// Doubles the capacity (16 slots at first), re-placing every entry.
    fn grow(&mut self) {
        let capacity = (self.capacity() * 2).max(16);
        let old_tags = std::mem::replace(&mut self.tags, vec![0; capacity + GROUP]);
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::default(); capacity]);
        self.mask = capacity - 1;
        for (index, slot) in old_slots.iter().enumerate() {
            if old_tags[index] != 0 {
                let to = self.find_empty(&slot.key);
                self.set_tag(to, tag(&slot.key));
                self.slots[to] = *slot;
            }
        }
    }

    /// Empties the table, keeping its capacity (as `HashMap::clear`).
    pub(crate) fn clear(&mut self) {
        if self.len != 0 {
            self.tags.fill(0);
            self.len = 0;
        }
    }

    /// Keeps the entries whose value satisfies `keep`.  One pass in probe
    /// order from an empty slot: a dropped entry's tag is emptied, a kept one
    /// is moved to the first empty slot from its home (never past where it is),
    /// so every probe chain stays unbroken.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&V) -> bool) {
        if self.len == 0 {
            return;
        }
        let capacity = self.capacity();
        let start = (0..capacity)
            .find(|&index| self.tags[index] == 0)
            .expect("the table is never full");
        for step in 1..=capacity {
            let index = (start + step) & self.mask;
            if self.tags[index] == 0 {
                continue;
            }
            let slot = self.slots[index];
            self.set_tag(index, 0);
            if keep(&slot.value) {
                let to = self.find_empty(&slot.key);
                self.set_tag(to, tag(&slot.key));
                self.slots[to] = slot;
            } else {
                self.len -= 1;
            }
        }
    }

    /// Every value, in slot order (callers only count them).
    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.slots
            .iter()
            .zip(self.tags.iter())
            .filter(|(_, &tag)| tag != 0)
            .map(|(slot, _)| &slot.value)
    }

    /// Asks the CPU to bring `key`'s tags and home slot into cache.
    #[inline]
    pub(crate) fn prefetch(&self, key: &[u64; 2]) {
        if self.len == 0 {
            return;
        }
        let index = home(key, self.mask);
        prefetch(self.tags.as_ptr().wrapping_add(index));
        prefetch(self.slots.as_ptr().wrapping_add(index).cast::<u8>());
    }
}

impl<V: Copy + Default> Default for Slot<V> {
    fn default() -> Self {
        Slot {
            key: [0; 2],
            value: V::default(),
        }
    }
}

/// Asks the CPU to bring the cache line at `pointer` into L1 (`prefetcht0` on
/// x86-64, `prfm pldl1keep` on AArch64, nothing elsewhere).
#[inline]
pub(crate) fn prefetch(pointer: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint; it never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch(pointer.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above; `prfm` neither faults nor touches memory or flags.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) pointer, options(nostack, readonly, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = pointer;
}

#[cfg(test)]
mod tests {
    use super::KeyTable;
    use std::collections::HashMap;

    fn key(rng: &mut u64) -> [u64; 2] {
        let mut next = || {
            *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        [next(), next()]
    }

    /// The table holds exactly what a `HashMap` holds through inserts,
    /// updates, retains and clears, including keys that share their home slot
    /// and their tag.
    #[test]
    fn matches_a_hash_map() {
        let mut rng = 7u64;
        let mut table: KeyTable<u32> = KeyTable::new();
        let mut map: HashMap<[u64; 2], u32> = HashMap::new();
        let mut keys = Vec::new();
        for round in 0..40_000u32 {
            let k = if round % 5 == 0 && !keys.is_empty() {
                keys[(round as usize * 31) % keys.len()]
            } else if round % 7 == 0 {
                // Same home slot and tag, different key.
                let mut k = key(&mut rng);
                k[0] &= 0xFF;
                k[1] |= 0xFE00_0000_0000_0000;
                k
            } else {
                key(&mut rng)
            };
            keys.push(k);
            *table.get_or_insert(k, 0) += round;
            *map.entry(k).or_insert(0) += round;
            if round % 9_000 == 8_999 {
                table.retain(|&v| v % 3 != 0);
                map.retain(|_, v| *v % 3 != 0);
            }
            if round == 30_000 {
                table.clear();
                map.clear();
            }
            assert_eq!(table.len(), map.len());
        }
        for k in &keys {
            assert_eq!(table.get(k), map.get(k));
            assert_eq!(table.contains_key(k), map.contains_key(k));
        }
        let mut a: Vec<u32> = table.values().copied().collect();
        let mut b: Vec<u32> = map.values().copied().collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}
