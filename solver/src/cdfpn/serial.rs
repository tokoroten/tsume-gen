//! The one-thread df-pn loops: with the children's numbers kept and the
//! 1+ε threshold (`prober-v3`, the default), and reading every child from the
//! table (the leaf search off, `TSUME_CDFPN_LEAF=0`, the TypeScript order).

use super::*;

/// The 1+ε threshold of the one-thread prober (Pawlewicz and Lew 2007,
/// 2026-10-28, `prober-v3`, always on since 1.0.0; `TSUME_CDFPN_EPS=0`, which
/// turned it off, is no longer read): a child is searched until its number passes the
/// second-best sibling's by `max(1, value >> EPS_SHIFT)` instead of by 1,
/// so the loop switches between siblings less often.  ε = 1/4: 1.13x on one
/// thread; the parallel loop keeps [`bump`] (no gain there, 0.97x at P=8;
/// `docs/experiments/solver-scaling/README.md` §15.8).
///
/// Always above `value` (a child's threshold must pass its number, or the
/// loop picks it again without expanding anything): where the step would
/// reach [`NEAR_INF`] it is [`bump`]'s `value + 1` instead (so `NEAR_INF`
/// gives `INF`, as before; Codex review 2026-10-28: clamping at `NEAR_INF`
/// returned `NEAR_INF` for `NEAR_INF` and looped on two saturated children).
#[inline]
pub(super) fn bump_eps(value: u32) -> u32 {
    if value == INF {
        return INF;
    }
    let stepped = value.saturating_add((value >> EPS_SHIFT).max(1));
    if stepped < NEAR_INF {
        stepped
    } else {
        value + 1
    }
}

/// [`bump_eps`]'s ε as a shift: `1 / 2^EPS_SHIFT`.
pub(super) const EPS_SHIFT: u32 = 2;

impl ClassicalProber {
    /// A child's `(pn, dn, pure)` as the parent reads it on one thread (a
    /// `NOT` edge swaps).
    #[inline]
    pub(super) fn child_numbers_serial(&self, child: &Child) -> (u32, u32, bool) {
        let (pn, dn, pure) = self
            .table
            .get(&child.key)
            .map_or((1, 1, false), |entry| (entry.pn, entry.dn, entry.pure));
        if child.negate {
            (dn, pn, false)
        } else {
            (pn, dn, pure)
        }
    }

    /// The df-pn loop over the children `kids[kbase..]` of `node`: on one
    /// thread [`ClassicalProber::search_children_serial`] (the loop as it
    /// was), on several [`ClassicalProber::search_children_par`].
    #[inline]
    pub(super) fn search_children(
        &mut self,
        pos: &mut Position,
        node: Node,
        kbase: usize,
        th_pn: u32,
        th_dn: u32,
    ) -> (u32, u32, bool) {
        if self.par.is_some() {
            self.search_children_par(pos, node, kbase, th_pn, th_dn)
        } else if child_cache() {
            self.search_children_cached(pos, node, kbase, th_pn, th_dn)
        } else {
            self.search_children_serial(pos, node, kbase, th_pn, th_dn)
        }
    }

    /// The one-thread df-pn loop keeping the children's numbers (2026-10-27,
    /// the default): read from the table once, then only the child just
    /// searched changes, to the numbers its search returned.  A descent may
    /// also change a sibling's entry (a transposition); the loop sees that
    /// only when the node is searched again, which changes the order of the
    /// search and the node counts, never a verdict (`stats.nodesKind`
    /// `prober-v2`; with the 1+ε threshold, [`bump_eps`], the default since
    /// 2026-10-28, `prober-v3`).
    pub(super) fn search_children_cached(
        &mut self,
        pos: &mut Position,
        node: Node,
        kbase: usize,
        th_pn: u32,
        th_dn: u32,
    ) -> (u32, u32, bool) {
        let count = self.kids.len() - kbase;
        let or = node.kind.is_or();
        let vbase = self.child_values.len();
        #[cfg(test)]
        let eps = !self.no_epsilon;
        #[cfg(not(test))]
        let eps = true;
        for index in 0..count {
            let value = self.child_numbers_serial(&self.kids[kbase + index]);
            self.child_values.push(value);
        }
        let result = loop {
            let mut best = 0usize;
            let mut best_pair = (1u32, 1u32);
            let (mut min1, mut min2) = (INF, INF);
            let mut sum = 0u32;
            let mut pure_any = false;
            let mut pure_all = true;
            for (index, &(pn, dn, pure)) in self.child_values[vbase..].iter().enumerate() {
                let (lead, other) = if or { (pn, dn) } else { (dn, pn) };
                sum = add(sum, other);
                if lead < min1 {
                    min2 = min1;
                    min1 = lead;
                    best = index;
                    best_pair = (pn, dn);
                } else if lead < min2 {
                    min2 = lead;
                }
                if or {
                    pure_any |= pn == 0 && pure;
                } else {
                    pure_all &= pn == 0 && pure;
                }
            }
            let (pn, dn) = if or { (min1, sum) } else { (sum, min1) };
            let pure = if pn == 0 {
                if or {
                    pure_any
                } else {
                    pure_all
                }
            } else {
                false
            };
            if self.aborted || pn >= th_pn || dn >= th_dn {
                break (pn, dn, pure && self.pure_node(node));
            }
            let child_th = if or {
                (
                    th_pn.min(if eps { bump_eps(min2) } else { bump(min2) }),
                    if th_dn == INF {
                        INF
                    } else {
                        th_dn - dn + best_pair.1
                    },
                )
            } else {
                (
                    if th_pn == INF {
                        INF
                    } else {
                        th_pn - pn + best_pair.0
                    },
                    th_dn.min(if eps { bump_eps(min2) } else { bump(min2) }),
                )
            };
            let child = self.kids[kbase + best];
            let (cpn, cdn, cpure) = self.descend(pos, &child, child_th);
            self.child_values[vbase + best] = if child.negate {
                (cdn, cpn, false)
            } else {
                (cpn, cdn, cpure)
            };
        };
        self.child_values.truncate(vbase);
        result
    }

    /// The one-thread df-pn loop (unchanged from before the parallel prober,
    /// so the default path's work per node is what it was).
    pub(super) fn search_children_serial(
        &mut self,
        pos: &mut Position,
        node: Node,
        kbase: usize,
        th_pn: u32,
        th_dn: u32,
    ) -> (u32, u32, bool) {
        let count = self.kids.len() - kbase;
        let or = node.kind.is_or();
        loop {
            let mut best = 0usize;
            let mut best_pair = (1u32, 1u32);
            let (mut min1, mut min2) = (INF, INF);
            let mut sum = 0u32;
            let mut pure_any = false;
            let mut pure_all = true;
            for index in 0..count {
                let (pn, dn, pure) = self.child_numbers_serial(&self.kids[kbase + index]);
                let (lead, other) = if or { (pn, dn) } else { (dn, pn) };
                sum = add(sum, other);
                if lead < min1 {
                    min2 = min1;
                    min1 = lead;
                    best = index;
                    best_pair = (pn, dn);
                } else if lead < min2 {
                    min2 = lead;
                }
                if or {
                    pure_any |= pn == 0 && pure;
                } else {
                    pure_all &= pn == 0 && pure;
                }
            }
            let (pn, dn) = if or { (min1, sum) } else { (sum, min1) };
            let pure = if pn == 0 {
                if or {
                    pure_any
                } else {
                    pure_all
                }
            } else {
                false
            };
            if self.aborted || pn >= th_pn || dn >= th_dn {
                return (pn, dn, pure && self.pure_node(node));
            }
            let child_th = if or {
                (
                    th_pn.min(bump(min2)),
                    if th_dn == INF {
                        INF
                    } else {
                        th_dn - dn + best_pair.1
                    },
                )
            } else {
                (
                    if th_pn == INF {
                        INF
                    } else {
                        th_pn - pn + best_pair.0
                    },
                    th_dn.min(bump(min2)),
                )
            };
            let child = self.kids[kbase + best];
            self.descend(pos, &child, child_th);
        }
    }
}

#[cfg(test)]
mod eps_tests {
    use super::*;

    /// A child's threshold always passes its number (else the one-thread
    /// loop descends into it, returns at once and picks it again forever),
    /// never by less than [`bump`] does, and saturates as [`bump`] does.
    #[test]
    fn the_eps_threshold_always_passes_the_number() {
        let near = [NEAR_INF - 3, NEAR_INF - 2, NEAR_INF - 1, NEAR_INF];
        for value in (0..2_000u32).chain([1 << 20, 1 << 30, 3 << 30]).chain(near) {
            let t = bump_eps(value);
            assert!(t > value, "{value}");
            assert!(t >= bump(value), "{value}");
        }
        assert_eq!(bump_eps(NEAR_INF), INF);
        assert_eq!(bump_eps(NEAR_INF - 1), NEAR_INF);
        assert_eq!(bump_eps(INF), INF);
        assert_eq!(bump_eps(0), 1);
        assert_eq!(bump_eps(8), 10);
        // Where `value + value / 4` would reach NEAR_INF: `value + 1`.
        let big = NEAR_INF / 5 * 4 + 7;
        assert_eq!(bump_eps(big), big + 1);
        // The thresholds the loop gives the best child, with two children
        // saturated at NEAR_INF (`min1 == min2`) and an open parent: the
        // child's threshold is above its number.
        let (min1, min2, th_pn) = (NEAR_INF, NEAR_INF, INF);
        assert!(min1 < th_pn);
        assert!(th_pn.min(bump_eps(min2)) > min1);
    }
}
