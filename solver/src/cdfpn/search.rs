//! A node's search: the table reads and writes (entries, intervals,
//! dominance), `mid`, descending into a child, and expanding a node into its
//! children (real and credit-line virtual nodes).

use super::*;

/// The credit line of a defense node whose legal replies are `moves`: an
/// interposable check with both interposition drops and other replies.
pub(super) fn credit_line(pos: &Position, attacker: u8, moves: &[Move]) -> Option<CheckLine> {
    if !moves.iter().any(|m| m.is_drop()) || moves.iter().all(|m| m.is_drop()) {
        return None;
    }
    let line = sliding_check_line(pos, attacker)?;
    let drops = moves.iter().filter(|&&m| line.interposes(m)).count();
    (drops > 0 && drops < moves.len()).then_some(line)
}

impl ClassicalProber {
    /// The table's numbers for `key`: `(pn, dn, pure, visitors)`, `(1, 1,
    /// false, 0)` when absent.
    #[inline]
    pub(super) fn read(&self, key: &[u64; 2]) -> (u32, u32, bool, u8) {
        match self.par.as_ref() {
            None => self.table.get(key).map_or((1, 1, false, 0), |entry| {
                (entry.pn, entry.dn, entry.pure, 0)
            }),
            Some(par) => par.tt.get(key).map_or((1, 1, false, 0), |hit| {
                (
                    hit.d as u32,
                    (hit.d >> 32) as u32,
                    hit.meta & 1 == 1,
                    hit.visitors,
                )
            }),
        }
    }

    /// The interval / dominance answer for a real node, `None` when they do not
    /// decide it.  `Some((true, pure))` is a proof.
    pub(super) fn known(&mut self, pos: &Position, bound: u32) -> Option<(bool, bool)> {
        let key = crate::solver::oracle_key_of(pos.hash());
        if let Some(par) = self.par.as_ref() {
            // The shared fact table instead of the interval table.
            if let Some(hit) = par.facts.get(&key) {
                let (mate, refuted_below, pure) = unpack_fact(hit.d);
                if mate != u8::MAX && u32::from(mate) <= bound {
                    self.stats.interval_hits += 1;
                    return Some((true, pure != u8::MAX && u32::from(pure) <= bound));
                }
                if bound < u32::from(refuted_below) {
                    self.stats.interval_hits += 1;
                    return Some((false, false));
                }
            }
            let board = board_key(pos);
            if let Some(entries) = self.dominance.get(&board) {
                if dominated(entries, pos.hands[self.attacker as usize], bound) {
                    self.stats.dominance_hits += 1;
                    return Some((true, true));
                }
            }
            return None;
        }
        if let Some(entry) = self.intervals.get(&key) {
            if entry.mate != u8::MAX && u32::from(entry.mate) <= bound {
                self.stats.interval_hits += 1;
                return Some((
                    true,
                    entry.pure != u8::MAX && u32::from(entry.pure) <= bound,
                ));
            }
            if bound < u32::from(entry.refuted_below) {
                self.stats.interval_hits += 1;
                return Some((false, false));
            }
        }
        let board = board_key(pos);
        if let Some(entries) = self.dominance.get(&board) {
            if dominated(entries, pos.hands[self.attacker as usize], bound) {
                self.stats.dominance_hits += 1;
                return Some((true, true));
            }
        }
        None
    }

    /// Records a real node's verdict in the interval (and, for a pure proof,
    /// the dominance) table.
    pub(super) fn settle(
        &mut self,
        pos: &Position,
        bound: u32,
        proven: bool,
        pure: bool,
        work: u64,
    ) {
        if bound > INTERVAL_BOUND_MAX {
            return;
        }
        let key = crate::solver::oracle_key_of(pos.hash());
        if let Some(par) = self.par.as_ref() {
            // The shared fact table instead of the interval table; the
            // dominance table stays this thread's.
            let b = bound as u8;
            let class = work_class(work.min(u64::from(u32::MAX)) as u32) as u32;
            par.facts.put(
                &key,
                |old| {
                    let (mut mate, mut refuted_below, mut pure_at, mut classes) = match old {
                        Some((d, meta)) => {
                            let (m, r, p) = unpack_fact(d);
                            (m, r, p, meta)
                        }
                        None => (u8::MAX, 0, u8::MAX, 0),
                    };
                    if proven {
                        mate = mate.min(b);
                        if pure {
                            pure_at = pure_at.min(b);
                        }
                    } else {
                        refuted_below = refuted_below.max(b + 1);
                    }
                    classes = classes.max(class);
                    (pack_fact_word(mate, refuted_below, pure_at), classes)
                },
                |_, meta| u64::from(meta),
                false,
            );
            if proven && pure {
                let board = board_key(pos);
                if self.dominance.len() >= self.caps[2] && !self.dominance.contains_key(&board) {
                    self.dominance.clear();
                    self.stats.collections += 1;
                }
                let entries = self.dominance.entry(board).or_default();
                record_dominance(entries, pos.hands[self.attacker as usize], bound);
            }
            return;
        }
        if self.intervals.len() >= self.caps[1] && !self.intervals.contains_key(&key) {
            collect_by(&mut self.intervals, |entry: &Interval| {
                work_class(entry.work)
            });
            self.stats.collections += 1;
        }
        let entry = self.intervals.get_or_insert(
            key,
            Interval {
                mate: u8::MAX,
                refuted_below: 0,
                pure: u8::MAX,
                work: 0,
            },
        );
        let b = bound as u8;
        if proven {
            entry.mate = entry.mate.min(b);
            if pure {
                entry.pure = entry.pure.min(b);
            }
        } else {
            entry.refuted_below = entry.refuted_below.max(b + 1);
        }
        entry.work = entry
            .work
            .saturating_add(work.min(u64::from(u32::MAX)) as u32);
        if proven && pure {
            let board = board_key(pos);
            if self.dominance.len() >= self.caps[2] && !self.dominance.contains_key(&board) {
                self.dominance.clear();
                self.stats.collections += 1;
            }
            let entries = self.dominance.entry(board).or_default();
            record_dominance(entries, pos.hands[self.attacker as usize], bound);
        }
    }

    pub(super) fn store(&mut self, key: [u64; 2], pn: u32, dn: u32, pure: bool, work: u64) {
        self.store_leaving(key, pn, dn, pure, work, false);
    }

    /// [`ClassicalProber::store`]; with `leaving` (a parallel prober that
    /// counted itself under `key`), the write also undoes that visit.
    /// `true` when the visit is undone (never on one thread, or without
    /// `leaving`).
    pub(super) fn store_leaving(
        &mut self,
        key: [u64; 2],
        pn: u32,
        dn: u32,
        pure: bool,
        work: u64,
        leaving: bool,
    ) -> bool {
        if let Some(par) = self.par.as_ref() {
            let class = work_class(work.min(u64::from(u32::MAX)) as u32) as u32;
            let late = std::cell::Cell::new(false);
            let merge = |old: Option<(u64, u32)>| {
                // The class of the work spent under the node so far, or
                // of this search's (the larger).
                let class = old.map_or(0, |(_, meta)| meta >> 1).max(class);
                if let Some((od, ometa)) = old {
                    // Another thread settled it (its numbers are real
                    // ones, item 3): an open value of this thread's --
                    // its children's entries may have been evicted --
                    // does not replace it.
                    let settled = od as u32 == 0 || (od >> 32) as u32 == 0;
                    if settled && pn != 0 && dn != 0 {
                        return (od, pack_meta(ometa & 1 == 1, class));
                    }
                    if settled {
                        late.set(true);
                    }
                }
                (pack_numbers(pn, dn), pack_meta(pure, class))
            };
            let written = if leaving {
                par.tt.put_leaving(&key, merge, entry_priority)
            } else {
                par.tt.put(&key, merge, entry_priority, false)
            };
            if !written.written() {
                self.stats.drops += 1;
            }
            if late.get() && (pn == 0 || dn == 0) {
                self.stats.late_stores += 1;
            }
            return leaving && written.written();
        }
        if self.table.len() >= self.caps[0] && !self.table.contains_key(&key) {
            collect_by(&mut self.table, |entry: &Entry| {
                work_class(entry.work)
                    + if entry.pn == 0 || entry.dn == 0 {
                        33
                    } else {
                        0
                    }
            });
            self.stats.collections += 1;
        }
        let entry = self.table.get_or_insert(
            key,
            Entry {
                pn,
                dn,
                work: 0,
                pure,
            },
        );
        entry.pn = pn;
        entry.dn = dn;
        entry.pure = pure;
        entry.work = entry
            .work
            .saturating_add(work.min(u64::from(u32::MAX)) as u32);
        false
    }

    /// A child's `(pn, dn, pure)` as the parent reads it (a `NOT` edge
    /// swaps), and the same numbers inflated by the other threads under it
    /// (the same, on one thread).
    #[inline]
    pub(super) fn child_numbers(
        &self,
        child: &Child,
        fresh: (u32, u32, bool),
    ) -> ((u32, u32, bool), (u32, u32)) {
        let hit = self
            .par
            .as_ref()
            .and_then(|par| par.tt.get(&child.key))
            .map(|hit| {
                (
                    hit.d as u32,
                    (hit.d >> 32) as u32,
                    hit.meta & 1 == 1,
                    hit.visitors,
                )
            });
        let (pn, dn, pure, visitors) = match hit {
            Some(hit) => hit,
            // Not in the table: what the child last returned to this
            // thread, if anything (`par_fresh`), else a fresh node.
            None if fresh != (0, 0, false) => (fresh.0, fresh.1, fresh.2, 0),
            None => (1, 1, false, 0),
        };
        let (ipn, idn) = (self.inflate(pn, visitors), self.inflate(dn, visitors));
        if child.negate {
            ((dn, pn, false), (idn, ipn))
        } else {
            ((pn, dn, pure), (ipn, idn))
        }
    }

    /// df-pn `MID` on `node`, anchored at `pos`.  Returns `(pn, dn, pure)` once
    /// one number reaches its threshold, and leaves them in the table.
    pub(super) fn mid(
        &mut self,
        pos: &mut Position,
        node: Node,
        th_pn: u32,
        th_dn: u32,
    ) -> (u32, u32, bool) {
        let key = node_key(&node, pos.hash());
        let (pn, dn, pure, visitors) = self.read(&key);
        let entry = (pn, dn, pure);
        if self.par.is_some() {
            if self.par_finished() {
                self.aborted = true;
            }
            if !self.par_leaves.is_empty() {
                self.retry_leaves(false);
            }
            if self.inflate(pn, visitors) >= th_pn
                || self.inflate(dn, visitors) >= th_dn
                || self.aborted
            {
                return entry;
            }
        } else if pn >= th_pn || dn >= th_dn || self.aborted {
            return entry;
        }
        let real = matches!(node.kind, Kind::Attack | Kind::Defense);
        if real {
            if let Some((proven, pure)) = self.known(pos, node.bound) {
                let result = if proven {
                    (0, INF, pure)
                } else {
                    (INF, 0, false)
                };
                self.store(key, result.0, result.1, result.2, 0);
                return result;
            }
        }
        if self.budget == 0 && !self.refill() {
            self.aborted = true;
            return entry;
        }
        self.budget -= 1;
        self.stats.nodes += 1;
        let started = self.stats.nodes;
        // On several threads: count this thread under the node (creating the
        // entry if need be, so the others see it from the start), and wake
        // the helpers once the main prober has done its share alone.
        let visited = match self.par.as_ref() {
            None => false,
            Some(par) => match par.tt.visit(&key) {
                Visit::Counted(before) => {
                    if before > 0 {
                        self.stats.concurrent += 1;
                    }
                    true
                }
                Visit::Absent => {
                    let put = par.tt.put(
                        &key,
                        |old| old.unwrap_or((pack_numbers(pn, dn), pack_meta(false, 0))),
                        entry_priority,
                        true,
                    );
                    let made = put.written();
                    if made {
                        self.stats.creations += 1;
                        // Every so often, see whether the shared tables are
                        // due for a sweep (their garbage collection).
                        if self.stats.creations & 0xFF == 0 {
                            if par.tt.needs_sweep() && par.tt.sweep(entry_priority) {
                                self.stats.sweeps += 1;
                            }
                            if par.facts.needs_sweep() && par.facts.sweep(|_, meta| u64::from(meta))
                            {
                                self.stats.sweeps += 1;
                            }
                        }
                    } else {
                        self.stats.drops += 1;
                    }
                    // A saturated count (255 threads under one node) is
                    // not ours to undo.
                    if let crate::pardfpn::Put::Visited(before) = put {
                        if before > 0 {
                            // Another thread made and entered the node
                            // between this thread's visit and its write.
                            self.stats.concurrent += 1;
                        }
                    }
                    put.visited()
                }
            },
        };
        if self.par_index == 0
            && !self.par_q.published
            && self.par.is_some()
            && self.stats.nodes - self.par_q.own_start >= PAR_JOIN_AFTER
        {
            self.publish();
        }
        let kbase = self.kids.len();
        // The leaf search (2026-10-26): an attack node at bound 1, and a plain
        // defense node at bound 2, are decided in place without table entries
        // for their children (`docs/experiments/solver-scaling/README.md` §6).
        let leaf = LEAF_SEARCH.load(Ordering::Relaxed);
        let expanded = if leaf && matches!(node.kind, Kind::Attack) && node.bound == 1 {
            Expanded::Terminal(self.leaf_attack1(pos))
        } else {
            match self.expand(pos, node) {
                Expanded::Children { impure }
                    if leaf
                        && matches!(node.kind, Kind::Defense)
                        && node.bound == 2
                        && self.kids[kbase..].iter().all(|kid| {
                            matches!(kid.node.kind, Kind::Attack) && kid.len == 1 && !kid.negate
                        }) =>
                {
                    let result = self.leaf_defense2(pos, kbase, impure);
                    self.kids.truncate(kbase);
                    Expanded::Terminal(result)
                }
                other => other,
            }
        };
        let result = match expanded {
            Expanded::Terminal(terminal) => terminal,
            Expanded::Children { impure } => {
                let (pn, dn, pure) = self.search_children(pos, node, kbase, th_pn, th_dn);
                (pn, dn, pure && !impure)
            }
        };
        self.kids.truncate(kbase);
        let mut left = false;
        if !self.aborted || result.0 == 0 || result.1 == 0 {
            let work = self.stats.nodes - started + 1;
            left = self.store_leaving(key, result.0, result.1, result.2, work, visited);
            if real && (result.0 == 0 || result.1 == 0) {
                self.settle(pos, node.bound, result.0 == 0, result.2, work);
            }
        }
        if visited && !left {
            if let Some(par) = self.par.as_ref() {
                if par.tt.leave(&key) == Leave::Busy {
                    self.par_leaves.push(key);
                }
            }
        }
        result
    }

    /// Whether a proof of `node` can be pure at all: the credit nodes are rule F.
    #[inline]
    pub(super) fn pure_node(&self, node: Node) -> bool {
        matches!(node.kind, Kind::Attack | Kind::Defense | Kind::Gate)
    }

    pub(super) fn descend(
        &mut self,
        pos: &mut Position,
        child: &Child,
        (th_pn, th_dn): (u32, u32),
    ) -> (u32, u32, bool) {
        let (th_pn, th_dn) = if child.negate {
            (th_dn, th_pn)
        } else {
            (th_pn, th_dn)
        };
        match child.len {
            0 => self.mid(pos, child.node, th_pn, th_dn),
            1 => {
                let undo = pos.make(child.path[0]);
                let value = self.mid(pos, child.node, th_pn, th_dn);
                pos.unmake(child.path[0], undo);
                value
            }
            _ => {
                let undo0 = pos.make(child.path[0]);
                let undo1 = pos.make(child.path[1]);
                let value = self.mid(pos, child.node, th_pn, th_dn);
                pos.unmake(child.path[1], undo1);
                pos.unmake(child.path[0], undo0);
                value
            }
        }
    }

    /// Pushes a real child reached by `path` from `pos`.
    pub(super) fn push_real(
        &mut self,
        pos: &mut Position,
        path: &[Move],
        kind: Kind,
        bound: u32,
        negate: bool,
    ) {
        let node = Node {
            kind,
            bound,
            drop: Move::drop(0, 0),
        };
        // The child's hash without making its last move.
        let hash = match *path {
            [m] => pos.hash_after(m),
            [m0, m1] => {
                let undo = pos.make(m0);
                let hash = pos.hash_after(m1);
                pos.unmake(m0, undo);
                hash
            }
            _ => unreachable!("a real child is one or two moves away"),
        };
        let key = node_key(&node, hash);
        // The child's entry, and its position's interval (the fact table on
        // several threads), which `known` reads first when it is expanded.
        let oracle = crate::solver::oracle_key_of(hash);
        match self.par.as_ref() {
            None => {
                self.table.prefetch(&key);
                self.intervals.prefetch(&oracle);
            }
            Some(par) => {
                par.tt.prefetch(&key);
                par.facts.prefetch(&oracle);
            }
        }
        let mut stored = [Move::drop(0, 0); 2];
        stored[..path.len()].copy_from_slice(path);
        self.kids.push(Child {
            node,
            path: stored,
            len: path.len() as u8,
            negate,
            key,
        });
    }

    /// Pushes a virtual child anchored at `pos`.
    pub(super) fn push_virtual(&mut self, pos: &Position, kind: Kind, bound: u32, drop: Move) {
        let node = Node { kind, bound, drop };
        let key = node_key(&node, pos.hash());
        match self.par.as_ref() {
            None => self.table.prefetch(&key),
            Some(par) => par.tt.prefetch(&key),
        }
        self.kids.push(Child {
            node,
            path: [Move::drop(0, 0); 2],
            len: 0,
            negate: false,
            key,
        });
    }

    /// Expands `node`: either its terminal value, or its children pushed onto
    /// [`ClassicalProber::kids`].
    pub(super) fn expand(&mut self, pos: &mut Position, node: Node) -> Expanded {
        match node.kind {
            Kind::Attack => {
                if node.bound == 0 {
                    return Expanded::Terminal((INF, 0, false));
                }
                let range = self.buf.gen_checks(pos, true);
                Move::sort_usi(&mut self.buf.buf[range.clone()]);
                // A bishop's, rook's or pawn's promotion before its twin
                // (solver 1.7.0): node counts only, never a verdict.
                Move::promotion_first(&mut self.buf.buf[range.clone()]);
                let mut moves = std::mem::take(&mut self.moves);
                moves.clear();
                moves.extend_from_slice(&self.buf.buf[range.clone()]);
                self.buf.buf.truncate(range.start);
                for &m in moves.iter() {
                    self.push_real(pos, &[m], Kind::Defense, node.bound - 1, false);
                }
                let empty = moves.is_empty();
                self.moves = moves;
                if empty {
                    return Expanded::Terminal((INF, 0, false));
                }
                Expanded::Children { impure: false }
            }
            Kind::Defense => self.expand_defense(pos, node.bound),
            Kind::Gate => {
                // b >= 2 on a credit line.
                let q = node.bound & !1;
                self.push_real(pos, &[node.drop], Kind::Attack, node.bound - 1, false);
                self.push_virtual(pos, Kind::Sub, q, node.drop);
                Expanded::Children { impure: false }
            }
            Kind::Sub => {
                self.push_virtual(pos, Kind::Ecap, node.bound, node.drop);
                // W does not depend on the drop: one per (D, q).
                self.push_virtual(pos, Kind::Witness, node.bound, Move::drop(0, 0));
                Expanded::Children { impure: true }
            }
            Kind::Ecap => {
                let line = sliding_check_line(pos, self.attacker).expect("a credit line");
                let undo = pos.make(node.drop);
                let forms = capture_forms_of(
                    &mut self.buf,
                    self.attacker,
                    pos,
                    line.checker,
                    node.drop.to,
                );
                pos.unmake(node.drop, undo);
                let mut any = false;
                for form in forms.iter().flatten() {
                    any = true;
                    self.push_real(pos, &[node.drop, *form], Kind::Defense, node.bound, false);
                }
                if any {
                    Expanded::Children { impure: true }
                } else {
                    Expanded::Terminal((INF, 0, false))
                }
            }
            Kind::Witness => {
                let line = sliding_check_line(pos, self.attacker).expect("a credit line");
                let range = self.buf.gen_legal(pos, true);
                Move::sort_usi(&mut self.buf.buf[range.clone()]);
                let mut moves = std::mem::take(&mut self.moves);
                moves.clear();
                moves.extend(
                    self.buf.buf[range.clone()]
                        .iter()
                        .copied()
                        .filter(|&m| !line.interposes(m)),
                );
                self.buf.buf.truncate(range.start);
                for m in moves.iter() {
                    self.push_real(pos, &[*m], Kind::Attack, node.bound - 2, true);
                }
                let empty = moves.is_empty();
                self.moves = moves;
                if empty {
                    Expanded::Terminal((INF, 0, false))
                } else {
                    Expanded::Children { impure: true }
                }
            }
        }
    }

    pub(super) fn expand_defense(&mut self, pos: &mut Position, bound: u32) -> Expanded {
        let mut moves = std::mem::take(&mut self.moves);
        let mut futile = std::mem::take(&mut self.futile);
        let expanded = self.expand_defense_into(pos, bound, &mut moves, &mut futile);
        self.moves = moves;
        self.futile = futile;
        expanded
    }

    /// [`ClassicalProber::expand_defense`] on the reused buffers `moves` and
    /// `futile` (no allocation per node).
    pub(super) fn expand_defense_into(
        &mut self,
        pos: &mut Position,
        bound: u32,
        moves: &mut Vec<Move>,
        futile: &mut Vec<bool>,
    ) -> Expanded {
        let safety = Safety::new(pos);
        if bound == 0 {
            if self.buf.has_legal_board_move_with(pos, &safety) {
                return Expanded::Terminal((INF, 0, false));
            }
            // Only drops are left.  Out of check none is legal (no reply:
            // not a mate) -- the same answer as the lists below; in check, the
            // verdict is rule F2's for the whole node, which the futility memo
            // keeps per position and stops at the first drop that is not
            // futile (2026-10-27; the same verdicts, so the same search).
            if !pos.is_in_check(pos.side) {
                return Expanded::Terminal((INF, 0, false));
            }
            return Expanded::Terminal(
                match self.futility.futile_defense_without_board_moves(
                    &mut self.buf,
                    self.attacker,
                    pos,
                    &safety,
                ) {
                    Some(0) => (0, INF, true),
                    Some(_) => (0, INF, false),
                    None => (INF, 0, false),
                },
            );
        }
        let range = self.buf.gen_legal_with(pos, true, &safety);
        Move::sort_usi(&mut self.buf.buf[range.clone()]);
        let count = range.len();
        if count == 0 {
            self.buf.buf.truncate(range.start);
            return Expanded::Terminal(if pos.is_in_check(pos.side) {
                (0, INF, true)
            } else {
                (INF, 0, false)
            });
        }
        let fbase = self.futile_buf.len();
        let classical = self.futility.classify_on(
            &mut self.buf,
            self.attacker,
            pos,
            range.clone(),
            &mut self.futile_buf,
            safety.check_line(),
        );
        moves.clear();
        moves.extend_from_slice(&self.buf.buf[range.clone()]);
        futile.clear();
        if classical {
            futile.extend(
                self.futile_buf[fbase..fbase + count]
                    .iter()
                    .map(Option::is_some),
            );
        }
        self.futile_buf.truncate(fbase);
        self.buf.buf.truncate(range.start);
        if classical && futile.iter().all(|&f| f) {
            // Mate by rule F2: not pure.
            return Expanded::Terminal((0, INF, false));
        }
        if bound == 0 {
            return Expanded::Terminal((INF, 0, false));
        }
        if classical {
            let mut skipped = false;
            for (m, &f) in moves.iter().zip(futile.iter()) {
                if f {
                    skipped = true;
                    continue;
                }
                self.push_real(pos, &[*m], Kind::Attack, bound - 1, false);
            }
            // A futile skip makes any proof of this node impure.
            return Expanded::Children { impure: skipped };
        }
        let line = credit_line(pos, self.attacker, moves);
        match line {
            Some(line) => {
                if bound < 2 {
                    return Expanded::Terminal((INF, 0, false));
                }
                for &m in moves.iter() {
                    if line.interposes(m) {
                        continue;
                    }
                    self.push_real(pos, &[m], Kind::Attack, bound - 1, false);
                }
                for &m in moves.iter() {
                    if line.interposes(m) {
                        self.push_virtual(pos, Kind::Gate, bound, m);
                    }
                }
                Expanded::Children { impure: false }
            }
            None => {
                for &m in moves.iter() {
                    self.push_real(pos, &[m], Kind::Attack, bound - 1, false);
                }
                Expanded::Children { impure: false }
            }
        }
    }
}
