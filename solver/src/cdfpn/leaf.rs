//! The leaf search (2026-10-26): an attack node at bound 1 and a defense
//! node at bound 2 are decided without table entries.

use super::*;

impl ClassicalProber {
    /// An attack node at bound 1, decided without the table: mate when some
    /// check's bound-0 defense node mates (no reply in check, or every reply
    /// F2-futile), which is what [`ClassicalProber::expand_defense_into`]
    /// answers at bound 0 (always a terminal).  A pure mate (one using no
    /// rule F) is preferred to an impure one; purity only feeds the dominance
    /// table, never a verdict.
    pub(super) fn leaf_attack1(&mut self, pos: &mut Position) -> (u32, u32, bool) {
        let range = self.buf.gen_checks(pos, true);
        let mut checks = std::mem::take(&mut self.leaf_checks);
        checks.clear();
        checks.extend_from_slice(&self.buf.buf[range.clone()]);
        self.buf.buf.truncate(range.start);
        // A bishop's, rook's or pawn's promotion before its twin (solver 1.7.0).
        Move::promotion_first(&mut checks);
        let mut moves = std::mem::take(&mut self.leaf_moves);
        let mut futile = std::mem::take(&mut self.leaf_futile);
        let (mut pure, mut impure) = (false, false);
        for &m in checks.iter() {
            // A king step out of the check: no mate, decided without playing
            // the check (the verdict `expand_defense_into` gives at bound 0
            // when the defender has a legal board move; 2026-10-27).
            if crate::movegen::king_escapes_after(pos, m) {
                continue;
            }
            let undo = pos.make(m);
            let verdict = self.expand_defense_into(pos, 0, &mut moves, &mut futile);
            pos.unmake(m, undo);
            match verdict {
                Expanded::Terminal((0, _, true)) => {
                    pure = true;
                    break;
                }
                Expanded::Terminal((0, _, false)) => impure = true,
                Expanded::Terminal(_) => {}
                Expanded::Children { .. } => unreachable!("a bound-0 defense node is terminal"),
            }
        }
        self.leaf_checks = checks;
        self.leaf_moves = moves;
        self.leaf_futile = futile;
        if pure {
            (0, INF, true)
        } else if impure {
            (0, INF, false)
        } else {
            (INF, 0, false)
        }
    }

    /// A defense node at bound 2 whose children `kids[kbase..]` are all real
    /// attack nodes at bound 1 (no credit line), decided without the table:
    /// mate when every reply's attack node mates ([`ClassicalProber::leaf_attack1`]),
    /// pure when they all are and no reply was skipped as futile (`impure`).
    pub(super) fn leaf_defense2(
        &mut self,
        pos: &mut Position,
        kbase: usize,
        impure: bool,
    ) -> (u32, u32, bool) {
        let mut pure = !impure;
        for index in kbase..self.kids.len() {
            let reply = self.kids[index].path[0];
            let undo = pos.make(reply);
            let (pn, _, child_pure) = self.leaf_attack1(pos);
            pos.unmake(reply, undo);
            if pn != 0 {
                return (INF, 0, false);
            }
            pure &= child_pure;
        }
        (0, INF, pure)
    }
}
