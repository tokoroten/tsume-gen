//! Negatives from a reduction (`reduce --emit-negatives`, `distill
//! --emit-negatives`; owner order 2026-10-10 22:30 JST,
//! `docs/experiments/reduce-negatives/README.md`).
//!
//! A reducer judges many diagrams it never publishes: a removal whose result
//! the acceptance test refuses, an intermediate diagram of an accepted chain
//! from which a further piece still came off (rule D), the entry diagram a
//! relaxed entry let in.  Those are the *negatives* the generator's training
//! wants: diagrams close to a sound puzzle that fail one of the composition
//! rules (B yozume, C koma-amari, G inert pieces, D not minimal).
//!
//! # What this module is, and what it is not
//!
//! **An observer.**  The reducers report what they judged through
//! [`record`]; nothing here feeds back into a verdict, so every main output is
//! byte-identical with the feature on or off, and the TypeScript reducer needs
//! no counterpart (as with `reduce --cg-entry`).  Off (the default) a report is
//! one thread-local read.
//!
//! * The caller of one record opens a collection with [`begin`] and closes it
//!   with [`take`], on the same thread.  The reducers run their record on the
//!   caller's thread (the `--jobs` pool's coordinator included), so the events
//!   of one record never mix with another's.
//! * [`crate::reduce`] buffers its chains' events and hands them over in a
//!   fixed chain order when it returns, so the order does not depend on
//!   `--jobs` (see `NegativeScope` there).
//! * [`finalize`] picks at most `cap` events per source (closest to a sound
//!   diagram first), deduplicates them by canonical hash, and judges each
//!   picked diagram once more by the full publishing rules at its target, so
//!   every written row carries one verdict under one rule set whatever rules
//!   the reducer stepped by.
//!
//! Rule F (無駄合, classical) is a *measurement* in the gate, not a reason: a
//! futile interposition shows up as a changed length or as a rule B / C
//! verdict on the measured line, never under a reason of its own.

use std::cell::RefCell;
use std::collections::HashSet;

use serde_json::{json, Map, Value};

use crate::canonical::canonicalize;
use crate::exact::{exact_validate_with_rules, TsumeRules};
use crate::jsonl::{PipelineRecord, RemovedPiece};
use crate::position::Position;
use crate::sfen::to_sfen;
use crate::solver::Solver;

/// The negatives' format version (`negativesVersion` of every row).
pub const NEGATIVES_VERSION: &str = "negatives-v1";

/// Rule B's reasons.
const RULE_B: [&str; 2] = ["first_move_not_unique", "yozume_in_main_line"];
/// Rule C's.
const RULE_C: [&str; 1] = ["koma_amari"];
/// Rule G's.
const RULE_G: [&str; 2] = ["inert_attacker_piece", "inert_defender_piece"];

/// Where in a reducer an event was observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// The input diagram, refused by the entry check (or let in by a relaxed
    /// entry with its reasons).
    Entry,
    /// A greedy step the chain's rule refused: the candidate's removal (and
    /// the inert sweep it made possible) on a diagram of the chain.
    Refused,
    /// A diagram an accepted step led to (`accepted` tells whether it passes
    /// the full test; a repair step's may not).
    Step,
    /// A trial variant of `distill` (its step or final gate).
    Variant,
}

impl EventKind {
    pub const fn label(self) -> &'static str {
        match self {
            EventKind::Entry => "entry",
            EventKind::Refused => "refused",
            EventKind::Step => "step",
            EventKind::Variant => "variant",
        }
    }
}

/// One diagram a reducer judged.
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: EventKind,
    /// `reduce` or `distill`.
    pub tool: &'static str,
    pub position: Position,
    /// The length the reducer judged it against.
    pub target: Option<u32>,
    /// The reducer's reasons (under the rules it stepped by).
    pub reasons: Vec<String>,
    /// It passed the reducer's test.
    pub accepted: bool,
    /// The diagram the step started from passed the full test, so this one is
    /// one step away from a sound diagram ([`EventKind::Refused`]).
    pub parent_sound: bool,
    /// The removal of this step ([`EventKind::Refused`], [`EventKind::Step`]).
    pub step: Option<RemovedPiece>,
    /// Everything the chain had removed from the input before this step.
    pub path: Vec<RemovedPiece>,
    /// A chain of the reducer, for ordering.
    pub chain: u32,
    /// [`Position::hash`] of the diagram the step started from
    /// ([`EventKind::Refused`], [`EventKind::Step`]): a sound diagram that some
    /// accepted step started from is not minimal (rule D), whichever chain
    /// took the step.
    pub parent: Option<u128>,
    /// An accepted step of a main-line-preserving chain: the full test passed
    /// and no main-line path outside those of `parent` appeared (since 0.14.0
    /// the displayed main line itself may change to another path of `parent`).
    /// Only such a step makes `parent` not minimal: rule D v5 calls a diagram
    /// minimal when no main-line-preserving step from it is accepted
    /// (`docs/rules.md`, "Reduction v5"); a sound child with another line is a
    /// different puzzle (a length-preserving reduction), not a defect.
    pub keeps_line: bool,
}

/// Events kept per collection (one input) at most; later ones are counted in
/// [`Collected::dropped`] and not kept, so a long reduction cannot hold an
/// unbounded number of diagrams.  `distill` reports at most
/// `max_validations` (4,000, fixed on the CLI) of its own.
pub const MAX_EVENTS: usize = 20_000;

/// Events kept per reducer chain at most ([`crate::reduce`] buffers by chain,
/// so its cut is the same on every `--jobs` path).
pub const MAX_EVENTS_PER_CHAIN: usize = 4_096;

/// What one collection gathered.
#[derive(Clone, Debug, Default)]
pub struct Collected {
    /// In report order.
    pub events: Vec<Event>,
    /// Events not kept for [`MAX_EVENTS`] / [`MAX_EVENTS_PER_CHAIN`].
    pub dropped: u64,
}

thread_local! {
    static SINK: RefCell<Option<Collected>> = const { RefCell::new(None) };
}

/// Opens a collection on this thread (discarding any left open).
pub fn begin() {
    SINK.with(|sink| *sink.borrow_mut() = Some(Collected::default()));
}

/// Closes the collection and returns what it gathered.
pub fn take() -> Collected {
    SINK.with(|sink| sink.borrow_mut().take().unwrap_or_default())
}

/// Whether a collection is open on this thread.
pub fn active() -> bool {
    SINK.with(|sink| sink.borrow().is_some())
}

/// Runs `f` with a collection of its own and reopens this thread's afterwards:
/// what `f` reports is held back, not reported.  `None` when no collection was
/// open (nothing to report into).  For a check that re-runs a gate or a reducer
/// on a copy (`distill`'s gate at the mate length): the caller drops what was
/// held when the check passes, so the negatives stay what they were without it,
/// and hands it over with [`release`] when the check's diagrams matter.
pub fn held<T>(f: impl FnOnce() -> T) -> (T, Option<Collected>) {
    let saved = SINK.with(|sink| sink.borrow_mut().take());
    let open = saved.is_some();
    if open {
        begin();
    }
    let out = f();
    let held = SINK.with(|sink| std::mem::replace(&mut *sink.borrow_mut(), saved));
    (out, held.filter(|_| open))
}

/// Reports what [`held`] held back, in its order (a no-op unless a collection
/// is open).
pub fn release(collected: Collected) {
    record_all(collected.events, collected.dropped);
}

/// Reports one event (a no-op unless a collection is open).
pub fn record(event: Event) {
    record_all(std::iter::once(event), 0);
}

/// Reports several events in order, and `dropped` events a caller did not
/// keep.  Past [`MAX_EVENTS`] events are counted, not kept.
pub fn record_all(events: impl IntoIterator<Item = Event>, dropped: u64) {
    SINK.with(|sink| {
        if let Some(sink) = sink.borrow_mut().as_mut() {
            sink.dropped += dropped;
            for event in events {
                if sink.events.len() < MAX_EVENTS {
                    sink.events.push(event);
                } else {
                    sink.dropped += 1;
                }
            }
        }
    });
}

/// The composition rule a reason belongs to, `None` for a reason that is not
/// a composition defect (no mate, a changed length, a budget, structure, A).
pub fn soft_rule(reason: &str) -> Option<char> {
    if RULE_B.contains(&reason) {
        Some('B')
    } else if RULE_C.contains(&reason) {
        Some('C')
    } else if RULE_G.contains(&reason) {
        Some('G')
    } else {
        None
    }
}

/// `reasons` name composition defects (B, C, G) and nothing else.
pub fn soft_only(reasons: &[String]) -> bool {
    !reasons.is_empty() && reasons.iter().all(|reason| soft_rule(reason).is_some())
}

/// The class of a judged diagram: `yozume` (any rule B reason, the rest C/G),
/// `defect` (C/G only), `not_minimal` (sound, but a piece came off it),
/// `sound`, or `other` (anything else: no mate, length, budget, structure, A).
pub fn class_of(reasons: &[String], not_minimal: bool) -> &'static str {
    if reasons.is_empty() {
        return if not_minimal { "not_minimal" } else { "sound" };
    }
    if !soft_only(reasons) {
        return "other";
    }
    if reasons
        .iter()
        .any(|reason| RULE_B.contains(&reason.as_str()))
    {
        "yozume"
    } else {
        "defect"
    }
}

/// What [`finalize`] is told about the run.
#[derive(Clone, Copy, Debug)]
pub struct FinalizeOptions {
    /// Rows per source at most.
    pub cap: usize,
    /// Node budget of the re-judgement.
    pub node_budget: u64,
    /// `reduce` or `distill`.
    pub command: &'static str,
}

/// One event's summary key for the per-source counts: `<kind>:<class>`.
fn count_key(event: &Event, not_minimal: bool) -> String {
    let class = if event.accepted {
        if not_minimal {
            "not_minimal"
        } else {
            "sound"
        }
    } else {
        class_of(&event.reasons, false)
    };
    format!("{}:{}", event.kind.label(), class)
}

/// The sound diagrams (an accepted entry or step) that an accepted
/// main-line-preserving step started from ([`Event::keeps_line`]): a piece
/// came off them with the line kept, so they are sound but not minimal (rule
/// D v5).  Matched by the diagram's hash, so a step of an attempt chain counts
/// against a pre-pass diagram of its start chain too.
fn not_minimal_flags(events: &[Event]) -> Vec<bool> {
    let parents: HashSet<u128> = events
        .iter()
        .filter(|event| event.accepted && event.keeps_line)
        .filter_map(|event| event.parent)
        .collect();
    events
        .iter()
        .map(|event| {
            event.accepted
                && matches!(event.kind, EventKind::Entry | EventKind::Step)
                && parents.contains(&event.position.hash())
        })
        .collect()
}

/// Preference of an event as a negative, lower first; `None` when it is not
/// one (a sound diagram, or a failure that is no composition defect).
fn preference(event: &Event, not_minimal: bool) -> Option<u8> {
    if event.accepted {
        return not_minimal.then_some(1);
    }
    if !soft_only(&event.reasons) {
        return None;
    }
    Some(match event.kind {
        EventKind::Refused if event.parent_sound => 0,
        EventKind::Refused | EventKind::Step => 2,
        EventKind::Variant => 2,
        EventKind::Entry => 3,
    })
}

/// The primary rule of an event, for spreading the picks over the rules.
fn primary(event: &Event, not_minimal: bool) -> char {
    if event.accepted && not_minimal {
        return 'D';
    }
    event
        .reasons
        .iter()
        .find_map(|reason| soft_rule(reason))
        .unwrap_or('?')
}

/// Picks, deduplicates and re-judges the negatives of one source, returning
/// the rows (JSON values) to write and the per-source summary row.  `record`
/// is the *input* record (its provenance is the rows' root); `outcome` is what
/// the reducer made of it, for the summary.
///
/// Picking: events in preference order (one step from a sound diagram first,
/// then sound-but-not-minimal intermediates, then other B/C/G failures, then
/// the entry), round-robin over their primary rule (B, C, D, G) so one rule
/// does not crowd the others out, each canonical hash once.  Every pick is
/// judged again by [`TsumeRules::puzzle`] at its target (bound = target); a
/// pick the re-judgement finds sound is written with class `sound` (catalogue
/// candidate), one it finds no composition defect in (`other`) is dropped and
/// counted.
pub fn finalize(
    solver: &mut Solver,
    record: &PipelineRecord,
    collected: &Collected,
    outcome: Value,
    options: &FinalizeOptions,
) -> (Vec<Value>, Value) {
    let _phase = crate::prof::enter(crate::prof::Phase::Negatives);
    let events = collected.events.as_slice();
    let flags = not_minimal_flags(events);
    // The rows' provenance: the input's, with one lineage entry naming it
    // (`docs/legal-data.md` §3; every row of a YaneuraOu input keeps
    // `yaneuraou-2020` in its origins).
    let mut provenance = serde_json::to_value(&record.provenance).unwrap_or(Value::Null);
    if let Some(map) = provenance.as_object_mut() {
        let lineage = map
            .entry("lineage")
            .or_insert_with(|| Value::Array(Vec::new()));
        if !lineage.is_array() {
            *lineage = Value::Array(Vec::new());
        }
        if let Some(list) = lineage.as_array_mut() {
            list.push(json!({
                "sourceId": record.source_id,
                "tool": format!("{}-negatives", options.command),
                "version": NEGATIVES_VERSION,
            }));
        }
    }
    let mut counts: Map<String, Value> = Map::new();
    for (event, &not_minimal) in events.iter().zip(&flags) {
        let key = count_key(event, not_minimal);
        let entry = counts.entry(key).or_insert(json!(0));
        *entry = json!(entry.as_u64().unwrap_or(0) + 1);
    }
    // Candidates by preference, then by report order.
    let mut ranked: Vec<(u8, usize)> = events
        .iter()
        .zip(&flags)
        .enumerate()
        .filter_map(|(index, (event, &not_minimal))| {
            preference(event, not_minimal).map(|rank| (rank, index))
        })
        .collect();
    ranked.sort();
    // Preference rank by rank (a better rank is used up before a worse one);
    // within one rank, round-robin over the primary rule in rule order,
    // keeping each rule's report order.
    let mut order: Vec<usize> = Vec::new();
    for rank in ranked.chunk_by(|a, b| a.0 == b.0) {
        let mut by_rule: Vec<(char, Vec<usize>)> = Vec::new();
        for &(_, index) in rank {
            let rule = primary(&events[index], flags[index]);
            match by_rule.iter_mut().find(|(r, _)| *r == rule) {
                Some((_, list)) => list.push(index),
                None => by_rule.push((rule, vec![index])),
            }
        }
        by_rule.sort_by_key(|(rule, _)| *rule);
        let mut cursor = vec![0usize; by_rule.len()];
        loop {
            let mut progressed = false;
            for (slot, (_, list)) in by_rule.iter().enumerate() {
                if let Some(&index) = list.get(cursor[slot]) {
                    order.push(index);
                    cursor[slot] += 1;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut rows: Vec<Value> = Vec::new();
    let mut dropped = 0u64;
    let mut duplicates = 0u64;
    for index in order {
        if rows.len() >= options.cap {
            break;
        }
        let event = &events[index];
        let Some(target) = event.target else {
            continue;
        };
        let canonical = canonicalize(&event.position);
        if !seen.insert(canonical.hash.clone()) {
            duplicates += 1;
            continue;
        }
        let validation = exact_validate_with_rules(
            solver,
            &event.position,
            Some(target),
            target,
            options.node_budget,
            false,
            TsumeRules::puzzle(),
        );
        let class = class_of(&validation.reasons, flags[index]);
        if class == "other" {
            dropped += 1;
            continue;
        }
        let rules: Vec<String> = {
            let mut rules: Vec<String> = validation
                .reasons
                .iter()
                .filter_map(|reason| soft_rule(reason))
                .map(String::from)
                .collect();
            if class == "not_minimal" {
                rules.push("D".to_string());
            }
            rules.dedup();
            rules
        };
        let line = validation.main_line.as_ref();
        rows.push(json!({
            "type": "negative",
            "negativesVersion": NEGATIVES_VERSION,
            "sourceId": record.source_id,
            "sourceSfen": record.normalized_sfen.as_deref().unwrap_or(&record.source_sfen),
            "sfen": to_sfen(&event.position),
            "canonicalHash": canonical.hash,
            "matePly": target,
            "foundMatePly": validation.result.distance,
            "class": class,
            "rules": rules,
            "reasons": validation.reasons,
            "reducerReasons": event.reasons,
            "event": event.kind.label(),
            "tool": event.tool,
            "chain": event.chain,
            "oneStepFromSound": event.kind == EventKind::Refused && event.parent_sound,
            "step": event.step.as_ref().map(|piece| json!({"square": piece.square, "piece": piece.piece})),
            "pathFromSource": event.path.iter().map(|piece| json!({"square": piece.square, "piece": piece.piece})).collect::<Vec<_>>(),
            "mainLine": line.map(|line| line.moves.iter().map(|m| m.as_str().to_string()).collect::<Vec<_>>()),
            "yozumeNodes": line.map(|line| line.yozume_nodes),
            "leftoverHand": line.map(|line| line.leftover_hand),
            "firstMoveUnique": validation.unique_first_move,
            "maxAttackerBranching": validation.max_attacker_branching,
            "inertPieces": validation.inert_pieces.iter().map(|piece| json!({"square": piece.square, "piece": piece.piece})).collect::<Vec<_>>(),
            "gate": {
                "rules": "puzzle",
                "maxPly": target,
                "nodeBudget": options.node_budget,
                "nodes": validation.result.stats.nodes + validation.extra_nodes,
                "solverVersion": crate::jsonl::SOLVER_VERSION,
                "pipelineVersion": crate::jsonl::PIPELINE_VERSION,
            },
            "origins": record.provenance.origins,
            "originClass": record.provenance.origin_class,
            "rootSourceId": record.provenance.root_source_id,
            "provenance": provenance.clone(),
        }));
    }
    let summary = json!({
        "type": "source",
        "negativesVersion": NEGATIVES_VERSION,
        "sourceId": record.source_id,
        "targetMatePly": record.target_mate_ply,
        // The length the events were judged at (the input's target, else the
        // first event's), so the summary of an input without a target still
        // has a mate length.
        "matePly": record
            .target_mate_ply
            .or_else(|| events.iter().find_map(|event| event.target)),
        "outcome": outcome,
        "events": events.len(),
        "eventsDropped": collected.dropped,
        "counts": Value::Object(counts),
        "written": rows.len(),
        "droppedAfterRejudge": dropped,
        "duplicatesInSource": duplicates,
        "origins": record.provenance.origins,
        "originClass": record.provenance.origin_class,
        "rootSourceId": record.provenance.root_source_id,
        "provenance": provenance,
    });
    (rows, summary)
}
