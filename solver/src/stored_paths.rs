//! `Π` and the final-move alternative of a **stored** solution graph
//! (`main-line-paths`, catalog schema v2.5 `n_main_paths` / `has_final_alt`),
//! and its designated main line (`stored-main-line`, 1.8.0: the line the gate
//! of this build would display for the stored graph, without solving).
//!
//! The catalog keeps each positive row's runtime graph (`solution.graph`, the
//! [`SolverGraphJson`] shape [`crate::stages::pruned_graph_to_json`] writes).
//! [`graph_from_json`] turns it back into a [`Graph`] node for node, so the
//! counts come from the very functions the reducer calls on the in-memory
//! graph: [`main_line_paths`] and [`has_final_move_alternative`], both of which
//! walk the main-line region `R` through [`crate::exact::SolutionFacts`].
//!
//! The runtime graph keeps everything `R` reads: every correct move `C(n)` with
//! its child, every reply of a defence node of the solution tree except rule F's
//! 無駄合 and the 変同駒余り replies outside `P(m)`
//! ([`crate::exact::prune_solver_graph`]).  A defence node whose replies were
//! all 無駄合 keeps none, which [`crate::exact::is_mated_leaf`] reads the same
//! way.  So `R`, `Π` and the counts are those of the complete search graph.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use std::collections::{HashMap, HashSet};

use crate::exact::{
    correct_moves, designated_main_line, has_final_move_alternative, is_mated_leaf,
    longest_replies, main_line_paths,
};
use crate::jsonl::SolverGraphJson;
use crate::position::Move;
use crate::sfen::parse_sfen;
use crate::solver::{Graph, GraphMove, NodeSpec, Outcome, MAX_GRAPH_DISTANCE};

/// What `main-line-paths` reports for one stored graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredPaths {
    /// `|Π|`, `None` when it is over the cap.
    pub n_main_paths: Option<usize>,
    pub has_final_alt: bool,
}

fn parse_outcome(text: &str) -> Result<Outcome> {
    match text {
        "MATE" => Ok(Outcome::Mate),
        "REFUTED" => Ok(Outcome::Refuted),
        "UNKNOWN" => Ok(Outcome::Unknown),
        other => bail!("unknown outcome {other:?}"),
    }
}

/// The [`Graph`] of a stored solution graph: node `i` is the `i`-th key of
/// `graph.nodes`, with the position its `sfen` gives.  A `next` naming a node
/// the JSON does not hold points past the last node, which [`Graph::node`]
/// answers with `None` -- what the walks do with a node the graph does not
/// hold.  `refutation` is dropped (no reader of `R` looks at it).
pub fn graph_from_json(graph: &SolverGraphJson) -> Result<Graph> {
    let missing = u32::try_from(graph.nodes.len()).context("too many nodes")?;
    let index = |name: &str| -> u32 {
        graph
            .nodes
            .get_index_of(name)
            .map_or(missing, |index| index as u32)
    };
    let root = graph
        .nodes
        .get_index_of(graph.root.as_str())
        .ok_or_else(|| anyhow!("root node {} missing", graph.root))? as u32;
    let mut specs = Vec::with_capacity(graph.nodes.len());
    for (name, node) in &graph.nodes {
        let attack = match node.side.as_str() {
            "attack" => true,
            "defense" => false,
            other => bail!("node {name}: unknown side {other:?}"),
        };
        let position =
            parse_sfen(&node.sfen).map_err(|error| anyhow!("node {name}: sfen: {error}"))?;
        let outcome = parse_outcome(&node.outcome).with_context(|| format!("node {name}"))?;
        let mut moves = Vec::with_capacity(node.moves.len());
        for m in &node.moves {
            let mv = Move::from_usi(&m.usi).map_err(|error| anyhow!("node {name}: {error}"))?;
            let outcome = parse_outcome(&m.outcome).with_context(|| format!("node {name}"))?;
            let next = m.next.as_deref().map(index);
            check_distance(m.distance).with_context(|| format!("node {name}: move {}", m.usi))?;
            moves.push(GraphMove::new(
                mv, outcome, m.distance, next, None, m.useless,
            ));
        }
        check_distance(node.distance).with_context(|| format!("node {name}"))?;
        specs.push(NodeSpec {
            attack,
            position,
            outcome,
            distance: node.distance,
            moves,
        });
    }
    let graph = Graph::from_specs(root, specs);
    check_region_walk(&graph)?;
    Ok(graph)
}

/// A distance the graph can store ([`GraphMove::new`] asserts it).
fn check_distance(distance: Option<u32>) -> Result<()> {
    match distance {
        Some(d) if d > MAX_GRAPH_DISTANCE => {
            bail!("distance {d} out of range (at most {MAX_GRAPH_DISTANCE})")
        }
        _ => Ok(()),
    }
}

/// The deepest walk from the root a stored graph may need, far beyond any
/// real mate (the catalog stops at 15 plies): the region's walks recurse once
/// per ply, so a deeper (corrupt) graph is refused rather than risk the stack.
pub const MAX_REGION_DEPTH: usize = 1024;

/// Refuses a graph the region's walks could not finish: a cycle, or a path
/// deeper than [`MAX_REGION_DEPTH`], through the edges they recurse on --
/// every correct move `C(n)` at an attack node and every longest reply `L(m)`
/// at a defence node that is not mated (`SolutionFacts::amari` and
/// `SolutionFacts::clean` read these; `P(m)` and so `R` are subsets).  A solved graph never
/// has either: the mate distance strictly decreases along those edges.
fn check_region_walk(graph: &Graph) -> Result<()> {
    fn edges(graph: &Graph, id: u32) -> Vec<u32> {
        match graph.node(id) {
            Some(node) if node.attack() => correct_moves(node)
                .iter()
                .filter_map(|m| m.next())
                .collect(),
            Some(node) if !is_mated_leaf(node) => longest_replies(node)
                .iter()
                .filter_map(|m| m.next())
                .collect(),
            _ => Vec::new(),
        }
    }
    // Iterative depth-first search: `depth` is set when a node is finished
    // (the longest path below it), `open` while it is on the current path.
    let mut depth: HashMap<u32, usize> = HashMap::new();
    let mut open: HashSet<u32> = HashSet::new();
    let mut stack: Vec<(u32, Vec<u32>, usize)> = vec![(graph.root, edges(graph, graph.root), 0)];
    open.insert(graph.root);
    while let Some((id, children, next)) = stack.last_mut() {
        if let Some(&child) = children.get(*next) {
            *next += 1;
            if open.contains(&child) {
                bail!("cycle in the solution graph");
            }
            if !depth.contains_key(&child) {
                if stack.len() > MAX_REGION_DEPTH {
                    bail!("solution graph deeper than {MAX_REGION_DEPTH} plies");
                }
                open.insert(child);
                let below = edges(graph, child);
                stack.push((child, below, 0));
            }
            continue;
        }
        let id = *id;
        let below = children
            .iter()
            .filter_map(|child| depth.get(child))
            .max()
            .map_or(0, |d| d + 1);
        if below > MAX_REGION_DEPTH {
            bail!("solution graph deeper than {MAX_REGION_DEPTH} plies");
        }
        open.remove(&id);
        depth.insert(id, below);
        stack.pop();
    }
    Ok(())
}

/// `|Π|` (under `cap`) and the final-move alternative of a stored graph.  The
/// attacker is the side to move at the root.
pub fn measure_graph(graph: &SolverGraphJson, cap: usize) -> Result<StoredPaths> {
    let graph = graph_from_json(graph)?;
    let root = graph
        .node(graph.root)
        .expect("graph_from_json checks the root");
    let attacker = root.position().side;
    Ok(StoredPaths {
        n_main_paths: main_line_paths(&graph, attacker, cap).map(|paths| paths.len()),
        has_final_alt: has_final_move_alternative(&graph, attacker),
    })
}

/// The stored graph of a record's `solution`: the solution object, or a string
/// holding its JSON (the catalog's `puzzles.solution` text).
fn graph_of_solution(solution: &Value) -> Result<SolverGraphJson> {
    let parsed;
    let solution = match solution {
        Value::String(text) => {
            parsed = serde_json::from_str::<Value>(text).context("solution is not JSON")?;
            &parsed
        }
        Value::Null => bail!("no solution"),
        other => other,
    };
    let graph = match solution.get("graph") {
        None | Some(Value::Null) => bail!("no graph"),
        Some(graph) => graph,
    };
    SolverGraphJson::deserialize(graph).map_err(|error| anyhow!("malformed graph: {error}"))
}

/// [`measure_graph`] of a record's `solution`: the solution object, or a string
/// holding its JSON (the catalog's `puzzles.solution` text).
pub fn measure_solution(solution: &Value, cap: usize) -> Result<StoredPaths> {
    measure_graph(&graph_of_solution(solution)?, cap)
}

/// The designated main line of a stored graph ([`designated_main_line`], the
/// gate's own walk: inside `R`, the `Clean` child first, ties on
/// [`crate::exact::main_line_key`], so a promotion before its non-promoting
/// twin since `PIPELINE_VERSION` 0.17.0).  The walk reads only `R`, which the
/// stored graph holds whole (see the module comment), so the line is the one
/// `solve --emit-main-line` of this build writes for the same diagram: a graph
/// stored by an older build gets this build's tie-break.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredMainLine {
    /// Attacker and defender moves alternating, as USI.
    pub moves: Vec<String>,
    /// Pieces in the attacker's hand at the line's terminal mate.
    pub leftover_hand: usize,
    /// `false` when the walk did not reach a terminal mate.
    pub complete: bool,
}

/// [`StoredMainLine`] of a stored graph.  The attacker is the side to move at
/// the root.
pub fn main_line_of_graph(graph: &SolverGraphJson) -> Result<StoredMainLine> {
    let graph = graph_from_json(graph)?;
    let attacker = graph
        .node(graph.root)
        .expect("graph_from_json checks the root")
        .position()
        .side;
    let line = designated_main_line(&graph, attacker);
    Ok(StoredMainLine {
        moves: line
            .moves
            .iter()
            .map(|usi| usi.as_str().to_string())
            .collect(),
        leftover_hand: line.leftover_hand,
        complete: line.complete,
    })
}

/// [`main_line_of_graph`] of a record's `solution` (object or JSON text).
pub fn main_line_of_solution(solution: &Value) -> Result<StoredMainLine> {
    main_line_of_graph(&graph_of_solution(solution)?)
}

/// One output line of `stored-main-line` for one input line:
/// `{"canonicalHash", "mainLine", "leftoverHand", "complete", "error"}`.  A
/// failure is reported in `error` (with `mainLine: null`), never raised.
pub fn stored_main_line_line(line: &str) -> String {
    let (hash, result) = match serde_json::from_str::<Value>(line) {
        Err(error) => (Value::Null, Err(anyhow!("input is not JSON: {error}"))),
        Ok(record) => {
            let hash = record.get("canonicalHash").cloned().unwrap_or(Value::Null);
            let result = match record.get("solution") {
                None => Err(anyhow!("no solution")),
                Some(solution) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    main_line_of_solution(solution)
                }))
                .unwrap_or_else(|_| Err(anyhow!("internal error (panic)"))),
            };
            (hash, result)
        }
    };
    let value = match result {
        Ok(line) => json!({
            "canonicalHash": hash,
            "mainLine": line.moves,
            "leftoverHand": line.leftover_hand,
            "complete": line.complete,
            "error": null,
        }),
        Err(error) => json!({
            "canonicalHash": hash,
            "mainLine": null,
            "leftoverHand": null,
            "complete": false,
            "error": format!("{error:#}"),
        }),
    };
    value.to_string()
}

/// One output line of `main-line-paths` for one input line:
/// `{"canonicalHash", "nMainPaths", "hasFinalAlt", "error"}`.  A failure is
/// reported in `error` (with `nMainPaths: null, hasFinalAlt: 0`), never raised.
pub fn main_line_paths_line(line: &str, cap: usize) -> String {
    let (hash, result) = match serde_json::from_str::<Value>(line) {
        Err(error) => (Value::Null, Err(anyhow!("input is not JSON: {error}"))),
        Ok(record) => {
            let hash = record.get("canonicalHash").cloned().unwrap_or(Value::Null);
            let result = match record.get("solution") {
                None => Err(anyhow!("no solution")),
                // A corrupt row must not stop the batch: a panic below (none is
                // known; the checks above refuse what would assert) becomes
                // this row's `error`.
                Some(solution) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    measure_solution(solution, cap)
                }))
                .unwrap_or_else(|_| Err(anyhow!("internal error (panic)"))),
            };
            (hash, result)
        }
    };
    let value = match result {
        Ok(paths) => json!({
            "canonicalHash": hash,
            "nMainPaths": paths.n_main_paths,
            "hasFinalAlt": u8::from(paths.has_final_alt),
            "error": null,
        }),
        Err(error) => json!({
            "canonicalHash": hash,
            "nMainPaths": null,
            "hasFinalAlt": 0,
            "error": format!("{error:#}"),
        }),
    };
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::{exact_validate_with_rules, prune_solver_graph, TsumeRules};
    use crate::solver::Solver;
    use crate::stages::{full_graph_to_json, pruned_graph_to_json};

    fn solved(sfen: &str, max_ply: u32) -> Graph {
        let position = parse_sfen(sfen).unwrap();
        let validation = exact_validate_with_rules(
            &mut Solver::new(),
            &position,
            None,
            max_ply,
            2_000_000,
            true,
            TsumeRules::puzzle(),
        );
        assert!(validation.accepted, "{sfen}: {:?}", validation.reasons);
        validation.result.graph.expect("graph")
    }

    fn direct(graph: &Graph, cap: usize) -> StoredPaths {
        let attacker = graph.node(graph.root).unwrap().position().side;
        StoredPaths {
            n_main_paths: main_line_paths(graph, attacker, cap).map(|paths| paths.len()),
            has_final_alt: has_final_move_alternative(graph, attacker),
        }
    }

    #[test]
    fn final_move_alternatives_and_their_mirror() {
        // Three plies, two mating moves at the end; and its left-right mirror.
        for sfen in [
            "9/9/9/9/9/5+B+R2/5G3/2G6/4k4 b - 1",
            "9/9/9/9/9/2+R+B5/3G5/6G2/4k4 b - 1",
        ] {
            let graph = solved(sfen, 3);
            let expected = direct(&graph, 512);
            assert!(expected.has_final_alt, "{sfen}");
            for json in [
                pruned_graph_to_json(&prune_solver_graph(&graph)),
                full_graph_to_json(&graph),
            ] {
                assert_eq!(measure_graph(&json, 512).unwrap(), expected, "{sfen}");
            }
        }
    }

    #[test]
    fn a_reducible_record_keeps_a_graph_whose_pi_holds_its_line() {
        // owner 2026-11-11 10:0x (catalog_record::keeps_graph): the line must be a path of Π.
        let graph = solved("9/9/9/9/9/5+B+R2/5G3/2G6/4k4 b - 1", 3);
        let json = pruned_graph_to_json(&prune_solver_graph(&graph));
        let attacker = graph.node(graph.root).unwrap().position().side;
        let paths = main_line_paths(&graph, attacker, 512).unwrap();
        for path in &paths {
            let line: Vec<String> = path.iter().map(|usi| usi.as_str().to_string()).collect();
            assert!(crate::catalog_record::keeps_graph(&json, &line), "{line:?}");
        }
        let mut other: Vec<String> = paths[0]
            .iter()
            .map(|usi| usi.as_str().to_string())
            .collect();
        other.pop();
        assert!(!crate::catalog_record::keeps_graph(&json, &other));
        other.push("1a1b".to_string());
        assert!(!crate::catalog_record::keeps_graph(&json, &other));
        assert!(!crate::catalog_record::keeps_graph(&json, &[]));
    }

    #[test]
    fn the_cap_counts_like_the_reducer() {
        let graph = solved("9/9/9/9/9/5+B+R2/5G3/2G6/4k4 b - 1", 3);
        let json = pruned_graph_to_json(&prune_solver_graph(&graph));
        let open = measure_graph(&json, 512).unwrap().n_main_paths.unwrap();
        assert!(open >= 2);
        assert_eq!(measure_graph(&json, open).unwrap().n_main_paths, Some(open));
        assert_eq!(measure_graph(&json, open - 1).unwrap().n_main_paths, None);
        assert_eq!(direct(&graph, open - 1).n_main_paths, None);
    }

    #[test]
    fn solution_text_and_errors() {
        let graph = solved("9/9/9/9/9/5+B+R2/5G3/2G6/4k4 b - 1", 3);
        let solution = json!({ "graph": pruned_graph_to_json(&prune_solver_graph(&graph)) });
        let as_object = json!({ "canonicalHash": "h", "sfen": "x", "solution": solution });
        let as_text = json!({ "canonicalHash": "h", "solution": solution.to_string() });
        let a: Value =
            serde_json::from_str(&main_line_paths_line(&as_object.to_string(), 512)).unwrap();
        let b: Value =
            serde_json::from_str(&main_line_paths_line(&as_text.to_string(), 512)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a["error"], Value::Null);
        assert_eq!(a["hasFinalAlt"], json!(1));
        for (line, why) in [
            (r#"{"canonicalHash":"h"}"#, "no solution"),
            (r#"{"canonicalHash":"h","solution":null}"#, "no solution"),
            (
                r#"{"canonicalHash":"h","solution":{"matePly":1}}"#,
                "no graph",
            ),
            (
                r#"{"canonicalHash":"h","solution":"{"}"#,
                "solution is not JSON",
            ),
            (
                r#"{"canonicalHash":"h","solution":{"graph":{"root":"n0","nodes":{}}}}"#,
                "root node n0 missing",
            ),
            (
                r#"{"canonicalHash":"h","solution":{"graph":{"root":3}}}"#,
                "malformed graph",
            ),
            ("not json", "input is not JSON"),
        ] {
            let out: Value = serde_json::from_str(&main_line_paths_line(line, 512)).unwrap();
            let error = out["error"].as_str().unwrap();
            assert!(error.starts_with(why), "{line}: {error}");
            assert_eq!(out["nMainPaths"], Value::Null);
            assert_eq!(out["hasFinalAlt"], json!(0));
        }
    }

    /// The stored graph of the 3-ply fixture, as JSON to corrupt.
    fn fixture_json() -> Value {
        let graph = solved("9/9/9/9/9/5+B+R2/5G3/2G6/4k4 b - 1", 3);
        serde_json::to_value(pruned_graph_to_json(&prune_solver_graph(&graph))).unwrap()
    }

    fn error_of(graph: Value) -> String {
        let line = json!({ "canonicalHash": "h", "solution": { "graph": graph } }).to_string();
        let out: Value = serde_json::from_str(&main_line_paths_line(&line, 512)).unwrap();
        assert_eq!(out["nMainPaths"], Value::Null);
        assert_eq!(out["hasFinalAlt"], json!(0));
        out["error"].as_str().expect("an error").to_string()
    }

    /// The root's first correct move, as a JSON pointer.
    fn first_correct(graph: &Value) -> String {
        let root = graph["root"].as_str().unwrap();
        let index = graph["nodes"][root]["moves"]
            .as_array()
            .unwrap()
            .iter()
            .position(|m| m["outcome"] == "MATE" && m.get("next").is_some())
            .unwrap();
        format!("/nodes/{root}/moves/{index}")
    }

    #[test]
    fn out_of_range_distances_are_errors() {
        let mut graph = fixture_json();
        let at = first_correct(&graph);
        *graph.pointer_mut(&format!("{at}/distance")).unwrap() = json!(9000);
        let error = error_of(graph);
        assert!(error.contains("distance 9000 out of range"), "{error}");

        let mut graph = fixture_json();
        let root = graph["root"].as_str().unwrap().to_string();
        graph["nodes"][&root]["distance"] = json!(8191);
        let error = error_of(graph);
        assert!(error.contains("distance 8191 out of range"), "{error}");
    }

    #[test]
    fn a_cycle_is_an_error() {
        // The root's correct move leads back to the root.
        let mut graph = fixture_json();
        let at = first_correct(&graph);
        let root = graph["root"].clone();
        *graph.pointer_mut(&format!("{at}/next")).unwrap() = root;
        assert_eq!(error_of(graph), "cycle in the solution graph");
    }

    #[test]
    fn a_graph_deeper_than_the_limit_is_an_error() {
        // A chain of attack nodes, each with one correct move to the next.
        let sfen = "9/9/9/9/9/9/9/9/4k4 b G 1";
        let depth = MAX_REGION_DEPTH + 2;
        let mut nodes = serde_json::Map::new();
        for i in 0..depth {
            let mut moves = Vec::new();
            if i + 1 < depth {
                moves.push(json!({ "usi": "G*5h", "outcome": "MATE", "distance": 1, "next": format!("n{}", i + 1) }));
            }
            nodes.insert(
                format!("n{i}"),
                json!({ "id": format!("n{i}"), "side": "attack", "sfen": sfen, "outcome": "MATE", "distance": 1, "moves": moves }),
            );
        }
        let graph = json!({ "root": "n0", "nodes": nodes });
        let error = error_of(graph);
        assert!(error.contains("deeper than"), "{error}");
    }
}
