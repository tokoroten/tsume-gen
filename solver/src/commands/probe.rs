//! The measurement commands: `probe`, `rule-b-probe` and `main-line-paths`.

use super::*;

/// Streams JSONL records through `stage` on a bounded worker pool, writing one
/// output line per input line in input order.
///
/// See [`tsume_solver::schedule`]: there is no barrier between records, so a
/// single expensive record no longer idles the other workers, and the reorder
/// buffer keeps stdout byte-identical to a single-threaded run.
/// `probe --engine dfpn` / `--engine exact-probe` / `--engine cdfpn`: a
/// **measurement** mode.
///
/// It publishes nothing.  Every input line -- a `PipelineRecord` or a bare SFEN --
/// is probed for "is there a forced mate within `--max-ply` plies, and how long is
/// the shortest one", and one small JSON object comes back:
///
/// ```text
/// {"ordinal":0,"sourceId":"…","sfen":"…","engine":"dfpn","verdict":"PROVEN",
///  "matePly":9,"pv":["…"],"nodes":123456,"budgetExhausted":false,"cpuSeconds":1.23}
/// ```
///
/// `exact-probe` answers the same question with `exact::minimum_mate_validate`
/// (rules off, so rule F is `keep` on both sides), which is what makes the two
/// columns of the comparison in `reports/dfpn-verifier-2026-09-14.md` measure the
/// same thing with the same harness.  `cpuSeconds` is the worker thread's own
/// elapsed time for that position; the search never blocks, so it is CPU time, and
/// the run's total process CPU is reported on stderr as the authoritative figure.
///
/// `cdfpn` answers it under rule F classical with the classical df-pn prober
/// ([`entry_probe`]: the root normalised as the publishing gate does, one budget
/// per line shared by "mate within `--max-ply`?" and the climb 1, 3, ... that
/// finds the length, the prober reset per line).  It is the RL entry since
/// 2026-10-10 (`docs/experiments/ply13/README.md` §12).  Its `pv` is empty, and
/// a budget cut while climbing is `UNKNOWN` with no length.
///
/// `cdfpn` rows also carry `disproofKind` (owner request 2026-11-16; null
/// unless `DISPROVEN`): `"checksExhausted"` when the refutation holds at every
/// bound (no deeper probe can find a mate), `"horizon"` otherwise (some
/// checking line reached `--max-ply`; a deeper mate may exist).  It is a walk
/// over the prober's tables after the search ([`entry_disproof_kind`]) and
/// conservative: `checksExhausted` is a proof, `horizon` may also mean the walk
/// found none.  The other fields are byte-identical to before.
pub(crate) fn run_probe(
    engine: &str,
    max_ply: u32,
    node_budget: u64,
    parallel: Parallel,
    position: PositionParallel,
) -> Result<()> {
    let dfpn = engine == "dfpn";
    let cdfpn = engine == "cdfpn";
    // `--engine cdfpn` builds its probers without `PositionParallel::solver`.
    if cdfpn {
        position.configure_prober();
    } else if position.prober_threads > 1 {
        bail!("--prober-threads > 1 applies to --engine cdfpn and to solve (dfpn and exact-probe have no classical prober)");
    }
    // `cdfpn` searches each position on `--prober-threads` threads (`T / P`
    // positions at a time, as `run_stage_multi_with`); `dfpn` and
    // `exact-probe` on one thread (`--prober-threads > 1` refused above).
    let workers = (resolve_threads(parallel) / position.threads_per_record()).max(1);
    let schedule = Schedule::new(workers, 0);
    let started = Instant::now();
    let cpu_before = process_cpu_seconds();
    let stats = run_jsonl(
        &schedule,
        // Each worker builds its engine before any timing, as before, and only
        // the one this mode uses.
        || {
            (
                (!dfpn && !cdfpn).then(|| position.solver()),
                dfpn.then(Prober::new),
                // `--prober-threads` applies here too (`T / P` positions at
                // a time, each on `P` threads).
                cdfpn.then(ClassicalProber::configured),
            )
        },
        |state, job| {
            let (source_id, sfen) = probe_input(job.line())?;
            let position = parse_sfen(&sfen)?;
            let (solver, prober, classical) = state;
            let at = Instant::now();
            // `--engine cdfpn` only: the kind of a `DISPROVEN` (null otherwise).
            let mut disproof_kind: Option<&str> = None;
            // `cpuSeconds` is the search's time only (the kind's walk after it
            // is not counted, so the field means what it meant before 1.6.0).
            let mut search_seconds: Option<f64> = None;
            let (verdict, mate_ply, pv, nodes, exhausted) = if cdfpn {
                let classical = classical.as_mut().expect("cdfpn worker");
                let probe = entry_probe(classical, &position, max_ply, node_budget);
                search_seconds = Some(at.elapsed().as_secs_f64());
                if probe.outcome == Outcome::Refuted {
                    // After the search, from its tables: no verdict, length
                    // or node count changes.
                    disproof_kind =
                        Some(entry_disproof_kind(classical, &position, max_ply).as_str());
                }
                let verdict = match probe.outcome {
                    Outcome::Mate => Verdict::Proven,
                    Outcome::Refuted => Verdict::Disproven,
                    Outcome::Unknown => Verdict::Unknown,
                };
                (
                    verdict.as_str(),
                    probe.mate_ply,
                    Vec::new(),
                    probe.nodes,
                    probe.budget_exhausted,
                )
            } else if dfpn {
                let prober = prober.as_mut().expect("dfpn worker");
                let probe = prober.minimal_mate_ply(
                    &position,
                    &ProbeOptions::default()
                        .max_ply(max_ply)
                        .node_budget(node_budget),
                );
                (
                    probe.verdict.as_str(),
                    probe.mate_ply,
                    probe.pv.iter().map(|usi| usi.to_string()).collect(),
                    probe.nodes,
                    probe.budget_exhausted,
                )
            } else {
                let solver = solver.as_mut().expect("exact-probe worker");
                let (_, validation) = tsume_solver::exact::minimum_mate_validate(
                    solver,
                    &position,
                    max_ply,
                    node_budget,
                    true,
                );
                let exhausted = validation.result.stats.budget_exhausted;
                let mate = validation.result.outcome == tsume_solver::solver::Outcome::Mate;
                let verdict = if exhausted && !mate {
                    Verdict::Unknown
                } else if mate {
                    Verdict::Proven
                } else {
                    Verdict::Disproven
                };
                let pv: Vec<String> = match (mate, validation.result.graph.as_ref()) {
                    (true, Some(graph)) => {
                        // The df-pn prober's line, not the designated main line:
                        // the two engines of this measurement mode must report the
                        // same `pv`, and the prober cannot judge koma-amari.
                        tsume_solver::exact::usi_first_line(graph)
                            .iter()
                            .map(|usi| usi.to_string())
                            .collect()
                    }
                    _ => Vec::new(),
                };
                (
                    verdict.as_str(),
                    validation.result.distance.filter(|_| mate),
                    pv,
                    validation.result.stats.nodes,
                    exhausted,
                )
            };
            let mut row = json!({
                "ordinal": job.ordinal,
                "sourceId": source_id,
                "sfen": sfen,
                "engine": engine,
                "maxPly": max_ply,
                "nodeBudget": node_budget,
                "verdict": verdict,
                "matePly": mate_ply,
                "pv": pv,
                "nodes": nodes,
                "budgetExhausted": exhausted,
                "cpuSeconds": search_seconds.unwrap_or_else(|| at.elapsed().as_secs_f64()),
            });
            if cdfpn {
                row["disproofKind"] = json!(disproof_kind);
            }
            let line = serde_json::to_string(&row)?;
            Ok(vec![line])
        },
        &mut SideFiles::default(),
        || {},
    )?;
    let cpu = cpu_seconds_since(cpu_before);
    eprintln!(
        "tsume-solver probe --engine {engine}: {} positions, {:.1} s wall, {} process CPU-s",
        stats.outputs,
        started.elapsed().as_secs_f64(),
        cpu.map_or("?".to_string(), |value| format!("{value:.1}")),
    );
    Ok(())
}

/// One `rule-b-probe` input line: the `id` to echo, the SFEN and the target.
pub(crate) fn rule_b_probe_input(
    line: &str,
    default_target: Option<u32>,
) -> Result<(Value, String, u32)> {
    let line = line.trim();
    if !line.starts_with('{') {
        let target = default_target.context("a bare SFEN line needs --mate-ply")?;
        return Ok((Value::Null, line.to_string(), target));
    }
    let value: Value = serde_json::from_str(line).context("rule-b-probe input is not JSON")?;
    let id = value
        .get("id")
        .or_else(|| value.get("sourceId"))
        .cloned()
        .unwrap_or(Value::Null);
    let sfen = ["sfen", "normalizedSfen", "sourceSfen"]
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .context("rule-b-probe input needs an sfen field")?
        .to_string();
    let target = ["matePly", "targetMatePly"]
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_u64))
        .map(|ply| ply as u32)
        .or(default_target)
        .context("rule-b-probe input needs matePly (or --mate-ply)")?;
    if target > tsume_solver::dfpn::MAX_PROBE_PLY {
        bail!("matePly {target} is out of range");
    }
    Ok((id, sfen, target))
}

/// Order-independent counters of a `rule-b-probe` run.
#[derive(Default)]
pub(crate) struct RuleBProbeTotals {
    pub(crate) positions: u64,
    pub(crate) nodes: u64,
    /// `reject:<reason>` / `inconclusive:<cause>` -> count, sorted for the JSON.
    pub(crate) outcomes: std::collections::BTreeMap<String, u64>,
}

/// `main-line-paths`; see [`Command::MainLinePaths`].
pub(crate) fn run_main_line_paths(cap: usize, parallel: Parallel) -> Result<()> {
    let threads = resolve_threads(parallel);
    // A light count over rows that may each hold a large graph: a window of
    // two rows per worker, not the solvers' 32 (a slow row would otherwise
    // keep up to 32 x threads whole solutions in memory).
    let schedule = Schedule::new(threads, 2 * threads);
    let started = Instant::now();
    let stats = run_jsonl(
        &schedule,
        || (),
        |_, job: &Job| {
            Ok(vec![tsume_solver::stored_paths::main_line_paths_line(
                job.line(),
                cap,
            )])
        },
        &mut SideFiles::default(),
        || {},
    )?;
    eprintln!(
        "tsume-solver main-line-paths: {} graphs, {:.2}s on {} threads",
        stats.outputs,
        started.elapsed().as_secs_f64(),
        threads
    );
    Ok(())
}

/// `stored-main-line`; see [`Command::StoredMainLine`].
pub(crate) fn run_stored_main_line(parallel: Parallel) -> Result<()> {
    let threads = resolve_threads(parallel);
    // As `main-line-paths`: a window of two rows per worker (large graphs).
    let schedule = Schedule::new(threads, 2 * threads);
    let started = Instant::now();
    let stats = run_jsonl(
        &schedule,
        || (),
        |_, job: &Job| {
            Ok(vec![tsume_solver::stored_paths::stored_main_line_line(
                job.line(),
            )])
        },
        &mut SideFiles::default(),
        || {},
    )?;
    eprintln!(
        "tsume-solver stored-main-line: {} graphs, {:.2}s on {} threads",
        stats.outputs,
        started.elapsed().as_secs_f64(),
        threads
    );
    Ok(())
}

/// `rule-b-probe`; see [`Command::RuleBProbe`].  Every count is a sum, so the
/// stats do not depend on which worker saw which line.
pub(crate) fn run_rule_b_probe(
    node_budget: u64,
    default_target: Option<u32>,
    parallel: Parallel,
    stats_out: Option<&str>,
) -> Result<()> {
    use tsume_solver::rule_b_probe::{rule_b_reject, RuleBOutcome};
    let rules = TsumeRules::puzzle();
    let threads = resolve_threads(parallel);
    let schedule = Schedule::new(threads, 0);
    let totals = std::sync::Mutex::new(RuleBProbeTotals::default());
    let started = Instant::now();
    let cpu_before = process_cpu_seconds();
    let stats = run_jsonl(
        &schedule,
        Prober::new,
        |prober, job: &Job| {
            let (id, sfen, target) = rule_b_probe_input(job.line(), default_target)
                .with_context(|| format!("input line {}", job.ordinal + 1))?;
            let position = parse_sfen(&sfen).with_context(|| format!("SFEN: {sfen}"))?;
            let probe = rule_b_reject(prober, &position, target, rules, node_budget);
            let (verdict, reason, cause) = match probe.outcome {
                RuleBOutcome::Reject(reason) => ("reject", Some(reason.as_str()), None),
                RuleBOutcome::Inconclusive(cause) => ("inconclusive", None, Some(cause.as_str())),
            };
            let line = serde_json::to_string(&json!({
                "id": id,
                "verdict": verdict,
                "reason": reason,
                "cause": cause,
                "matePly": target,
                "nodes": probe.nodes,
                "plyReached": probe.ply_reached,
                "rounds": probe.max_round,
            }))?;
            let mut totals = totals
                .lock()
                .map_err(|_| anyhow::anyhow!("rule-b-probe counters poisoned"))?;
            totals.positions += 1;
            totals.nodes += probe.nodes;
            *totals
                .outcomes
                .entry(format!("{verdict}:{}", reason.or(cause).unwrap_or("?")))
                .or_insert(0) += 1;
            Ok(vec![line])
        },
        &mut SideFiles::default(),
        || {},
    )?;
    let wall = started.elapsed().as_secs_f64();
    let cpu = cpu_seconds_since(cpu_before);
    let totals = totals
        .into_inner()
        .map_err(|_| anyhow::anyhow!("rule-b-probe counters poisoned"))?;
    let rejected: u64 = totals
        .outcomes
        .iter()
        .filter(|(key, _)| key.starts_with("reject:"))
        .map(|(_, count)| count)
        .sum();
    eprintln!(
        "tsume-solver rule-b-probe: {} positions, {} rejected, {} df-pn nodes, {:.2}s on {} threads",
        stats.outputs, rejected, totals.nodes, wall, threads
    );
    if let Some(path) = stats_out {
        let allocated = wall * threads as f64;
        write_stats(
            path,
            &with_profile(json!({
                "command": "rule-b-probe",
                "threads": threads,
                "inFlight": schedule.in_flight,
                "nodeBudget": node_budget,
                "positions": totals.positions,
                "rejected": rejected,
                "inconclusive": totals.positions - rejected,
                "outcomes": totals.outcomes,
                "nodes": totals.nodes,
                "wallSeconds": wall,
                "cpuSeconds": cpu,
                "allocatedCoreSeconds": allocated,
                "utilization": cpu.map(|cpu| cpu / allocated.max(1e-9)),
            })),
        )?;
    }
    Ok(())
}

/// `(sourceId, sfen)` of one probe input line: a `PipelineRecord`, or a bare SFEN.
pub(crate) fn probe_input(line: &str) -> Result<(String, String)> {
    if line.trim_start().starts_with('{') {
        let record = read_json_line(line)?;
        let sfen = record
            .normalized_sfen
            .clone()
            .unwrap_or_else(|| record.source_sfen.clone());
        return Ok((record.source_id, sfen));
    }
    Ok((String::new(), line.trim().to_string()))
}
