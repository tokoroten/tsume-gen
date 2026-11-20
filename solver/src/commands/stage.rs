//! The JSONL stages (`validate`, `solve`, `reduce`, `reduce-accepted`), `solve-sfen` and `movegen`.

use super::*;

/// Worker count for the streaming stages: `--threads 0` means every logical core,
/// exactly like the flag's default.
pub(crate) fn resolve_threads(parallel: Parallel) -> usize {
    if parallel.threads == 0 {
        default_threads()
    } else {
        parallel.threads
    }
}

pub(crate) fn read_stdin_lines() -> Result<Vec<String>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    Ok(text
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

pub(crate) fn run_movegen(sfen: Option<String>, stdin: bool) -> Result<()> {
    let out = io::stdout();
    let mut out = BufWriter::new(out.lock());
    if stdin || sfen.is_none() {
        for line in read_stdin_lines()? {
            let position = parse_sfen(&line).with_context(|| format!("SFEN: {line}"))?;
            let moves: Vec<String> = legal_moves_sorted(&position)
                .iter()
                .map(|m| m.to_usi().to_string())
                .collect();
            writeln!(out, "{}", json!({ "sfen": line, "moves": moves }))?;
        }
        return Ok(());
    }
    let sfen = sfen.unwrap();
    let position = parse_sfen(&sfen)?;
    for m in legal_moves_sorted(&position) {
        writeln!(out, "{}", m.to_usi())?;
    }
    Ok(())
}

/// The shape of `SolveResult` in `packages/solver`, for differential testing.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SolveSfenOutput {
    pub(crate) sfen: String,
    /// Echoed so a differential run can tell the two rule F modes apart.
    pub(crate) useless_interposition: String,
    pub(crate) outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) distance: Option<u32>,
    pub(crate) stats: tsume_solver::jsonl::SolveStatsJson,
    pub(crate) moves: Vec<tsume_solver::jsonl::GraphMoveJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) graph: Option<SolverGraphJson>,
    /// `non_futile_node_count` of the search graph at the proven distance, for
    /// `diff-solve.ts`; present only for a mate solved with a graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) nodes_non_futile: Option<u64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_solve_sfen(
    sfen: Option<String>,
    max_ply: u32,
    node_budget: u64,
    attacker: Option<String>,
    strict_material: bool,
    graph_mode: &str,
    useless_mode: &str,
    stdin: bool,
    position: PositionParallel,
) -> Result<()> {
    let attacker = match attacker.as_deref() {
        Some("b") => Some(BLACK),
        Some("w") => Some(WHITE),
        _ => None,
    };
    let default_useless =
        UselessInterposition::parse(useless_mode).context("--useless-interposition")?;
    // A compact table keeps the solution region only, not the full graph.
    anyhow::ensure!(
        !(position.table.is_compact() && graph_mode == "full"),
        "--graph full needs --table exact (a compact table keeps the solution region only)"
    );
    let out = io::stdout();
    let mut out = BufWriter::new(out.lock());
    let mut solver = position.solver();
    let mut emit = |sfen: &str, rules: SearchRules, limits: SearchLimits| -> Result<()> {
        let position = parse_sfen(sfen).with_context(|| format!("SFEN: {sfen}"))?;
        let options = SolveOptions {
            rules,
            limits,
            build_graph: graph_mode != "none",
        };
        let result = solver.solve(&position, &options);
        let useless = rules.useless;
        let graph = match (graph_mode, result.graph.as_ref()) {
            ("full", Some(graph)) => Some(full_graph_to_json(graph)),
            ("pruned", Some(graph)) => Some(pruned_graph_to_json(&prune_solver_graph(graph))),
            _ => None,
        };
        let payload = SolveSfenOutput {
            sfen: sfen.to_string(),
            useless_interposition: useless.as_str().to_string(),
            outcome: result.outcome.as_str().to_string(),
            distance: result.distance,
            stats: stats_to_json(&result.stats),
            // Root moves in published shape; `next` ids refer to graph nodes.
            moves: result
                .moves
                .iter()
                .map(|m| tsume_solver::jsonl::GraphMoveJson {
                    usi: m.usi().to_string(),
                    outcome: m.outcome().as_str().to_string(),
                    distance: m.distance(),
                    next: m.next().map(|id| format!("n{id}")),
                    refutation: m.refutation().map(|usi| usi.to_string()),
                    useless: m.useless(),
                })
                .collect(),
            graph,
            nodes_non_futile: match (result.outcome, result.distance, result.graph.as_ref()) {
                (Outcome::Mate, Some(distance), Some(graph)) => {
                    Some(non_futile_node_count(graph, distance))
                }
                _ => None,
            },
        };
        writeln!(out, "{}", serde_json::to_string(&payload)?)?;
        Ok(())
    };
    let rules = SearchRules {
        attacker,
        strict_material,
        useless: default_useless,
    };
    let limits = SearchLimits {
        max_ply,
        node_budget,
        credit_budget: None,
    };
    if stdin || sfen.is_none() {
        for line in read_stdin_lines()? {
            let value: serde_json::Value =
                serde_json::from_str(&line).context("Invalid solve request")?;
            let sfen = value
                .get("sfen")
                .and_then(|v| v.as_str())
                .context("solve request needs an sfen field")?;
            let line_rules = SearchRules {
                strict_material: value
                    .get("strictMaterial")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(rules.strict_material),
                useless: match value.get("uselessInterposition").and_then(|v| v.as_str()) {
                    Some(text) => {
                        UselessInterposition::parse(text).context("uselessInterposition")?
                    }
                    None => rules.useless,
                },
                ..rules
            };
            let line_limits = SearchLimits {
                max_ply: value
                    .get("maxPly")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(limits.max_ply),
                node_budget: value
                    .get("nodeBudget")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(limits.node_budget),
                credit_budget: value
                    .get("creditBudget")
                    .and_then(|v| v.as_u64())
                    .or(limits.credit_budget),
            };
            emit(sfen, line_rules, line_limits)?;
        }
        return Ok(());
    }
    emit(&sfen.unwrap(), rules, limits)
}

/// `reduce-accepted` on one record; see [`Command::ReduceAccepted`].
pub(crate) fn reduce_accepted_record(
    record: &mut PipelineRecord,
    node_budget: u64,
    attempts: u32,
    length_k: usize,
    jobs: usize,
    solver: &mut Solver,
) -> Result<Vec<PipelineRecord>> {
    let Some(solution) = record.solution.as_ref().filter(|_| record.is_accepted()) else {
        return Ok(Vec::new());
    };
    let bound = solution.stats.max_ply;
    // A generator scores its puzzles before this pass; the score stays with the
    // record (flagged below) rather than vanishing with the re-solve.
    let quality = solution.quality.clone();
    let difficulty = solution.difficulty.clone();
    let report = reduce_accepted(
        record,
        tsume_solver::reduce::ReduceOptions {
            attempts,
            max_ply: bound,
            node_budget,
            jobs,
            ..Default::default()
        },
        tsume_solver::reduce::TwoKindsOptions {
            length_k,
            ..Default::default()
        },
        &SolveStageOptions {
            max_ply: bound,
            node_budget,
            rules: TsumeRules::puzzle(),
        },
        solver,
    )?;
    let mut summary = json!({ "bound": bound, "nodeBudget": node_budget });
    let map = summary.as_object_mut().expect("a JSON object");
    // Written only off the defaults, so the stock already reduced keeps its bytes.
    // `untilFixpoint` stays `false` (the flag went in stage 4, 2026-10-22).
    if attempts != 8 {
        map.insert("attempts".to_string(), json!(attempts));
        map.insert("untilFixpoint".to_string(), json!(false));
    }
    let fallback = matches!(report.outcome, ReduceAcceptedOutcome::Fallback { .. });
    match report.outcome {
        ReduceAcceptedOutcome::Unchanged => {
            map.insert("outcome".to_string(), json!("unchanged"));
        }
        ReduceAcceptedOutcome::Reduced { removed } => {
            map.insert("outcome".to_string(), json!("reduced"));
            map.insert("removed".to_string(), json!(removed));
            map.insert("resolveNodes".to_string(), json!(report.resolve_nodes));
            map.insert("scoresPredateReduction".to_string(), json!(true));
            if let Some(solution) = record.solution.as_mut() {
                solution.quality = quality;
                solution.difficulty = difficulty;
            }
        }
        ReduceAcceptedOutcome::Fallback { reasons } => {
            map.insert("outcome".to_string(), json!("fallback"));
            map.insert("reasons".to_string(), json!(reasons));
        }
    }
    // The minimality certificate (owner 2026-10-22): an `unchanged` or
    // `reduced` record is minimal only when `verified` (a removal the reducer
    // refused for lack of budget leaves `unchanged` / fewer pieces off, and
    // says `unverified`).  Also in `reduction`; repeated here for the callers
    // that read only this summary.
    if let Some(reduction) = record.reduction.as_ref().filter(|_| !fallback) {
        if let (Some(minimality), Some(checks)) =
            (reduction.minimality.as_ref(), reduction.budget_out_checks)
        {
            map.insert("minimality".to_string(), json!(minimality));
            map.insert("budgetOutChecks".to_string(), json!(checks));
        }
    }
    record.extra.insert("reduceAccepted".to_string(), summary);
    Ok(report.length)
}

/// `--prober-threads P` > 1 runs `--threads / P` records at a time (at least
/// one), each prober searching on its own `P` threads.
pub(crate) fn run_stage<F>(parallel: Parallel, position: PositionParallel, stage: F) -> Result<()>
where
    F: Fn(&mut PipelineRecord, &mut Solver) -> Result<()> + Send + Sync,
{
    run_stage_multi(parallel, position, move |record, solver| {
        stage(record, solver).map(|()| Vec::new())
    })
}

/// [`run_stage`] for a stage that may emit records of its own after the one it
/// transforms (rule D v5's length-preserving reductions), in the order it
/// returns them; the output still follows the input order.
pub(crate) fn run_stage_multi<F>(
    parallel: Parallel,
    position: PositionParallel,
    stage: F,
) -> Result<()>
where
    F: Fn(&mut PipelineRecord, &mut Solver) -> Result<Vec<PipelineRecord>> + Send + Sync,
{
    run_stage_multi_with(parallel, position, None, None, None, stage)
}

/// [`run_stage_multi`] with the optional `--emit-negatives` side file: each
/// record's reducer events are collected on its worker thread
/// ([`tsume_solver::negatives::begin`]) and written, picked and re-judged, to
/// the side file in input order.  The main output is the same either way.
/// Likewise the optional `--emit-record` side file (`records`: its path and
/// the records of one job).
pub(crate) fn run_stage_multi_with<F>(
    parallel: Parallel,
    position: PositionParallel,
    negatives: Option<(String, tsume_solver::negatives::FinalizeOptions)>,
    records: Option<(RecordFiles, RecordEmit)>,
    budget_outs: Option<&BudgetOutsArgs>,
    stage: F,
) -> Result<()>
where
    F: Fn(&mut PipelineRecord, &mut Solver) -> Result<Vec<PipelineRecord>> + Send + Sync,
{
    // Created in this order (the first that fails names its file).
    let mut sides = SideFiles {
        negatives: match &negatives {
            Some((path, _)) => Some(NegativesSink::create(path)?),
            None => None,
        },
        records: match &records {
            Some((files, _)) => Some(RecordSink::create(files)?),
            None => None,
        },
        budget_outs: match budget_outs {
            Some(args) => BudgetOutSink::create(args)?,
            None => None,
        },
    };
    let finalize = negatives.map(|(_, options)| options);
    let emit = records.map(|(_, emit)| emit);
    // The reducer's commands only: the budget-out measurement.
    let meter = budget_outs.map(BudgetOutMeter::new);
    let workers = (resolve_threads(parallel) / position.threads_per_record()).max(1);
    let schedule = Schedule::new(workers, 0);
    let stats = run_jsonl(
        &schedule,
        || position.solver(),
        |solver, job| {
            let mut record = read_json_line(job.line())?;
            let input = finalize.as_ref().map(|_| {
                tsume_solver::negatives::begin();
                record.clone()
            });
            let as_read = emit.as_ref().map(|_| record.clone());
            let measured = meter.as_ref().map(|meter| {
                meter.begin();
                record.clone()
            });
            let extra = stage(&mut record, solver)?;
            let mut lines = Vec::with_capacity(1 + extra.len());
            lines.push(write_json_line(&record)?);
            for record in extra.iter() {
                lines.push(write_json_line(record)?);
            }
            if let (Some(meter), Some(input)) = (meter.as_ref(), measured.as_ref()) {
                lines.extend(meter.finish(input)?);
            }
            if let (Some(emit), Some(as_read)) = (emit.as_ref(), as_read.as_ref()) {
                lines.extend(record_lines(emit(as_read, &record, &extra, solver)?)?);
            }
            if let (Some(finalize), Some(input)) = (finalize.as_ref(), input) {
                let outcome = match &record.rejection {
                    Some(rejection) => json!({ "rejected": rejection.reasons }),
                    None => json!({ "accepted": 1 + extra.len() }),
                };
                lines.extend(negative_lines(solver, &input, outcome, finalize)?);
            }
            Ok(lines)
        },
        &mut sides,
        || {},
    )?;
    sides.finish_budget_outs()?;
    if let Some(meter) = meter.as_ref() {
        meter.report();
    }
    sides.finish_negatives()?;
    sides.finish_records()?;
    eprintln!("tsume-solver: {} records", stats.outputs);
    Ok(())
}
