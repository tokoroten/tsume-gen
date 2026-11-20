//! `extend`.

use super::*;

/// Streams source puzzles through [`extend_record`] on the ordered worker pool.
///
/// The counters are accumulated behind a mutex; every one of them is a sum or a
/// set, so the aggregate does not depend on which worker saw which record and a run
/// is reproducible at any `--threads`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_extend(
    options: &ExtendOptions,
    parallel: Parallel,
    stats_out: Option<&str>,
    origin: Option<&BareSfenOrigin>,
    order: Option<&ExtendOrder>,
    records: Option<(RecordFiles, RecordContext)>,
    budget_outs: &BudgetOutsArgs,
) -> Result<()> {
    let threads = resolve_threads(parallel);
    let schedule = Schedule::new(threads, 0);
    let meter = BudgetOutMeter::new(budget_outs);
    let mut sides = SideFiles {
        budget_outs: BudgetOutSink::create(budget_outs)?,
        records: match &records {
            Some((files, _)) => Some(RecordSink::create(files)?),
            None => None,
        },
        ..SideFiles::default()
    };
    let record_ctx = records.map(|(_, ctx)| ctx);
    let totals = std::sync::Mutex::new(ExtendStats::default());
    let mut last_report = Instant::now();
    let mut written = 0usize;
    let started = Instant::now();
    let cpu_before = process_cpu_seconds();
    let stats = run_jsonl(
        &schedule,
        || (Solver::new(), Prober::new()),
        |state, job: &Job| {
            let (solver, prober) = state;
            let record = extend_input_record(job, origin)?;
            meter.begin();
            let produced = extend_record_probed(solver, prober, &record, options, order)?;
            let budget_line = meter.finish(&record)?;
            let mut lines = Vec::with_capacity(produced.records.len() + produced.candidates.len());
            for candidate in produced.records.iter() {
                lines.push(write_json_line(candidate)?);
            }
            // `--emit candidates` writes lean rows, not PipelineRecords: they are
            // not puzzles and nothing downstream should mistake them for one.
            for row in produced.candidates.iter() {
                lines.push(serde_json::to_string(row)?);
            }
            if let Some(ctx) = record_ctx.as_ref() {
                // every judged candidate, written or not (`--emit accepted`)
                for candidate in
                    catalog_record::in_judged_order(&produced.records, &produced.unwritten)
                {
                    lines.extend(record_lines(catalog_record::generated(candidate, ctx))?);
                }
            }
            lines.extend(budget_line);
            totals
                .lock()
                .map_err(|_| anyhow::anyhow!("extend counters poisoned"))?
                .merge(&produced.stats);
            Ok(lines)
        },
        &mut sides,
        || {
            written += 1;
            if last_report.elapsed().as_secs_f64() >= 2.0 {
                eprintln!("tsume-solver extend: {written} records");
                last_report = Instant::now();
            }
        },
    )?;
    sides.finish_records()?;
    sides.finish_budget_outs()?;
    meter.report();
    let wall = started.elapsed().as_secs_f64();
    let cpu = cpu_seconds_since(cpu_before);
    let totals = totals
        .into_inner()
        .map_err(|_| anyhow::anyhow!("extend counters poisoned"))?;
    eprintln!(
        "tsume-solver extend: {} sources, {} candidates, {} verified, {} skipped ({} cap + {} unlisted), {} accepted, {} nodes, {} records, {:.2}s on {} threads",
        stats.inputs,
        totals.candidates,
        totals.solved,
        totals.skipped_by_cap + totals.skipped_unlisted,
        totals.skipped_by_cap,
        totals.skipped_unlisted,
        totals.accepted,
        totals.nodes,
        stats.outputs,
        wall,
        threads
    );
    if let Some(path) = stats_out {
        let allocated = wall * threads as f64;
        let mut payload = totals.to_json();
        if let serde_json::Value::Object(map) = &mut payload {
            map.insert("command".to_string(), json!("extend"));
            map.insert("threads".to_string(), json!(threads));
            map.insert("inFlight".to_string(), json!(schedule.in_flight));
            map.insert("wallSeconds".to_string(), json!(wall));
            map.insert("cpuSeconds".to_string(), json!(cpu));
            map.insert("allocatedCoreSeconds".to_string(), json!(allocated));
            map.insert(
                "utilization".to_string(),
                json!(cpu.map(|cpu| cpu / allocated.max(1e-9))),
            );
            map.insert("records".to_string(), json!(stats.outputs));
            map.insert("options".to_string(), options.run_json());
            map.insert(
                "ranker".to_string(),
                match order {
                    Some(order) => json!({
                        "id": order.id,
                        "rows": order.rows,
                        "parents": order.parents(),
                    }),
                    None => Value::Null,
                },
            );
            map.insert(
                "cpuSecondsPerAccepted".to_string(),
                json!(cpu.map(|cpu| if totals.accepted == 0 {
                    0.0
                } else {
                    cpu / totals.accepted as f64
                })),
            );
        }
        write_stats(path, &payload)?;
    }
    Ok(())
}

/// An `extend` input line: see [`bare_sfen_input_record`].
pub(crate) fn extend_input_record(
    job: &Job,
    origin: Option<&BareSfenOrigin>,
) -> Result<PipelineRecord> {
    bare_sfen_input_record(
        job,
        origin,
        "reverse-extended",
        tsume_solver::extend::imported_record_from_sfen,
    )
}
