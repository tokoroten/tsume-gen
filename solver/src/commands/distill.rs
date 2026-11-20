//! `distill`.

use super::*;

/// `--origin` / `--origin-class`: the provenance a **bare SFEN** input line has
/// to be given, because the line itself carries none (`docs/legal-data.md` §3).
pub(crate) struct BareSfenOrigin {
    pub(crate) id: String,
    pub(crate) class: String,
}

/// Streams sources through `distill` on the ordered worker pool, each source
/// under the full `--max-nodes` budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_distill(
    options: &DistillOptions,
    parallel: Parallel,
    stats_out: Option<&str>,
    origin: Option<&BareSfenOrigin>,
    negatives: Option<(String, tsume_solver::negatives::FinalizeOptions)>,
    records: Option<(RecordFiles, RecordContext)>,
    budget_outs: &BudgetOutsArgs,
) -> Result<()> {
    let threads = resolve_threads(parallel);
    let schedule = Schedule::new(threads, 0);
    let meter = BudgetOutMeter::new(budget_outs);
    let mut sides = SideFiles {
        budget_outs: BudgetOutSink::create(budget_outs)?,
        negatives: match &negatives {
            Some((path, _)) => Some(NegativesSink::create(path)?),
            None => None,
        },
        records: match &records {
            Some((files, _)) => Some(RecordSink::create(files)?),
            None => None,
        },
    };
    let record_ctx = records.map(|(_, ctx)| ctx);
    let finalize = negatives.map(|(_, options)| options);
    // Wall time of the negatives' re-judgement, summed over the workers: the
    // run's own wall and CPU below include it, so it is reported beside them
    // (and `TSUME_PROFILE` charges it to the `negatives` phase).
    let negatives_ns = std::sync::atomic::AtomicU64::new(0);

    let mut last_report = Instant::now();
    let mut written = 0usize;
    let started = Instant::now();
    let cpu_before = process_cpu_seconds();
    let stats = run_jsonl(
        &schedule,
        Solver::new,
        |solver, job: &Job| {
            let record = distill_input_record(job, origin)?;
            if finalize.is_some() {
                tsume_solver::negatives::begin();
            }
            meter.begin();
            let (produced, folded) = distill_record_judged(solver, &record, options)?;
            let mut lines = Vec::with_capacity(produced.len());
            for variant in produced.iter() {
                lines.push(write_json_line(variant)?);
            }
            lines.extend(meter.finish(&record)?);
            if let Some(ctx) = record_ctx.as_ref() {
                // every judged variant, written or folded by rule D
                for variant in catalog_record::in_judged_order(&produced, &folded) {
                    lines.extend(record_lines(catalog_record::generated(variant, ctx))?);
                }
            }
            if let Some(finalize) = finalize.as_ref() {
                let accepted = produced.iter().filter(|r| r.is_accepted()).count();
                let rejected: Vec<String> = produced
                    .iter()
                    .filter_map(|r| r.rejection.as_ref())
                    .flat_map(|rejection| rejection.reasons.iter().cloned())
                    .collect();
                let outcome = json!({ "accepted": accepted, "rejected": rejected });
                let started = Instant::now();
                lines.extend(negative_lines(solver, &record, outcome, finalize)?);
                negatives_ns.fetch_add(
                    started.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            Ok(lines)
        },
        &mut sides,
        || {
            written += 1;
            if last_report.elapsed().as_secs_f64() >= 2.0 {
                eprintln!("tsume-solver distill: {written} records");
                last_report = Instant::now();
            }
        },
    )?;
    // The negatives' lines went through the same ordered stream; the counters
    // below are the main output's.
    let negatives_on = sides.negatives.is_some();
    sides.finish_negatives()?;
    sides.finish_budget_outs()?;
    meter.report();
    sides.finish_records()?;
    let wall = started.elapsed().as_secs_f64();
    let cpu = cpu_seconds_since(cpu_before);
    eprintln!(
        "tsume-solver distill: {} sources, {} records, {:.2}s on {} threads",
        stats.inputs, stats.outputs, wall, threads
    );
    if let Some(path) = stats_out {
        let allocated = wall * threads as f64;
        let mut payload = json!({
                "command": "distill",
                "seed": options.seed.to_string(),
                "threads": threads,
                "inFlight": schedule.in_flight,
                // The two-pass schedule was removed (2026-10-08); the keys keep
                // the single-pass values so the file keeps its shape.
                "firstPassNodes": Value::Null,
                "wallSeconds": wall,
                // Measured user + kernel CPU of this process, every thread
                // included; `allocatedCoreSeconds` is the `wall x threads` figure
                // this field used to carry.  Their ratio is the utilization.
                "cpuSeconds": cpu,
                "allocatedCoreSeconds": allocated,
                "utilization": cpu.map(|cpu| cpu / allocated.max(1e-9)),
                "sources": stats.inputs,
                "records": stats.outputs,
                "secondPassSources": 0,
                "options": options.to_json(),
        });
        // `--emit-negatives` only (the key is absent otherwise): the part of
        // the wall time above, summed over the workers, spent re-judging the
        // negatives rather than distilling.
        if negatives_on {
            payload["negativesWorkerSeconds"] =
                json!(negatives_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        }
        write_stats(path, &with_profile(payload))?;
    }
    Ok(())
}

/// A `distill` input line: see [`bare_sfen_input_record`].
pub(crate) fn distill_input_record(
    job: &Job,
    origin: Option<&BareSfenOrigin>,
) -> Result<PipelineRecord> {
    bare_sfen_input_record(job, origin, "distilled", imported_record_from_sfen)
}

/// A `distill` or `extend` input line (one reader since 2026-10-27): a JSONL
/// `PipelineRecord`, or a bare SFEN turned into an imported-stage record keyed
/// by its position in the stream (`build`, the command's own source-id prefix).
///
/// A JSONL record brings its own provenance, so `--origin` / `--origin-class` are
/// ignored for it.  A bare SFEN brings none, so the flags are **required**: the
/// run fails rather than inventing an origin, since a puzzle `verb` from an
/// unattributed position cannot be published (`docs/legal-data.md` §3; under
/// `extend` every candidate would be rejected `missing_origin` anyway).
pub(crate) fn bare_sfen_input_record(
    job: &Job,
    origin: Option<&BareSfenOrigin>,
    verb: &str,
    build: fn(&tsume_solver::Position, usize, &str, &str) -> PipelineRecord,
) -> Result<PipelineRecord> {
    let line = job.line();
    if line.trim_start().starts_with('{') {
        return read_json_line(line);
    }
    let Some(origin) = origin else {
        bail!(
            "bare SFEN input needs --origin <dataset-id> and --origin-class <{}>: \
             a puzzle {verb} from an unattributed position cannot be published \
             (solver/README.md, Provenance)",
            tsume_solver::jsonl::ORIGIN_CLASSES.join("|")
        );
    };
    let position = parse_sfen(line.trim()).with_context(|| format!("SFEN: {line}"))?;
    Ok(build(&position, job.ordinal, &origin.id, &origin.class))
}
