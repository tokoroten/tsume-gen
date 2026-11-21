//! The side files a command writes besides stdout: catalog records (`--emit-record`), budget-outs (`--emit-budget-outs`), negatives (`--emit-negatives`), and the stats file.

use super::*;

/// Marks a worker's output line as a budget-out line (`--emit-budget-outs`).
pub(crate) const BUDGET_OUT_MARK: char = '\u{3}';

/// The budget-out measurement of a reducer command: every input's checks are
/// logged on its worker thread and counted for the stderr summary; with
/// `--emit-budget-outs` also written, one marked line per input.
pub(crate) struct BudgetOutMeter {
    pub(crate) file: bool,
    /// Inputs, inputs with a check out of budget, checks out of budget.
    pub(crate) totals: [std::sync::atomic::AtomicU64; 3],
}

impl BudgetOutMeter {
    pub(crate) fn new(args: &BudgetOutsArgs) -> Self {
        Self {
            file: args.emit_budget_outs.is_some(),
            totals: Default::default(),
        }
    }

    /// Opens the log for one input, on its worker thread.
    pub(crate) fn begin(&self) {
        tsume_solver::reduce::budget_outs_begin();
    }

    /// Closes the log of `input` and returns its marked line, if written.
    pub(crate) fn finish(&self, input: &PipelineRecord) -> Result<Option<String>> {
        use std::sync::atomic::Ordering::Relaxed;
        let entries = tsume_solver::reduce::budget_outs_take();
        let still_out = entries.len() as u64;
        self.totals[0].fetch_add(1, Relaxed);
        self.totals[1].fetch_add(u64::from(still_out > 0), Relaxed);
        self.totals[2].fetch_add(still_out, Relaxed);
        if !self.file {
            return Ok(None);
        }
        let line = budget_out_line(input, &entries, still_out);
        Ok(Some(format!(
            "{BUDGET_OUT_MARK}{}",
            serde_json::to_string(&line)?
        )))
    }

    /// The run's stderr summary.
    pub(crate) fn report(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let load = |index: usize| self.totals[index].load(Relaxed);
        eprintln!(
            "tsume-solver: budget-outs: {} inputs, {} with a check still out of budget, {} checks still out",
            load(0),
            load(1),
            load(2)
        );
    }
}

/// The `--emit-budget-outs` line of `input`.
pub(crate) fn budget_out_line(
    input: &PipelineRecord,
    entries: &[tsume_solver::reduce::BudgetOut],
    still_out: u64,
) -> Value {
    let hash = input.canonical_hash.clone().or_else(|| {
        let sfen = input
            .normalized_sfen
            .as_deref()
            .unwrap_or(&input.source_sfen);
        parse_sfen(sfen)
            .ok()
            .map(|position| tsume_solver::canonical::canonicalize(&position).hash)
    });
    let piece = |piece: &tsume_solver::jsonl::RemovedPiece| json!({ "square": piece.square, "piece": piece.piece });
    let entries: Vec<Value> = entries
        .iter()
        .map(|entry| {
            json!({
                "stage": entry.stage,
                "kind": entry.kind,
                "step": entry.step.as_ref().map(piece),
                "removed": entry.removed.iter().map(piece).collect::<Vec<_>>(),
                "sfen": entry.sfen,
                "targetMatePly": entry.target,
                "paths": entry.paths,
                "budget": entry.budget,
                "threadMode": entry.mode.label(),
                "nodesUsed": entry.first.nodes,
                "reasons": entry.first.reasons,
                // Fixed since the one-thread re-run went (2026-10-27); kept
                // so the line keeps its shape for the readers.
                "rerun": Value::Null,
                "finalOutcome": "still-out",
            })
        })
        .collect();
    json!({
        "type": "budget-outs",
        "sourceId": input.source_id,
        "canonicalHash": hash,
        "budgetOutChecks": still_out,
        "checks": entries.len(),
        "entries": entries,
    })
}

/// The `--emit-budget-outs` file: the marked lines, in input order.
pub(crate) struct BudgetOutSink {
    pub(crate) out: BufWriter<std::fs::File>,
    pub(crate) lines: usize,
}

impl BudgetOutSink {
    pub(crate) fn create(args: &BudgetOutsArgs) -> Result<Option<Self>> {
        args.emit_budget_outs
            .as_deref()
            .map(|path| {
                Ok(Self {
                    out: create_side_file(path)?,
                    lines: 0,
                })
            })
            .transpose()
    }

    /// Writes a marked line and returns `true`; `false` for any other line.
    pub(crate) fn route(&mut self, line: &str) -> Result<bool> {
        let Some(rest) = line.strip_prefix(BUDGET_OUT_MARK) else {
            return Ok(false);
        };
        self.lines += 1;
        writeln!(self.out, "{rest}")?;
        Ok(true)
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

/// Marks a worker's output line as a negatives' line (never the first byte of
/// a JSON line), followed by the row's canonical hash (`-` for a summary row)
/// and a tab.
pub(crate) const NEGATIVE_MARK: char = '\u{1}';

/// Where `--emit-record` writes: the records, and with `--emit-unresolved`
/// the records of status `unknown`.
#[derive(Clone, Debug)]
pub(crate) struct RecordFiles {
    pub(crate) records: String,
    pub(crate) unresolved: Option<String>,
}

/// What an output path names, for telling two outputs apart: the absolute
/// path, `.` and `..` folded lexically, with its nearest existing ancestor
/// canonicalized (symlinks and junctions resolved) and the rest appended,
/// case-folded on Windows.  A hard link to another output is not seen.
/// Refuses a command whose side files (`--emit-negatives`, `--emit-record`,
/// `--emit-budget-outs`, `--stats-out`) name the same file twice: each is
/// opened on its own and would overwrite the other.
pub(crate) fn distinct_outputs(command: &str, outputs: &[(&str, Option<&str>)]) -> Result<()> {
    let given: Vec<(&str, &str, String)> = outputs
        .iter()
        .filter_map(|(flag, path)| path.map(|path| (*flag, path, output_key(path))))
        .collect();
    for (index, (flag, path, key)) in given.iter().enumerate() {
        if let Some((other, _, _)) = given[..index].iter().find(|(_, _, seen)| seen == key) {
            bail!("{command}: {other} and {flag} name the same file {path}");
        }
    }
    // Hard links ([`output_key`] does not see them): two existing paths may
    // be one file.  Compared by file identity, without touching the files.
    let mut identities: Vec<(&str, &str, (u64, u64))> = Vec::new();
    for (flag, path, _) in &given {
        let Some(identity) = file_identity(path) else {
            continue;
        };
        if let Some((other, earlier, _)) = identities.iter().find(|(_, _, seen)| *seen == identity)
        {
            bail!("{command}: {other} and {flag} name the same file ({earlier} and {path} are hard links)");
        }
        identities.push((flag, path, identity));
    }
    Ok(())
}

/// The identity of an existing file (`None` when it does not exist or the
/// platform gives none): the volume serial number and file index on Windows
/// (`GetFileInformationByHandle`), the device and inode on Unix.  Two paths
/// with the same identity are one file, hard links included.
pub(crate) fn file_identity(path: &str) -> Option<(u64, u64)> {
    let file = std::fs::File::open(path).ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    platform_identity(&file)
}

#[cfg(windows)]
pub(crate) fn platform_identity(file: &std::fs::File) -> Option<(u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    /// `BY_HANDLE_FILE_INFORMATION` (every field a 4-byte word; the three
    /// `FILETIME`s as two halves each).
    #[repr(C)]
    #[derive(Default)]
    struct Info {
        attributes: u32,
        times: [u32; 6],
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        links: u32,
        index_high: u32,
        index_low: u32,
    }
    extern "system" {
        fn GetFileInformationByHandle(file: *mut std::ffi::c_void, info: *mut Info) -> i32;
    }
    let mut info = Info::default();
    // SAFETY: the handle is the open `file`'s, live for the call, and `info`
    // is a live, correctly laid out `BY_HANDLE_FILE_INFORMATION`.
    let ok = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) };
    (ok != 0).then(|| {
        (
            u64::from(info.volume_serial),
            (u64::from(info.index_high) << 32) | u64::from(info.index_low),
        )
    })
}

#[cfg(unix)]
pub(crate) fn platform_identity(file: &std::fs::File) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata().ok()?;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(any(windows, unix)))]
pub(crate) fn platform_identity(_file: &std::fs::File) -> Option<(u64, u64)> {
    None
}

pub(crate) fn output_key(path: &str) -> String {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let mut folded = std::path::PathBuf::new();
    for part in absolute.components() {
        match part {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other.as_os_str()),
        }
    }
    let absolute = folded;
    let mut rest = Vec::new();
    let mut base = absolute.as_path();
    let resolved = loop {
        if let Ok(real) = std::fs::canonicalize(base) {
            break rest.iter().rev().fold(real, |path, name| path.join(name));
        }
        match (base.parent(), base.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                base = parent;
            }
            _ => break absolute.clone(),
        }
    };
    let text = resolved.to_string_lossy().into_owned();
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

/// Marks a worker's output line as a side record (`--emit-record`), with a
/// following `?` a record of status `unknown` (`--emit-unresolved`), or with
/// a following `!` the reason a judged diagram got none.
pub(crate) const RECORD_MARK: char = '\u{2}';

/// The marked lines of [`catalog_record`] results.
pub(crate) fn record_lines(
    results: Vec<Result<Value, catalog_record::Skip>>,
) -> Result<Vec<String>> {
    results
        .into_iter()
        .map(|result| {
            Ok(match result {
                Ok(record) => {
                    let unresolved = record["judgement"]["status"] == "unknown";
                    let mark = if unresolved { "?" } else { "" };
                    format!("{RECORD_MARK}{mark}{}", serde_json::to_string(&record)?)
                }
                Err(skip) => format!("{RECORD_MARK}!{skip}"),
            })
        })
        .collect()
}

/// Creates a side file, and its directory.
pub(crate) fn create_side_file(path: &str) -> Result<BufWriter<std::fs::File>> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
    Ok(BufWriter::new(file))
}

/// The `--emit-record` file: every record in input order (those of status
/// `unknown` in their own file with `--emit-unresolved`), and the counts of
/// the diagrams that got none.
pub(crate) struct RecordSink {
    pub(crate) out: BufWriter<std::fs::File>,
    pub(crate) unresolved: Option<BufWriter<std::fs::File>>,
    pub(crate) lines: usize,
    pub(crate) written: usize,
    pub(crate) unresolved_written: usize,
    pub(crate) skipped: std::collections::BTreeMap<String, usize>,
}

impl RecordSink {
    pub(crate) fn create(files: &RecordFiles) -> Result<Self> {
        Ok(Self {
            out: create_side_file(&files.records)?,
            unresolved: files
                .unresolved
                .as_deref()
                .map(create_side_file)
                .transpose()?,
            lines: 0,
            written: 0,
            unresolved_written: 0,
            skipped: std::collections::BTreeMap::new(),
        })
    }

    /// Writes a marked line and returns `true`; `false` for any other line.
    pub(crate) fn route(&mut self, line: &str) -> Result<bool> {
        let Some(rest) = line.strip_prefix(RECORD_MARK) else {
            return Ok(false);
        };
        self.lines += 1;
        if let Some(skip) = rest.strip_prefix('!') {
            *self.skipped.entry(skip.to_string()).or_default() += 1;
        } else if let Some(record) = rest.strip_prefix('?') {
            match self.unresolved.as_mut() {
                Some(unresolved) => {
                    writeln!(unresolved, "{record}")?;
                    self.unresolved_written += 1;
                }
                None => {
                    writeln!(self.out, "{record}")?;
                    self.written += 1;
                }
            }
        } else {
            writeln!(self.out, "{rest}")?;
            self.written += 1;
        }
        Ok(true)
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.out.flush()?;
        let unresolved = match self.unresolved.as_mut() {
            Some(unresolved) => {
                unresolved.flush()?;
                format!(", {} unresolved", self.unresolved_written)
            }
            None => String::new(),
        };
        eprintln!(
            "tsume-solver: catalog records: {} written{unresolved}, not written {:?}",
            self.written, self.skipped
        );
        Ok(())
    }
}

/// The records of one job of a stage runner (`--emit-record`): from the input
/// as read, the stage's record and its extra records.
pub(crate) type RecordEmit = Box<
    dyn Fn(
            &PipelineRecord,
            &PipelineRecord,
            &[PipelineRecord],
            &mut Solver,
        ) -> Result<Vec<Result<Value, catalog_record::Skip>>>
        + Send
        + Sync,
>;

/// The negatives' side of one job: the events the reducer reported for
/// `input`, picked and re-judged, as marked lines.
pub(crate) fn negative_lines(
    solver: &mut Solver,
    input: &PipelineRecord,
    outcome: Value,
    options: &tsume_solver::negatives::FinalizeOptions,
) -> Result<Vec<String>> {
    let collected = tsume_solver::negatives::take();
    let (rows, summary) =
        tsume_solver::negatives::finalize(solver, input, &collected, outcome, options);
    let mut lines = Vec::with_capacity(1 + rows.len());
    lines.push(format!(
        "{NEGATIVE_MARK}-\t{}",
        serde_json::to_string(&summary)?
    ));
    for row in rows {
        let hash = row
            .get("canonicalHash")
            .and_then(Value::as_str)
            .unwrap_or("-")
            .to_string();
        lines.push(format!(
            "{NEGATIVE_MARK}{hash}\t{}",
            serde_json::to_string(&row)?
        ));
    }
    Ok(lines)
}

/// The negatives' file: rows in input order, each canonical hash once.
pub(crate) struct NegativesSink {
    pub(crate) out: BufWriter<std::fs::File>,
    pub(crate) seen: std::collections::HashSet<String>,
    pub(crate) lines: usize,
    pub(crate) duplicates: usize,
}

impl NegativesSink {
    pub(crate) fn create(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
        Ok(Self {
            out: BufWriter::new(file),
            seen: std::collections::HashSet::new(),
            lines: 0,
            duplicates: 0,
        })
    }

    /// Writes a marked line and returns `true`; `false` for a main line.
    pub(crate) fn route(&mut self, line: &str) -> Result<bool> {
        let Some(rest) = line.strip_prefix(NEGATIVE_MARK) else {
            return Ok(false);
        };
        self.lines += 1;
        let (hash, json) = rest.split_once('\t').context("a negatives line")?;
        if hash != "-" && !self.seen.insert(hash.to_string()) {
            self.duplicates += 1;
            return Ok(true);
        }
        writeln!(self.out, "{json}")?;
        Ok(true)
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.out.flush()?;
        eprintln!(
            "tsume-solver: negatives: {} rows ({} duplicates over the run dropped)",
            self.seen.len(),
            self.duplicates
        );
        Ok(())
    }
}

/// `--reduce-budget-ladder` of `reduce` and `reduce-accepted`.
#[derive(Args, Clone, Debug)]
pub(crate) struct BudgetLadder {
    // The ladder: owner 2026-10-15.  The top rung and no one-thread re-run:
    // owner 2026-10-23.
    /// Budget ladder of the reducer's checks: ascending node budgets,
    /// comma-separated, the last equal to `--node-budget`, e.g.
    /// `10000000,50000000` (50,000,000 is the top rung the production runs
    /// use since 2026-10-23; a convention, not enforced here).  Every
    /// acceptance test and entry check runs on the first budget; one that ran out of it
    /// (`solver_budget_exhausted` / `incomplete_response_graph`, or
    /// `minimum_length_not_proven`, which the shorter minimality search gives
    /// only by running out) runs again on the next.  Every rung runs on the
    /// same threads: one thread, or a parallel prober (`--prober-threads` >
    /// 1).  On one prober thread the output is byte-identical to the full
    /// budget alone (`src/reduce.rs` `BUDGET_LADDER`); with a parallel prober
    /// only whether a check resolved within its budget may differ, as without
    /// the ladder.  A check out of budget on the last rung stays a budget-out
    /// (2026-10-23: never re-run on one thread; the record's
    /// `reduction.minimality` is then `unverified`, and `--emit-budget-outs`
    /// lists it).  At most 3 smaller rungs.
    #[arg(long = "reduce-budget-ladder", value_delimiter = ',')]
    pub(crate) reduce_budget_ladder: Vec<u64>,
}

impl BudgetLadder {
    /// Sets the reducer's ladder for a full budget of `node_budget`.
    pub(crate) fn setup(&self, node_budget: u64) -> Result<()> {
        let rungs = &self.reduce_budget_ladder;
        let Some((&last, smaller)) = rungs.split_last() else {
            return Ok(());
        };
        if last != node_budget {
            bail!("--reduce-budget-ladder must end with --node-budget ({node_budget}), not {last}");
        }
        if !rungs.windows(2).all(|pair| pair[0] < pair[1]) || rungs[0] == 0 {
            bail!("--reduce-budget-ladder must be positive and strictly ascending");
        }
        if smaller.len() > tsume_solver::reduce::LADDER_RUNGS {
            bail!(
                "--reduce-budget-ladder takes at most {} rungs below --node-budget",
                tsume_solver::reduce::LADDER_RUNGS
            );
        }
        tsume_solver::reduce::set_budget_ladder(smaller);
        Ok(())
    }
}

/// Writes a `--stats-out` document, creating its directory if needed.
pub(crate) fn write_stats(path: &str, payload: &serde_json::Value) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let mut text = serde_json::to_string_pretty(payload)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("writing {path}"))
}

/// Adds the `profile` object of [`tsume_solver::prof`] when `TSUME_PROFILE` is set.
pub(crate) fn with_profile(mut stats: Value) -> Value {
    if let (Some(profile), Some(map)) = (tsume_solver::prof::report(), stats.as_object_mut()) {
        map.insert("profile".to_string(), profile);
    }
    stats
}
