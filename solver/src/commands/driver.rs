//! The streaming driver every JSONL command shares: stdin through
//! [`run_ordered`] on the worker pool, each output line routed to its side
//! file (by its mark) or to stdout, in input order.

use super::*;

/// The side files of one run (each `None` when its flag is absent).  A worker
/// marks a side file's line with that file's mark (`RECORD_MARK`,
/// `NEGATIVE_MARK`, `BUDGET_OUT_MARK`); every other line is the main output.
#[derive(Default)]
pub(crate) struct SideFiles {
    pub(crate) records: Option<RecordSink>,
    pub(crate) negatives: Option<NegativesSink>,
    pub(crate) budget_outs: Option<BudgetOutSink>,
}

impl SideFiles {
    /// Writes a side file's line and returns `true`; `false` for a main line.
    fn route(&mut self, line: &str) -> Result<bool> {
        if let Some(sink) = self.records.as_mut() {
            if sink.route(line)? {
                return Ok(true);
            }
        }
        if let Some(sink) = self.negatives.as_mut() {
            if sink.route(line)? {
                return Ok(true);
            }
        }
        if let Some(sink) = self.budget_outs.as_mut() {
            if sink.route(line)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Flushes `--emit-record` (and reports its counts), if any.
    pub(crate) fn finish_records(&mut self) -> Result<()> {
        self.records.take().map_or(Ok(()), RecordSink::finish)
    }

    /// Flushes `--emit-negatives` (and reports its counts), if any.
    pub(crate) fn finish_negatives(&mut self) -> Result<()> {
        self.negatives.take().map_or(Ok(()), NegativesSink::finish)
    }

    /// Flushes `--emit-budget-outs`, if any.
    pub(crate) fn finish_budget_outs(&mut self) -> Result<()> {
        self.budget_outs
            .take()
            .map_or(Ok(()), BudgetOutSink::finish)
    }
}

/// Streams stdin through `work` on `schedule` ([`run_ordered`]) and writes the
/// results in input order: a side file's lines to `sides`, the others to
/// stdout (flushed before returning), calling `on_main` after each main line.
/// The returned `outputs` counts the main lines only.  The side files are left
/// open; the caller finishes them (each reports on stderr).
pub(crate) fn run_jsonl<T, W>(
    schedule: &Schedule,
    init: impl Fn() -> T + Sync,
    work: W,
    sides: &mut SideFiles,
    mut on_main: impl FnMut(),
) -> Result<StreamStats>
where
    T: Send,
    W: Fn(&mut T, &Job) -> Result<Vec<String>> + Sync,
{
    let stdin = io::stdin();
    let out = io::stdout();
    let mut out = BufWriter::new(out.lock());
    let mut side_lines = 0usize;
    let mut stats = run_ordered(stdin.lock().lines(), schedule, init, work, |line| {
        if sides.route(line)? {
            side_lines += 1;
            return Ok(());
        }
        writeln!(out, "{line}")?;
        on_main();
        Ok(())
    })?;
    out.flush()?;
    stats.outputs -= side_lines;
    Ok(stats)
}
