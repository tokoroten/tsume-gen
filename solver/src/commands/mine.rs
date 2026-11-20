//! `mine`.

use super::*;

pub(crate) fn run_mine(
    options: &MineOptions,
    spec: &PolicySpec,
    threads: usize,
    stats_out: Option<&str>,
) -> Result<()> {
    let out = io::stdout();
    let mut out = BufWriter::new(out.lock());
    let threads = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };
    let cpu_before = process_cpu_seconds();
    let run = tsume_solver::mine::run(options, spec, threads, &mut out, true)?;
    let cpu = cpu_seconds_since(cpu_before);
    out.flush()?;
    let stats = run.stats;
    eprintln!(
        "tsume-solver mine: {} candidates in {:.2}s on {} threads ({} games, {} mate searches, {} mates, {} strict-valid)",
        stats.emitted,
        run.wall_seconds,
        run.threads,
        stats.games,
        stats.mate_searches,
        stats.mates_found,
        stats.strict_valid
    );
    if let Some(path) = stats_out {
        let allocated = run.wall_seconds * run.threads as f64;
        write_stats(
            path,
            &with_profile(json!({
                "family": options.family.as_str(),
                "seed": options.seed.to_string(),
                "threads": run.threads,
                "wallSeconds": run.wall_seconds,
                // `cpuSeconds` used to be `wall x threads`, i.e. the CPU the run
                // was *allocated*.  It is measured process CPU now (user + kernel,
                // every thread), and the old figure keeps its own name, so
                // `cpuSeconds / allocatedCoreSeconds` is the utilization.
                "cpuSeconds": cpu,
                "allocatedCoreSeconds": allocated,
                "utilization": cpu.map(|cpu| cpu / allocated.max(1e-9)),
                "options": options.to_json(),
                "generatorConfig": options.generator.to_json(),
                "stats": stats.to_json(),
            })),
        )?;
    }
    Ok(())
}
