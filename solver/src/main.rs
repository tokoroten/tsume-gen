//! `tsume-solver` CLI.
//!
//! JSONL `PipelineRecord` documents on stdin/stdout, logs on stderr, non-zero
//! exit on bad input.  The judging stages run in this order (the record shape
//! and the full commands are in `examples/generate_and_judge.py`):
//!
//! ```text
//! tsume-solver validate < records.jsonl \
//!   | tsume-solver solve --max-ply 15 --node-budget 10000000 \
//!   | tsume-solver reduce --max-ply 15 --node-budget 10000000
//! ```
//!
//! `mine`, `distill` and `extend` make diagrams; see `tsume-solver <command> --help`.

mod cli;
mod commands;
mod largepage;
mod record_output;
mod startup;

use cli::*;
use commands::*;
use record_output::*;
use startup::*;

use std::io::{self, BufRead, BufWriter, Read, Write};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use serde_json::{json, Value};

use tsume_solver::catalog_record::{self, Extras, RecordContext};
use tsume_solver::cdfpn::{entry_disproof_kind, entry_probe, ClassicalProber};
use tsume_solver::cpu::{cpu_seconds_since, process_cpu_seconds};
use tsume_solver::dfpn::{ProbeEngine, ProbeOptions, Prober, Verdict};
use tsume_solver::distill::{distill_record_judged, imported_record_from_sfen, DistillOptions};
use tsume_solver::engine::EngineConfig;
use tsume_solver::exact::{prune_solver_graph, TsumeRules};
use tsume_solver::extend::{extend_record_probed, Emit, ExtendOptions, ExtendOrder, ExtendStats};
use tsume_solver::generate::Distribution;
use tsume_solver::jsonl::{read_json_line, write_json_line, PipelineRecord, SolverGraphJson};
use tsume_solver::mine::{Family, MineGeneratorConfig, MineOptions, PolicySpec};
use tsume_solver::movegen::{legal_moves_sorted, perft};
use tsume_solver::piece::{BLACK, WHITE};
use tsume_solver::schedule::{run_ordered, Job, Schedule, StreamStats};
use tsume_solver::sfen::parse_sfen;
use tsume_solver::solver::{
    non_futile_node_count, Outcome, SearchLimits, SearchRules, SolveOptions, Solver, TableMode,
    UselessInterposition,
};
use tsume_solver::stages::{
    full_graph_to_json, pruned_graph_to_json, reduce_accepted, reduce_stage, solve_stage_reported,
    stats_to_json, ReduceAcceptedOutcome, ReduceStageOptions, SolveStageOptions,
};

fn main() {
    startup::configure();
    let result = run();
    startup::report();
    if let Err(error) = result {
        eprintln!("tsume-solver: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    // Before any command opens an output.
    if let Some(position) = cli.command.position() {
        position.refuse_unsupported();
    }
    match cli.command {
        Command::Movegen { sfen, stdin } => run_movegen(sfen, stdin),
        Command::Probe {
            engine,
            max_ply,
            node_budget,
            parallel,
            position,
        } => run_probe(&engine, max_ply, node_budget, parallel, position),
        Command::Perft { sfen, depth } => {
            let position = parse_sfen(&sfen)?;
            println!("{}", perft(&position, depth));
            Ok(())
        }
        Command::SolveSfen {
            sfen,
            max_ply,
            node_budget,
            attacker,
            no_strict_material,
            graph,
            useless_interposition,
            stdin,
            position,
        } => {
            note_experimental("--no-strict-material", no_strict_material);
            run_solve_sfen(
                sfen,
                max_ply,
                node_budget,
                attacker,
                !no_strict_material,
                &graph,
                &useless_interposition,
                stdin,
                position,
            )
        }
        Command::Validate { parallel } => {
            run_stage(parallel, PositionParallel::SINGLE, |record, _solver| {
                tsume_solver::stages::validate_stage(record);
                Ok(())
            })
        }
        Command::Reduce {
            attempts,
            max_ply,
            node_budget,
            seed,
            length_k,
            no_rule_b_entry,
            cg_entry,
            jobs,
            ladder,
            negatives,
            budget_outs,
            record,
            parallel,
            position,
        } => {
            note_experimental("--no-rule-b-entry", no_rule_b_entry);
            ladder.setup(node_budget)?;
            distinct_outputs(
                "reduce",
                &[
                    ("--emit-negatives", negatives.emit_negatives.as_deref()),
                    ("--emit-record", record.emit_record.as_deref()),
                    ("--emit-unresolved", record.emit_unresolved.as_deref()),
                    (
                        "--emit-budget-outs",
                        budget_outs.emit_budget_outs.as_deref(),
                    ),
                ],
            )?;
            let negatives = negatives.setup("reduce", node_budget);
            let records = position.tag_records(record.setup(
                "reduce",
                Some("reduce"),
                Some(catalog_record::rule_d_version(node_budget)),
                &[
                    negatives.as_ref().map(|(path, _)| path.as_str()),
                    budget_outs.emit_budget_outs.as_deref(),
                ],
            )?);
            let options = ReduceStageOptions {
                reduce: tsume_solver::reduce::ReduceOptions {
                    attempts,
                    max_ply,
                    node_budget,
                    rules: TsumeRules::puzzle(),
                    // Rule D is frozen for the `reduce` stage.
                    hand_transfer: true,
                    // Rule D v4 (owner decision 2026-09-18): strict entry.
                    strict_entry: true,
                    // A no-op since `PIPELINE_VERSION` 0.13.0; the CLI flag went
                    // in stage 4 (2026-10-22).
                    until_fixpoint: false,
                    jobs: position.reduce_jobs(jobs),
                },
                seed,
                kinds: tsume_solver::reduce::TwoKindsOptions {
                    length_k: length_k as usize,
                    length_k_yozume: tsume_solver::reduce::LENGTH_K_YOZUME,
                    rule_b_entry: !no_rule_b_entry,
                    cg_entry,
                },
            };
            // `--emit-record`: the gate judges each diagram at the reducer's bound
            // and budget, under the reducer's rules.
            let gate = SolveStageOptions {
                max_ply,
                node_budget,
                rules: options.reduce.rules,
            };
            let records = records.map(|(path, ctx)| {
                let emit: RecordEmit = Box::new(move |input, main, extra, solver| {
                    catalog_record::reduce_records(input, main, extra, &gate, &ctx, solver)
                });
                (path, emit)
            });
            run_stage_multi_with(
                parallel,
                position,
                negatives,
                records,
                Some(&budget_outs),
                move |record, solver| reduce_stage(record, &options, solver),
            )
        }
        Command::Solve {
            max_ply,
            node_budget,
            emit_main_line: _,
            record,
            parallel,
            position,
        } => {
            // The published gate's rule F (`classical` since `PIPELINE_VERSION`
            // 0.10.0).  The measurement-only override `--useless-interposition`
            // went on 2026-10-27 (never used on `solve`; `solve-sfen` keeps it),
            // and with it `extra.uselessInterposition` on `solve`'s records.
            let options = SolveStageOptions {
                max_ply,
                node_budget,
                rules: TsumeRules::puzzle(),
            };
            distinct_outputs(
                "solve",
                &[
                    ("--emit-record", record.emit_record.as_deref()),
                    ("--emit-unresolved", record.emit_unresolved.as_deref()),
                ],
            )?;
            let records = position
                .tag_records(record.setup("solve", None, None, &[])?)
                .map(|(path, ctx)| {
                    let emit: RecordEmit = Box::new(move |_input, record, _extra, _solver| {
                        Ok(vec![catalog_record::build(
                            record,
                            &ctx,
                            &Extras::default(),
                        )])
                    });
                    (path, emit)
                });
            let stage = move |record: &mut PipelineRecord, solver: &mut Solver| -> Result<()> {
                let report = solve_stage_reported(record, &options, solver)?;
                if report.solved {
                    record
                        .extra
                        .insert("designatedMainLine".to_string(), json!(report.main_line));
                    record.extra.insert(
                        "designatedMainLineLeftoverHand".to_string(),
                        json!(report.leftover_hand),
                    );
                }
                Ok(())
            };
            run_stage_multi_with(
                parallel,
                position,
                None,
                records,
                None,
                move |record, solver| stage(record, solver).map(|()| Vec::new()),
            )
        }
        Command::ReduceAccepted {
            node_budget,
            attempts,
            length_k,
            jobs,
            ladder,
            budget_outs,
            record,
            parallel,
            position,
        } => {
            ladder.setup(node_budget)?;
            distinct_outputs(
                "reduce-accepted",
                &[
                    ("--emit-record", record.emit_record.as_deref()),
                    ("--emit-unresolved", record.emit_unresolved.as_deref()),
                    (
                        "--emit-budget-outs",
                        budget_outs.emit_budget_outs.as_deref(),
                    ),
                ],
            )?;
            let records = position
                .tag_records(record.setup(
                    "reduce-accepted",
                    Some("reduce"),
                    Some(catalog_record::rule_d_version(node_budget)),
                    &[budget_outs.emit_budget_outs.as_deref()],
                )?)
                .map(|(files, ctx)| {
                    let emit: RecordEmit = Box::new(move |input, record, extra, solver| {
                        catalog_record::reduce_accepted_records(
                            input,
                            record,
                            extra,
                            node_budget,
                            &ctx,
                            solver,
                        )
                    });
                    (files, emit)
                });
            run_stage_multi_with(
                parallel,
                position,
                None,
                records,
                Some(&budget_outs),
                move |record, solver| {
                    reduce_accepted_record(
                        record,
                        node_budget,
                        attempts,
                        length_k as usize,
                        position.reduce_jobs(jobs),
                        solver,
                    )
                },
            )
        }
        Command::Distill {
            seed,
            attempts,
            distill_v2,
            allow_length_change,
            origin,
            origin_class,
            max_ply,
            node_budget,
            max_nodes,
            prestrip,
            probe,
            dfpn_node_budget,
            stats_out,
            negatives,
            budget_outs,
            record,
            parallel,
        } => {
            distinct_outputs(
                "distill",
                &[
                    ("--emit-negatives", negatives.emit_negatives.as_deref()),
                    ("--emit-record", record.emit_record.as_deref()),
                    ("--emit-unresolved", record.emit_unresolved.as_deref()),
                    (
                        "--emit-budget-outs",
                        budget_outs.emit_budget_outs.as_deref(),
                    ),
                    ("--stats-out", stats_out.as_deref()),
                ],
            )?;
            // One inner-loop trial's budget, 60,000, and 4,000 exact validations
            // per source are fixed (`--search-node-budget`, `--max-validations`
            // went on 2026-10-27: never given).
            let negatives =
                negatives.setup("distill", DistillOptions::default().search_node_budget);
            let records = record.setup(
                "distill",
                Some("distill"),
                Some(catalog_record::rule_d_version(node_budget)),
                &[
                    negatives.as_ref().map(|(path, _)| path.as_str()),
                    stats_out.as_deref(),
                    budget_outs.emit_budget_outs.as_deref(),
                ],
            )?;
            let probe = ProbeEngine::parse(&probe).expect("clap restricts --probe");
            // Both or neither: an origin id without a class states half of
            // `docs/legal-data.md` §3, which is not a provenance at all.
            let origin = match (origin, origin_class) {
                (Some(id), Some(class)) => Some(BareSfenOrigin { id, class }),
                (None, None) => None,
                _ => bail!("--origin and --origin-class must be given together"),
            };
            let options = DistillOptions {
                seed,
                base_seed: seed,
                attempts,
                // `distill-v2` (`docs/generation.md` §2.6): rule B relaxed on
                // chain steps and one `relocate`.
                relax_step_yozume: distill_v2,
                max_relocate: usize::from(distill_v2),
                allow_length_change,
                max_ply,
                node_budget,
                max_nodes,
                probe,
                dfpn_node_budget,
                prestrip,
                ..DistillOptions::default()
            };
            run_distill(
                &options,
                parallel,
                stats_out.as_deref(),
                origin.as_ref(),
                negatives,
                records,
                &budget_outs,
            )
        }
        Command::RuleBProbe {
            node_budget,
            mate_ply,
            stats_out,
            parallel,
        } => run_rule_b_probe(node_budget, mate_ply, parallel, stats_out.as_deref()),
        Command::MainLinePaths { cap, parallel } => run_main_line_paths(cap, parallel),
        Command::StoredMainLine { parallel } => run_stored_main_line(parallel),
        Command::Extend {
            node_budget,
            source_max_ply,
            emit,
            order,
            ranker_id,
            max_verify,
            stop_after_accepted,
            probe,
            dfpn_node_budget,
            origin,
            origin_class,
            stats_out,
            budget_outs,
            record,
            parallel,
        } => {
            distinct_outputs(
                "extend",
                &[
                    ("--emit-record", record.emit_record.as_deref()),
                    ("--emit-unresolved", record.emit_unresolved.as_deref()),
                    (
                        "--emit-budget-outs",
                        budget_outs.emit_budget_outs.as_deref(),
                    ),
                    ("--stats-out", stats_out.as_deref()),
                ],
            )?;
            let records = record.setup(
                "extend",
                Some("extend"),
                Some(catalog_record::rule_d_version(node_budget)),
                &[
                    stats_out.as_deref(),
                    budget_outs.emit_budget_outs.as_deref(),
                ],
            )?;
            // Both or neither, as in `distill`: an origin id without a class states
            // half of `docs/legal-data.md` §3, which is not a provenance at all.
            let origin = match (origin, origin_class) {
                (Some(id), Some(class)) => Some(BareSfenOrigin { id, class }),
                (None, None) => None,
                _ => bail!("--origin and --origin-class must be given together"),
            };
            let emit = Emit::parse(&emit).context("--emit")?;
            // Every ranked record names the model that ranked it, so the id is not
            // optional once an order file is in play.
            let order = match (order, ranker_id) {
                (Some(path), Some(id)) => Some(ExtendOrder::read(&path, &id)?),
                (None, _) => None,
                (Some(_), None) => bail!(
                    "--order needs --ranker-id <id>: every ranked record records \
                     which model ordered it (provenance.extra.ranker.id)"
                ),
            };
            if emit == Emit::Candidates
                && (order.is_some() || max_verify != 0 || stop_after_accepted != 0)
            {
                bail!(
                    "--emit candidates only enumerates; it runs no search, so \
                     --order / --ranker-id / --max-verify / \
                     --stop-after-accepted have nothing to cut"
                );
            }
            if let Some(order) = order.as_ref() {
                eprintln!(
                    "tsume-solver extend: --order {} rows over {} parents",
                    order.rows,
                    order.parents()
                );
            }
            let options = ExtendOptions {
                node_budget,
                source_max_ply,
                emit,
                max_verify,
                stop_after_accepted,
                probe: ProbeEngine::parse(&probe).expect("clap restricts --probe"),
                dfpn_node_budget,
                reduce: true,
            };
            run_extend(
                &options,
                parallel,
                stats_out.as_deref(),
                origin.as_ref(),
                order.as_ref(),
                records,
                &budget_outs,
            )
        }
        Command::Mine {
            family,
            count,
            seed,
            plies,
            max_ply,
            check_from,
            node_budget,
            max_games,
            prestrip,
            probe,
            dfpn_node_budget,
            config,
            distribution,
            policy,
            stats_out,
            parallel,
        } => {
            let family = Family::parse(&family).context("unknown --family")?;
            let mut generator = MineGeneratorConfig::default();
            if let Some(text) = config.as_deref() {
                let value: serde_json::Value =
                    serde_json::from_str(text).context("--config must be JSON")?;
                generator.overlay_json(&value)?;
            }
            // Rule E: only a family that actually plays moves needs an attacker
            // king, and an explicit `attackerKing` in --config always wins.
            generator.apply_family_default(family.plays());
            // The built-in engine at its fixed depth 4 and node cap 30,000 per
            // decision (the provenance keeps the values; the flags that set
            // them: `docs/experiments/solver-cli-audit/README.md` §8).
            let engine = EngineConfig::default();
            let spec = PolicySpec::parse(&policy, engine)?;
            let options = MineOptions {
                family,
                count,
                seed,
                plies: parse_ply_range(&plies)?,
                max_ply,
                check_from: check_from.unwrap_or(u32::from(family.plays())),
                node_budget,
                max_games,
                prestrip,
                probe: ProbeEngine::parse(&probe).expect("clap restricts --probe"),
                dfpn_node_budget,
                generator,
                distribution: distribution.as_deref().and_then(Distribution::parse),
                ..MineOptions::default()
            };
            run_mine(&options, &spec, parallel.threads, stats_out.as_deref())
        }
    }
}
