# Copyright (c) 2026 tokoroten
# SPDX-License-Identifier: MIT
"""Generate tsume-shogi puzzles with a tsume-gen model and judge them with the solver.

    cargo build --release --manifest-path solver/Cargo.toml   # or: cd solver && cargo build --release
    pip install -r requirements.txt
    python examples/generate_and_judge.py --model branch-rl-large --ply 3 --count 8

Steps:

1. Download ``<model>/model.safetensors`` and ``<model>/config.json`` from the
   Hugging Face model repository (default ``tokoroten-jp/tsume-gen``,
   anonymous; ``--weights-dir`` uses local files instead).
2. Sample ``--count`` diagrams with ``generate_min.py``, conditioned on the
   target mate length ``--ply``, the band ``sound`` (no alternative mate on
   the main line) and optionally the defender's board pieces
   (``--defenders``).  The default decoding is the paper's (24 rounds,
   temperature 1.0, random order, top-p 0.95, redo-check, defender king on
   files 1-5); ``--rounds``, ``--temperature``, ``--top-p`` and
   ``--no-redo-check`` change it, ``--plain`` drops top-p and redo-check.
3. Judge every diagram with the Rust solver: ``validate`` (strict tsume
   structure, the defender holds the remaining pieces), ``solve`` (forced mate
   and the composition rules A-C, F, G) and ``reduce`` (rule D: no piece can
   be removed).  ``solver/README.md`` lists the rules.

The verdict ``sound`` means the diagram passed all of them; with ``--skip-reduce`` a
diagram that passed ``solve`` is ``mate`` (rule D not checked).

Batches: ``--count`` diagrams are made ``--batch-size`` at a time (default
256): generate a batch, judge it, print its rows and flush, then the next.  A
run that stops early keeps the rows already printed, and the memory follows
the batch size, not ``--count``.  Batch 0 uses ``--seed`` itself and batch
``k`` the seed :func:`batch_seed` derives from (``--seed``, ``k``), so a run
with ``--count`` up to ``--batch-size`` gives the same diagrams as one batch
of that size.  The output is deterministic for (weights, conditions,
``--seed``, ``--batch-size``, device type); another batch size gives other
diagrams once ``--count`` exceeds it.

The provenance written into the solver's input records below is a
placeholder for judging only; when you share generated diagrams, state
where they came from (see the README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "generator"))   # generator/generate_min.py

import generate_min as gm  # noqa: E402

#: The sample's default solver bound (for speed; the paper judged up to 15
#: plies) and node budget.
GATE_MAX_PLY = 13
GATE_BUDGET = 10_000_000
#: Diagrams per batch (generate, judge, print).  Large enough that the
#: solver's threads stay busy and a GPU gets a useful batch; small enough
#: that a stop loses at most one batch.  It does not change the output of a
#: run with ``--count`` up to it (one batch, as before batches existed).
DEFAULT_BATCH_SIZE = 256


def batch_seed(seed: int, batch: int) -> int:
    """The generation seed of batch ``batch``: ``seed`` for batch 0, else a
    63-bit hash of both (so batch 1 of seed 1 is not batch 0 of seed 2)."""
    if batch == 0:
        return seed
    digest = hashlib.sha256(("tsume-gen-batch:%d:%d" % (seed, batch)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def default_solver() -> Path:
    exe = "tsume-solver.exe" if os.name == "nt" else "tsume-solver"
    env = os.environ.get("TSUME_SOLVER_BIN")
    return Path(env) if env else REPO_ROOT / "solver" / "target" / "release" / exe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="branch-rl-large", choices=gm.MODELS)
    p.add_argument("--repo-id", default=gm.DEFAULT_REPO_ID)
    p.add_argument("--revision", default="main", help="HF revision (branch, tag or commit)")
    p.add_argument("--weights-dir", default=None,
                   help="local dir with model.safetensors and config.json (no download)")
    p.add_argument("--ply", type=int, default=3, help="target mate length (odd, 1-11)")
    p.add_argument("--defenders", default=None, choices=dict(gm.EXTRA_CONDITIONS)["T13"],
                   help="defender pieces on the board, king not counted (default unspecified)")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="diagrams generated, judged and printed at a time (default %d); "
                        "each batch is flushed before the next starts" % DEFAULT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or mps")
    gm.add_decoding_arguments(p)
    p.add_argument("--solver", default=None,
                   help="tsume-solver binary (default: $TSUME_SOLVER_BIN or "
                        "solver/target/release/tsume-solver)")
    p.add_argument("--max-ply", type=int, default=GATE_MAX_PLY, help="solver search bound")
    p.add_argument("--node-budget", type=int, default=GATE_BUDGET, help="solver node budget")
    p.add_argument("--threads", type=int, default=0, help="solver threads (0 = all cores)")
    p.add_argument("--skip-reduce", action="store_true",
                   help="skip rule D (reduce); puzzles that pass solve are reported as mate")
    p.add_argument("--json", action="store_true", help="one JSON object per puzzle")
    p.add_argument("--sfen", action="append", default=[], metavar="SFEN",
                   help="judge this position instead of generating (repeatable; no model "
                        "is loaded). Attacker (black) to move; the defender's hand may be "
                        "left out")
    return p


# ---------------------------------------------------------------- judge

def imported_record(sfen: str, index: int, model: str) -> dict:
    """A minimal ``stage: "imported"`` record for the solver's JSONL stages."""
    digest = hashlib.sha256(sfen.encode("utf-8")).hexdigest()
    sid = "gen-%d-%s" % (index, digest[:12])
    return {"schemaVersion": 1, "stage": "imported", "sourceId": sid, "sourceSfen": sfen,
            "originalCandidateId": sid,
            "provenance": {"sourceType": "random_direct", "generatorVersion": "tsume-gen:" + model,
                           "generatorConfigHash": digest, "sourcePositionHash": digest,
                           "mateSolverVersion": "bounded-and-or-v1",
                           "createdAt": "2026-01-01T00:00:00.000Z",
                           "historicalReachabilityChecked": False, "transformations": [],
                           "originClass": "own_generated", "origins": ["tsume-gen"],
                           "rootSourceId": sid, "lineage": []}}


def run_stage(solver: Path, args: list, records: list) -> list:
    """Run one solver stage: JSONL records on stdin, JSONL records on stdout."""
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    proc = subprocess.run([str(solver)] + args, input=payload, capture_output=True,
                          text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit("tsume-solver %s failed:\n%s" % (" ".join(args), proc.stderr))
    return [json.loads(t) for t in proc.stdout.splitlines() if t.strip()]


def judge(samples: list, args, solver: Path, first: int = 0) -> list:
    """One verdict dict per sample, in order (``first``: the run-wide index
    of ``samples[0]``, for the records' ids)."""
    threads = ["--threads", str(args.threads)] if args.threads else []
    bound = ["--max-ply", str(args.max_ply), "--node-budget", str(args.node_budget)]
    results = [dict(s, verdict=None, reasons=[], matePly=None, mainLine=None) for s in samples]
    todo = [(i, s["sfen"]) for i, s in enumerate(samples) if s["sfen"] is not None]
    for s in results:
        if s["sfen"] is None:
            s["verdict"] = "undecodable"
    if not todo:
        return results
    records = [imported_record(sfen, first + i, "input" if args.sfen else args.model)
               for i, sfen in todo]
    validated = run_stage(solver, ["validate"] + threads, records)
    solved = run_stage(solver, ["solve"] + bound + threads, validated)
    accepted = {}
    for (i, _), rec in zip(todo, solved):
        res = results[i]
        res["normalizedSfen"] = rec.get("normalizedSfen")
        rejection = rec.get("rejection") or {}
        reasons = rejection.get("reasons") or []
        res["reasons"] = reasons
        res["matePly"] = (rec.get("solution") or {}).get("matePly")
        res["mainLine"] = rec.get("designatedMainLine")
        if reasons:
            res["verdict"] = "invalid" if rejection.get("stage") == "validate" else (
                "unknown" if "solver_budget_exhausted" in reasons else "rejected")
        else:
            # rules A-C, F, G passed; rule D (reduce) not run yet, or skipped
            res["verdict"] = "mate"
            accepted.setdefault(rec["canonicalHash"], []).append(i)
    if accepted and not args.skip_reduce:
        firsts = {}
        for r in solved:
            if r.get("canonicalHash") in accepted:
                firsts.setdefault(r["canonicalHash"], r)   # duplicates: reduce once
        reduced = run_stage(solver, ["reduce"] + bound + threads, list(firsts.values()))
        for rec in reduced:
            red = rec.get("reduction") or {}
            if red.get("kind", "mainline") != "mainline" or red.get("rank", 0):
                continue
            for i in accepted.get(red.get("sourceCanonicalHash"), ()):
                res = results[i]
                if red.get("removed"):
                    res["verdict"] = "reducible"
                    res["reasons"] = ["rule_d_removable:" + ",".join(
                        "%s@%s" % (p.get("piece"), usi_square(p.get("square")))
                        if isinstance(p, dict)
                        else str(p) for p in red["removed"])]
                elif red.get("minimality") == "verified":
                    res["verdict"] = "sound"
                else:
                    res["verdict"] = "unknown"
                    res["reasons"] = ["rule_d_" + str(red.get("minimality"))]
        for res in results:
            if res["verdict"] == "mate":   # reduce wrote nothing for it
                res["verdict"] = "unknown"
                res["reasons"] = ["rule_d_no_record"]
    return results


def usi_square(square):
    """The solver's square number (0-80: ``rank * 9 + column``, rank 0 = USI
    rank ``a``, column 0 = file 9) as a USI square, e.g. 60 -> ``3g``; other
    values unchanged."""
    if isinstance(square, int) and 0 <= square < 81:
        return "%d%s" % (9 - square % 9, "abcdefghi"[square // 9])
    return square


def row_origin(args, meta) -> dict:
    """Where a ``--json`` row came from: the model (its ``config.json`` name),
    the target mate length, ``--seed`` and ``--batch-size`` (the diagrams
    depend on both); all null for ``--sfen``, which generates nothing."""
    if meta is None:
        return {"model": None, "targetPly": None, "seed": None, "batchSize": None}
    return {"model": meta.get("name", args.model), "targetPly": args.ply, "seed": args.seed,
            "batchSize": args.batch_size}


# ---------------------------------------------------------------- main

def print_rows(results: list, args, first: int, out, origin=None) -> None:
    """Print one batch (``first``: the run-wide index before it) and flush.
    ``origin``: the keys every ``--json`` row ends with (:func:`row_origin`)."""
    for n, res in enumerate(results, first + 1):
        if args.json:
            row = dict(res, index=n, **(origin or {}))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue
        target = "" if res["matePly"] is None or args.sfen else (
            " (target %d)" % args.ply if res["matePly"] != args.ply else "")
        mate = "" if res["matePly"] is None else " mate in %d%s" % (res["matePly"], target)
        out.write("%2d %-11s%s  %s\n" % (n, res["verdict"], mate, res["sfen"]))
        if res["mainLine"]:
            out.write("   main line: %s\n" % " ".join(res["mainLine"]))
        if res["reasons"]:
            out.write("   reasons: %s\n" % ", ".join(res["reasons"]))
    out.flush()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    gm.lf_stdout()   # LF line ends on every OS
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.count < 0:
        raise SystemExit("--count must not be negative")
    solver = Path(args.solver) if args.solver else default_solver()
    if not solver.is_file():
        raise SystemExit("solver binary not found: %s (build it: cd solver && cargo build "
                         "--release, or pass --solver)" % solver)
    t0 = time.perf_counter()
    meta = model = condition = None
    total = len(args.sfen) if args.sfen else args.count
    if not args.sfen:   # --sfen: judge the given positions; no model, no download
        weights = Path(args.weights_dir) if args.weights_dir else gm.fetch_weights(
            args.model, args.repo_id, args.revision)
        model, meta = gm.load_model(weights, args.device)
        extra = {"T13": args.defenders} if args.defenders else {}
        try:
            condition = gm.make_condition(model, args.ply, "sound", extra=extra)
        except ValueError as error:
            raise SystemExit("--ply / --defenders: %s" % error)
    t1 = time.perf_counter()
    seconds_generate = seconds_judge = 0.0
    counts, on_target = {}, 0
    batches = -(-total // args.batch_size)
    for batch in range(batches):
        first = batch * args.batch_size
        size = min(args.batch_size, total - first)
        ta = time.perf_counter()
        if args.sfen:
            samples = [{"sfen": sfen, "line": None} for sfen in args.sfen[first:first + size]]
        else:
            try:
                samples = gm.generate(model, condition, size, batch_seed(args.seed, batch),
                                      args.device, **gm.decoding_kwargs(args))
            except ValueError as error:
                raise SystemExit(str(error))
        tb = time.perf_counter()
        results = judge(samples, args, solver, first)
        tc = time.perf_counter()
        seconds_generate += tb - ta
        seconds_judge += tc - tb
        print_rows(results, args, first, sys.stdout, row_origin(args, meta))
        for res in results:
            counts[res["verdict"]] = counts.get(res["verdict"], 0) + 1
            if meta is not None and res["verdict"] == "sound" and res["matePly"] == args.ply:
                on_target += 1
        if batches > 1:
            print("batch %d/%d: %d of %d done" % (batch + 1, batches, first + size, total),
                  file=sys.stderr, flush=True)
    done = sum(counts.values())
    if meta is None:
        print("%d positions: %s" % (done, json.dumps(counts, sort_keys=True)),
              file=sys.stderr)
        print("seconds: judge %.1f" % seconds_judge, file=sys.stderr)
        return 0
    print("model %s (%s parameters, %s steps), ply %d, %d samples: %s; sound at the target "
          "length: %d" % (meta.get("name", args.model), meta.get("parameters"),
                          meta.get("steps"), args.ply,
                          done, json.dumps(counts, sort_keys=True), on_target),
          file=sys.stderr)
    print("seconds: load %.1f, generate %.1f, judge %.1f" % (t1 - t0, seconds_generate,
                                                             seconds_judge), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
