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
diagram that passed ``solve`` is ``mate`` (rule D not checked).  Generation is
deterministic for a seed on one device type (CPU here).

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


def judge(samples: list, args, solver: Path) -> list:
    """One verdict dict per sample, in order."""
    threads = ["--threads", str(args.threads)] if args.threads else []
    bound = ["--max-ply", str(args.max_ply), "--node-budget", str(args.node_budget)]
    results = [dict(s, verdict=None, reasons=[], matePly=None, mainLine=None) for s in samples]
    todo = [(i, s["sfen"]) for i, s in enumerate(samples) if s["sfen"] is not None]
    for s in results:
        if s["sfen"] is None:
            s["verdict"] = "undecodable"
    if not todo:
        return results
    records = [imported_record(sfen, i, "input" if args.sfen else args.model) for i, sfen in todo]
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
                        "%s@%s" % (p.get("piece"), p.get("square")) if isinstance(p, dict)
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


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    solver = Path(args.solver) if args.solver else default_solver()
    if not solver.is_file():
        raise SystemExit("solver binary not found: %s (build it: cd solver && cargo build "
                         "--release, or pass --solver)" % solver)
    t0 = time.perf_counter()
    if args.sfen:   # judge the given positions; no model, no download
        meta = None
        samples = [{"sfen": sfen, "line": None} for sfen in args.sfen]
        t1 = t2 = time.perf_counter()
    else:
        weights = Path(args.weights_dir) if args.weights_dir else gm.fetch_weights(
            args.model, args.repo_id, args.revision)
        model, meta = gm.load_model(weights, args.device)
        extra = {"T13": args.defenders} if args.defenders else {}
        try:
            condition = gm.make_condition(model, args.ply, "sound", extra=extra)
        except ValueError as error:
            raise SystemExit("--ply / --defenders: %s" % error)
        t1 = time.perf_counter()
        try:
            samples = gm.generate(model, condition, args.count, args.seed, args.device,
                                  **gm.decoding_kwargs(args))
        except ValueError as error:
            raise SystemExit(str(error))
        t2 = time.perf_counter()
    results = judge(samples, args, solver)
    t3 = time.perf_counter()
    for n, res in enumerate(results, 1):
        if args.json:
            print(json.dumps(dict(res, index=n), ensure_ascii=False))
            continue
        target = "" if res["matePly"] is None or args.sfen else (
            " (target %d)" % args.ply if res["matePly"] != args.ply else "")
        mate = "" if res["matePly"] is None else " mate in %d%s" % (res["matePly"], target)
        print("%2d %-11s%s  %s" % (n, res["verdict"], mate, res["sfen"]))
        if res["mainLine"]:
            print("   main line: %s" % " ".join(res["mainLine"]))
        if res["reasons"]:
            print("   reasons: %s" % ", ".join(res["reasons"]))
    counts = {}
    for res in results:
        counts[res["verdict"]] = counts.get(res["verdict"], 0) + 1
    if meta is None:
        print("%d positions: %s" % (len(results), json.dumps(counts, sort_keys=True)),
              file=sys.stderr)
        print("seconds: judge %.1f" % (t3 - t2), file=sys.stderr)
        return 0
    on_target = sum(1 for r in results if r["verdict"] == "sound" and r["matePly"] == args.ply)
    print("model %s (%s parameters, %s steps), ply %d, %d samples: %s; sound at the target "
          "length: %d" % (meta.get("name", args.model), meta.get("parameters"),
                          meta.get("steps"), args.ply,
                          len(results), json.dumps(counts, sort_keys=True), on_target),
          file=sys.stderr)
    print("seconds: load %.1f, generate %.1f, judge %.1f" % (t1 - t0, t2 - t1, t3 - t2),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
