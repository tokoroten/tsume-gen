# AGENTS.md — guide for AI coding agents

This file is for an AI coding agent (Claude Code, Codex, Cursor, Copilot and similar) that a user has asked to set up or use this repository. People can read [README.md](README.md) instead.

What a user usually wants from this repository is one of these:
- make it run and see generated puzzles;
- generate many puzzles and save them;
- judge their own positions;
- call the generator or the solver from their own code.

Section 1 explains the parts, sections 2–5 get everything running with a check after each step, and sections 6–7 cover the common tasks. Report the checks' results to the user as you go.

## 1. What is here

- `solver/`: a Rust command-line program, `tsume-solver`. It searches for forced mates exactly and judges a position against the usual rules of tsume shogi (Japanese chess mate problems).
- `generator/generate_min.py`: one PyTorch file that runs the trained generator, a MaskGIT model. It fills in a board, the attacker's pieces in hand and the first three moves of the solution. There is no training code.
- `training/restore_m17.py`: rebuilds the pre-training data files from the Hugging Face dataset `tokoroten-jp/tsume-gen-m17-pretraining` (packages in `requirements-training.txt`; run with `--check` to compare SHA-256). Only needed to study or reuse the training data.
- `examples/generate_and_judge.py`: the end-to-end script. It downloads weights, generates positions and judges each one with the solver.
- The weights are not in this repository. They are downloaded at run time from the public Hugging Face repository `tokoroten-jp/tsume-gen`. The models are:
  - `branch-rl-large`: the default, the main weights of the paper.
  - `branch-rl-large-ema`
  - `sft-large`
  - `sft-medium`
  - `sft-small`: the smallest and fastest.
  - The comparison runs on the model card: `rl-indep-large-s0`, `rl-indep-large-s1`, `rl-branchonly-large-s0`, `rl-branchonly-large-s1`, each also with `-ema`.
  - Weights elsewhere: `--repo-id` (another Hugging Face repository) or `--weights-dir DIR` (local `config.json` and `model.safetensors`).

No account, token, GPU or database is needed. The first setup downloads from rustup, crates.io and PyPI (and download.pytorch.org if you pick the CPU build of PyTorch). After that, the only network access is the weight download from Hugging Face. Once the weights are cached, or with `--weights-dir`, it runs offline.

## 2. Prerequisites

| Need | Detail | Check |
| --- | --- | --- |
| Rust | Install with rustup. `solver/rust-toolchain.toml` selects the `stable` channel with `rustfmt` and `clippy` (a default rustup install has them). Windows: the MSVC toolchain needs the Visual Studio Build Tools with "Desktop development with C++". Linux: a C linker (`cc`). macOS: `xcode-select --install`. | `cargo --version` |
| Python | 3.12 (the tested version) | `python3 --version` (Windows: `python --version`) |
| Disk | With the CPU build of PyTorch, about 1.5 GB: the Rust build ~0.5 GB, PyTorch, and the weights (343 MB for a large model, about 30 MB for `sft-small`). With the CUDA build (Windows, cu128), the venv alone was about 4.4 GB and pip's cache about 2.6 GB (`--no-cache-dir` keeps none). | — |
| Memory | About 1.5 GB free for the default run on CPU. The measured peak, including the solver, was 0.6–1.3 GB. | — |

**x86_64 CPUs.** The solver is built for x86-64-v3, which needs AVX2 and BMI2. Most CPUs from about 2015 onward have them.
- If the solver fails with "illegal instruction" (`STATUS_ILLEGAL_INSTRUCTION` on Windows), the CPU lacks them.
- Rebuild with `RUSTFLAGS="-C target-cpu=x86-64" cargo build --release` (PowerShell: `$env:RUSTFLAGS="-C target-cpu=x86-64"; cargo build --release`). The results are the same, only a little slower.
- Apple silicon and other ARM machines are not affected.

## 3. Build the solver

```bash
cd solver
cargo build --release
```

- The binary is `solver/target/release/tsume-solver`, or `tsume-solver.exe` on Windows.
- Run cargo from inside `solver/`. The build settings in `solver/.cargo/config.toml` apply only there.

**Check 1:**

```bash
./target/release/tsume-solver --help
./target/release/tsume-solver solve-sfen "4k4/9/4P4/9/9/9/9/9/9 b G 1" --max-ply 3
```

Expected:
- The first command lists the subcommands.
- The second prints one JSON object containing `"outcome":"MATE"` and `"distance":1`. The mating move is `G*5b`.

**Check 2:**

```bash
cargo test --release --lib
```

Expected: every test passes (about 60; the exact count depends on the OS, because some tests run on Windows or Linux only).
- Plain `cargo test --release` also passes (it adds the command-line tests); `--lib` is the faster one.
- Then go back to the repository root (`cd ..`). The rest of this file runs from there.

## 4. Set up Python

From the repository root:

```bash
python3 -m venv .venv                                       # Windows: py -3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # Windows: .venv\Scripts\python -m pip install -r requirements.txt
```

- Call the venv's interpreter by its path (`.venv/bin/python`, or `.venv\Scripts\python` on Windows). This avoids depending on shell activation or aliases. The rest of this file writes `python` for that interpreter.
- Everything runs on the CPU (the default `--device cpu`); a GPU is optional.
- Windows: PyPI's `torch` is the CPU build (`+cpu`). For an NVIDIA GPU, install the CUDA build first, then `requirements.txt`: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128` (pick the `cu` number the driver supports; `nvidia-smi` shows it). If the `+cpu` build is already there, `python -m pip uninstall torch` first, or pip changes nothing.
- Linux without a GPU: PyPI's `torch` includes CUDA (several GB). Install the CPU build first: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Apple silicon: the default PyTorch works with `--device mps`.

**Check 3:**

```bash
python -c "import torch, numpy, safetensors, huggingface_hub; print(torch.__version__, torch.cuda.is_available())"
```

Expected: the version, then `True` if `--device cuda` will work (`False` with `+cpu` is the CPU build).

## 5. Run the sample

```bash
python examples/generate_and_judge.py --ply 3 --count 8 --seed 1
```

- `--ply` is the target mate length: 1, 3, 5, 7, 9 or 11.
- `--device` picks the hardware: `cpu` (the default), `cuda`, `cuda:N` or `mps`.
- `--model` picks the weights; for a quick first try, use `--model sft-small`.
- The first run downloads the weights. huggingface_hub caches them in `~/.cache/huggingface/hub` (Windows: `%USERPROFILE%\.cache\huggingface\hub`); set `HF_HOME` to move the cache.
- A warning about "unauthenticated requests" is harmless. Do not ask the user for a Hugging Face token.

**Check 4.** The script prints two summary lines on **stderr**, for example:

```
model branch-rl-large (85883664 parameters, 102000 steps), ply 3, 8 samples: {"reducible": 2, "rejected": 4, "sound": 2}; sound at the target length: 2
seconds: load 0.5, generate 7.2, judge 18.6
```

On **stdout**, it prints one entry per generated position:

```
 2 sound       mate in 3  9/7nl/7sk/9/8P/6SS1/8N/9/9 b - 1
   main line: 1g2e 1c2d 3f3e
```

Every position gets one verdict:

| Verdict | Meaning |
| --- | --- |
| `sound` | A valid problem. It mates, it has no alternative solution (余詰), no attacker piece is left in hand, every piece takes part, and no piece can be removed. |
| `reducible` | It mates, but a piece can be removed and the problem still works. The reason names the piece. |
| `mate` | It passed the mate and rule checks, but the removable-piece check was skipped (`--skip-reduce`). Not a confirmed `sound` problem. |
| `rejected` | It breaks a rule, or does not mate within the bound. The reasons say why. |
| `invalid` | Not a legal tsume position. |
| `unknown` | The solver ran out of its node budget before it could decide. For example, `rule_d_unverified` means it could not prove that no piece is removable. A larger `--node-budget` may decide it. |
| `undecodable` | The model's output could not be turned into a position. |

Notes:
- Some `rejected` positions are normal. The generator is not perfect, which is why every position is judged.
- Times on one machine (CPU): under 1 s to load once the weights are cached, 3–8 s to generate 8 positions, and 8–19 s to judge them.
- **Reproducibility.** The same seed on the same kind of device gives the same output. Different devices (cpu, cuda, mps) give different positions for the same seed.
- **Search bound.** The sample judges up to 13 plies by default (`--max-ply 13`), for speed. The paper's judge went to 15 plies; `--max-ply 15` sets the same bound. That matches the bound only: the paper's evaluation numbers also depend on the samples, conditions and decoding used there, so this sample does not reproduce them by itself. With 13, a position whose only mate takes 15 plies is reported as not mating.
- **Decoding.** The defaults are the paper's decoding: `--rounds 24`, `--temperature 1.0`, `--top-p 0.95` (1.0 turns it off) and redo-check (a round that broke a hard rule is drawn again; `--no-redo-check` turns it off). `--plain` is short for `--top-p 1.0 --no-redo-check`. `--defenders` (`0`–`5` or `6+`) conditions on the defender's pieces on the board, king not counted.

## 6. Common tasks

**Generate many puzzles and save them.** Use `--json` to get one JSON object per position on stdout, then keep the `sound` ones:

```bash
python examples/generate_and_judge.py --ply 5 --count 200 --seed 7 --json > out.jsonl
```

- The script works in batches of `--batch-size` (default 256): generate, judge, write and flush each batch, then the next. Memory follows the batch size, not `--count`, and a stopped run keeps its finished batches.
- The output is deterministic for (weights, conditions, `--seed`, `--batch-size`, device type). Batch 0 uses `--seed`, later batches a seed derived from it, so a stopped run's rows are the first batches of the full run. Up to `--batch-size` diagrams, the output is the same as without batches.
- For parallel jobs, give each its own `--seed` and output file.
- Judging takes more than 90 % of the time; a GPU speeds up only generation. Measured (Ryzen 9 5950X, RTX 3070, `branch-rl-large`, `--threads 8`, batch 256): `--ply 5 --count 1024` generated in 56 s and judged in 628 s.
- GPU memory grows with `--batch-size`: 1,397 MiB at 256 and 4,573 MiB at 1,024 (`branch-rl-large`, RTX 3070 8 GB). Lower `--batch-size` if CUDA runs out of memory.
- Each `--json` row ends with `model`, `targetPly`, `seed` and `batchSize` (null with `--sfen`); keep them when sharing puzzles.
- Use `--threads N` for the solver (the default is all logical cores) and a GPU for generation. Each solver thread has its own search tables, so memory grows with the threads: lower `--threads` if memory runs short (the verdicts do not depend on it).
- `--skip-reduce` skips the removable-piece check: faster, but such positions are reported as `mate`, not `sound`.

**Judge the user's own position.** Use `--sfen`; no model is loaded:

```bash
python examples/generate_and_judge.py --sfen "5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1" --max-ply 15
```

- Positions are SFEN with the attacker (black) to move, and the attacker has no king. The defender's hand may be left out, because the defender holds every piece that is not on the board or in the attacker's hand.
- `--sfen` can be given many times, and `--json` works too.
- The README's 15-ply example (section "Example") shows the expected result and explains the three solver stages it runs.
- Do not use `tsume-solver solve-sfen` for long problems. It is an exhaustive search meant for short ones; on the README's 15-ply example it stops at its node budget with `"outcome":"UNKNOWN"`.
- To call the stages from code, reuse `imported_record`, `run_stage` and `judge` in `examples/generate_and_judge.py` rather than writing the record format by hand.

**Use the generator from Python.** Everything is in `generator/generate_min.py`; `examples/generate_and_judge.py` shows the calls:
- `fetch_weights` downloads `<model>/config.json` and `<model>/model.safetensors`; `load_model(dir, device)` returns the model and the `config.json` metadata.
- `make_condition(model, ply, band="sound", defenders=None, extra=None)` builds the condition (`band` is `sound`, `yozume`, `defect` or `none`; `defenders` is `0`-`5` or `6+`; `extra` maps further names in `EXTRA_CONDITIONS` to values).
- `generate(model, condition, count, seed, device, rounds=24, temperature=1.0, top_p=0.95, redo_check=True)` returns `{"sfen", "line"}` per diagram. The defaults are the paper's decoding: random unmasking order, the defender king on files 1–5.
- `python generator/generate_min.py --help` runs it on its own (generation only, no judging). Besides the sample's options it takes `--band` (`sound`, `yozume`, `defect` or `none`) and `--cond NAME=VALUE,...` (names and values in `EXTRA_CONDITIONS`).
- Add `generator/` to `sys.path`, as the sample does.

**Use local weights.** If the machine has no network access, download `<model>/config.json` and `<model>/model.safetensors` from Hugging Face by other means, then pass `--weights-dir DIR`.

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `solver binary not found: ...` | Build the solver (section 3), or pass `--solver PATH`, or set `TSUME_SOLVER_BIN`. |
| Illegal instruction when the solver starts | Rebuild with the portable baseline (section 2). |
| `ModuleNotFoundError` for torch and the like | The packages went into a different Python. Call the venv's interpreter by path (section 4). |
| PyTorch download of several GB | That is the CUDA build. On a machine without a GPU, install the CPU build (section 4). |
| Hugging Face HTTP errors | Retry, check the proxy, or use `--weights-dir` (section 6). |
| Many `unknown` verdicts | Raise `--node-budget` (the default is 10,000,000 nodes). |
| What a reason means (`koma_amari`, `rule_d_removable:b+P@3g`, ...) | README.md, "Reasons" (the squares are USI). |
| `CUDA out of memory` | Lower `--batch-size`. |
| Out of memory while judging | Lower `--threads` (each solver thread has its own tables) or `--batch-size`. |
| `torch.cuda.is_available()` is `False` on a GPU machine | A CPU build of PyTorch is installed. Install the CUDA build (section 4). |

## 8. License and attribution

- The code is MIT ([LICENSE](LICENSE)). The weights and the dataset are CC BY 4.0.
- If the user redistributes the weights, or publishes puzzles or data made with them, keep the credit described in [README.md](README.md) ("Where the data comes from") and cite the paper.
- Issues and pull requests are welcome on the GitHub repository. This repository is synced from the authors' development repository, so accepted fixes come back with the next sync.
