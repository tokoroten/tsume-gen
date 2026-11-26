# tsume-gen: generating tsume shogi problems by filling the board and the solution together

[日本語](README.ja.md) · AI coding agents: see [AGENTS.md](AGENTS.md)

Code for a deep-learning generator of tsume shogi (Japanese chess mate problems) and for the exact solver that judges what it generates. This repository is the release described in §6 of the paper "盤面と正解手順を同時に生成する詰将棋の自動作問：MaskGIT と厳密ソルバによる強化学習" (Automatic Tsume-Shogi Composition by Jointly Generating the Board and the Solution: Reinforcement Learning with MaskGIT and an Exact Solver), 68th Programming Symposium (IPSJ).

- The generator is a bidirectional Transformer based on MaskGIT. Starting from a fully masked input, it fills the board, the attacker's pieces in hand and the first three moves of the solution together.
- The solver is a df-pn tsume solver written in Rust. It decides whether a generated diagram is a *minimal sound problem* in the paper's sense: a forced mate with no alternative solution (yozume), no attacker pieces left over at the end, no piece that takes no part, and no piece that can be removed with the solution kept.
- No human-composed tsume problem was used, either for training or for generation.

## What §6 of the paper refers to

| §6 of the paper | where | status |
| --- | --- | --- |
| Dataset (about 75.27 million diagrams in v5, 2026-11-26, with the solution trees of about 28.5 million; CC BY 4.0) | <https://huggingface.co/datasets/tokoroten-jp/generated-tsume-shogi> | released |
| Training data of the experiments (about 8.05 million diagrams; CC BY 4.0) | <https://huggingface.co/datasets/tokoroten-jp/tsume-gen-m17-pretraining> | released |
| Generator weights (small, medium and large after pre-training; the large one after 2,000 steps of RL with the solver's verdict as the reward) | <https://huggingface.co/tokoroten-jp/tsume-gen> | released |
| Solver | `solver/` in this repository | released |
| Python sample that runs the generator | `examples/` in this repository | released |
| Training code (pre-training and RL) | this repository | to be added |
| Mass-production pipeline (generation, judging, reduction, repair by adding a defender piece) | this repository | to be added |
| The paper itself (PDF and figures) | [`paper/`](paper/README.md) in this repository | released |
| Problem generation in the browser | <https://tsume.quest/generate> | live (not part of this repository) |

## Contents

| Path | What it is |
| --- | --- |
| [`solver/`](solver/README.md) | The Rust tsume solver (crate `tsume-solver`): exact forced-mate search, the composition rules, and the JSON-lines stages `validate`, `solve` and `reduce` that judge a diagram. The subcommands that built diagrams of the dataset (`distill`, `extend`, `mine`) are part of the solver and are included. |
| [`generator/generate_min.py`](generator/generate_min.py) | The generator's inference code (PyTorch) in one file: the model, weight loading, the condition inputs, the unmasking decode and the output as SFEN plus the first three moves of the main line. It also runs on its own (`python generator/generate_min.py --help`). No training code. |
| [`examples/generate_and_judge.py`](examples/generate_and_judge.py) | The sample: download weights from Hugging Face, generate diagrams, judge them with the solver. |
| [`paper/`](paper/README.md) | The paper (PDF, Japanese) and its figures. Copyright IPSJ; not under the MIT License. |
| [`requirements.txt`](requirements.txt) | Python packages for `generator/` and the sample. |
| [`training/restore_m17.py`](training/restore_m17.py) | Rebuilds the pre-training data files from tokoroten-jp/tsume-gen-m17-pretraining, byte for byte (packages in [`requirements-training.txt`](requirements-training.txt)). |
| [`LICENSE`](LICENSE) | MIT. |

## Usage

Run every command from the repository root unless it says otherwise.

### Setup

**Rust** (for the solver). Install it with [rustup](https://rustup.rs/). `solver/rust-toolchain.toml` asks for the `stable` channel (not a fixed version) with `rustfmt` and `clippy`, which is what a default rustup install has; `cargo` run inside `solver/` uses it.

- Windows: Rust's default toolchain there (`x86_64-pc-windows-msvc`) links with Microsoft's C++ build tools. Install the Visual Studio Build Tools with the "Desktop development with C++" workload (the rustup installer offers to install them when they are missing).
- Linux: a C linker (`cc`, e.g. from `build-essential` or `gcc`).
- macOS: the Xcode command line tools (`xcode-select --install`).

**Python** 3.12 and a virtual environment. The commands below call the venv's interpreter by its path, so no activation is needed: `.venv/bin/python` on Linux and macOS, `.venv\Scripts\python` on Windows.

```bash
python3 -m venv .venv                                   # Windows: py -3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python -m pip install -r requirements.txt
```

**PyTorch, CPU or GPU.** Everything here runs on the CPU (`--device cpu` is the default; the times below are CPU times), so the GPU build is optional. Which build `pip install -r requirements.txt` gives depends on the platform:

- Windows: PyPI's `torch` is the CPU build (its version ends in `+cpu`). For an NVIDIA GPU, install the CUDA build from PyTorch's own index first, then `requirements.txt` (pip keeps the installed `torch`):
  `.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`
  The CUDA build is large: on one Windows PC (cu128) the wheel was 2,753 MB, the venv about 4.4 GB and pip's cache about 2.6 GB; add `--no-cache-dir` to keep no cache.
  Pick the `cu` number that your NVIDIA driver supports (`nvidia-smi` shows the highest CUDA version the driver supports; <https://pytorch.org/get-started/locally/> lists the builds). If the `+cpu` build is already installed, remove it first (`.venv\Scripts\python -m pip uninstall torch`); otherwise pip sees `torch` as installed and changes nothing.
- Linux (x86_64): PyPI's `torch` already includes CUDA, and its download is several GB. Without a GPU, install the CPU build first: `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- macOS (Apple silicon): the default `torch` works; use `--device mps` for the GPU.

Check which build you have (on Windows, `.venv\Scripts\python`):

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` means `--device cuda` will work; `False` with a version ending in `+cpu` is the CPU build.

**Weights.** The sample downloads the weights from Hugging Face on first use and keeps them in huggingface_hub's cache: `~/.cache/huggingface/hub` (Windows: `%USERPROFILE%\.cache\huggingface\hub`). Set `HF_HOME` to move the cache (it then lives in `$HF_HOME/hub`), or download the files yourself and pass `--weights-dir`. Without a Hugging Face token, huggingface_hub prints a warning about "unauthenticated requests" on each download; it is harmless, and no token or account is needed.

The first setup downloads from rustup, crates.io and PyPI (and download.pytorch.org for the CPU or CUDA build); after that, the only network access is the weight download from Hugging Face.

### Solver

```bash
cd solver
cargo build --release            # target/release/tsume-solver (Windows: tsume-solver.exe)
./target/release/tsume-solver --help
./target/release/tsume-solver solve-sfen "4k4/9/4P4/9/9/9/9/9/9 b G 1" --max-ply 3
cd ..                            # back to the repository root
```

On x86_64 the build targets x86-64-v3 (AVX2, BMI2; `solver/.cargo/config.toml`, read only when cargo runs inside `solver/`). On a CPU without them the solver stops with an illegal-instruction error; rebuild with `RUSTFLAGS="-C target-cpu=x86-64" cargo build --release` (PowerShell: `$env:RUSTFLAGS="-C target-cpu=x86-64"; cargo build --release`), which gives the same output, a little slower. See [solver/README.md](solver/README.md) for the stages and the rules they check.

**Threads and memory.** The stages `validate`, `solve` and `reduce` take `--threads N`; the default is every logical core. Each thread judges one record at a time with its own search tables, which grow with the nodes the search visits (up to `--node-budget`), so the solver's peak memory grows with the number of threads. If the machine runs short of memory, lower `--threads` (the sample passes its own `--threads` on to the solver). The verdicts and the output order do not depend on the number of threads.

### Generation sample

```bash
.venv/bin/python examples/generate_and_judge.py --model branch-rl-large --ply 3 --count 8
# Windows: .venv\Scripts\python examples/generate_and_judge.py --model branch-rl-large --ply 3 --count 8
```

[`examples/generate_and_judge.py`](examples/generate_and_judge.py) downloads `<model>/model.safetensors` and `<model>/config.json` from [tokoroten-jp/tsume-gen](https://huggingface.co/tokoroten-jp/tsume-gen) (anonymously; `--weights-dir` uses local files instead), samples `--count` diagrams conditioned on the target mate length `--ply` with the decode of §4.2 and §4.4 of the paper (24 rounds, temperature 1.0, random unmasking order, illegal squares redrawn, top-p 0.95, the defender king on files 1–5), and judges every diagram with the solver: `validate`, `solve` (rules A–C, F, G) and `reduce` (rule D). It prints each diagram (SFEN) with its verdict (`sound`, `reducible`, `mate`, `rejected`, `invalid`, `unknown` or `undecodable`), its mate length and its main line on stdout, and two summary lines on stderr; `--json` prints one JSON object per diagram. `sound` means the diagram passed every rule; `mate` means it passed `solve` but rule D was not checked (`--skip-reduce`); `unknown` means the solver ran out of its node budget before deciding.

**Reasons.** A verdict other than `sound` comes with `reasons` (stdout: `reasons:`; `--json`: the `reasons` list). The common ones:

| Reason | Verdict | Meaning |
| --- | --- | --- |
| `no_forced_mate_within_bound` | `rejected` | No forced mate within `--max-ply`. |
| `defense_coverage_incomplete` | `rejected` | The search could not show a mate against every defence; it comes with `no_forced_mate_within_bound` and similar. |
| `first_move_not_unique` | `rejected` | More than one first move mates in the minimum number of moves (rule B, at the first move). |
| `yozume_in_main_line` | `rejected` | Later on the main line an attacker move has an alternative (余詰, rule B). |
| `koma_amari` | `rejected` | An attacker piece is left in hand at the end (駒余り, rule C). |
| `inert_attacker_piece`, `inert_defender_piece` | `rejected` | A piece on the board takes no part (rule G). |
| `rule_d_removable:<side><piece>@<square>` | `reducible` | A piece can be removed and the problem still works (rule D), e.g. `rule_d_removable:b+P@3g` (black's promoted pawn on 3g). Several are separated by commas. |
| `solver_budget_exhausted`, `incomplete_response_graph`, `minimum_length_not_proven` | `unknown` | The search ran out of `--node-budget` before it could decide. |
| `rule_d_unverified` | `unknown` | The removal check ran out of budget: no removable piece was found, but that is not proven. |
| `material`, `nifu`, `dead_piece`, `defender_king`, `defender_already_checked` | `invalid` | Not a legal tsume position: more pieces of a type than a set has, two unpromoted pawns on a file, a piece that can never move, not exactly one defender king, the defender already in check. |

Squares in the reasons are USI squares (file 1–9, rank `a`–`i`). In the solver's own records (the `reduce` stage's `reduction.removed`), a square is a number 0–80 instead: `rank * 9 + column`, with rank 0 = `a` and column 0 = file 9, so square `n` is file `9 - n % 9`, rank `"abcdefghi"[n // 9]` (60 is `3g`, 0 is `9a`, 80 is `1i`).

**`line` and the main line.** `line` is the model's own guess at the first three moves, made together with the diagram; `mainLine` is the solver's designated main line. They often differ, even for a `sound` diagram. For example (run with `--ply 5 --seed 7 --json` on a GPU), the model proposed `4a3a 3b3a 2e3c`, while the solver's main line plays `G*1b` as the third move and `2e3c+` only as the fifth:

```
{"sfen": "5Ggk1/5Sg2/8S/9/7N1/9/9/9/9 b - 1", "line": ["4a3a", "3b3a", "2e3c"], "verdict": "sound", "reasons": [], "matePly": 5, "mainLine": ["4a3a", "3b3a", "G*1b", "2a3b", "2e3c+"], "normalizedSfen": "5Ggk1/5Sg2/8S/9/7N1/9/9/9/9 b 2r2bg2s3n4l18p 1", "index": 19, "model": "branch-rl-large", "targetPly": 5, "seed": 7, "batchSize": 256}
```

Each `--json` row ends with where it came from: `model` (the name in the weights' `config.json`), `targetPly` (`--ply`), `seed` (`--seed`) and `batchSize` (`--batch-size`; the diagrams depend on it, see the next section). With `--sfen` these four are `null`. They are in every `--json` row so that the output of several runs can be mixed and still traced, and so that shared diagrams keep their origin (see "Where the data comes from"). The plain text output has none of them.

- `--model`: `sft-small`, `sft-medium`, `sft-large`, `branch-rl-large` (the default, the weights of the paper) or `branch-rl-large-ema`; also the comparison runs on the model card, `rl-indep-large-s0`, `rl-indep-large-s1`, `rl-branchonly-large-s0` and `rl-branchonly-large-s1`, each also with `-ema`. For weights in another Hugging Face repository use `--repo-id` (and `--revision`); for files on disk, `--weights-dir DIR` (a directory with `config.json` and `model.safetensors`; nothing is downloaded).
- `--ply` (1, 3, 5, 7, 9 or 11), `--count`, `--seed`, `--device` (`cpu`, `cuda`, `cuda:N` or `mps`); `--defenders` (`0`–`5` or `6+`) conditions on the number of the defender's pieces on the board, king not counted (default: unspecified).
- Decoding: `--rounds` (default 24), `--temperature` (default 1.0), `--top-p` (default 0.95; 1.0 turns it off) and `--no-redo-check` (turns off redrawing a round that broke a hard rule). `--plain` is short for `--top-p 1.0 --no-redo-check`. The defaults are the paper's decoding. With redo-check, a diagram never has more pieces of a type than a shogi set, two unpromoted pawns of one side on a file, or a piece that can never move (a check on the defender king can still slip through its last step); without it, such diagrams are reported as `invalid`.
- `--solver` (or the environment variable `TSUME_SOLVER_BIN`) names the solver binary; `--skip-reduce` skips rule D; `--threads` sets the solver's threads (default: all cores; see "Threads and memory").
- `--max-ply` is the solver's search bound. The default 13 is for speed; the paper judged up to 15 plies, and `--max-ply 15` sets the same bound. That matches the judge's bound only: it does not by itself reproduce the paper's evaluation numbers, which also depend on the samples, conditions and decoding used there. With 13, a diagram whose only mate takes 15 plies is reported as not mating.
- `--sfen SFEN` (repeatable) judges the given positions instead of generating; no model is loaded.
- `--batch-size` (default 256): see the next section.
- Generation is deterministic for a seed on one device type; different devices give different diagrams for the same seed.

On a CPU, `branch-rl-large` with `--ply 3 --count 8` takes under 1 s to load (once downloaded), 7–8 s to generate and 14–19 s to judge, with a peak memory of 0.6–1.3 GB including the solver (times measured on one Windows PC, memory on Windows and macOS). The first run downloads 343 MB of weights (cached, see "Weights" above). On a GPU (`--device cuda`, with a CUDA build of PyTorch) we have not measured the memory; expect 1–2 GB. `--model sft-small` (about 30 MB) also works and is faster.

### Generating many puzzles

Use `--json` and redirect stdout to a file; each line is one diagram with its verdict:

```bash
.venv/bin/python examples/generate_and_judge.py --ply 5 --count 10000 --seed 7 --json > out.jsonl
# Windows: .venv\Scripts\python examples/generate_and_judge.py --ply 5 --count 10000 --seed 7 --json > out.jsonl
```

- The script works in batches of `--batch-size` diagrams (default 256): it generates a batch, judges it, writes its rows, flushes the output, and goes on with the next batch. The memory depends on the batch size, not on `--count`, and a run that stops early keeps every batch it finished. With more than one batch, stderr gets one progress line per batch.
- The diagrams depend on the weights, the conditions, `--seed`, `--batch-size` and the device type. Batch 0 uses `--seed` itself, and each later batch a seed derived from `--seed` and its number, so the rows of a stopped run are the first complete batches of the full run. With `--count` at most `--batch-size`, the output is the same as before batches existed; another `--batch-size` gives other diagrams once `--count` is larger than it.
- To run several jobs side by side, give each its own `--seed` and output file. Mind memory: each job runs its own solver, which uses every core unless `--threads` says otherwise.
- Time: judging takes more than 90 % of a run, and a GPU speeds up only the generation. Measured on one PC (Ryzen 9 5950X, RTX 3070, `branch-rl-large`, `--device cuda`, `--threads 8`, `--batch-size 256`): `--ply 5 --count 1024` took 56 s to generate and 628 s to judge; `--ply 7 --count 512`, 28 s and 215 s. More solver threads (`--threads`, mind the memory) shorten the judging; long target mates take longer.
- GPU memory grows with `--batch-size`. Measured with `branch-rl-large` on an RTX 3070 (8 GB), peak `torch.cuda.max_memory_allocated`: 1,397 MiB at 256 diagrams per batch and 4,573 MiB at 1,024. On an 8 GB card that also drives the display, a batch of about 1,800 would likely not fit (not tried). If the GPU runs out of memory, lower `--batch-size`.
- Keep the rows whose `verdict` is `sound` (the ones that passed every rule). Judging dominates the run time for long mates, because the solver proves non-mates exhaustively; `--skip-reduce` is faster but reports such diagrams as `mate`, not `sound`.

### The generator on its own

`generator/generate_min.py` generates without judging and prints one JSON object per diagram (`{"index", "sfen", "line"}`):

```bash
.venv/bin/python generator/generate_min.py --model branch-rl-large --ply 5 --count 8 --seed 1
.venv/bin/python generator/generate_min.py --ply 7 --band sound --cond "T1=sac,H=1"
```

It takes the sample's `--model`, `--repo-id`, `--revision`, `--weights-dir`, `--ply`, `--defenders`, `--count`, `--seed`, `--device` and decoding options, and two more conditions:

- `--band` (default `sound`): the verdict class asked for: `sound` (no alternative mate on the main line), `yozume` (with such alternatives), `defect` (other rule defects) or `none` (unspecified).
- `--cond NAME=VALUE,...`: further conditions, listed in `EXTRA_CONDITIONS` in the file: `T1`–`T12` are techniques on the main line (for example `T1` = `none`, `sac` or `free_sac`; most others `no` / `yes`), `T13` the defender's board pieces (what `--defenders` sets), `A` the attacker's board pieces, `H` the attacker's pieces in hand, and `M` the route by which the training row was made. A name left out is unspecified.

The output is not judged; pass the SFENs to the sample with `--sfen`, or to the solver stages.

## Example: a 15-move problem and how the solver solves it

This problem was made by one of the project's generators and passed the solver's judge. It was also published as a puzzle on <https://tsume.quest>. The attacker (black, ▲) moves first and holds a gold and a silver; the defender (white, △) holds every other piece.

```
後手の持駒：飛二 角 金二 桂四 香四 歩十七
  ９ ８ ７ ６ ５ ４ ３ ２ １
+---------------------------+
| ・ ・ ・ ・ ・v玉 ・v全 ・|一
| ・ ・ ・ ・v金 ・ 歩 ・ ・|二
| ・ ・ ・ ・ ・ ・ ・ ・ 銀|三
| ・ ・ ・ ・ ・ 銀 ・ ・ ・|四
| 角 ・ ・ ・ ・ ・ ・ ・ ・|五
| ・ ・ ・ ・ ・ ・ ・ ・ ・|六
| ・ ・ ・ ・ ・ ・ ・ ・ ・|七
| ・ ・ ・ ・ ・ ・ ・ ・ ・|八
| ・ ・ ・ ・ ・ ・ ・ ・ ・|九
+---------------------------+
先手の持駒：金 銀
```

SFEN: `5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1` (the defender's hand may be left out; the solver gives the defender every remaining piece).

Solution (15 plies):

▲3一金 △同成銀 ▲同歩成 △同玉 ▲2二銀打 △4一玉 ▲3二銀打 △同玉 ▲3三銀成 △4一玉 ▲3一銀成 △同玉 ▲2二銀成 △4一玉 ▲3二成銀寄 まで15手

In USI: `G*3a 2a3a 3b3a+ 4a3a S*2b 3a4a S*3b 4a3b 4d3c+ 3b4a 2b3a+ 4a3a 1c2b+ 3a4a 2b3b`

### Judging it

```bash
.venv/bin/python examples/generate_and_judge.py --sfen "5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1" --max-ply 15
# Windows: .venv\Scripts\python examples/generate_and_judge.py --sfen ...
```

```
 1 sound       mate in 15  5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1
   main line: G*3a 2a3a 3b3a+ 4a3a S*2b 3a4a S*3b 4a3b 4d3c+ 3b4a 2b3a+ 4a3a 1c2b+ 3a4a 2b3b
```

`--sfen` (repeatable) judges the given positions instead of generating, so no model is downloaded. `--max-ply 15` is the paper's bound; the sample's default of 13 would report this problem as not mating within the bound. On one PC (Ryzen 9 5950X) the whole judge took about 4 s. It runs three solver stages, which read and write one JSON record per line:

1. **`validate`** checks that the position is a strict tsume position (for example, no attacker king) and fills in the defender's hand.
2. **`solve --max-ply 15`** first proves the forced mate with a df-pn search (the proof tree has 463 nodes). It then checks, within 15 plies, the rules that make a problem valid: the mate length is exact, no alternative mate on the main line (余詰), no attacker piece left in hand at the end (駒余り), every piece takes part, and futile interpositions (無駄合) are not counted. It writes the designated main line shown above. About 1.2 million nodes, 0.8 s.
3. **`reduce --max-ply 15`** tries removing each piece and re-solving, and finds that none can go (`reduction.minimality: verified`). About 3 s.

Plain `tsume-solver solve-sfen "<SFEN>" --max-ply 15` does an exhaustive search of every line and is meant for short problems. On this problem it stops at its default budget of 100,000 nodes (`--node-budget`) with `"outcome":"UNKNOWN"`. For long problems, use the stages, as the sample does.

## Where the data comes from

The generator's training examples were drawn from the database of diagrams that we publish as the dataset tokoroten-jp/generated-tsume-shogi (<https://huggingface.co/datasets/tokoroten-jp/generated-tsume-shogi>), as it stood on 2026-11-03. That is a later state than the dataset's v4 (2026-11-01): about 82 % of the training rows are in v4, and the rest were added to the database between the two dates. The training data is a selected subset of that database, not the published dataset itself. It is published as it was used, as tokoroten-jp/tsume-gen-m17-pretraining (<https://huggingface.co/datasets/tokoroten-jp/tsume-gen-m17-pretraining>); `training/restore_m17.py` rebuilds the trainer's input files from it byte for byte. The database contains diagrams derived from two third-party sources:

- **YaneuraOu practical mate positions** (やねうら王 実戦詰み局面集, 2020): positions from self-play games of the shogi engine YaneuraOu in which a forced mate of 3 to 11 plies was found. Its page, <https://yaneuraou.yaneu.com/2020/12/25/christmas-present/>, states 「著作権は主張しませんのでご自由にお使いください。」 ("We do not assert copyright; please feel free to use it"). We removed the attacker's king and unneeded pieces from these positions (distillation), and extended some of the results backwards to longer mates.
- **floodgate** (<https://wdoor.c.u-tokyo.ac.jp/shogi/>): game records of the online computer-shogi server. We distilled diagrams from positions near the end of the games. floodgate states no copyright terms for its game records, so please use the diagrams derived from them with care.

The other training diagrams were found by random placement or generated by earlier generators, and all of them were judged by the solver. The dataset records the origin of every row. Diagrams that the generator produces inherit these origins through its training data.

## License

- Code: MIT ([LICENSE](LICENSE)), except `paper/`.
- `paper/`: the paper and its figures; the copyright is retained by IPSJ (see [paper/README.md](paper/README.md)).
- Generator weights and dataset: CC BY 4.0 (see their Hugging Face pages).
- CC BY 4.0 is our license for the weights and the dataset we made. It is not a license of the upstream sources: the YaneuraOu collection is used under the author's statement quoted above, and floodgate states no copyright terms (see above).

## Citation

If you use this repository, the weights or the dataset, please cite the paper:

```bibtex
@inproceedings{nakayama2027tsume,
  author    = {Shinta Nakayama},
  title     = {盤面と正解手順を同時に生成する詰将棋の自動作問：{MaskGIT} と厳密ソルバによる強化学習},
  booktitle = {Proceedings of the 68th Programming Symposium (IPSJ)},
  year      = {2027}
}
```

If you use only the dataset, you may instead cite it as in the CITATION.cff on the dataset page.

## Planned

- Training code: extracting training examples from the dataset, pre-training, RL.
- Mass-production pipeline: generation, solver judging, board reduction, and repair of yozume by adding a defender piece.
- The full version of the paper (arXiv).
