# tsume-gen: generating tsume shogi problems by filling the board and the solution together

[日本語](README.ja.md) · AI coding agents: see [AGENTS.md](AGENTS.md)

Code for a deep-learning generator of tsume shogi (Japanese chess mate problems) and for the exact solver that judges what it generates. This repository is the release described in §6 of the paper "盤面と正解手順を同時に生成する詰将棋の自動作問：MaskGIT と厳密ソルバによる強化学習" (Generating tsume shogi problems with the board and the solution together: MaskGIT and reinforcement learning with an exact solver), 68th Programming Symposium (IPSJ).

- The generator is a bidirectional Transformer based on MaskGIT. Starting from a fully masked input, it fills the board, the attacker's pieces in hand and the first three moves of the solution together.
- The solver is a df-pn tsume solver written in Rust. It decides whether a generated diagram is a *minimal sound problem* in the paper's sense: a forced mate with no alternative solution (yozume), no attacker pieces left over at the end, no piece that takes no part, and no piece that can be removed with the solution kept.
- No human-composed tsume problem was used, either for training or for generation.

## What §6 of the paper refers to

| §6 of the paper | where | status |
| --- | --- | --- |
| Dataset (about 33.3 million diagrams in v4, 2026-11-01; CC BY 4.0) | <https://huggingface.co/datasets/tokoroten-jp/generated-tsume-shogi> | released |
| Training data of the experiments (about 8.05 million diagrams; CC BY 4.0) | <https://huggingface.co/datasets/tokoroten-jp/tsume-gen-m17-pretraining> | released |
| Generator weights (small, medium and large after pre-training; the large one after 2,000 steps of sibling RL) | <https://huggingface.co/tokoroten-jp/tsume-gen> | released |
| Solver | `solver/` in this repository | released |
| Python sample that runs the generator | `examples/` in this repository | released |
| Training code (pre-training and sibling RL) | this repository | to be added |
| Mass-production pipeline (generation, judging, reduction, repair by adding a defender piece) | this repository | to be added |
| Problem generation in the browser | <https://tsume.quest/generate> | live (not part of this repository) |

## Contents

| Path | What it is |
| --- | --- |
| [`solver/`](solver/README.md) | The Rust tsume solver (crate `tsume-solver`): exact forced-mate search, the composition rules, and the JSON-lines stages `validate`, `solve` and `reduce` that judge a diagram. The subcommands that built diagrams of the dataset (`distill`, `extend`, `mine`) are part of the solver and are included. |
| [`generator/generate_min.py`](generator/generate_min.py) | The generator's inference code (PyTorch) in one file: the model, weight loading, the condition inputs, the unmasking decode and the output as SFEN plus the first three moves of the main line. It also runs on its own (`python generator/generate_min.py --help`). No training code. |
| [`examples/generate_and_judge.py`](examples/generate_and_judge.py) | The sample: download weights from Hugging Face, generate diagrams, judge them with the solver. |
| [`requirements.txt`](requirements.txt) | Python packages for `generator/` and the sample. |
| [`training/restore_m17.py`](training/restore_m17.py) | Rebuilds the pre-training data files from tokoroten-jp/tsume-gen-m17-pretraining, byte for byte (packages in [`requirements-training.txt`](requirements-training.txt)). |
| [`LICENSE`](LICENSE) | MIT. |

## Usage

### Solver

```bash
cd solver
cargo build --release            # target/release/tsume-solver
./target/release/tsume-solver --help
./target/release/tsume-solver solve-sfen "4k4/9/4P4/9/9/9/9/9/9 b G 1" --max-ply 3
cd ..                            # back to the repository root
```

### Generation sample

Building the solver needs Rust; `solver/rust-toolchain.toml` selects the stable toolchain (with `rustup`, `cargo` picks it up by itself). On x86_64 the build targets x86-64-v3 (AVX2, BMI2); see [solver/README.md](solver/README.md) for the stages and the rules they check.

The sample needs Python 3.12 and the packages in [requirements.txt](requirements.txt) (PyTorch, NumPy, safetensors, huggingface_hub), and the solver built as above. Run these commands from the repository root.

```bash
python3 -m venv .venv                                   # Windows: py -3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python examples/generate_and_judge.py --model branch-rl-large --ply 3 --count 8
```

The first setup downloads from rustup, crates.io and PyPI; after that, the only network access is the weight download from Hugging Face.

[`examples/generate_and_judge.py`](examples/generate_and_judge.py) downloads `<model>/model.safetensors` and `<model>/config.json` from [tokoroten-jp/tsume-gen](https://huggingface.co/tokoroten-jp/tsume-gen) (anonymously; `--weights-dir` uses local files instead), samples `--count` diagrams conditioned on the target mate length `--ply` with the decode of §4.2 and §4.4 of the paper (24 rounds, temperature 1.0, random unmasking order, illegal squares redrawn, top-p 0.95, the defender king on files 1–5), and judges every diagram with the solver: `validate`, `solve` (rules A–C, F, G) and `reduce` (rule D). It prints each diagram (SFEN) with its verdict (`sound`, `reducible`, `mate`, `rejected`, `invalid`, `unknown` or `undecodable`), its mate length and its main line on stdout, and two summary lines on stderr; `--json` prints one JSON object per diagram. `sound` means the diagram passed every rule; `mate` means it passed `solve` but rule D was not checked (`--skip-reduce`); `unknown` means the solver ran out of its node budget before deciding.

- `--model`: `sft-small`, `sft-medium`, `sft-large`, `branch-rl-large` (the default, the weights of the paper) or `branch-rl-large-ema`.
- `--ply` (1, 3, 5, 7, 9 or 11), `--count`, `--seed`, `--device` (`cpu`, `cuda`, `cuda:N` or `mps`); `--defenders` (`0`–`5` or `6+`) conditions on the number of the defender's pieces on the board, king not counted (default: unspecified).
- Decoding: `--rounds` (default 24), `--temperature` (default 1.0), `--top-p` (default 0.95; 1.0 turns it off) and `--no-redo-check` (turns off redrawing a round that broke a hard rule). `--plain` is short for `--top-p 1.0 --no-redo-check`. The defaults are the paper's decoding.
- `--solver` (or the environment variable `TSUME_SOLVER_BIN`) names the solver binary; `--skip-reduce` skips rule D; `--threads` sets the solver's threads (default: all cores).
- `--max-ply` is the solver's search bound. The default 13 is for speed; the paper judged up to 15 plies, and `--max-ply 15` sets the same bound. That matches the judge's bound only: it does not by itself reproduce the paper's evaluation numbers, which also depend on the samples, conditions and decoding used there. With 13, a diagram whose only mate takes 15 plies is reported as not mating.
- `--sfen SFEN` (repeatable) judges the given positions instead of generating; no model is loaded.
- Generation is deterministic for a seed on one device type; different devices give different diagrams for the same seed.

On a CPU, `branch-rl-large` with `--ply 3 --count 8` takes under 1 s to load (once downloaded), 7–8 s to generate and 14–19 s to judge, with a peak memory of 0.6–1.3 GB including the solver (times measured on one Windows PC, memory on Windows and macOS). The first run downloads 343 MB of weights (cached by huggingface_hub). On a GPU (`--device cuda`, with a CUDA build of PyTorch) we have not measured the memory; expect 1–2 GB. `--model sft-small` (about 30 MB) also works and is faster.

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
python examples/generate_and_judge.py --sfen "5k1+s1/4g1P2/8S/5S3/B8/9/9/9/9 b GS 1" --max-ply 15
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

- Code: MIT ([LICENSE](LICENSE)).
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

- Training code: extracting training examples from the dataset, pre-training, sibling RL.
- Mass-production pipeline: generation, solver judging, board reduction, and repair of yozume by adding a defender piece.
- The full version of the paper (arXiv).
