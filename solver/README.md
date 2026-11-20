# tsume-solver

A strict tsume-shogi (詰将棋) solver and the JSONL stages that judge a diagram
against the orthodox composition rules.  Rust, no dependencies outside crates.io.

```bash
cd solver
cargo build --release            # target/release/tsume-solver
./target/release/tsume-solver --help
./target/release/tsume-solver solve-sfen "4k4/9/4P4/9/9/9/9/9/9 b G 1" --max-ply 3
```

`.cargo/config.toml` builds x86_64 for x86-64-v3 (AVX2, BMI2: Haswell, Zen or
later).  A build with `RUSTFLAGS` set, or with `--manifest-path` from another
directory, is the plain x86-64 baseline: the same output, a little slower.

## Judging a diagram

The stages read and write one JSON record per line (stdin to stdout, logs on
stderr); `examples/generate_and_judge.py` shows the record shape.

| Stage | What it does |
| --- | --- |
| `validate` | strict tsume structure; the defender holds every piece not on the board or in the attacker's hand; canonical hash |
| `solve --max-ply P --node-budget N` | exact forced-mate search and the composition rules below; writes the designated main line |
| `reduce --max-ply P --node-budget N` | rule D: tries to remove pieces; `reduction.minimality: verified` and no `removed` piece = minimal |

Rules (the attacker is black and moves first):

- **A** no attacker king on the board;
- **B** no alternative mate (余詰) on the main line;
- **C** no attacker piece left in hand at the end (駒余り);
- **D** no piece can be removed with the puzzle still sound (minimality);
- **F** futile interpositions (無駄合) are not counted, by the classical convention;
- **G** every piece on the board takes part.

## Provenance

Every record carries `provenance` with `originClass` (`own_authored`,
`own_generated` or `third_party`), a non-empty `origins` list of dataset ids
(for example `own` or `yaneuraou-2020-12-25`) and `rootSourceId`.  `validate`
rejects a record without them (`missing_origin`); diagrams made from a record
(by `distill` or `extend`) inherit its origins.  `distill` and `extend`
also read bare SFEN lines; those need `--origin <dataset-id>` and
`--origin-class <class>`.  When you share diagrams, keep their origins (see
the README, "Where the data comes from").

`solve-sfen <sfen>` and `movegen <sfen>` are the one-position tools.  Output
order never depends on `--threads`.

*Draft (2026-11-13): the full documentation of the rules and the stages comes
with a later release.*
