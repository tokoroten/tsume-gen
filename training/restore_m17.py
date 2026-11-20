"""Rebuild the M17 trainer's input files from the published parquet.

The M17 generators (``sft-small``, ``sft-medium``, ``sft-large``) were pre-trained on
three files: ``sft-m17.jsonl`` (one JSON object per diagram), ``sft-m17-fg-holdout.jsonl``
(floodgate diagrams scored during training, never trained on) and ``sft-m17-cond.npy``
(the 16 condition labels per row of ``sft-m17.jsonl``, int8).  The Hugging Face dataset
stores them as one parquet file per split; this script turns them back into the three
files, byte for byte.

    python training/restore_m17.py --parquet-dir DIR --out OUT [--check]

``DIR`` holds ``train.parquet``, ``validation.parquet``, ``test.parquet`` and
``floodgate_holdout.parquet``.  With ``--check`` the SHA-256 of each file is compared
with the published one and the script exits 1 on a mismatch.

Derived keys that the parquet leaves out are filled back in: ``pairHash`` equals
``canonicalHash``, ``source`` is ``"catalog"``, and a turned row (``mirrored``) gets
``lineage = [{"from": canonicalHash, "op": "orient-right"}]``.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MAIN_FILE = "sft-m17.jsonl"
HOLDOUT_FILE = "sft-m17-fg-holdout.jsonl"
LABELS_FILE = "sft-m17-cond.npy"
MAIN_SPLITS = ("train", "validation", "test")
N_LABELS = 16
KEYS = ("band", "canonicalHash", "floodgate", "king", "matePly", "mirrored", "originClass",
        "origins", "producedBy", "rootSourceId", "route", "sfen", "solution", "split")
PUBLISHED_SHA256 = {
    MAIN_FILE: "e03a714c6279e1ffb37426c4f3ee6998ddd6ae4d62085c72d63378408d78be3a",
    HOLDOUT_FILE: "f02244b9c7225c01482f133a183a1e5fb437ae5c14dcfb692ff47db9541b3d15",
    LABELS_FILE: "0bb48d722b63efee294dd2528f80218780059054c0b7b96a7901220f49d0e750",
}


def _rows(pdir: Path, name: str):
    pf = pq.ParquetFile(str(pdir / ("%s.parquet" % name)))
    for batch in pf.iter_batches(batch_size=100_000):
        yield from batch.to_pylist()


def to_json(rec: dict, main: bool) -> str:
    """One trainer line from one parquet row."""
    row = {k: rec[k] for k in KEYS}
    row["pairHash"], row["source"] = rec["canonicalHash"], "catalog"
    if rec["mirrored"]:
        row["lineage"] = [{"from": rec["canonicalHash"], "op": "orient-right"}]
    if main and rec["split"] == "train":
        row["drawWeight"], row["m17Cell"] = rec["drawWeight"], rec["m17Cell"]
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def restore(pdir: Path, out: Path) -> dict:
    """Write the three files into ``out`` and return their SHA-256.

    The main file's rows come back in ``row`` order by a k-way merge of the three main
    splits, each of which is stored in that order."""
    out.mkdir(parents=True, exist_ok=True)
    n_main = sum(pq.ParquetFile(str(pdir / ("%s.parquet" % n))).metadata.num_rows
                 for n in MAIN_SPLITS)
    streams = [((r["row"], k, r) for r in _rows(pdir, n)) for k, n in enumerate(MAIN_SPLITS)]
    labels = np.lib.format.open_memmap(str(out / LABELS_FILE), mode="w+", dtype=np.int8,
                                       shape=(n_main, N_LABELS))
    with open(out / MAIN_FILE, "w", encoding="utf-8", newline="\n") as fh:
        i = -1
        for i, (pos, _k, rec) in enumerate(heapq.merge(*streams)):
            if pos != i:
                raise ValueError("row %d is missing or repeated (found %d)" % (i, pos))
            fh.write(to_json(rec, True))
            labels[i] = rec["labels"]
        if i + 1 != n_main:
            raise ValueError("expected %d rows, wrote %d" % (n_main, i + 1))
    labels.flush()
    del labels
    with open(out / HOLDOUT_FILE, "w", encoding="utf-8", newline="\n") as fh:
        for rec in _rows(pdir, "floodgate_holdout"):
            fh.write(to_json(rec, False))
    return {name: sha256(out / name) for name in (MAIN_FILE, HOLDOUT_FILE, LABELS_FILE)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true",
                    help="compare the SHA-256 of each file with the published one")
    a = ap.parse_args(argv)
    sha = restore(a.parquet_dir, a.out)
    print(json.dumps(sha, indent=1))
    if a.check:
        bad = [n for n, h in sha.items() if h != PUBLISHED_SHA256[n]]
        if bad:
            print("SHA-256 mismatch: %s" % ", ".join(bad), file=sys.stderr)
            return 1
        print("all three files match the published SHA-256", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
