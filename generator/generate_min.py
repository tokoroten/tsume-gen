# Copyright (c) 2026 tokoroten
# SPDX-License-Identifier: MIT
"""Minimal generator for the tsume-gen MaskGIT models (tsume shogi puzzles).

One self-contained file: the model definition, weight loading, the condition
inputs, the 24-round unmasking decode and the output as SFEN plus the first
three moves of the main line the model proposes.  It needs only torch,
safetensors and (for downloading) huggingface_hub.

    python generator/generate_min.py --model branch-rl-large --ply 5 --count 8 --seed 1
    python generator/generate_min.py --model sft-small --ply 9 --defenders 6+ --plain

Each output line is one JSON object:

    {"index": 1, "sfen": "<board> b <attacker hand> 1", "line": ["G*5b", ...]}

``sfen`` holds the board and the attacker's hand only; the defender holds
every remaining piece (strict tsume convention), which the solver fills in.
``line`` is the model's guess at the first three moves of the solution, or
``null`` when the three move slots do not decode.  Nothing here checks that
a diagram is a mate: judge the output with the Rust solver (see
``examples/generate_and_judge.py``).

Token layout of a diagram (97 slots):

* 0-80: board squares, ``rank * 9 + column`` with rank 0 = USI rank ``a`` and
  column 0 = USI file 9.  Ids 0 empty, 1-7 attacker ``P L N S G B R``, 8-13
  attacker ``+P +L +N +S +B +R``, 14-21 defender ``p l n s g b r k``, 22-27
  defender ``+p +l +n +s +b +r``.  The attacker has no king.
* 81-87: the attacker's hand counts of ``P L N S G B R``, id ``28 + count``.
* 88-96: three moves of three slots each, FROM / TO / PROMO.  FROM is
  ``48 + square`` for a board move or ``129 + piece`` for a drop (piece in
  ``P L N S G B R`` order), TO is ``136 + square``, PROMO is 217 (no) or 218
  (yes), and 219 in all three slots means "no such move".
* 47 is the input-only MASK id.

Decoding: the defender king is placed first, uniformly on files 1-5 (the
models were trained on diagrams oriented that way).  Then ``--rounds`` rounds
(default 24, i.e. 4 slots per round) fill the remaining 96 slots in a random
order; each filled slot draws its id from the model's distribution at
``--temperature`` (default 1.0).  The defaults are the paper's decoding
("R24RCP"):

* ``--top-p`` (default 0.95): nucleus filtering of every slot's
  distribution; 1.0 turns it off.
* redo-check (on by default, ``--no-redo-check`` turns it off): a row
  whose round broke a hard rule (more pieces of a type than a shogi set has,
  two unpromoted pawns of one side on a file, a piece that can never move, or
  the defender king already in check) draws that round again, up to 8 times;
  if it is still illegal, the round's slots are drawn one at a time with the
  offending ids removed (a check is not looked at in that last step).
* ``--plain``: shorthand for ``--top-p 1.0 --no-redo-check``.

Same weights, conditions, seed and device type give the same output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn

DEFAULT_REPO_ID = "tokoroten-jp/tsume-gen"
MODELS = ("sft-small", "sft-medium", "sft-large", "branch-rl-large", "branch-rl-large-ema")

# ---------------------------------------------------------------- tokens

BOARD_SLOTS = 81
DIAGRAM_SLOTS = 88            # board + attacker hand
LINE_MOVES = 3
TOTAL_SLOTS = DIAGRAM_SLOTS + 3 * LINE_MOVES
N_BOARD_IDS = 28              # empty + 27 piece ids
HAND_ID0 = 28                 # hand count n has id 28 + n (n <= 18)
N_DIAGRAM_IDS = 47
MASK_ID = 47
FROM_BASE = 48
FROM_DROP_BASE = FROM_BASE + 81
TO_BASE = FROM_DROP_BASE + 7
PROMO_NO = TO_BASE + 81
PROMO_YES = PROMO_NO + 1
NONE_ID = PROMO_NO + 2
VOCAB = NONE_ID + 1           # 220 ids
KING_ID = 21                  # the defender king

BOARD_SYMBOLS = ((None, "P", "L", "N", "S", "G", "B", "R",
                  "+P", "+L", "+N", "+S", "+B", "+R",
                  "p", "l", "n", "s", "g", "b", "r", "k",
                  "+p", "+l", "+n", "+s", "+b", "+r"))
HAND_PIECES = ("P", "L", "N", "S", "G", "B", "R")
SFEN_HAND_ORDER = ("R", "B", "G", "S", "N", "L", "P")


def slot_allowed() -> torch.Tensor:
    """``[97, 220]`` bool: the ids each slot may hold (the model's head is
    masked to these): board ids on the board, hand counts on the hand slots,
    each move slot its own range plus "no such move"."""
    allowed = torch.zeros(TOTAL_SLOTS, VOCAB, dtype=torch.bool)
    allowed[:BOARD_SLOTS, :N_BOARD_IDS] = True
    allowed[BOARD_SLOTS:DIAGRAM_SLOTS, HAND_ID0:N_DIAGRAM_IDS] = True
    ranges = ((FROM_BASE, TO_BASE), (TO_BASE, PROMO_NO), (PROMO_NO, NONE_ID))
    for i in range(3 * LINE_MOVES):
        lo, hi = ranges[i % 3]
        allowed[DIAGRAM_SLOTS + i, lo:hi] = True
        allowed[DIAGRAM_SLOTS + i, NONE_ID] = True
    return allowed


def square_name(slot: int) -> str:
    rank, column = divmod(int(slot), 9)
    return "%d%s" % (9 - column, "abcdefghi"[rank])


def diagram_sfen(ids) -> str:
    """The 88 diagram ids -> ``"<board> b <attacker hand> 1"``."""
    ranks = []
    for rank in range(9):
        out, empty = [], 0
        for column in range(9):
            symbol = BOARD_SYMBOLS[ids[rank * 9 + column]]
            if symbol is None:
                empty += 1
                continue
            if empty:
                out.append(str(empty))
                empty = 0
            out.append(symbol)
        if empty:
            out.append(str(empty))
        ranks.append("".join(out))
    counts = dict(zip(HAND_PIECES, (i - HAND_ID0 for i in ids[BOARD_SLOTS:DIAGRAM_SLOTS])))
    hand = "".join(("" if counts[p] == 1 else str(counts[p])) + p
                   for p in SFEN_HAND_ORDER if counts[p] > 0)
    return "%s b %s 1" % ("/".join(ranks), hand or "-")


def decode_move(frm: int, to: int, promo: int):
    """One move's three ids -> USI (``None`` for "no such move"); ``ValueError``
    for a mix of "no such move" and real ids."""
    nones = (frm == NONE_ID, to == NONE_ID, promo == NONE_ID)
    if all(nones):
        return None
    if any(nones):
        raise ValueError("partial move")
    plus = "+" if promo == PROMO_YES else ""
    dest = square_name(to - TO_BASE)
    if frm >= FROM_DROP_BASE:
        return "%s*%s%s" % (HAND_PIECES[frm - FROM_DROP_BASE], dest, plus)
    return "%s%s%s" % (square_name(frm - FROM_BASE), dest, plus)


def decode_line(ids):
    """The 9 move ids -> the USI moves before the first "no such move", or
    ``None`` if they do not decode (a real move after a missing one, or a
    partly missing move)."""
    out, ended = [], False
    for lo in range(0, len(ids), 3):
        try:
            move = decode_move(*ids[lo:lo + 3])
        except ValueError:
            return None
        if move is None:
            ended = True
        elif ended:
            return None
        else:
            out.append(move)
    return out


# ---------------------------------------------------------------- conditions

#: Target mate lengths and their condition ids (the model's ``cond_plies``).
#: Band ids: 0 sound (no alternative mate on the main line), 1 with such
#: alternatives ("yozume"), 2 other rule defects; 3 = unspecified.
BANDS = {"sound": 0, "yozume": 1, "defect": 2, "none": 3}

#: The extra conditions, in the model's order: (name, value names).  Each
#: also has an "unspecified" id equal to its number of values.  T1-T12 are
#: techniques on the main line, T13 the defender's board pieces (king not
#: counted), A the attacker's board pieces, H the attacker's hand pieces,
#: M the route by which the training row was made.
EXTRA_CONDITIONS = (
    ("T1", ("none", "sac", "free_sac")),   # sacrifice / free sacrifice
    ("T2", ("no", "yes")),                 # focal-point move
    ("T3", ("no", "yes")),                 # non-promotion before the last move
    ("T4", ("no", "yes")),                 # bishop / rook non-promotion
    ("T5", ("no", "yes")),                 # long-distance drop
    ("T6", ("no", "yes")),                 # interposition with a gap
    ("T7", ("no", "yes")),                 # discovered check
    ("T8", ("no", "yes")),                 # defender non-promotion
    ("T9", ("no", "yes")),                 # interposing a captured piece
    ("T10", ("no", "yes")),                # mate by lining up pieces (5 plies and up)
    ("T11", ("no", "yes")),                # piece left hanging
    ("T12", ("no", "yes")),                # dropping a captured piece
    ("T13", ("0", "1", "2", "3", "4", "5", "6+")),
    ("A", ("<=1", "2", "3", "4", "5+")),
    ("H", ("0", "1", "2", "3+")),
    ("M", ("model", "mine", "distill-yane", "distill-other", "extend")),
)


def make_condition(model, ply: int, band: str = "sound", defenders=None, extra=None) -> list:
    """The condition passed to :func:`generate`: the ``[2 + 16]`` condition
    ids of the target ply, the band, then the extra conditions (``defenders``:
    the defender's board pieces other than the king, ``0``-``5`` or ``6+``;
    ``extra``: ``{name: value name}``; the rest unspecified)."""
    extra = dict(extra or {})
    if defenders is not None:
        extra["T13"] = str(defenders)
    plies = model.config.cond_plies
    if ply not in plies:
        raise ValueError("ply %d: this model is conditioned on %s" % (ply, plies))
    names = [name for name, _ in model.config.extra_conds]
    if names != [name for name, _ in EXTRA_CONDITIONS]:
        raise ValueError("unexpected condition list %s" % names)
    condition = [plies.index(ply), BANDS[band]] + [len(v) for _, v in EXTRA_CONDITIONS]
    for name, value in extra.items():
        values = dict(EXTRA_CONDITIONS).get(name)
        if values is None:
            raise ValueError("unknown condition %r" % name)
        if str(value) not in values:
            raise ValueError("condition %s: %r is not one of %s" % (name, value, values))
        condition[2 + names.index(name)] = values.index(str(value))
    return condition


# ---------------------------------------------------------------- model

@dataclass
class ModelConfig:
    """The ``modelConfig`` of a published ``config.json``."""
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    cond_plies: tuple
    line_prefix: int
    extra_conds: tuple
    dropout: float = 0.0
    conditions: bool = True
    rel_attn_bias: bool = True
    rel_bias_mirror: bool = False
    sdpa_attention: bool = True
    adaln: str = "none"
    count_feat: bool = False
    mlp: str = "swiglu"
    domain_mask: bool = True

    @classmethod
    def from_json(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("unsupported model options %s" % unknown)
        cfg = cls(**data)
        cfg.cond_plies = tuple(int(p) for p in cfg.cond_plies)
        cfg.extra_conds = tuple((str(n), int(k)) for n, k in cfg.extra_conds)
        expected = dict(vocab_size=N_DIAGRAM_IDS, seq_len=DIAGRAM_SLOTS, line_prefix=LINE_MOVES,
                        conditions=True, rel_attn_bias=True, rel_bias_mirror=False,
                        adaln="none", count_feat=False, mlp="swiglu", domain_mask=True)
        for key, want in expected.items():
            if getattr(cfg, key) != want:
                raise ValueError("this file supports %s=%r only, got %r"
                                 % (key, want, getattr(cfg, key)))
        if cfg.extra_conds != tuple((n, len(v)) for n, v in EXTRA_CONDITIONS):
            raise ValueError("unexpected extra conditions %s" % (cfg.extra_conds,))
        return cfg


def attention_bias_index() -> torch.Tensor:
    """``[97, 97]`` ids into each layer's learned attention-bias table, per
    (query, key) slot pair: board-board pairs by their displacement
    (``(drank + 8) * 17 + (dcolumn + 8)``), then one id each for board->hand,
    hand->board, hand->hand, board->move, move->board, hand->move and
    move->hand, then one id per (query move slot, key move slot) pair."""
    squares = torch.arange(BOARD_SLOTS)
    rank, column = squares // 9, squares % 9
    dr = rank.unsqueeze(0) - rank.unsqueeze(1)
    dc = column.unsqueeze(0) - column.unsqueeze(1) + 8
    b = 17 * 17
    d, t = DIAGRAM_SLOTS, TOTAL_SLOTS
    index = torch.empty(t, t, dtype=torch.long)
    index[:81, :81] = (dr + 8) * 17 + dc
    index[:81, 81:d] = b
    index[81:d, :81] = b + 1
    index[81:d, 81:d] = b + 2
    index[:81, d:] = b + 3
    index[d:, :81] = b + 4
    index[81:d, d:] = b + 5
    index[d:, 81:d] = b + 6
    line = t - d
    index[d:, d:] = b + 7 + torch.arange(line * line).reshape(line, line)
    return index


class Attention(nn.Module):
    """Multi-head self-attention with an additive per-head bias."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h, bias):
        batch, length, width = h.shape
        qkv = nn.functional.linear(h, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(batch, length, 3, self.n_heads, width // self.n_heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    """``down(SiLU(gate) * up)``, gate and up from one stacked projection."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


class Block(nn.Module):
    """Pre-LayerNorm transformer block (bidirectional: no causal mask)."""

    def __init__(self, cfg: ModelConfig, bias_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)
        self.rel_attn = nn.Parameter(torch.zeros(cfg.n_heads, bias_size))

    def forward(self, x, bias_index):
        bias = self.rel_attn[:, bias_index].unsqueeze(0)
        x = x + self.attn(self.norm1(x), bias)
        return x + self.mlp(self.norm2(x))


class TsumeGen(nn.Module):
    """Masked-token transformer over the 97 slots.

    The input of each slot is the sum of: its token embedding, its position
    embedding, its offset from the defender king (board slots; one shared id
    for the hand slots and one for the move slots), the king's four distances
    to the board edges, and the condition embeddings."""

    REL_HAND = 17 * 17
    REL_LINE = 17 * 17 + 1

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        d = cfg.d_model
        self.token_embedding = nn.Embedding(VOCAB, d)
        self.position_embedding = nn.Embedding(TOTAL_SLOTS, d)
        self.rel_embedding = nn.Embedding(17 * 17 + 2, d)
        self.edge_embedding = nn.Embedding(9, d)
        self.ply_embedding = nn.Embedding(len(cfg.cond_plies) + 1, d)
        self.band_embedding = nn.Embedding(len(BANDS), d)
        self.extra_embeddings = nn.ModuleList(nn.Embedding(n + 1, d) for _, n in cfg.extra_conds)
        bias_size = 17 * 17 + 3 + 4 + (TOTAL_SLOTS - DIAGRAM_SLOTS) ** 2
        self.blocks = nn.ModuleList(Block(cfg, bias_size) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.register_buffer("bias_index", attention_bias_index(), persistent=False)
        self.register_buffer("allowed", slot_allowed(), persistent=False)

    def king_channels(self, king):
        """``[B]`` king squares -> ``[B, 97, D]`` king-relative embeddings."""
        kr, kc = king // 9, king % 9
        squares = torch.arange(BOARD_SLOTS, device=king.device)
        rel = (squares // 9 - kr.unsqueeze(1) + 8) * 17 + (squares % 9 - kc.unsqueeze(1) + 8)
        rest = torch.tensor([self.REL_HAND] * (DIAGRAM_SLOTS - BOARD_SLOTS)
                            + [self.REL_LINE] * (TOTAL_SLOTS - DIAGRAM_SLOTS), device=king.device)
        rel = torch.cat([rel, rest.unsqueeze(0).expand(king.shape[0], -1)], dim=1)
        edge = torch.stack([kr, 8 - kr, kc, 8 - kc], dim=1)
        return self.rel_embedding(rel) + self.edge_embedding(edge).sum(dim=1, keepdim=True)

    def forward(self, tokens, king, cond):
        """``tokens [B, 97]`` (MASK where unknown), ``king [B]``, ``cond
        [B, 18]`` -> logits ``[B, 97, 220]`` (ids a slot may not hold: -inf)."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = x + self.king_channels(king)
        added = self.ply_embedding(cond[:, 0]) + self.band_embedding(cond[:, 1])
        for i, table in enumerate(self.extra_embeddings):
            added = added + table(cond[:, 2 + i])
        x = x + added.unsqueeze(1)
        for block in self.blocks:
            x = block(x, self.bias_index)
        logits = self.head(self.norm(x))
        return logits.masked_fill(~self.allowed[:tokens.shape[1]], float("-inf"))


#: Weight-file tensors this file does not use: the head that predicts the
#: king square (here the king square is drawn uniformly instead).
UNUSED_KEYS = ("king_head.weight",)


def fetch_weights(model: str, repo_id: str = DEFAULT_REPO_ID, revision: str = "main") -> Path:
    """Download (or reuse the cached) ``<model>/config.json`` and
    ``<model>/model.safetensors`` anonymously; returns their directory."""
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(repo_id, "%s/%s" % (model, name), revision=revision, token=False)
             for name in ("config.json", "model.safetensors")]
    return Path(paths[0]).parent


def load_model(weights_dir, device="cpu"):
    """``(model, meta)`` from a directory with ``config.json`` and
    ``model.safetensors``."""
    from safetensors.torch import load_file
    weights_dir = Path(weights_dir)
    meta = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    model = TsumeGen(ModelConfig.from_json(meta["modelConfig"]))
    state = load_file(str(weights_dir / "model.safetensors"))
    for key in UNUSED_KEYS:
        state.pop(key, None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), meta


# ---------------------------------------------------------------- legality (redo-check)

#: Pieces per base type ``P L N S G B R`` in a shogi set.
SET_SIZE = (18, 4, 4, 4, 4, 2, 2)
#: Board id -> base type index (-1: empty or the defender king).
BASE_TYPE = (-1, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 5, 6,
             0, 1, 2, 3, 4, 5, 6, -1, 0, 1, 2, 3, 5, 6)
ATT_PAWN, DEF_PAWN = 1, 14
#: Attacker board id -> its one-square steps (drank, dcolumn); rank 0 is the
#: attacker's far side, so "forward" is drank -1.
_GOLD = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0))
STEPS = {1: ((-1, 0),), 4: ((-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)),
         5: _GOLD, 8: _GOLD, 9: _GOLD, 10: _GOLD, 11: _GOLD,
         12: ((-1, 0), (1, 0), (0, -1), (0, 1)), 13: ((-1, -1), (-1, 1), (1, -1), (1, 1))}
ORTHO_RAYS = (7, 13)           # R, +R
DIAG_RAYS = (6, 12)            # B, +B
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def immobile_ban(device) -> torch.Tensor:
    """``[81, 28]``: True where a piece could never move (attacker P / L on
    rank 1, N on ranks 1-2; defender p / l on rank 9, n on ranks 8-9)."""
    ban = torch.zeros(81, N_BOARD_IDS, dtype=torch.bool, device=device)
    ban[0:9, 1] = ban[0:9, 2] = True
    ban[0:18, 3] = True
    ban[72:81, 14] = ban[72:81, 15] = True
    ban[63:81, 16] = True
    return ban


def piece_counts(tokens):
    """``(per-type counts [B, 7], attacker pawns per column [B, 9], defender
    pawns per column [B, 9])`` of the filled diagram slots (MASK counts
    nothing).  The counts cover attacker board + attacker hand + defender
    board; the defender's hand is the remainder of the set."""
    board = tokens[:, :81]
    base = torch.tensor(BASE_TYPE, dtype=torch.long, device=tokens.device)
    kind = torch.where(board < N_BOARD_IDS, base[board.clamp(0, N_BOARD_IDS - 1)],
                       torch.full_like(board, -1))
    counts = nn.functional.one_hot(kind + 1, 8)[..., 1:].sum(dim=1)
    hand = tokens[:, 81:DIAGRAM_SLOTS]
    filled = (hand >= HAND_ID0) & (hand < N_DIAGRAM_IDS)
    counts = counts + torch.where(filled, hand - HAND_ID0, torch.zeros_like(hand))
    grid = board.reshape(-1, 9, 9)
    return counts, (grid == ATT_PAWN).sum(dim=1), (grid == DEF_PAWN).sum(dim=1)


def king_in_check(tokens) -> torch.Tensor:
    """``[B]`` bool: an attacker board piece already gives check to the
    defender king.  Each of the 8 directions is followed from the king to its
    first filled square (MASK blocks the look, as an unknown square); the
    piece there checks if it can move back onto the king.  The two knight
    squares are looked at too."""
    board = tokens[:, :81]
    n, dev = board.shape[0], board.device
    is_king = board == KING_ID
    one = is_king.sum(dim=1) == 1
    ks = is_king.float().argmax(dim=1)
    kr, kc = ks // 9, ks % 9
    rows = torch.arange(n, device=dev)
    out = torch.zeros(n, dtype=torch.bool, device=dev)
    for dr, dc in DIRECTIONS:
        back = (-dr, -dc)
        step_ids = [i for i, steps in STEPS.items() if back in steps]
        ray_ids = list(ORTHO_RAYS if dr == 0 or dc == 0 else DIAG_RAYS)
        if (dr, dc) == (1, 0):
            ray_ids.append(2)    # a lance below the king looks up onto it
        found = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in range(1, 9):
            r, c = kr + dr * k, kc + dc * k
            inside = (r >= 0) & (r < 9) & (c >= 0) & (c < 9)
            tok = board[rows, r.clamp(0, 8) * 9 + c.clamp(0, 8)]
            filled = inside & ~found & (tok != 0)
            hit = torch.zeros(n, dtype=torch.bool, device=dev)
            for i in ray_ids:
                hit |= tok == i
            if k == 1:
                for i in step_ids:
                    hit |= tok == i
            out |= filled & hit
            found |= filled | ~inside
    for dc in (-1, 1):
        r, c = kr + 2, kc + dc
        inside = (r < 9) & (c >= 0) & (c < 9)
        out |= inside & (board[rows, r.clamp(0, 8) * 9 + c.clamp(0, 8)] == 3)
    return out & one


def violations(tokens, immobile) -> torch.Tensor:
    """``[B]`` number of hard-rule violations of the filled slots: pieces
    over the set, extra unpromoted pawns on a file, a check on the defender
    king, immobile pieces."""
    counts, att, dfn = piece_counts(tokens)
    cap = torch.tensor(SET_SIZE, dtype=torch.long, device=tokens.device)
    out = (counts - cap).clamp_min(0).sum(dim=1)
    out = out + (att - 1).clamp_min(0).sum(dim=1) + (dfn - 1).clamp_min(0).sum(dim=1)
    out = out + king_in_check(tokens).long()
    board = tokens[:, :81]
    hit = immobile[torch.arange(81, device=tokens.device).unsqueeze(0),
                   board.clamp(0, N_BOARD_IDS - 1)] & (board < N_BOARD_IDS)
    return out + hit.sum(dim=1)


def banned_ids(tokens, immobile):
    """``(board [B, 81, 28], hand [B, 7, 19])``: ids that would break the set,
    a pawn file or immobility given the filled slots (empty and a zero hand
    count are never banned)."""
    counts, att, dfn = piece_counts(tokens)
    left = torch.tensor(SET_SIZE, dtype=torch.long, device=tokens.device) - counts
    base = torch.tensor(BASE_TYPE, dtype=torch.long, device=tokens.device)
    full = (left <= 0)[:, base.clamp_min(0)] & (base >= 0)
    board = full.unsqueeze(1).expand(-1, 81, -1).clone()
    board |= immobile.unsqueeze(0)
    board[:, :, ATT_PAWN] |= (att > 0).unsqueeze(1).expand(-1, 9, -1).reshape(-1, 81)
    board[:, :, DEF_PAWN] |= (dfn > 0).unsqueeze(1).expand(-1, 9, -1).reshape(-1, 81)
    n = torch.arange(N_DIAGRAM_IDS - N_BOARD_IDS, device=tokens.device)
    hand = n.view(1, 1, -1) > left.clamp_min(0).unsqueeze(-1)
    return board, hand


def slot_ban(ban, slot):
    """``[B, 220]`` banned ids of each row's ``slot`` (move slots: none)."""
    board, hand = ban
    rows = torch.arange(slot.shape[0], device=slot.device)
    out = torch.zeros(slot.shape[0], VOCAB, dtype=torch.bool, device=slot.device)
    out[:, :N_BOARD_IDS] = board[rows, slot.clamp(0, 80)] & (slot < 81).unsqueeze(1)
    on_hand = (slot >= 81) & (slot < DIAGRAM_SLOTS)
    out[:, N_BOARD_IDS:N_DIAGRAM_IDS] = hand[rows, (slot - 81).clamp(0, 6)] & on_hand.unsqueeze(1)
    return out


def draw(dist, generator, keep=None):
    """One id per row of ``dist [n, 220]``.  ``keep [n]``: a row with no
    probability mass left keeps that id."""
    if keep is not None:
        empty = dist.sum(dim=-1) <= 0
        if bool(empty.any()):
            dist = torch.where(empty.unsqueeze(-1), nn.functional.one_hot(
                keep.reshape(-1), dist.shape[-1]).to(dist.dtype), dist)
    dist = torch.where(dist > 0, dist.clamp_min(1e-20), dist)
    return torch.multinomial(dist, 1, generator=generator).reshape(-1)


def fill_one_by_one(tokens, chosen, order, drawn, probs, generator, immobile):
    """Fix the round's chosen slots one at a time in the round's order; a
    slot whose drawn id is banned by the slots fixed before it is drawn again
    from its own distribution without the banned ids."""
    cur = tokens.clone()
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    for j in range(order.shape[1]):
        slot = order[:, j]
        take = chosen[rows, slot]
        if not bool(take.any()):
            continue
        tok = drawn[rows, slot]
        ban = slot_ban(banned_ids(cur, immobile), slot)
        bad = take & ban[rows, tok]
        if bool(bad.any()):
            idx = bad.nonzero(as_tuple=True)[0]
            dist = probs[idx, slot[idx]].masked_fill(ban[idx], 0.0)
            tok = tok.clone()
            tok[idx] = draw(dist, generator, keep=tok[idx])
        cur[rows, slot] = torch.where(take, tok, cur[rows, slot])
    return cur


def redo_check_round(tokens, chosen, order, drawn, probs, side, immobile, tries=8):
    """The round's fill under redo-check (see the module docstring).  All
    extra draws come from ``side``, so rows the rule never touches come out
    exactly as in the plain decode."""
    before = violations(tokens, immobile)
    out = torch.where(chosen, drawn, tokens)
    bad = violations(out, immobile) > before
    for _ in range(tries):
        if not bool(bad.any()):
            break
        idx = bad.nonzero(as_tuple=True)[0]
        dist = probs[idx]
        again = draw(dist.reshape(-1, dist.shape[-1]), side).reshape(dist.shape[:2])
        cand = torch.where(chosen[idx], again, tokens[idx])
        still = violations(cand, immobile) > before[idx]
        out[idx] = torch.where(still.unsqueeze(1), out[idx], cand)
        bad[idx] = still
    if bool(bad.any()):
        idx = bad.nonzero(as_tuple=True)[0]
        board_ban, hand_ban = banned_ids(tokens[idx], immobile)
        dist = probs[idx].clone()
        dist[:, :81, :N_BOARD_IDS] = dist[:, :81, :N_BOARD_IDS].masked_fill(board_ban, 0.0)
        dist[:, 81:DIAGRAM_SLOTS, N_BOARD_IDS:N_DIAGRAM_IDS] = \
            dist[:, 81:DIAGRAM_SLOTS, N_BOARD_IDS:N_DIAGRAM_IDS].masked_fill(hand_ban, 0.0)
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        again = draw(dist.reshape(-1, dist.shape[-1]), side,
                     keep=out[idx].reshape(-1)).reshape(dist.shape[:2])
        out[idx] = fill_one_by_one(tokens[idx], chosen[idx], order[idx], again, dist, side,
                                   immobile)
    return out


def side_stream(generator) -> torch.Generator:
    """The redo-check's own random stream, seeded from (not drawn from) the
    main generator's initial state."""
    state = generator.get_state().cpu().numpy().tobytes()
    seed = int(hashlib.sha256(b"legal-decode:" + state).hexdigest()[:15], 16)
    return torch.Generator(device=generator.device).manual_seed(seed)


# ---------------------------------------------------------------- decode

def top_p_filter(probs, top_p: float):
    """Keep the most probable ids up to cumulative mass ``top_p`` (the top
    id always kept), renormalized."""
    sp, idx = probs.sort(dim=-1, descending=True)
    keep = torch.zeros_like(sp, dtype=torch.bool).scatter(-1, idx, (sp.cumsum(dim=-1) - sp) < top_p)
    out = torch.where(keep, probs, torch.zeros_like(probs))
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-30)


#: Board slots on files 1-5 (columns 4-8), where the defender king is placed.
KING_SLOTS = tuple(s for s in range(81) if s % 9 >= 4)


@torch.no_grad()
def generate(model, condition, count: int, seed: int, device="cpu", rounds: int = 24,
             temperature: float = 1.0, top_p: float = 0.95, redo_check: bool = True) -> list:
    """``count`` diagrams for one condition (:func:`make_condition`):
    ``[{"sfen", "line"}]``.  The defaults are the paper's decoding (R24RCP);
    ``top_p=1.0, redo_check=False`` is the plain decode.  Draw order: the king squares from a CPU
    generator, then per round one uniform score per slot (which slots to
    fill) and one draw per slot from a generator on ``device``, both seeded
    with ``seed``."""
    device = torch.device(device)
    king_gen = torch.Generator().manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    king = torch.tensor(KING_SLOTS)[torch.randint(0, len(KING_SLOTS), (count,),
                                                  generator=king_gen)].to(device)
    cond = torch.tensor([list(condition)] * count, device=device)
    rows = torch.arange(count, device=device)
    tokens = torch.full((count, TOTAL_SLOTS), MASK_ID, dtype=torch.long, device=device)
    tokens[rows, king] = KING_ID
    frozen = tokens != MASK_ID
    if rounds < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("need rounds >= 1, temperature > 0 and 0 < top_p <= 1")
    side = immobile = None
    if redo_check:
        side, immobile = side_stream(gen), immobile_ban(device)
    for round_ in range(rounds):
        logits = model(tokens, king, cond).float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
        open_slots = (tokens == MASK_ID) & ~frozen
        n_fill = max(1, open_slots.sum(dim=1).min().item() // max(1, rounds - round_))
        flat = probs.reshape(-1, VOCAB)
        flat = torch.where(flat > 0, flat.clamp_min(1e-20), flat)
        score = torch.rand(open_slots.shape, generator=gen, device=gen.device).to(device)
        _, order = torch.where(open_slots, score, torch.full_like(score, -1.0)).topk(n_fill, dim=1)
        pick = rows.unsqueeze(1).expand_as(order)
        chosen = torch.zeros_like(open_slots)
        chosen[pick, order] = open_slots[pick, order]
        drawn = torch.multinomial(flat, 1, generator=gen).reshape(count, TOTAL_SLOTS)
        if redo_check:
            tokens = redo_check_round(tokens, chosen, order, drawn, probs, side, immobile)
        else:
            tokens = torch.where(chosen, drawn, tokens)
    out = []
    for row in tokens.cpu().tolist():
        out.append({"sfen": diagram_sfen(row[:DIAGRAM_SLOTS]),
                    "line": decode_line(row[DIAGRAM_SLOTS:])})
    return out


# ---------------------------------------------------------------- command line

def parse_conditions(text: str) -> dict:
    """``"T1=sac,T5=yes"`` -> ``{"T1": "sac", "T5": "yes"}``."""
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            name, _, value = item.partition("=")
            out[name.strip()] = value.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="branch-rl-large", choices=MODELS)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default="main", help="HF revision (branch, tag or commit)")
    p.add_argument("--weights-dir", default=None,
                   help="local dir with model.safetensors and config.json (no download)")
    p.add_argument("--ply", type=int, default=3, help="target mate length: 1, 3, 5, 7, 9 or 11")
    p.add_argument("--band", default="sound", choices=tuple(BANDS),
                   help="verdict class asked for (default sound)")
    p.add_argument("--defenders", default=None, choices=dict(EXTRA_CONDITIONS)["T13"],
                   help="defender pieces on the board, king not counted (default unspecified)")
    p.add_argument("--cond", default="",
                   help="further conditions, e.g. 'T1=sac,H=1' (names in EXTRA_CONDITIONS)")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or mps")
    add_decoding_arguments(p)
    return p


def add_decoding_arguments(p: argparse.ArgumentParser) -> None:
    """The decoding options; the defaults are the paper's (R24RCP)."""
    p.add_argument("--rounds", type=int, default=24, help="unmasking rounds (default 24)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default 1.0)")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus filter; 1.0 = off (default 0.95)")
    p.add_argument("--no-redo-check", dest="redo_check", action="store_false",
                   help="turn off redo-check (on by default)")
    p.add_argument("--plain", action="store_true", help="same as --top-p 1.0 --no-redo-check")


def decoding_kwargs(args) -> dict:
    """``generate`` keyword arguments from :func:`add_decoding_arguments`."""
    top_p, redo = (1.0, False) if args.plain else (args.top_p, args.redo_check)
    return dict(rounds=args.rounds, temperature=args.temperature, top_p=top_p, redo_check=redo)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    weights = Path(args.weights_dir) if args.weights_dir else fetch_weights(
        args.model, args.repo_id, args.revision)
    model, _meta = load_model(weights, args.device)
    extra = parse_conditions(args.cond)
    if args.defenders is not None:
        extra["T13"] = args.defenders
    try:
        condition = make_condition(model, args.ply, args.band, extra=extra)
    except ValueError as error:
        raise SystemExit(str(error))
    try:
        results = generate(model, condition, args.count, args.seed, args.device, **decoding_kwargs(args))
    except ValueError as error:
        raise SystemExit(str(error))
    for n, res in enumerate(results, 1):
        sys.stdout.write(json.dumps(dict(index=n, **res), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
