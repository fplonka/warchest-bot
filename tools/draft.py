import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import warchest
from export_weights import read_bin
from gpu_batch import attention_layout

SNAKE = [0, 1, 1, 0, 0, 1, 1, 0]


def load(path):
    return read_bin(path).eval().cuda()


def values(net, drafts):
    rows, cc, cw, seg = [], [], [], []
    for b, (white, black, first) in enumerate(drafts):
        row, c, w, (n0, n1) = warchest.opening(list(white), list(black), first)
        rows.append(np.frombuffer(bytes(row), np.uint8))
        cc.append(np.frombuffer(bytes(c), np.uint8).reshape(-1, warchest.CCOUNTS))
        cw.append(np.asarray(w, np.float32))
        seg.append(np.repeat([2 * b, 2 * b + 1], [n0, n1]))
    rows, cc, cw, seg = np.stack(rows), np.concatenate(cc), np.concatenate(cw), np.concatenate(seg)
    x = np.asarray(warchest.expand_rows(rows.ravel()), np.float32).reshape(len(drafts), warchest.PUBFEAT)
    slot, width = attention_layout(seg // 2, len(drafts))
    gpu = lambda a: torch.from_numpy(a).cuda()
    with torch.no_grad():
        v = net(gpu(x), gpu(cc.astype(np.float32) / warchest.CNORM), gpu(cw), gpu(seg.astype(np.int64)),
                2 * len(drafts), (gpu(slot), width)).cpu().numpy()
    out = np.zeros((len(drafts), 2))
    np.add.at(out, (seg // 2, seg % 2), v * cw)
    return out


def plan(net, seat, pool, mine, theirs, eliminated, me_first_picker):
    avail = frozenset(pool) - set(mine) - set(theirs) - set(eliminated)
    eliminations = len(pool) - 8 - len(eliminated)
    turns = [("elim", (k % 2 == 0) == me_first_picker) for k in range(len(eliminated), len(pool) - 8)]
    turns += [("pick", (SNAKE[k] == 0) == me_first_picker) for k in range(len(mine) + len(theirs), 8)]

    leaves = set()
    for gone in combinations(sorted(avail), eliminations):
        rest = sorted(avail - set(gone))
        for extra in combinations(rest, 4 - len(mine)):
            leaves.add((frozenset(mine) | set(extra), frozenset(theirs) | (set(rest) - set(extra))))
    leaves = sorted(leaves, key=lambda x: (sorted(x[0]), sorted(x[1])))
    drafts = []
    for m, t in leaves:
        w, b = (sorted(m), sorted(t)) if seat == 0 else (sorted(t), sorted(m))
        drafts += [(w, b, 0), (w, b, 1)]
    vals = np.concatenate([values(net, drafts[i:i + 2048])[:, seat] for i in range(0, len(drafts), 2048)])
    table = {leaf: (vals[2 * i] + vals[2 * i + 1]) / 2 for i, leaf in enumerate(leaves)}

    @lru_cache(maxsize=None)
    def value(avail, m, t, i):
        if i == len(turns):
            return table[(m, t)], None
        kind, mine_turn = turns[i]
        best = None
        for u in sorted(avail):
            rest = avail - {u}
            if kind == "elim":
                v = value(rest, m, t, i + 1)[0]
            elif mine_turn:
                v = value(rest, m | {u}, t, i + 1)[0]
            else:
                v = value(rest, m, t | {u}, i + 1)[0]
            if best is None or (v > best[0] if mine_turn else v < best[0]):
                best = (v, u)
        return best

    v, u = value(avail, frozenset(mine), frozenset(theirs), 0)
    return u, v, turns[0]
