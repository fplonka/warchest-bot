import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import warchest
from export_weights import read_bin

SNAKE = [0, 1, 1, 0, 0, 1, 1, 0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    return read_bin(path).eval().to(DEVICE)


def values(net, drafts):
    rows = np.stack([np.frombuffer(bytes(warchest.opening(list(w), list(b), f)), np.uint8) for w, b, f in drafts])
    x = np.asarray(warchest.expand_rows(rows.ravel()), np.float32).reshape(len(drafts), warchest.PUBFEAT)
    with torch.no_grad():
        v = net.value(net.evaluate(torch.from_numpy(x).to(DEVICE))[0]).cpu().numpy()
    first = np.asarray([f for _, _, f in drafts])
    out = np.zeros((len(drafts), 2))
    out[np.arange(len(drafts)), first] = v
    out[np.arange(len(drafts)), 1 - first] = -v
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
