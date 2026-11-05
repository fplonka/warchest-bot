from functools import lru_cache
from itertools import combinations

import numpy as np

SNAKE = [0, 1, 1, 0, 0, 1, 1, 0]
DEALS = 8


def plan(player, seat, pool, mine, theirs, eliminated, me_first_picker):
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
    vals = np.asarray(player.values(drafts, DEALS)) * (1 if seat == 0 else -1)
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
