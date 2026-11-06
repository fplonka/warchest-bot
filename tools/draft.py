from functools import lru_cache
from itertools import combinations

import numpy as np

SNAKE = [0, 1, 1, 0, 0, 1, 1, 0]


def key(first, other):
    return sum(int(u) << (6 * i) for i, u in enumerate(sorted(first) + sorted(other)))


class Table:
    def __init__(self, path):
        z = np.load(path)
        self.keys, self.values = z["keys"], z["values"]

    def __call__(self, pairs):
        keys = np.array([key(a, b) for a, b in pairs], np.uint64)
        at = np.searchsorted(self.keys, keys)
        assert (self.keys[at] == keys).all(), "a draft is missing from the table"
        return self.values[at]


def turns(pool):
    return [("elim", k % 2) for k in range(pool - 8)] + [("pick", k) for k in SNAKE]


def value(table, pool, mine, theirs, gone, draft_first, play_first):
    avail = frozenset(pool) - set(mine) - set(theirs) - set(gone)
    todo = turns(len(pool))[len(gone) + len(mine) + len(theirs):]
    leaves = set()
    for out in combinations(sorted(avail), len(pool) - 8 - len(gone)):
        rest = avail - set(out)
        for extra in combinations(sorted(rest), 4 - len(mine)):
            leaves.add((frozenset(mine) | set(extra), frozenset(theirs) | (rest - set(extra))))
    leaves = list(leaves)
    sign = 1 if play_first else -1
    worth = dict(zip(leaves, sign * table([(m, t) if play_first else (t, m) for m, t in leaves])))

    @lru_cache(maxsize=None)
    def best(avail, m, t, i):
        if i == len(todo):
            return worth[(m, t)], None
        kind, actor = todo[i]
        ours = (actor == 0) == draft_first
        options = []
        for u in sorted(avail):
            m2, t2 = (m, t) if kind == "elim" else (m | {u}, t) if ours else (m, t | {u})
            options.append((best(avail - {u}, m2, t2, i + 1)[0], u))
        return max(options) if ours else min(options)

    return best(avail, frozenset(mine), frozenset(theirs), 0)


def play_order(table, pool, draft_first):
    return value(table, pool, [], [], [], draft_first, True)[0] >= value(table, pool, [], [], [], draft_first, False)[0]


def draft_order(table, pool):
    worst = lambda d: min(value(table, pool, [], [], [], d, f)[0] for f in (True, False))
    return worst(True) >= worst(False)
