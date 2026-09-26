import sys
from itertools import combinations

import draft
import numpy as np

POOL = [[1], [2, 52], [3], [4], [5], [6, 53], [7], [8], [9], [10], [11], [12], [16], [17], [18, 54], [19]]
NAMES = {1: "Archer", 2: "Berserker", 3: "Cavalry", 4: "Crossbowman", 5: "Ensign", 6: "Footman", 7: "Knight",
         8: "Lancer", 9: "Light Cavalry", 10: "Marshal", 11: "Mercenary", 12: "Pikeman", 16: "Scout",
         17: "Swordsman", 18: "Warrior Priest", 19: "Royal Guard", 52: "Berserker V2", 53: "Footman V2",
         54: "Warrior Priest V2"}
UNITS = sorted(NAMES)
PAIRS = list(combinations(UNITS, 2))
PAIRS_AT = list(combinations(range(len(UNITS)), 2))


def deal(rng):
    slots = rng.permutation(len(POOL))[:8]
    units = [int(rng.choice(POOL[s])) for s in slots]
    return sorted(units[:4]), sorted(units[4:]), int(rng.integers(2))


def features(drafts):
    side = lambda k: np.array([[u in d[k] for u in UNITS] for d in drafts], np.float64)
    w, b = side(0), side(1)
    i, j = np.array(PAIRS_AT).T
    first = np.array([[1 - 2 * d[2]] for d in drafts], np.float64)
    return np.hstack([first, w - b, w[:, i] * w[:, j] - b[:, i] * b[:, j], w[:, i] * b[:, j] - b[:, i] * w[:, j]])


def fit(x, y, ridge=1.0):
    live = np.abs(x).sum(0) > 0
    coef = np.zeros(x.shape[1])
    a = x[:, live]
    coef[live] = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y)
    return coef, 1 - ((y - x @ coef) ** 2).mean() / y.var()


def main(weights, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    drafts = [deal(rng) for _ in range(n)]
    net = draft.load(weights)
    v = np.concatenate([draft.values(net, drafts[i:i + 256]) for i in range(0, n, 256)])
    y = 50 * (v[:, 0] - v[:, 1]) / 2
    x = features(drafts)
    k = 1 + len(UNITS)
    for name, cols in [("first mover + units", k), ("+ own pairs", k + len(PAIRS)), ("+ matchups", x.shape[1])]:
        print(f"R² {name}: {fit(x[:, :cols], y)[1]:.3f}")
    coef = fit(x, y)[0]
    unit, pair, match = coef[1:k], coef[k:k + len(PAIRS)], coef[k + len(PAIRS):]
    print(f"\nfirst mover: {coef[0]:+.2f} pts of win% for white when white starts")
    print("\nunit value (win% pts, vs average unit):")
    for i in np.argsort(-unit):
        print(f"  {NAMES[UNITS[i]]:18} {unit[i] - unit.mean():+6.2f}")
    print("\nown pairs, best and worst:")
    for i in [*np.argsort(-pair)[:10], *np.argsort(pair)[:10]]:
        u, v = PAIRS[i]
        print(f"  {NAMES[u]:18} + {NAMES[v]:18} {pair[i]:+6.2f}")
    print("\nmatchups, strongest counters (row beats column):")
    for i in np.argsort(-np.abs(match))[:20]:
        u, v = PAIRS[i] if match[i] > 0 else PAIRS[i][::-1]
        print(f"  {NAMES[u]:18} > {NAMES[v]:18} {abs(match[i]):6.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
