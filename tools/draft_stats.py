import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {1: "Archer", 2: "Berserker", 3: "Cavalry", 4: "Crossbowman", 5: "Ensign", 6: "Footman", 7: "Knight",
         8: "Lancer", 9: "Light Cavalry", 10: "Marshal", 11: "Mercenary", 12: "Pikeman", 16: "Scout",
         17: "Swordsman", 18: "Warrior Priest", 19: "Royal Guard", 52: "Berserker II", 53: "Footman II",
         54: "Warrior Priest II"}
UNITS = sorted(NAMES)


def sides(keys):
    index = np.full(64, -1)
    index[UNITS] = np.arange(len(UNITS))
    units = index[((keys[:, None] >> np.arange(0, 48, 6, dtype=np.uint64)) & np.uint64(63)).astype(np.int64)]
    rows = np.arange(len(keys))[:, None]
    first, other = (np.zeros((len(keys), len(UNITS)), np.float32) for _ in range(2))
    first[rows, units[:, :4]] = 1
    other[rows, units[:, 4:]] = 1
    return first, other


def mean_residual(*parts):
    total = sum(a.T @ (b * r[:, None]) for a, b, r in parts)
    count = sum(a.T @ b for a, b, _ in parts)
    return total / np.where(count > 0, count, np.nan)


def main(table, plot):
    z = np.load(table)
    v = 50 * z["values"].astype(np.float32)
    first, other = sides(z["keys"])
    x = np.hstack([first, -other])
    fit = np.linalg.lstsq(x.T @ x, x.T @ v, rcond=None)[0]
    r = v - x @ fit
    p, q = fit[:19], fit[19:]
    lead, p, q = 4 * (p.mean() - q.mean()), p - p.mean(), q - q.mean()
    synergy = mean_residual((first, first, r), (other, other, -r))
    counter = mean_residual((first, other, r), (other, first, -r))
    np.fill_diagonal(synergy, np.nan)

    strength = (p + q) / 2
    order = np.argsort(-strength)
    names = [NAMES[UNITS[i]] for i in order]
    print(f"{len(v)} drafts; first mover {lead:+.1f} points; additive fit explains {1 - r.var() / v.var():.0%} of the variance")
    for i in order:
        print(f"{NAMES[UNITS[i]]:>18} {strength[i]:+5.1f}  first {p[i]:+5.1f}  second {q[i]:+5.1f}")
    pick = lambda m, sep: sorted((m[i, j], f"{NAMES[UNITS[i]]}{sep}{NAMES[UNITS[j]]}") for i in range(19) for j in range(i + 1, 19) if not np.isnan(m[i, j]))
    for title, m, sep in (("synergy", synergy, " + "), ("counter", counter, " vs ")):
        ranked = pick(m, sep)
        print(f"{title} top: " + ", ".join(f"{n} {s:+.1f}" for s, n in ranked[::-1][:6]))
        print(f"{title} bottom: " + ", ".join(f"{n} {s:+.1f}" for s, n in ranked[:6]))

    fig, axes = plt.subplots(3, 1, figsize=(8, 22), gridspec_kw={"height_ratios": [1, 1.25, 1.25]})
    y = np.arange(19)
    axes[0].barh(y, strength[order], color="#2a6fdb")
    axes[0].scatter(p[order], y, color="#1a1a1a", marker="|", s=120, label="on the first mover's side")
    axes[0].scatter(q[order], y, color="#b3261e", marker="|", s=120, label="on the other side")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#888", lw=0.8)
    axes[0].set_xlabel("points of win chance the unit adds to its side")
    axes[0].set_title(f"Unit strength (first mover {lead:+.1f} points)")
    axes[0].legend(loc="lower right", frameon=False)
    for ax, m, title in ((axes[1], synergy, "Synergy: extra points when both are on one side"),
                         (axes[2], counter, "Counters: extra points for the row unit against the column unit")):
        m = m[np.ix_(order, order)]
        lim = np.nanmax(np.abs(m))
        im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_xticks(y, names, rotation=90)
        ax.set_yticks(y, names)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7, label="points of win chance")
    fig.tight_layout()
    fig.savefig(plot, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
