import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def student(shape):
    l1, r = shape.split(":")
    return f"w{l1}r{r}"


def colors(names):
    order = sorted(names - {"teacher"}, key=lambda n: tuple(map(int, n[1:].split("r"))))
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(["teacher"] + order)}


def progress(ax, rows, key, color):
    sources = sorted({r["held"] for r in rows})
    for s in sorted({s for r in rows for s in r["kl"]}):
        for held, dash in zip(sources, ("-", "--", ":", "-.")):
            pts = [(r["step"], r["kl"][s][key]) for r in rows if r["held"] == held and s in r["kl"]]
            if pts:
                ax.plot(*zip(*pts), dash, color=color[student(s)], linewidth=2, label=f"{student(s)} {held}")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)


def family(rating):
    return "teacher" if rating["search"]["mind"] == "search" else rating["name"].split(".")[1].split("@")[0].split("-")[0]


def steps(rating):
    return int(rating["name"].split("@")[0].rsplit("-", 1)[1][:-1]) if family(rating) != "teacher" else 0


def ladder(ax, ratings, color):
    anchor = next((r["elo"] for r in ratings if family(r) == "teacher" and r["search"]["sims"] == 256), 0.0)
    newest = {}
    for r in ratings:
        newest[family(r)] = max(newest.get(family(r), 0), steps(r))
    families = {}
    for r in ratings:
        if steps(r) == newest[family(r)]:
            families.setdefault(family(r), []).append(r)
    for family_name, rs in sorted(families.items()):
        rs.sort(key=lambda r: r["search"]["sims"])
        sims = [r["search"]["sims"] for r in rs]
        elo = [r["elo"] - anchor for r in rs]
        ax.errorbar(sims, elo, yerr=[1.96 * r["se"] for r in rs], fmt="-o", color=color[family_name],
                    linewidth=2, markersize=6, capsize=3,
                    label=family_name if family_name == "teacher" else f"{family_name} at {newest[family_name]}k steps")
    ax.set_xscale("log", base=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def main():
    runs = ROOT / "runs"
    distill = sorted(runs.glob("*/progress.jsonl"))
    ratings = sorted(runs.glob("*-ladder/ladder.json"))
    rows = [json.loads(line) for line in distill[-1].read_text().splitlines()] if distill else []
    rs = json.loads(ratings[-1].read_text()) if ratings else []
    color = colors({student(s) for r in rows for s in r["kl"]} | {family(r) for r in rs} | {"teacher"})
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    if distill:
        run = distill[-1].parent
        progress(axes[0], rows, "value", color)
        progress(axes[1], rows, "policy", color)
        style(axes[0], f"{run.name}: held-out value KL to the teacher", "training step", "nats")
        style(axes[1], f"{run.name}: held-out policy KL to the teacher", "training step", "nats")
    if ratings:
        ladder(axes[2], rs, color)
        style(axes[2], f"{ratings[-1].parent.name}: Elo against sims per move (0 = teacher at 256), 95% CI", "simulations per move", "Elo")
    now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%a %H:%M")
    fig.suptitle(f"NNUE vs teacher, updated {now} Amsterdam", x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    out = runs / "plots" / "nnue.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
