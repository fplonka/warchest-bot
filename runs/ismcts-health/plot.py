import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUNS = {"ismcts30": "#eb6834", "ismcts60": "#2a78d6"}
PLAYS = {"attack": "#2a78d6", "pass": "#e34948", "deploy": "#1baf7a", "bolster": "#eda100",
         "maneuver": "#4a3aa7", "recruit": "#e87ba4", "claim_initiative": "#008300"}
DECIDED_CE, TIMEOUT_CE, DECIDED_SQ = 0.5072, 1.2003, 0.0074
WINDOW = 6


def load(run):
    rows = [json.loads(l) for l in open(HERE.parent / run / "epochs.jsonl")]
    d = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in rows[-1] if k != "phase"}
    d["dt"] = np.diff(d["t"], prepend=0.0)
    return d


def smooth(x):
    c = np.cumsum(np.insert(x, 0, 0.0))
    n = np.minimum(np.arange(1, len(x) + 1), WINDOW)
    return (c[1:] - c[np.arange(1, len(x) + 1) - n]) / n


def ratio(num, den):
    return smooth(num) / np.maximum(smooth(den), 1e-9)


def figure(name, panels):
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), squeeze=False)
    for ax, (title, fn) in zip(axes.flat, panels):
        for run, color in RUNS.items():
            ax.plot(data[run]["t"] / 60.0, fn(data[run]), color=color, lw=1.6, label=run)
        ax.axvline(3, color="#888", lw=0.8, ls="--")
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("minutes")
    for ax in axes.flat[len(panels):]:
        ax.remove()
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / f"{name}.png", dpi=110)
    plt.close(fig)


data = {run: load(run) for run in RUNS}
figure("mix", [(f"{p} per finished game", lambda d, p=p: ratio(d[f"plays_{p}"], d["games"])) for p in PLAYS]
       + [("pass share of plays", lambda d: ratio(d["plays_pass"], sum(d[f"plays_{p}"] for p in PLAYS)))])
figure("games", [
    ("decisions per finished game", lambda d: ratio(d["decisions"], d["games"])),
    ("rows per finished game", lambda d: ratio(d["rows"], d["games"])),
    ("play-limit timeout fraction", lambda d: ratio(d["timeouts"], d["games"])),
    ("white share of decided games", lambda d: ratio(d["white_wins"], d["white_wins"] + d["black_wins"])),
    ("mean z of rows", lambda d: smooth(d["tgt_mean"])),
    ("finished games per second", lambda d: ratio(d["games"], d["dt"])),
])
figure("losses", [
    ("value loss", lambda d: smooth(d["value_loss"])),
    ("value loss above its entropy floor", lambda d: smooth(d["value_loss"]) - DECIDED_CE
     - (TIMEOUT_CE - DECIDED_CE) * ratio(d["timeouts"], d["games"])),
    (f"log10 value squared error (floor {np.log10(DECIDED_SQ):.2f})", lambda d: np.log10(smooth(d["value_sq_err"]))),
    ("policy loss", lambda d: smooth(d["policy_loss"])),
    ("policy target entropy", lambda d: smooth(d["policy_target_entropy"])),
    ("policy KL (loss - entropy)", lambda d: smooth(d["policy_kl"])),
    ("grad norm", lambda d: smooth(d["grad_norm"])),
    ("learning rate", lambda d: d["lr"]),
])
figure("throughput", [
    ("decisions per second", lambda d: ratio(d["decisions"], d["dt"])),
    ("rows per second", lambda d: ratio(d["rows"], d["dt"])),
    ("evals per second", lambda d: smooth(d["evals_per_s"])),
    ("GPU 0 busy", lambda d: smooth(d["gpu_busy"])),
    ("buffer span in seconds (oldest row age)", lambda d: d["buf_s"]),
    ("optimizer rows / generated rows", lambda d: d["optimizer_rows"] / np.maximum(d["replay_rows"], 1)),
])
