import argparse
import glob
import json
import math
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np
from matplotlib.figure import Figure

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(HERE, "tools")
PAGE = os.path.join(TOOLS, "monitor.html")
ASSETS = {
    "/vendor/uPlot.iife.min.js": ("uPlot.iife.min.js", "text/javascript"),
    "/vendor/uPlot.min.css": ("uPlot.min.css", "text/css"),
}

LIVE = 300
RECENT = 2 * 86400
EPOCH_LIMIT = 2048
SNAPSHOT = re.compile(r"/runs/([^/]+)/snap_(\d+)\.pt$")
IDENTITY = {"out", "note", "init_weights", "git"}


def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


def read_json(path):
    try:
        with open(path) as f:
            got = json.load(f)
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def read_text(path, limit=64 << 10):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def jsonl_record(raw):
    try:
        got = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def last_jsonl(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (64 << 10)))
            lines = f.read().splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        got = jsonl_record(raw)
        if got is not None:
            return got
    return {}


def read_epochs(path, limit=EPOCH_LIMIT):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if not size:
                return []
            if size <= (8 << 20):
                f.seek(0)
                return [got for raw in f
                        if (got := jsonl_record(raw)) is not None]
            out, starts = [], set()
            for i in range(limit):
                offset = size * i // limit
                f.seek(offset)
                if offset:
                    f.readline()
                start = f.tell()
                got = jsonl_record(f.readline())
                if got is not None and start not in starts:
                    starts.add(start)
                    out.append(got)
            last = last_jsonl(path) or None
            if last is not None:
                if len(out) == limit:
                    out[-1] = last
                elif not out or out[-1] != last:
                    out.append(last)
            return out
    except OSError:
        return []


def finite(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def ema(y, span=12):
    a, p, out = 2 / (span + 1), None, []
    for v in y:
        if v is not None:
            p = v if p is None else a * v + (1 - a) * p
        out.append(None if v is None else p)
    return out


def rounded(y):
    return [None if v is None else float(f"{v:.5g}") for v in y]


def series(label, y, smooth=False, dots=False, err=None):
    y = [finite(v) for v in y]
    return {"n": label, "y": rounded(y), "m": rounded(ema(y)) if smooth else None,
            "dots": dots, "e": err}


def panel(title, ylabel, x, ss, zero=False, hlines=()):
    ss = [s for s in ss if any(v is not None for v in s["y"])]
    if not ss or len(x) < 2:
        return None
    return {"t": title, "y": ylabel, "x": [finite(v) for v in x], "s": ss,
            "zero": zero, "h": list(hlines)}


HELD = (
    ("Held-out value loss", "same rows at every snapshot", (("value", "holdout_value_loss"),), False, ()),
    ("Held-out policy KL", "nats, same rows at every snapshot", (("policy", "holdout_policy_kl"),), False, ()),
    ("Held-out move contrast", "relative spread of move embeddings",
     (("move contrast", "holdout_move_contrast"),), False, ()),
    ("Held-out gradient ratio", "policy / value gradient at the last block",
     (("ratio", "holdout_grad_ratio"),), False, ()),
)

PANELS = (
    ("Value loss", "mean per row",
     (("training batch", "loss"), ("old rows", "loss_old"), ("fresh rows", "loss_new")), False, ()),
    ("Policy loss", "cross-entropy",
     (("training batch", "policy_loss"), ("fresh rows", "fresh_policy_loss")), True, ()),
    ("Short-horizon value heads", "squared error",
     (("6 decisions", "short_6_loss"), ("16 decisions", "short_16_loss"),
      ("50 decisions", "short_50_loss")), True, ()),
    ("Per-head objective", "weighted loss",
     (("policy", "policy_weighted_loss"), ("total", "total_loss")), True, ()),
    ("Policy entropy", "nats / information state",
     (("network prior", "policy_prior_entropy"),
      ("search target", "policy_target_entropy")), True, ()),
    ("Stored search target vs current prior", "KL divergence (nats)",
     (("target KL", "policy_search_kl"),), True, ()),
    ("Learning rate", "lr", (("learning rate", "lr"),), True, ()),
    ("Spread of predictions", "std",
     (("prediction", "probe_std"), ("target configs", "tgt_std"),
      ("belief-weighted target", "tgt_belief_std")), True, ()),
    ("Value targets", "target",
     (("p05", "tgt_p05"), ("median", "tgt_p50"), ("p95", "tgt_p95"),
      ("belief mean", "tgt_belief_mean")), False, (("zero", 0),)),
    ("Value head against game outcomes", "error",
     (("RMSE", "value_outcome_rmse"), ("MAE", "value_outcome_mae")), True, ()),
    ("Value calibration", "statistic",
     (("correlation", "value_outcome_corr"), ("slope", "value_calibration_slope"),
      ("bias", "value_outcome_bias")), False, (("ideal slope", 1), ("zero", 0))),
    ("Replay buffer fill", "rows", (("buffer", "buf"),), True, ()),
    ("Replay composition", "fraction of live rows",
     (("warm", "replay_warm_frac"), ("main line", "replay_play_frac"),
      ("query", "replay_query_frac"), ("TD(1) rows", "replay_td1_row_frac"),
      ("TD(1) target cells", "replay_td1_target_frac")), True, ()),
    ("Sampled replay composition", "fraction of sampled rows",
     (("warm", "sample_warm_frac"), ("main line", "sample_play_frac"),
      ("query", "sample_query_frac"), ("TD(1) target cells", "sample_td1_target_frac")),
     True, ()),
    ("Target age when sampled", "seconds",
     (("mean", "sample_age_mean"), ("median", "sample_age_p50"),
      ("p90", "sample_age_p90"), ("oldest target", "target_age_max")), True, ()),
    ("Target delivery delay", "seconds",
     (("p90", "sample_delay_p90"), ("warm mean", "sample_warm_delay"),
      ("main-line mean", "sample_play_delay"), ("query mean", "sample_query_delay")),
     True, ()),
    ("Gradient norm before clipping", "L2 norm",
     (("mean", "grad_norm"), ("max", "grad_norm_max")), True, (("clip", 5),)),
    ("Weight norm", "L2 norm", (("weights", "weight_norm"),), True, ()),
    ("Zero-sum residual", "|E[v0] + E[v1]|",
     (("RMS", "zero_sum_rms"), ("batch max", "zero_sum_max")), True, ()),
    ("Replay generation throughput", "rows/s", (("replay generation throughput", "rows_per_s"),),
     True, ()),
    ("Effective training ratio", "optimizer rows / solve",
     (("effective training ratio", "effective_train_ratio"),), True, ()),
    ("Passes per generated row", "optimizer rows / replay row",
     (("passes per generated row", "train_row_ratio"),), True, ()),
    ("Gradient clipping", "fraction of steps", (("gradient clipping", "grad_clip_frac"),),
     True, ()),
    ("Replay age", "seconds retained", (("replay age", "buf_s"),), True, ()),
    ("Games stopped at the play limit", "fraction", (("games stopped", "timeout_frac"),),
     True, ()),
)


def build(rows, spec, **style):
    title, ylabel, keys, zero, hlines = spec
    return panel(title, ylabel, [e["h"] for e in rows],
                 [series(label, [e.get(key) for e in rows], **style) for label, key in keys],
                 zero, hlines)


def panels(reports, snapshots):
    m = [e["h"] for e in reports]
    out = ([build(snapshots, spec, dots=True) for spec in HELD]
           + [build(reports, spec, smooth=True) for spec in PANELS])

    relative_loss = [e["loss"] / e["tgt_std"] ** 2
                     if e.get("tgt_std") and "loss" in e else None for e in reports]
    out.append(panel("Relative value loss", "huber / target variance", m,
                     [series("value", relative_loss, True)], zero=True))

    now = [e["solves"] * (e["epoch"] - a["epoch"]) / (e["t"] - a["t"])
           if (a := e.get("prev")) and e["t"] > a["t"] else None for e in reports]
    out.append(panel("Generation throughput", "searches/s", m,
                     [series("now", now, True),
                      series("run average", [e.get("solves_per_s") for e in reports])],
                     zero=True))

    out.append(panel("Game length", "decisions / game", m,
                     [series("game length", [e["decisions"] / e["games"] if e.get("games") and e.get("decisions")
                                             else None for e in reports], True)]))

    kinds = ("attack", "maneuver", "deploy", "bolster", "recruit", "pass",
             "claim_initiative")
    kinds = [kind for kind in kinds
             if any(kind in (e.get("plays") or {}) for e in reports)]
    out.append(panel("Move mix", "% of decisions", m,
                     [series(kind.replace("_", " "),
                             [100 * e.get("plays", {}).get(kind, 0)
                              / max(e.get("decisions", 1), 1) for e in reports], True)
                      for kind in kinds], zero=True))

    def taken(e, kind):
        if not (a := e.get("prev")):
            return None
        (c0, o0), (c1, o1) = [(row.get("kinds") or {}).get(kind, (0, 0)) for row in (a, e)]
        return 100 * (c1 - c0) / (o1 - o0) if o1 > o0 else None

    def chances(title, kinds):
        return panel(title, "% of chances taken", m,
                     [series(kind, [taken(e, kind) for e in reports], True) for kind in kinds],
                     zero=True)

    seen = sorted({kind for e in reports for kind in e.get("kinds") or {}})
    out.append(chances("Tactic use",
                       [k for k in seen if k.startswith("Tac") and k != "TacCavalryAttack"]))
    out.append(chances("Optional maneuvers declined",
                       [k for k in seen if k.endswith(("Decline", "Stop"))]))

    censuses = [e.get("stop_census") or {} for e in reports]
    reasons = sorted({reason for census in censuses for stops in census.values()
                      for reason in stops})

    def stop_share(census, reason):
        groups = [stops for stops in census.values() if isinstance(stops, dict)]
        total = sum(item.get("count", 0) for stops in groups
                    for item in stops.values())
        count = sum(stops.get(reason, {}).get("count", 0) for stops in groups)
        return 100 * count / total if total else None

    out.append(panel("Solver stop mix", "% of solves", m,
                     [series(reason.removeprefix("budget_"),
                             [stop_share(c, reason) for c in censuses], True)
                      for reason in reasons], zero=True))
    return out


def elo(report, names):
    if not report:
        return None
    bots = [b for b in report["bots"] if b["name"].split(".")[0] in names]
    fit = report["fit"]
    return panel("Elo", "rating ± standard error", [b["hours"] for b in bots],
                 [series(name, [b["elo"] if b["name"].split(".")[0] == name else None for b in bots],
                         dots=True, err=[b["se"] for b in bots]) for name in names]
                 + [series(f"log fit, {fit['b']:+.0f} per doubling",
                           [fit["a"] + fit["b"] * math.log2(b["hours"]) for b in bots])],
                 hlines=report["anchors"].items())


def origin(cfg):
    hit = SNAPSHOT.search(cfg.get("init_weights", ""))
    return hit and (hit[1], f"s{int(hit[2])}" if int(hit[2]) else "init")


def branch(runs_dir, cfg):
    got = origin(cfg)
    path = got and run_dir(runs_dir, got[0])
    bot = path and read_json(os.path.join(path, "bots", got[1], "bot.json"))
    return (got[0], bot["minutes"] * 60) if bot and "minutes" in bot else ("", math.inf)


def lineage(runs_dir, name, until=math.inf, seen=()):
    path = os.path.join(runs_dir, name)
    cfg = (read_json(f"{path}/log.json") or {}).get("cfg") or {}
    parent, at = branch(runs_dir, cfg)
    stages = lineage(runs_dir, parent, at, seen + (name,)) if parent and parent not in seen else []
    start = stages[-1]["start"] + at / 3600 if stages else 0.0
    rows = [e for e in map(schema, read_epochs(f"{path}/epochs.jsonl")) if e["t"] < until]
    for e in rows:
        e["h"] = round(start + e["t"] / 3600, 4)
    reports = [e for e in rows if e.get("phase") == "sog" and e.get("solves", 0) > 0
               and e.get("steps", 1) > 0]
    for a, b in zip(reports, reports[1:]):
        b["prev"] = a
    return stages + [{"name": name, "cfg": cfg, "start": start, "reports": reports,
                      "snapshots": [e for e in rows if e.get("phase") == "snapshot"]}]


SUMMARY = {}


def index(runs_dir):
    out = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.json")):
        folder = os.path.dirname(path)
        name, epochs = os.path.basename(folder), os.path.join(folder, "epochs.jsonl")
        mt = max(mtime(path), mtime(epochs))
        hit = SUMMARY.get(name)
        if not hit or hit[0] != mt:
            cfg = (read_json(path) or {}).get("cfg") or {}
            hit = (mt, {"name": name, "mtime": mt, "note": cfg.get("note", ""),
                        "hours": last_jsonl(epochs).get("t", 0) / 3600,
                        "budget": cfg.get("minutes", 0) / 60,
                        "parent": ".".join(origin(cfg) or ())})
            SUMMARY[name] = hit
        out.append(dict(hit[1], running=time.time() - mt < LIVE))
    return sorted(out, key=lambda r: -r["mtime"])


def run_dir(runs_dir, name):
    path = os.path.abspath(os.path.join(runs_dir, name))
    return path if os.path.dirname(path) == runs_dir and os.path.isdir(path) else ""


ISMCTS = {"decisions": "solves", "decisions_per_s": "solves_per_s", "value_loss": "loss",
          "fresh_value_loss": "loss_new", "policy_kl": "policy_search_kl",
          "prior_entropy": "policy_prior_entropy"}


def schema(e):
    if e.get("phase") != "search":
        return e
    got = {ISMCTS.get(k, k): v for k, v in e.items()}
    got["phase"], got["decisions"] = "sog", e.get("decisions")
    got["plays"] = {k[6:]: v for k, v in e.items() if k.startswith("plays_")}
    if e.get("replay_rows"):
        got["train_row_ratio"] = e.get("optimizer_rows", 0) / e["replay_rows"]
    return got


def detail(runs_dir, name):
    path = run_dir(runs_dir, name)
    if not path:
        return None
    stages = lineage(runs_dir, name)
    cfg = stages[-1]["cfg"]
    base = stages[-2]["cfg"] if len(stages) > 1 else cfg
    names = [s["name"] for s in stages]
    reports = [e for s in stages for e in s["reports"]]
    snapshots = [e for s in stages for e in s["snapshots"]]
    return {"name": name, "note": cfg.get("note", ""), "git": cfg.get("git", ""),
            "stages": [[s["name"], s["start"]] for s in stages],
            "panels": [p for p in [elo(read_json(f"{path}/elo.json"), names)]
                       + panels(reports, snapshots) if p],
            "cfg": [[k, str(v), k not in IDENTITY and v != base.get(k, v)] for k, v in cfg.items()],
            "log": read_text(f"{path}/train.log"),
            "notes": read_text(f"{path}/NOTES.md"),
            "ladder": read_json(f"{path}/ladder.json"),
            "comparisons": [report for report in
                            (read_json(p) for p in sorted(glob.glob(
                                f"{path}/comparisons/*.json"))) if report]}


def draw(p, stages, span, path):
    fig = Figure(figsize=(8, 3.2), dpi=100, layout="constrained")
    ax = fig.add_subplot(title=f"{p['t']} · {p['y']}", xlabel="training hours")
    x = np.array(p["x"], float)
    for i, s in enumerate(p["s"]):
        y, color = np.array(s["y"], float), f"C{i % 10}"
        if s["m"]:
            ax.plot(x, y, color=color, alpha=.25, lw=.8)
            y = np.array(s["m"], float)
        keep = ~np.isnan(y) | (not s["dots"])
        ax.errorbar(x[keep], y[keep], np.array(s["e"], float)[keep] if s["e"] else None,
                    color=color, lw=1.4, marker="o" if s["dots"] else None, ms=3, label=s["n"])
    for i, (label, v) in enumerate(p["h"], len(p["s"])):
        ax.axhline(v, color=f"C{i % 10}", ls="--", lw=1, label=label)
    for name, start in stages[1:]:
        ax.axvline(start, color="gray", ls=":", lw=1)
        ax.text(start, 1, f" {name}", transform=ax.get_xaxis_transform(), va="top",
                fontsize=8, color="gray")
    if p["zero"]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), max(hi, 0))
    ax.set_xlim(span)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.savefig(path + ".tmp", format="png")
    os.replace(path + ".tmp", path)


def plot(runs_dir, name):
    d = detail(runs_dir, name)
    if not d["panels"]:
        return
    folder = os.path.join(runs_dir, name, "plots")
    os.makedirs(folder, exist_ok=True)
    files = {re.sub(r"[^a-z0-9]+", "-", p["t"].lower()).strip("-") + ".png": p
             for p in d["panels"]}
    for stale in set(os.listdir(folder)) - set(files):
        os.remove(os.path.join(folder, stale))
    span = min(p["x"][0] for p in d["panels"]), max(p["x"][-1] for p in d["panels"])
    for file, p in files.items():
        draw(p, d["stages"], span, os.path.join(folder, file))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/":
            return self.body(read_text(PAGE, 1 << 20).encode(), "text/html")
        if route in ASSETS:
            filename, ctype = ASSETS[route]
            with open(os.path.join(TOOLS, "vendor", filename), "rb") as f:
                return self.body(f.read(), ctype)
        if route == "/api/runs":
            return self.body(json.dumps(index(self.server.runs)).encode(),
                             "application/json")
        if route.startswith("/api/run/"):
            got = detail(self.server.runs, unquote(route[len("/api/run/"):]))
            return (self.body(json.dumps(got).encode(), "application/json")
                    if got else self.send_error(404))
        if route.startswith("/run/") and route.endswith("/train.log"):
            name = unquote(route[len("/run/"):-len("/train.log")])
            raw = read_text(os.path.join(run_dir(self.server.runs, name),
                                         "train.log"), 1 << 24)
            return (self.body(raw.encode(), "text/plain; charset=utf-8")
                    if raw else self.send_error(404))
        self.send_error(404)

    def body(self, raw, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def refresh(runs_dir, every=30):
    env = os.environ.copy()
    env["WARCHEST_BOX_LOCAL_DIR"] = os.path.dirname(runs_dir)
    drawn = {}
    while True:
        r = subprocess.run([os.path.join(TOOLS, "box.sh"), "pull"], env=env,
                           capture_output=True, text=True)
        if r.returncode:
            print(f"[monitor] pull {r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        for run in index(runs_dir):
            stamp = run["mtime"], mtime(os.path.join(runs_dir, run["name"], "elo.json"))
            if time.time() - run["mtime"] < RECENT and drawn.get(run["name"]) != stamp:
                try:
                    plot(runs_dir, run["name"])
                except Exception as err:
                    print(f"[monitor] plot {run['name']}: {err!r}", flush=True)
                drawn[run["name"]] = stamp
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description="Live view of runs/, read from disk on every request.")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--pull", action="store_true",
                    help="pull runs from the box every 30s and redraw runs/<run>/plots/*.png "
                         "for each run that changed in the last two days")
    args = ap.parse_args()
    runs = os.path.abspath(args.runs)
    if args.pull:
        threading.Thread(target=refresh, args=(runs,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.runs = runs
    print(f"[monitor] http://{args.host}:{args.port} · {runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
