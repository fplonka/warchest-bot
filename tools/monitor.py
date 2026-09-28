import argparse
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(HERE, "tools")
PAGE = os.path.join(TOOLS, "monitor.html")
ASSETS = {
    "/vendor/uPlot.iife.min.js": ("uPlot.iife.min.js", "text/javascript"),
    "/vendor/uPlot.min.css": ("uPlot.min.css", "text/css"),
}
sys.path.insert(0, os.path.join(HERE, "train"))
import config

LIVE = 120
EPOCH_LIMIT = 2048


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


def series(label, y, smooth=False):
    return {"n": label, "y": [finite(v) for v in y], "sm": smooth}


def panel(title, ylabel, x, ss, zero=False, hlines=()):
    ss = [s for s in ss if any(v is not None for v in s["y"])]
    if not ss:
        return None
    return {"t": title, "y": ylabel, "x": [finite(v) for v in x], "s": ss,
            "zero": zero, "h": list(hlines)}


PANELS = (
    ("Value loss", "cross-entropy", (("value", "value_loss"),), False, ()),
    ("Value error against game results", "squared error", (("value", "value_sq_err"),), True, ()),
    ("Policy loss", "cross-entropy", (("policy", "policy_loss"),), True, ()),
    ("Search target vs prior", "nats",
     (("KL", "policy_kl"), ("target entropy", "policy_target_entropy")), True, ()),
    ("Network evaluations", "rows/s", (("evals", "evals_per_s"),), True, ()),
    ("Simulations", "sims/s", (("sims", "sims_per_s"),), True, ()),
    ("Replay generation throughput", "rows/s", (("rows", "rows_per_s"),), True, ()),
    ("Rows per GPU call", "rows", (("rows per call", "rows_per_call"),), True, ()),
    ("Gradient norm before clipping", "L2 norm", (("mean", "grad_norm"),), True, (("clip", 5),)),
    ("Replay buffer fill", "rows", (("buffer", "buf"),), True, ()),
    ("Replay age", "seconds retained", (("replay age", "buf_s"),), True, ()),
    ("Games stopped at the play limit", "fraction", (("games stopped", "timeout_frac"),), True, ()),
)


def panels(eps):
    m = [e["t"] / 60.0 for e in eps]
    out = [
        panel(title, ylabel, m,
              [series(label, [e.get(key) for e in eps], True)
               for label, key in specs], zero=zero, hlines=hlines)
        for title, ylabel, specs, zero, hlines in PANELS]
    kinds = ("attack", "maneuver", "deploy", "bolster", "recruit", "pass", "claim_initiative")
    out.append(panel("Move mix", "% of decisions", m,
                     [series(kind.replace("_", " "),
                             [100 * e.get(f"plays_{kind}", 0) / max(e.get("decisions", 1), 1)
                              for e in eps], True)
                      for kind in kinds], zero=True))
    return [p for p in out if p]


def health(eps):
    if not eps:
        return []
    last, tot = eps[-1], lambda k: sum(e.get(k, 0) for e in eps)
    return [("wall clock", f"{last.get('t', 0) / 60:.0f} min"),
            ("games", f"{tot('games'):,}"),
            ("evals/s", f"{last.get('evals_per_s', 0):,.0f}"),
            ("sims/s", f"{last.get('sims_per_s', 0):,.0f}"),
            ("buffer", f"{last.get('buf', 0):,}"),
            ("games stopped", f"{100 * tot('timeouts') / max(tot('games'), 1):.1f}%"),
            ("gradient norm", f"{last.get('grad_norm', 0):.2f}")]


SUMMARY = {}


def index(runs_dir):
    out = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.json")):
        name = os.path.basename(os.path.dirname(path))
        epoch_path = os.path.join(os.path.dirname(path), "epochs.jsonl")
        evaluation = glob.glob(os.path.join(os.path.dirname(path),
                                            "comparisons", "*.json"))
        mt = max([mtime(path), mtime(epoch_path),
                  mtime(os.path.join(os.path.dirname(path), "ladder.json"))]
                 + [mtime(p) for p in evaluation])
        hit = SUMMARY.get(name)
        if not hit or hit[0] != mt:
            log = read_json(path) or {}
            last = last_jsonl(epoch_path)
            epoch = last.get("epoch", -1)
            hit = (mt, {"name": name, "mtime": mt,
                        "epochs": int(epoch) + 1 if isinstance(epoch, (int, float)) else 0,
                        "phase": last.get("phase", ""),
                        "minutes": round(last.get("t", 0) / 60.0),
                        "note": (log.get("cfg") or {}).get("note", "")})
            SUMMARY[name] = hit
        out.append(dict(hit[1], running=time.time() - mt < LIVE))
    return sorted(out, key=lambda r: -r["mtime"])


def run_dir(runs_dir, name):
    path = os.path.abspath(os.path.join(runs_dir, name))
    return path if os.path.dirname(path) == runs_dir and os.path.isdir(path) else ""


def detail(runs_dir, name):
    path = run_dir(runs_dir, name)
    if not path:
        return None
    log = read_json(os.path.join(path, "log.json")) or {}
    eps = [e for e in read_epochs(os.path.join(path, "epochs.jsonl"))
           if e.get("phase") == "search" and e.get("steps", 1) > 0]
    cfg = log.get("cfg") or {}
    return {"name": name, "epochs": len(eps), "panels": panels(eps),
            "cfg": [[k, str(v), ch] for k, v, ch in config.knobs(cfg)],
            "health": health(eps), "note": cfg.get("note", ""),
            "git": cfg.get("git", ""), "log": read_text(f"{path}/train.log"),
            "notes": read_text(f"{path}/NOTES.md"),
            "ladder": read_json(f"{path}/ladder.json"),
            "comparisons": [report for report in
                            (read_json(p) for p in sorted(glob.glob(
                                f"{path}/comparisons/*.json"))) if report],
            "snaps": [s.get("t", 0) / 60.0 for s in log.get("snapshots") or []]}


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


def puller(dest, every=30):
    env = os.environ.copy()
    env["WARCHEST_BOX_LOCAL_DIR"] = os.path.dirname(os.path.abspath(dest))
    command = [os.path.join(TOOLS, "box.sh"), "pull"]
    while True:
        r = subprocess.run(command, env=env, capture_output=True, text=True)
        if r.returncode:
            print(f"[monitor] pull {r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description="Live view of runs/, read from disk on every request.")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--pull", action="store_true",
                    help="pull runs from the box every 30s")
    args = ap.parse_args()
    if args.pull:
        threading.Thread(target=puller, args=(args.runs,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.runs = os.path.abspath(args.runs)
    print(f"[monitor] http://{args.host}:{args.port} · {srv.runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
