
import dataclasses
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Cfg:
    minutes: float = 60.0
    warm_minutes: float = 3.0
    snapshot_every: float = 15.0
    init_weights: str = ""
    resume: str = ""

    batch: int = 8192
    micro: int = 2048
    lr: float = 1e-3
    wd: float = 0.01
    lr_final: float = 1e-4
    lr_stable_frac: float = 0.75
    replay_ratio: float = 1.0
    target_every: float = 1.0
    cap: int = 4_000_000
    policy_w: float = 1.0
    warm_games: int = 96

    sims: int = 32
    games: int = 1024
    collect_rows: int = 65536

    device: str = "cuda:1"

    out: str = ""
    seed: int = 1
    git: str = ""
    note: str = ""


BASELINE = Cfg()
CAST = {"int": int, "float": float,
        "bool": lambda v: v not in ("0", "false", "False", "")}


def parse(kvs):
    over = dict(kv.split("=", 1) for kv in kvs)
    kinds = {f.name: getattr(f.type, "__name__", f.type)
             for f in dataclasses.fields(Cfg)}
    bad = set(over) - set(kinds)
    if bad:
        raise SystemExit(f"no such knob: {sorted(bad)}")
    return {k: CAST.get(kinds[k], str)(v) for k, v in over.items()}


def knobs(cfg):
    d = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    base = dataclasses.asdict(BASELINE)
    for f in dataclasses.fields(Cfg):
        v = d.get(f.name, base[f.name])
        yield f.name, v, f.name in base and v != base[f.name]


def git_sha():
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", ".gitsha")
    try:
        with open(marker) as f:
            return f.read().strip() or "unknown"
    except OSError:
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--", ".", ":!runs"],
            capture_output=True)
        return sha + ("+dirty" if dirty.returncode != 0 else "")
    except Exception:
        return "unknown"
