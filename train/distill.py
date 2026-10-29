import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import export_weights
import nnue
import shards
import warchest

HELD = 200_000
CHUNK = 32768


def batch(parts, device):
    rows, desc, parow, prior, value = parts
    t = lambda a: torch.as_tensor(a, dtype=torch.float32).pin_memory().to(device, non_blocking=True)
    return nnue.inputs(rows, desc, parow, device), (t(value) + 1) / 2, t(prior)


def gaps(student, b):
    x, value, prior = b
    win, logp = student(*x)
    entropy = lambda p: -(p * p.clamp(min=1e-30).log())
    return {"value": F.binary_cross_entropy_with_logits(win, value) - entropy(value).mean() - entropy(1 - value).mean(),
            "policy": (entropy(prior).neg() - prior * logp).sum() / len(value)}


def holdout(directory):
    files = shards.listed(directory, True)
    if not files:
        return None
    return shards.concat([s.gather(np.sort(np.random.default_rng(0).choice(len(s), min(HELD // len(files), len(s)), replace=False)))
                          for s in map(shards.Shard, files)])


def sub(held, lo, hi):
    rows, desc, parow, prior, value = held
    at = (parow >= lo) & (parow < hi)
    return rows[lo:hi], desc[at], parow[at] - lo, prior[at], value[lo:hi]


@torch.no_grad()
def scores(students, held, device):
    n = len(held[0])
    runs = [[gaps(s, batch(sub(held, lo, min(lo + CHUNK, n)), device)) for s in students] for lo in range(0, n, CHUNK)]
    return [{k: float(np.mean([r[i][k].item() for r in runs])) for k in ("value", "policy")} for i in range(len(students))]


def parity(student, held, device, weights):
    rows, desc, parow, _, _ = sub(held, 0, 4096)
    with torch.no_grad():
        win, logp = student(*nnue.inputs(rows, desc, parow, device))
    off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))]).astype(np.uint32)
    value, prior = warchest.nnue_eval(str(weights), rows.ravel(), desc.ravel(), off)
    return (f"rust vs torch mean |d|: value {np.abs(np.tanh(win.cpu().numpy() / 2) - value).mean():.1e} "
            f"prior {np.abs(logp.exp().cpu().numpy() - prior).mean():.1e}")


def latest(bots, shape):
    return max(bots.glob(f"{shape}-*k"), key=lambda b: int(b.name.rsplit("-", 1)[1][:-1]))


def pack(out, label, student, args, t0):
    bot = out / "bots" / label
    export_weights.pack(bot, f"{out.name}.{label}", student, config.git_sha(), {"mind": "nnue", "sims": args.sims},
                        (time.time() - t0) / 60, f"nnue {label}, {args.batch} a step from {' '.join(args.data)}")
    torch.save(student.state_dict(), bot / "student.pt")


def main():
    ap = argparse.ArgumentParser(description="Train NNUE students on labelled shards streamed from disk, and pack them as bots.")
    ap.add_argument("--data", nargs="+", required=True, help="shard directories, re-read every pass so they may still be growing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--students", nargs="+", required=True, help="width:rank:value-weight of each student, trained on the same batches")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--init", help="a bots directory whose latest checkpoint of each shape to start from")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=8.75e-4)
    ap.add_argument("--pool", type=int, default=24, help="shards held in memory at once")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0, out, device = time.time(), Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    stream = shards.Stream(args.data, args.pool, rng)
    rows, desc, parow, _, _ = stream.batch(1 << 20)
    phases = nnue.inputs(rows, desc, parow, "cpu")[3].numpy()
    thresholds = np.quantile(phases, np.arange(1, nnue.STACKS) / nnue.STACKS).round().astype(int)
    print(f"pool of {args.pool} shards loaded in {time.time() - t0:.0f}s; phase thresholds {thresholds.tolist()}", flush=True)
    shapes = [tuple(map(int, s.split(":"))) for s in args.students]
    names = [f"w{l1}r{r}v{v}" for l1, r, v in shapes]
    students = [nnue.Student(l1, r, thresholds).to(device) for l1, r, _ in shapes]
    if args.init:
        for name, s in zip(names, students):
            s.load_state_dict(torch.load(latest(Path(args.init), name) / "student.pt", map_location=device))
    opts = [torch.optim.Adam(s.parameters(), lr=args.lr) for s in students]
    decays = [torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1 - step / args.steps) for o in opts]
    ready = stream.prefetch(args.batch, lambda parts: batch(parts, device))
    every, train = max(args.steps // 40, 1), [collections.Counter() for _ in students]
    for step in range(1, args.steps + 1):
        b = ready.get()
        for (_, _, v), s, o, d, seen in zip(shapes, students, opts, decays, train):
            g = gaps(s, b)
            o.zero_grad(set_to_none=True)
            (v * g["value"] + g["policy"]).backward()
            o.step()
            d.step()
            s.clip()
            seen.update({k: v.detach() for k, v in g.items()})
        if step % max(args.steps // 5, 1) == 0 or step == args.steps:
            for name, s in zip(names, students):
                pack(out, f"{name}-{step // 1000}k", s, args, t0)
        if step % every == 0:
            sources = {"train": [{k: float(v) / every for k, v in seen.items()} for seen in train]}
            sources.update({Path(d).name: scores(students, held, device) for d in args.data if (held := holdout(d)) is not None})
            train = [collections.Counter() for _ in students]
            for name, gs in sources.items():
                kl = dict(zip(names, gs))
                with open(out / "progress.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "t": time.time() - t0, "held": name, "kl": kl}) + "\n")
                print(f"step {step} {time.time() - t0:.0f}s {name}, KL to the teacher: "
                      + "  ".join(f"{k} value {g['value']:.4f} policy {g['policy']:.4f}" for k, g in kl.items()), flush=True)
    held = holdout(args.data[0])
    for name, s in zip(names, students):
        print(f"{name} {parity(s, held, device, out / 'bots' / f'{name}-{args.steps // 1000}k' / 'weights.bin')}", flush=True)


if __name__ == "__main__":
    main()
