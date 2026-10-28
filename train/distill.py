import argparse
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
import gpu_batch
import nnue
import warchest
from train import Buffer

CHUNK = 32768
HELD = 0.01


def chunks(lo, hi):
    return np.array_split(np.arange(lo, hi), -(-(hi - lo) // CHUNK))


@torch.no_grad()
def label(teacher, buf, device):
    for ids in chunks(buf.lo, buf.rows):
        s, at, _ = buf.slots(ids)
        x, desc, parow, _, _ = gpu_batch.make_batch(buf.gather(ids), device)
        with torch.autocast("cuda", torch.bfloat16):
            win, _, pieces = teacher.evaluate(x)
            logit = teacher.actions(desc, pieces, parow)
        buf.z[s, 0] = teacher.value(win.float()).cpu().numpy()
        buf.pp[at] = nnue.log_softmax(logit.float(), parow, len(ids)).exp().cpu().numpy()


def batch(parts, device):
    rows, desc, parow, prior, z = parts
    t = lambda a: torch.as_tensor(a, dtype=torch.float32).to(device, non_blocking=True)
    return nnue.inputs(rows, desc, parow, device), (t(z[:, 0]) + 1) / 2, t(prior)


def gaps(student, b):
    x, value, prior = b
    win, logp = student(*x)
    entropy = lambda p: -(p * p.clamp(min=1e-30).log())
    return {"value": F.binary_cross_entropy_with_logits(win, value) - entropy(value).mean() - entropy(1 - value).mean(),
            "policy": (entropy(prior).neg() - prior * logp).sum() / len(value)}


@torch.no_grad()
def scores(students, buf, held, device):
    runs = [[gaps(s, b) for s in students] for b in (batch(buf.gather(ids), device) for ids in chunks(*held))]
    return [{k: np.mean([r[i][k].item() for r in runs]) for k in ("value", "policy")} for i in range(len(students))]


def parity(student, buf, held, device, weights):
    rows, desc, parow, _, _ = buf.gather(np.arange(*held)[:4096])
    with torch.no_grad():
        win, logp = student(*nnue.inputs(rows, desc, parow, device))
    off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))]).astype(np.uint32)
    value, prior = warchest.nnue_eval(str(weights), rows.ravel(), desc.ravel(), off)
    return (f"rust vs torch mean |d|: value {np.abs(np.tanh(win.cpu().numpy() / 2) - value).mean():.1e} "
            f"prior {np.abs(logp.exp().cpu().numpy() - prior).mean():.1e}")


def main():
    ap = argparse.ArgumentParser(description="Distil a teacher snapshot's raw value and prior into an NNUE student over its replay buffer, and pack it as a bot.")
    ap.add_argument("--teacher", required=True, help="a snapshot .pt; its .buffer holds the training positions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--students", nargs="+", required=True, help="width:rank of each student, trained on the same batches")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=8.75e-4)
    ap.add_argument("--cap", type=int, default=config.Cfg.cap)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0, out, device = time.time(), Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    buf = Buffer(args.cap).load(Path(args.teacher).with_suffix(".buffer"))
    label(export_weights.load(args.teacher).to(device).eval(), buf, device)
    held = (buf.rows - int(HELD * len(buf)), buf.rows)
    rows, desc, parow, _, _ = buf.gather(rng.integers(buf.lo, held[0], size=1 << 20))
    phases = nnue.inputs(rows, desc, parow, "cpu")[3].numpy()
    thresholds = np.quantile(phases, np.arange(1, nnue.STACKS) / nnue.STACKS).round().astype(int)
    print(f"{len(buf)} rows labelled in {time.time() - t0:.0f}s, newest {held[1] - held[0]} held out; "
          f"phase thresholds {thresholds.tolist()}", flush=True)
    shapes = [tuple(map(int, s.split(":"))) for s in args.students]
    students = [nnue.Student(l1, r, thresholds).to(device) for l1, r in shapes]
    opts = [torch.optim.Adam(s.parameters(), lr=args.lr) for s in students]
    decays = [torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1 - step / args.steps) for o in opts]
    for step in range(1, args.steps + 1):
        b = batch(buf.gather(rng.integers(buf.lo, held[0], size=args.batch)), device)
        for s, o, d in zip(students, opts, decays):
            g = gaps(s, b)
            o.zero_grad(set_to_none=True)
            (g["value"] + g["policy"]).backward()
            o.step()
            d.step()
            s.clip()
        if step % max(args.steps // 20, 1) == 0:
            report = "  ".join(f"{l1}:{r} value {g['value']:.4f} policy {g['policy']:.4f}"
                               for (l1, r), g in zip(shapes, scores(students, buf, held, device)))
            print(f"step {step} {time.time() - t0:.0f}s held-out KL to the teacher: {report}", flush=True)
    for (l1, r), s in zip(shapes, students):
        bot = out / "bots" / f"w{l1}r{r}"
        export_weights.pack(bot, f"{out.name}.w{l1}r{r}", s, config.git_sha(), {"mind": "nnue", "sims": args.sims},
                            (time.time() - t0) / 60, f"nnue {l1}/{r} of {args.teacher}, {args.steps} steps of {args.batch}")
        torch.save(s.state_dict(), bot / "student.pt")
        print(f"{l1}:{r} {parity(s, buf, held, device, bot / 'weights.bin')}", flush=True)


if __name__ == "__main__":
    main()
