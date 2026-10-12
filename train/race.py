import argparse
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config
import gpu_batch
import train
from value_net import Net

HOLD = 2_000_000
EVAL = 262_144
EVERY = 500
WARMUP = 500
KEYS = ("policy_kl", "value_loss", *train.SHORTS)


def main():
    ap = argparse.ArgumentParser(description="Train one net from scratch on a frozen replay buffer.")
    ap.add_argument("out", help="JSON lines report; the final net is saved next to it as .pt")
    ap.add_argument("--buffer", default="/workspace/big/runs/td2w/snap_0053.buffer")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Net().to(device)
    acting = copy.deepcopy(net).requires_grad_(False)
    opt = train.adam(net, args.lr, config.BASELINE.wd)
    print(f"[race] {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params", flush=True)

    t0 = time.time()
    buf = train.Buffer(config.BASELINE.cap)
    buf.load(args.buffer)
    print(f"[race] loaded {len(buf)} rows in {time.time() - t0:.0f}s", flush=True)
    held = buf.rows - HOLD + rng.choice(HOLD, EVAL, replace=False)
    seen = buf.lo + rng.choice(len(buf) - HOLD, EVAL, replace=False)
    buf.rows -= HOLD

    def measure(ids):
        total = dict.fromkeys(KEYS, 0.0)
        with torch.no_grad():
            for chunk in np.split(ids, len(ids) // config.BASELINE.micro):
                _, stats = train.losses(net, *gpu_batch.make_batch(buf.gather(chunk), device), 1.0)
                for k in KEYS:
                    total[k] += float(stats[k]) * len(chunk) / len(ids)
        return total

    log, last, clock = open(args.out, "a"), 0, time.time()

    def report(step):
        nonlocal last, clock
        rate = (step - last) * config.BASELINE.batch / (time.time() - clock)
        rec = {"step": step, "t": round(time.time() - t0), "rows_per_s": round(rate),
               **{f"held_{k}": round(v, 5) for k, v in measure(held).items()},
               **{f"train_{k}": round(v, 5) for k, v in measure(seen).items()}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        last, clock = step, time.time()

    report(0)
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / WARMUP) * (1 + math.cos(math.pi * step / args.steps)) / 2
        train.train_steps(net, acting, opt, buf, 1, config.BASELINE.batch, config.BASELINE.micro,
                          rng, device, gpu_batch.make_batch, 1.0)
        if (step + 1) % EVERY == 0 or step + 1 == args.steps:
            report(step + 1)
    torch.save({"acting": acting.state_dict()}, os.path.splitext(args.out)[0] + ".pt")


if __name__ == "__main__":
    main()
