import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config
import export_weights
import gpu_batch
import train
from value_net import Net

TEST_ROWS = 500_000
REPORT = 50


def main():
    replay, steps, *kvs = sys.argv[1:]
    steps = int(steps)
    args = config.Cfg(**{"minutes": steps / 60, "lr_warmup": steps / 1200, "lr_stable_frac": 0.5,
                         "lr": 2.5e-3, **config.parse(kvs)})
    os.makedirs(args.out)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    buf = train.Buffer(args.cap).load(replay)
    seen, test = copy.copy(buf), copy.copy(buf)
    seen.rows = test.lo = buf.rows - TEST_ROWS
    held = train.Buffer(train.HOLDOUT_ROWS).load(args.holdout)
    net = (export_weights.load(args.init_weights) if args.init_weights else Net()).to(dev)
    acting = copy.deepcopy(net).requires_grad_(False)
    opt = train.adam(net, args.lr, args.wd)

    def score(step):
        rec = {"step": step, "lr": opt.param_groups[0]["lr"],
               **{f"test_{k}": v for k, v in train.holdout(acting, test, args.micro, dev, gpu_batch.make_batch, args.policy_w).items()},
               **train.holdout(acting, held, args.micro, dev, gpu_batch.make_batch, args.policy_w)}
        with open(f"{args.out}/fit.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    t0 = time.time()
    score(0)
    for step in range(0, steps, REPORT):
        for group in opt.param_groups:
            group["lr"] = args.lr_at(step + REPORT)
        done, stats = train.train_steps(net, acting, opt, seen, min(REPORT, steps - step), args.batch, args.micro,
                                        rng, dev, gpu_batch.make_batch, args.policy_w)
        print(f"[{time.time() - t0:7.0f}s] step {step + done} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()), flush=True)
    score(steps)
    export_weights.bot(args.out, "final", acting, config.git_sha(), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
