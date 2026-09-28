import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

import numpy as np
import torch

import config
import gpu_batch
import train
from value_net import Net


def main():
    steps, cfg = int(sys.argv[1]), config.Cfg(**config.parse(sys.argv[2:]))
    dev = torch.device(cfg.device)
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")
    value, acting = Net().to(dev), Net().to(dev).requires_grad_(False)
    opt, buf, rng = train.adam(value, cfg.lr, cfg.wd), train.Buffer(cfg.cap), np.random.default_rng(cfg.seed)
    train.ingest(buf, train.warchest.warm(cfg.warm_games, cfg.seed, temp=cfg.warm_temp))
    run = lambda n: train.train_steps(value, acting, opt, buf, n, cfg.batch, rng, dev, gpu_batch.make_batch,
                                      cfg.policy_w, float("inf"))
    run(steps // 4)
    torch.cuda.synchronize()
    t = time.time()
    _, stats = run(steps)
    span = time.time() - t
    print(f"[bench_train] batch={cfg.batch} {steps} steps in {span:.1f}s: rows/s={steps * cfg.batch / span:.0f} "
          f"Lv={stats['value_loss']:.4f} Lp={stats['policy_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
