import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))

import torch

import config
import export_weights
import warchest
from value_net import Net

WARM = float(os.getenv("WARCHEST_BENCH_WARM", "30"))
FLOP = 1.43e9
PEAK = 142e12


def main():
    seconds = float(sys.argv[1])
    cfg = config.Cfg(**config.parse(sys.argv[2:]))
    torch.manual_seed(cfg.seed)
    net = export_weights.load(cfg.init_weights) if cfg.init_weights else Net()
    net.push()
    devices = list(range(torch.cuda.device_count()))
    farm = warchest.Farm(cfg.seed, cfg.games, devices, sims=cfg.sims, chunk=cfg.collect_rows)
    time.sleep(WARM)
    farm.collect()
    t, at, games, decisions, busy, samples = time.time(), farm.gpu_stats(), 0, 0, [0] * len(devices), 0
    while time.time() - t < seconds:
        d = farm.collect()
        games += d["games"]
        decisions += d["decisions"]
        busy = [b + torch.cuda.utilization(dev) for b, dev in zip(busy, devices)]
        samples += 1
    span = time.time() - t
    got = [(c - c0, r - r0) for (c, r), (c0, r0) in zip(farm.gpu_stats(), at)]
    print(f"[bench] games={cfg.games} sims={cfg.sims} over {span:.0f}s: games/s={games / span:.1f} "
          f"decisions/s={decisions / span:.0f} evals/s={sum(r for _, r in got) / span:.0f}", flush=True)
    for dev, (calls, rows), util in zip(devices, got, busy):
        evals = rows / span
        print(f"[bench] gpu {dev}: evals/s={evals:.0f} TFLOP/s={evals * FLOP / 1e12:.1f} "
              f"({evals * FLOP / PEAK:.0%} of peak) busy={util / samples:.0f}% rows/call={rows / max(calls, 1):.0f}", flush=True)


if __name__ == "__main__":
    main()
