import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))

import torch

import config
import warchest
from value_net import Net

WARM = float(os.getenv("WARCHEST_BENCH_WARM", "30"))
GPU = ("gpu_rows", "gpu_calls", "gpu_nanos")


def main():
    seconds = float(sys.argv[1])
    args = [kv for kv in sys.argv[2:] if kv != "prof"]
    prof = len(args) != len(sys.argv) - 2
    cfg = config.Cfg(**config.parse(args))
    torch.manual_seed(cfg.seed)
    net = Net()
    if cfg.init_weights:
        net.load_state_dict(torch.load(cfg.init_weights, map_location="cpu", weights_only=False)["acting"])
    net.push()
    devices = [int(d) for d in cfg.gen_devices.split(",")]
    farm = warchest.Farm(cfg.seed, cfg.games, devices, sims=cfg.sims, batch=cfg.search_batch,
                         puct=cfg.puct, explore=cfg.explore, temp=cfg.temp)
    t0, start, at, rows, decisions = time.time(), None, {}, 0, 0
    while True:
        d = farm.collect(cfg.collect_rows)
        now = time.time()
        if start is None:
            if now - t0 >= WARM:
                start, at = now, {k: int(d[k]) for k in GPU}
                if prof:
                    torch.cuda.cudart().cudaProfilerStart()
            continue
        rows += len(d["z"])
        decisions += int(d["decisions"])
        if now - start > seconds:
            break
    if prof:
        torch.cuda.cudart().cudaProfilerStop()
    span, lanes = now - start, warchest.PIPELINE * len(devices)
    got = {k: int(d[k]) - at[k] for k in GPU}
    print(f"[bench] games={cfg.games} sims={cfg.sims} batch={cfg.search_batch} over {span:.0f}s: "
          f"evals/s={got['gpu_rows'] / span:.0f} "
          f"sims/s={rows * cfg.sims / span:.0f} "
          f"decisions/s={decisions / span:.1f} rows/s={rows / span:.1f} "
          f"rows/call={got['gpu_rows'] / max(got['gpu_calls'], 1):.0f} "
          f"lane_busy={1e-9 * got['gpu_nanos'] / span / lanes:.2f}", flush=True)


if __name__ == "__main__":
    main()
