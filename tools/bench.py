import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))

import numpy as np
import torch

import config
import warchest
from value_net import Net

WARM = float(os.getenv("WARCHEST_BENCH_WARM", "45"))


def main():
    seconds = float(sys.argv[1])
    args = [kv for kv in sys.argv[2:] if kv != "prof"]
    prof = len(args) != len(sys.argv) - 2
    cfg = config.Cfg(**config.parse(args))
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    net = Net()
    if cfg.init_weights:
        net.load_state_dict(torch.load(cfg.init_weights, map_location="cpu", weights_only=False)["acting"])
    net.push()
    farm = warchest.SolveFarm(
        cfg.seed, config.workers(cfg.gen_workers), s=cfg.s, c=cfg.c, batch=cfg.round_batch,
        rounds=cfg.rounds, explore=cfg.explore, random_draft=cfg.random_draft,
        cfr=cfg.cfr, p_td1=cfg.p_td1, query_rate=cfg.query_rate,
        recursive_rate=cfg.recursive_rate,
        devices=[int(d) for d in cfg.gen_devices.split(",")], slots=cfg.gen_slots)

    keys = ("rounds", "round_calls", "round_rows", "round_nanos", "budget_hits", "host_nanos", "grow_nanos")
    solves, t0, warm, at = 0, time.time(), None, {}
    while True:
        d = farm.collect(cfg.gen_solves)
        now = time.time()
        if warm is None:
            if now - t0 < WARM:
                continue
            warm, solves = now, 0
            at = {k: int(d[k]) for k in keys}
            d["shapes"]
            print(f"[warm] slots={d['slots']} spc={d['slots_per_card']} "
                  f"slot={d['slot_bytes'] / (1 << 20):.0f}MiB", flush=True)
            if prof:
                torch.cuda.cudart().cudaProfilerStart()
            continue
        solves += int(d["solves"])
        if now - warm > seconds:
            break
    if prof:
        torch.cuda.cudart().cudaProfilerStop()
    span = {k: int(d[k]) - at[k] for k in keys}
    n = max(span["rounds"], 1)
    print(f"[bench] s={cfg.s} rounds={cfg.rounds} slots={d['slots']} "
          f"{solves / (now - warm):.2f} solves/s over {now - warm:.0f}s "
          f"rounds/s={n / (now - warm):.1f} "
          f"calls/round={span['round_calls'] / n:.1f} "
          f"rows/round={span['round_rows'] / n:.0f} "
          f"ms/round={1e-6 * span['round_nanos'] / n:.1f} "
          f"hits/solve={span['budget_hits'] / max(solves, 1):.3f} "
          f"host_cores={1e-9 * span['host_nanos'] / (now - warm):.2f} "
          f"grow_cores={1e-9 * span['grow_nanos'] / (now - warm):.2f}", flush=True)
    ent = list(warchest.ENT_NAMES)
    a = np.asarray(d["shapes"] or [[0] * 10], np.float64)
    mean = a.mean(0)
    caps = [max(1, c * cfg.s // 512) for c in warchest.ENT_CAPS_512]
    print("[caps] " + "  ".join(
        f"{n}={np.percentile(a[:, i], 99):.0f}/{a[:, i].max():.0f} of {caps[i]}"
        for i, n in enumerate(ent)), flush=True)
    print("[shape/solve] " + "  ".join(f"{n}={mean[i]:.0f}" for i, n in enumerate(ent))
          + f"  cell/node={mean[1] / max(mean[0], 1):.2f}"
          + f"  reach/node={mean[2] / max(mean[0], 1):.2f}"
          + f"  cidx/row={mean[7] / max(mean[4], 1):.2f}"
          + f"  row/board={mean[4] / max(mean[5], 1):.2f}", flush=True)


if __name__ == "__main__":
    main()
