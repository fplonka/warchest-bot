import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import export_weights
import warchest


def main():
    ap = argparse.ArgumentParser(description="Play a frozen snapshot's self-play at its own search settings and write every recorded row to shards.")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    shards = Path(args.out) / "shards"
    shards.mkdir(parents=True, exist_ok=False)
    cfg = config.Cfg(**torch.load(args.teacher, map_location="cpu", weights_only=False)["cfg"])
    playout = cfg.playout_at(float("inf"))
    export_weights.load(args.teacher).push()
    farm = warchest.Farm(args.seed, cfg.games, list(range(torch.cuda.device_count())), *playout, chunk=cfg.collect_rows)
    print(f"{args.teacher} at sims, fast_sims, p_full = {playout}, {cfg.games} games at once", flush=True)
    t0, rows, games, shard = time.time(), 0, 0, 0
    while rows < args.rows:
        d = farm.collect()
        np.savez_compressed(shards / f"{shard:06d}.npz",
                            **{k: np.asarray(d[k]) for k in ("rows", "desc", "off", "target", "value", "z")})
        rows, games, shard = rows + len(d["value"]), games + d["games"], shard + 1
        print(f"{rows} rows from {games} games, {rows / (time.time() - t0):.0f} rows/s", flush=True)


if __name__ == "__main__":
    main()
