import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import export_weights
import shards
import warchest

SHARD = 1 << 20


def main():
    ap = argparse.ArgumentParser(description="Play a frozen snapshot's self-play at its own settings and write a sample of every position its net evaluates, with the net's value and prior, to shards.")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, required=True, help="share of evaluated positions to write")
    ap.add_argument("--devices", type=int, nargs="+", required=True)
    ap.add_argument("--free-gb", type=float, default=8.0, help="stop when the disk has this little left")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.Cfg(**torch.load(args.teacher, map_location="cpu", weights_only=False)["cfg"])
    playout = cfg.playout_at(float("inf"))
    export_weights.load(args.teacher).push()
    gen = warchest.Generator(args.seed, cfg.games, args.devices, *playout, args.keep, cfg.collect_rows)
    print(f"{args.teacher} at sims, fast_sims, p_full = {playout}, {cfg.games} games, keeping {args.keep:.0%}", flush=True)
    t0, parts, total, index = time.time(), [], 0, len(list(out.glob("*.npz")))
    while shutil.disk_usage(out).free > args.free_gb * 2 ** 30:
        d = gen.take()
        parts.append(d)
        if sum(len(p["value"]) for p in parts) < SHARD:
            continue
        shard = shards.merge(parts)
        shards.write(out / f"{index:06d}.npz", shard)
        total, index, parts = total + len(shard["value"]), index + 1, []
        print(f"{total} positions in {index} shards, {total / (time.time() - t0):.0f}/s, "
              f"{shutil.disk_usage(out).free / 2 ** 30:.1f} GB free", flush=True)


if __name__ == "__main__":
    main()
