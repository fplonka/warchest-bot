import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_weights
import shards
import warchest


def main():
    ap = argparse.ArgumentParser(description="Play an NNUE bot against itself on every CPU core, keep some of its search's leaf positions per move, label them with the teacher's net and write shards.")
    ap.add_argument("--bot", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, nargs="+", required=True)
    ap.add_argument("--positions", type=int, required=True, help="leaf positions to write at each budget")
    ap.add_argument("--per-move", type=float, default=8.0, help="leaf positions to keep per move, on average")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_weights.load(args.teacher).push()
    weights = str(Path(args.bot) / "weights.bin")
    for sims in args.sims:
        t0, kept, seed = time.time(), 0, args.seed
        while kept < args.positions:
            d = warchest.nnue_leaves(weights, sims, args.per_move / sims, 256, seed)
            d["value"], d["prior"] = warchest.evaluate(d["rows"], d["desc"], d["off"].astype(np.uint32), args.device)
            shards.write(out / f"{Path(args.bot).name}-{sims}-{seed:06d}.npz", d)
            kept, seed = kept + len(d["value"]), seed + 1
        print(f"{sims} sims: {kept} leaves in {seed - args.seed} shards, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
