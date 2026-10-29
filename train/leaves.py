import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_weights
import shards
import warchest

SHARD = 1 << 20


def main():
    ap = argparse.ArgumentParser(description="Play an NNUE bot against itself on every CPU core, keep some of its search's leaf positions per move, label them with the teacher's net and write shards.")
    ap.add_argument("--bot", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--per-move", type=float, required=True, help="leaf positions to keep per move, on average")
    ap.add_argument("--free-gb", type=float, default=8.0, help="stop when the disk has this little left")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_weights.load(args.teacher).push()
    weights, name = str(Path(args.bot) / "weights.bin"), Path(args.bot).name
    t0, parts, kept, games, seed, index = time.time(), [], 0, 0, args.seed, 0
    while shutil.disk_usage(out).free > args.free_gb * 2 ** 30:
        d = warchest.nnue_leaves(weights, args.sims, args.per_move / args.sims, 256, seed)
        d["value"], d["prior"] = warchest.evaluate(d["rows"], d["desc"], d["off"].astype(np.uint32), args.device)
        parts.append(d)
        games, seed = games + 256, seed + 1
        if sum(len(p["value"]) for p in parts) < SHARD:
            continue
        shard = shards.merge(parts)
        shards.write(out / f"{name}-{args.sims}-s{args.seed}-{index:06d}.npz", shard)
        kept, parts, index = kept + len(shard["value"]), [], index + 1
        print(f"{kept} leaves from {games} games at {args.sims} sims, {kept / (time.time() - t0):.0f}/s, "
              f"{shutil.disk_usage(out).free / 2 ** 30:.1f} GB free", flush=True)


if __name__ == "__main__":
    main()
