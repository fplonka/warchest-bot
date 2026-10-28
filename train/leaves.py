import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warchest


def main():
    ap = argparse.ArgumentParser(description="Play an NNUE bot against itself on every CPU core and keep about two of its search's leaf positions per move, at each budget.")
    ap.add_argument("--bot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, nargs="+", required=True)
    ap.add_argument("--positions", type=int, required=True, help="leaf positions to keep at each budget")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sims in args.sims:
        t0, parts, kept, seed = time.time(), [], 0, args.seed
        while kept < args.positions:
            rows, desc, off = warchest.nnue_leaves(str(Path(args.bot) / "weights.bin"), sims, 2 / sims, 256, seed)
            parts.append((rows, desc, off.astype(np.int64)))
            kept, seed = kept + len(off) - 1, seed + 1
        base = np.cumsum([0] + [p[2][-1] for p in parts[:-1]])
        np.savez_compressed(out / f"{Path(args.bot).name}-{sims}.npz", rows=np.concatenate([p[0] for p in parts]),
                 desc=np.concatenate([p[1] for p in parts]), off=np.concatenate([[0]] + [p[2][1:] + b for p, b in zip(parts, base)]))
        print(f"{sims} sims: {kept} leaves from {256 * len(parts)} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
