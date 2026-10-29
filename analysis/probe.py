import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import distill
import nnue
import shards


class Probe(nn.Module):

    def __init__(self, width, rank):
        super().__init__()
        self.ft = nn.EmbeddingBag(nnue.FEATURES, width, mode="sum")
        nn.init.normal_(self.ft.weight, std=0.065)
        self.value = nn.Sequential(nn.ReLU(), nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.proj = nn.Linear(width, rank)
        self.emb = nn.EmbeddingBag(nnue.MOVES, rank, mode="sum")
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(2 * rank, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, feats, moves, parow, phases):
        h = self.ft(*feats)
        logit = self.head(torch.cat([self.proj(F.relu(h))[parow], self.emb(*moves)], 1)).squeeze(1)
        return self.value(h).squeeze(1), nnue.log_softmax(logit, parow, len(h))


def main():
    ap = argparse.ArgumentParser(description="Train a float probe net with deep value and move tails on the NNUE's own inputs, to see whether the inputs or the NNUE's shape limit its accuracy.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0, rng = time.time(), np.random.default_rng(0)
    probe = Probe(args.width, args.rank).to(args.device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    decay = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - step / args.steps)
    ready = shards.Stream([args.data], 24, rng).prefetch(16384, lambda parts: distill.batch(parts, args.device))
    held = distill.holdout(args.data)
    for step in range(1, args.steps + 1):
        g = distill.gaps(probe, ready.get())
        opt.zero_grad(set_to_none=True)
        (4 * g["value"] + g["policy"]).backward()
        opt.step()
        decay.step()
        if step % (args.steps // 20) == 0:
            g = distill.scores([probe], held, args.device)[0]
            print(f"step {step} {time.time() - t0:.0f}s held-out KL to the teacher: value {g['value']:.4f} policy {g['policy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
