import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

FEATURES, MOVES, STACKS, L2 = warchest.NNUE_FEATURES, warchest.NNUE_MOVES, warchest.NNUE_STACKS, warchest.NNUE_L2
FT_ONE, FT_CLIP, ONE = 256.0, 0.99, 128.0
OUT = ONE * ONE


def straight(x, hard):
    return x + (hard - x).detach()


def weights(w, scale):
    return straight(w, (w * scale).round() / scale)


def acts(x):
    return straight(x, (x * ONE + 1e-5).floor() / ONE).clamp(0.0, 127.0 / ONE)


def bag(table, ids, scale):
    return F.embedding_bag(ids, F.pad(weights(table, scale), (0, 0, 0, 1)), mode="sum", padding_idx=len(table))


def log_softmax(logit, parow, n):
    top = torch.full((n,), -1e30, device=logit.device).scatter_reduce(0, parow, logit.detach(), reduce="amax")
    shifted = logit - top[parow]
    return shifted - torch.zeros(n, device=logit.device).index_add_(0, parow, shifted.exp())[parow].log()


def inputs(rows, desc, parow, device):
    off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))]).astype(np.uint32)
    feats, moves, phases = warchest.nnue_features(rows.ravel(), desc.ravel(), off)
    t = lambda a: torch.as_tensor(a.astype(np.int64)).to(device, non_blocking=True)
    return t(feats.reshape(len(rows), -1)), t(moves.reshape(len(desc), -1)), t(parow), t(phases)


class Student(nn.Module):

    def __init__(self, l1, r, thresholds):
        super().__init__()
        half = l1 // 2
        self.l1, self.r = l1, r
        self.register_buffer("thresholds", torch.as_tensor(thresholds, dtype=torch.long))
        self.ft = nn.Parameter(torch.randn(FEATURES, l1) * 0.065)
        self.ft_bias = nn.Parameter(torch.full((l1,), 0.5))
        self.psqt = nn.Parameter(torch.zeros(FEATURES, STACKS))
        self.fc0 = nn.Linear(half, STACKS * L2)
        self.fc1 = nn.Linear(2 * L2, STACKS * L2)
        self.out = nn.Linear(4 * L2, STACKS)
        self.proj = nn.Linear(half, r)
        self.emb = nn.Parameter(torch.randn(MOVES, r) * 0.1)
        self.head = nn.Parameter(torch.randn(r) * r ** -0.5)
        self.move_bias = nn.Parameter(torch.zeros(MOVES, 1))
        self.clip()

    def stacked(self, layer, x, wscale, stack):
        y = F.linear(x, weights(layer.weight, wscale), weights(layer.bias, wscale * ONE))
        return y.view(len(x), STACKS, -1)[torch.arange(len(x), device=x.device), stack]

    def forward(self, feats, moves, parow, phases):
        acc = bag(self.ft, feats, FT_ONE) + weights(self.ft_bias, FT_ONE)
        a, b = acc.clamp(0.0, 255.0 / FT_ONE).split(self.l1 // 2, dim=1)
        h = acts(a * b)
        stack = torch.searchsorted(self.thresholds, phases, right=True)
        c0 = self.stacked(self.fc0, h, ONE, stack)
        x0 = torch.cat([acts(c0 * c0), acts(c0)], 1)
        c1 = self.stacked(self.fc1, x0, 64.0, stack)
        x1 = torch.cat([acts(c1 * c1), acts(c1)], 1)
        out = self.stacked(self.out, torch.cat([x0, x1], 1), ONE, stack).squeeze(1)
        psqt = bag(self.psqt, feats, OUT)[torch.arange(len(feats), device=feats.device), stack]
        value = out + c0[:, L2 - 2] - c0[:, L2 - 1] + psqt
        z = F.linear(h, weights(self.proj.weight, ONE), weights(self.proj.bias, OUT))[parow] + bag(self.emb, moves, OUT)
        logit = acts(z) @ weights(self.head, ONE) + bag(self.move_bias, moves, OUT).squeeze(1)
        return value, log_softmax(logit, parow, len(feats))

    def clip(self):
        with torch.no_grad():
            for p, bound in ((self.ft, FT_CLIP), (self.ft_bias, FT_CLIP), (self.fc0.weight, 127 / ONE),
                             (self.fc1.weight, 127 / 64), (self.out.weight, 127 / ONE), (self.proj.weight, 127 / ONE),
                             (self.head, 127 / ONE)):
                p.clamp_(-bound, bound)

    def flat(self):
        arrays = (self.ft, self.ft_bias, self.psqt, self.fc0.weight, self.fc0.bias, self.fc1.weight, self.fc1.bias,
                  self.out.weight, self.out.bias, self.proj.weight, self.proj.bias, self.emb, self.head, self.move_bias)
        head = np.array([self.l1, self.r, *self.thresholds.tolist()], np.float32)
        return np.concatenate([head, *(a.detach().cpu().float().numpy().ravel() for a in arrays)])
