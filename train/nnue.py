import numpy as np
import torch
import torch.nn as nn

import warchest

H, FEATURES, MOVES, CLIP = warchest.NNUE_H, warchest.NNUE_FEATURES, warchest.NNUE_MOVES, warchest.NNUE_CLIP


def bag(n, width):
    return nn.EmbeddingBag(n + 1, width, mode="sum", padding_idx=n)


def log_softmax(logit, parow, n):
    top = torch.full((n,), -1e30, device=logit.device).scatter_reduce(0, parow, logit.detach(), reduce="amax")
    shifted = logit - top[parow]
    return shifted - torch.zeros(n, device=logit.device).index_add_(0, parow, shifted.exp())[parow].log()


def inputs(rows, desc, parow, device):
    off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))]).astype(np.uint32)
    feats, moves = warchest.nnue_features(rows.ravel(), desc.ravel(), off)
    t = lambda a: torch.as_tensor(a.astype(np.int64)).to(device, non_blocking=True)
    return t(feats.reshape(len(rows), -1)), t(moves.reshape(len(desc), -1)), t(parow)


class Student(nn.Module):

    def __init__(self):
        super().__init__()
        self.ft = bag(FEATURES, H)
        self.ft_bias = nn.Parameter(torch.zeros(H))
        self.value = nn.Linear(H, 1)
        self.policy = bag(MOVES, H)
        self.policy_bias = bag(MOVES, 1)
        for weight, fan_in in ((self.ft.weight, 64), (self.value.weight, H), (self.policy.weight, H)):
            nn.init.normal_(weight, std=(2 / fan_in) ** 0.5)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.policy_bias.weight)
        self.clip()

    def forward(self, feats, moves, parow):
        hl = (self.ft(feats) + self.ft_bias).clamp(0, 1)
        logit = (hl[parow] * self.policy(moves)).sum(-1) + self.policy_bias(moves).squeeze(-1)
        return self.value(hl).squeeze(-1), log_softmax(logit, parow, len(hl))

    def clip(self):
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(-CLIP, CLIP)

    def flat(self):
        arrays = (self.ft.weight[:FEATURES], self.ft_bias, self.value.weight, self.value.bias,
                  self.policy.weight[:MOVES], self.policy_bias.weight[:MOVES])
        return np.concatenate([a.detach().cpu().float().numpy().ravel() for a in arrays])
