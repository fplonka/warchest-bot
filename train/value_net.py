import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

N_HEXES = warchest.N_HEXES
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
PILE_COUNTS = warchest.PILE_COUNTS
CARD_FEATS = warchest.CARD_FEATS
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE
LOOSE = warchest.LOOSE
N_KINDS = warchest.N_KINDS

TYPE = 64
C = 192
BLOCKS = 12
D = 384
HIDDEN = 192
BINS = 16


def gelu(x):
    return F.gelu(x, approximate="tanh")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.card1 = nn.Linear(CARD_FEATS, TYPE)
        self.card2 = nn.Linear(TYPE, TYPE)
        self.pile = nn.Linear(PILE_COUNTS, TYPE, bias=False)
        self.seat = nn.Embedding(2, TYPE)

        self.hex_stem = nn.Linear(HEX_FACTS, C)
        self.tok_stem = nn.Linear(TYPE, C, bias=False)
        self.pos = nn.Embedding(N_HEXES, C)
        self.glob_stem = nn.Linear(LOOSE, C, bias=False)

        self.blk1 = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blkg = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blk2 = nn.ModuleList([nn.Linear(C, C) for _ in range(BLOCKS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln_trunk = nn.LayerNorm(C)
        self.board_out = nn.Linear(2 * C + LOOSE, D)

        self.value_hidden = nn.Linear(D, HIDDEN)
        self.value_out = nn.Linear(HIDDEN, BINS)
        nn.init.zeros_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)
        self.act_kind = nn.Embedding(N_KINDS, C)
        self.act_role = nn.Embedding(5, C)
        self.act_board = nn.Linear(D, C, bias=False)
        self.ln_act = nn.LayerNorm(C)
        self.act_out = nn.Linear(C, 1)
        nn.init.normal_(self.act_kind.weight, std=C ** -0.5)
        nn.init.ones_(self.act_role.weight)

        nb = torch.as_tensor(warchest.hex_neighbours(), dtype=torch.long).view(N_HEXES, 6)
        adj = F.one_hot(nb, N_HEXES + 1).sum(1)[:, :N_HEXES].float()
        self.register_buffer("adj", adj, persistent=False)
        self.register_buffer("seat_of", torch.arange(NTYPE) // NSLOT, persistent=False)
        self.register_buffer("centres", (torch.arange(BINS) + 0.5) / (BINS / 2) - 1, persistent=False)

    def cards(self, xpub):
        facts = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS].reshape(-1, NTYPE, CARD_FEATS)
        return self.card2(gelu(self.card1(facts)))

    def tokens(self, xpub, cards):
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(-1, NTYPE, PILE_COUNTS)
        return cards + self.pile(piles) + self.seat(self.seat_of)

    def trunk(self, xpub, projected):
        batch = xpub.shape[0]
        hexes = xpub[:, :N_HEXES * HEX_CH].reshape(batch, N_HEXES, HEX_CH)
        occupant = hexes[:, :, HEX_FACTS:] @ projected
        type_pool = gelu(projected).mean(1)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        x = (self.hex_stem(hexes[:, :, :HEX_FACTS])
             + occupant
             + self.pos.weight
             + self.glob_stem(loose).unsqueeze(1)
             + type_pool.unsqueeze(1))
        for i in range(BLOCKS):
            a = gelu(self.ln1[i](x))
            y = self.blk1[i](torch.cat([a, self.adj @ a], -1))
            y = y + self.blkg[i](torch.cat([a.mean(1), a.amax(1)], -1)).unsqueeze(1)
            x = x + self.blk2[i](gelu(self.ln2[i](y)))
        return gelu(self.ln_trunk(x))

    def evaluate(self, xpub):
        projected = self.tok_stem(self.tokens(xpub, self.cards(xpub)))
        spatial = self.trunk(xpub, projected)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        board = self.board_out(torch.cat([spatial.mean(1), spatial.amax(1), loose], -1))
        return self.value_out(gelu(self.value_hidden(board))), (board, projected, spatial)

    def actions(self, desc, pieces, row):
        board, projected, spatial = pieces
        card = torch.cat([projected, projected.new_zeros(projected.shape[0], 1, C)], 1)
        hexes = torch.cat([spatial, spatial.new_zeros(spatial.shape[0], 1, C)], 1)
        coin = desc[:, 1:3].long().clamp_max(NTYPE)
        where = desc[:, 3:6].long().clamp_max(N_HEXES)
        entity = torch.stack([
            card[row, coin[:, 0]], card[row, coin[:, 1]],
            hexes[row, where[:, 0]], hexes[row, where[:, 1]], hexes[row, where[:, 2]],
        ], 1)
        present = desc[:, 1:6].lt(255).unsqueeze(-1)
        local = (entity * self.act_role.weight * present).sum(1)
        z = self.act_kind(desc[:, 0].long()) + local + self.act_board(board)[row]
        return self.act_out(gelu(self.ln_act(z))).squeeze(-1)

    def value(self, logits):
        return logits.softmax(-1) @ self.centres

    def tensors(self):
        blocks = [m for i in range(BLOCKS) for m in (self.blk1[i], self.blkg[i], self.blk2[i])]
        mats = [
            self.card1, self.card2, self.pile, self.seat,
            self.hex_stem, self.tok_stem, self.pos, self.glob_stem,
            *blocks,
            self.board_out, self.value_hidden, self.value_out,
            self.act_kind, self.act_role, self.act_board, self.act_out,
        ]
        norms = [n for i in range(BLOCKS) for n in (self.ln1[i], self.ln2[i])]
        norms += [self.ln_trunk, self.ln_act]
        w = [m.weight if isinstance(m, nn.Embedding) else m.weight.t() for m in mats]
        b = [m.bias for m in mats if not isinstance(m, nn.Embedding) and m.bias is not None]
        ln = [x for n in norms for x in (n.weight, n.bias)]
        return w, b, ln

    def flat(self):
        def raw(t):
            return t.detach().cpu().contiguous().numpy().ravel()
        return tuple(np.ascontiguousarray(np.concatenate([raw(t) for t in ts]), np.float32)
                     for ts in self.tensors())

    def load_flat(self, arrays):
        with torch.no_grad():
            for flat, ts in zip(arrays, self.tensors(), strict=True):
                sizes = [t.numel() for t in ts]
                assert sum(sizes) == len(flat), "weights layout mismatch"
                for t, chunk in zip(ts, torch.from_numpy(flat.copy()).split(sizes)):
                    t.copy_(chunk.reshape(t.shape))
        return self

    def push(self):
        warchest.set_weights(*self.flat())
