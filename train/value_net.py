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
SHORT = len(warchest.HORIZONS) - 1

TYPE = 64
D = 384
DH = 64
LAYERS = 10
TOKENS = N_HEXES + NTYPE + 1
STATE = TOKENS - 1
NONE = TOKENS
INDEX = torch.tensor(warchest.ATTENTION_INDEX).view(TOKENS, TOKENS)


def gelu(x):
    return F.gelu(x, approximate="tanh")


def widen(v, shape, d, f, fan_in):
    for axis, (n, m) in enumerate(zip(v.shape, shape)):
        if m == f * n != n:
            v = torch.stack([v.unflatten(axis, (-1, min(n, d)))] * f, axis + 1)
            if fan_in and axis == 1:
                e = 0.1 * torch.randn_like(v)
                v = v * (1 + e - e.mean(axis + 1, keepdim=True)) / f
            v = v.flatten(axis, axis + 2)
    return v


class Block(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(d // DH, int(INDEX.max()) + 1))
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)

    def forward(self, x, index):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, -1, DH).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, index].to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(gelu(self.up(self.ln2(x))))


class Net(nn.Module):

    def __init__(self, d=D):
        super().__init__()
        self.card1 = nn.Linear(CARD_FEATS, TYPE)
        self.card2 = nn.Linear(TYPE, d)
        self.pile = nn.Linear(PILE_COUNTS, d, bias=False)
        self.seat = nn.Embedding(2, d)
        self.hex_stem = nn.Linear(HEX_FACTS, d)
        self.pos = nn.Embedding(N_HEXES, d)
        self.state_stem = nn.Linear(LOOSE, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(LAYERS))
        self.ln = nn.LayerNorm(d)

        self.value_hidden = nn.Linear(d, d)
        self.value_out = nn.Linear(d, 1)
        self.short_out = nn.Linear(d, SHORT)
        for head in (self.value_out, self.short_out):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.kind = nn.Embedding(N_KINDS, d)
        self.none = nn.Parameter(torch.randn(d))
        self.move1 = nn.Linear(7 * d, d)
        self.move2 = nn.Linear(d, 1)

        self.register_buffer("index", INDEX.clone(), persistent=False)
        self.register_buffer("seat_of", torch.arange(NTYPE) // NSLOT, persistent=False)

    def evaluate(self, xpub):
        b = xpub.shape[0]
        hexes = xpub[:, :N_HEXES * HEX_CH].reshape(b, N_HEXES, HEX_CH)
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(b, NTYPE, PILE_COUNTS)
        cards = xpub[:, OFF_CARDS:OFF_LOOSE].reshape(b, NTYPE, CARD_FEATS)
        coins = self.card2(gelu(self.card1(cards))) + self.pile(piles) + self.seat(self.seat_of)
        board = self.hex_stem(hexes[..., :HEX_FACTS]) + hexes[..., HEX_FACTS:] @ coins + self.pos.weight
        state = self.state_stem(xpub[:, OFF_LOOSE:]).unsqueeze(1)
        x = torch.cat([board, coins, state], 1)
        for block in self.blocks:
            x = block(x, self.index)
        x = self.ln(x)
        hidden = gelu(self.value_hidden(x[:, STATE]))
        return self.value_out(hidden).squeeze(-1), torch.tanh(self.short_out(hidden)), x

    def moves(self, desc, x, row):
        tokens = torch.cat([x, self.none.expand(x.shape[0], 1, -1)], 1)
        desc = desc.long()
        coins = torch.where(desc[:, 1:3] < NTYPE, desc[:, 1:3] + N_HEXES, NONE)
        hexes = torch.where(desc[:, 3:6] < N_HEXES, desc[:, 3:6], NONE)
        state = torch.full_like(coins[:, :1], STATE)
        picked = tokens[row.unsqueeze(1), torch.cat([coins, hexes, state], 1)]
        return self.move1(torch.cat([self.kind(desc[:, 0]), picked.flatten(1)], 1))

    def actions(self, desc, x, row):
        return self.move2(gelu(self.moves(desc, x, row))).squeeze(-1)

    def value(self, logit):
        return 2 * torch.sigmoid(logit) - 1

    def flat(self):
        return nn.utils.parameters_to_vector(self.parameters()).detach().float().cpu().numpy()

    def load_flat(self, flat):
        nn.utils.vector_to_parameters(torch.as_tensor(flat), self.parameters())
        return self

    def grow(self, state):
        d = len(state["ln.weight"])
        f = len(self.ln.weight) // d
        linear = {f"{name}.weight" for name, m in self.named_modules() if isinstance(m, nn.Linear)}
        wide = {k: widen(v, self.get_parameter(k).shape, d, f, k in linear) for k, v in state.items()}
        depth = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
        fresh = {f"blocks.{i}.{k}": torch.zeros_like(v) if k.startswith(("proj.", "down.")) else v
                 for i in range(depth, LAYERS) for k, v in self.blocks[i].state_dict().items()}
        self.load_state_dict(wide | fresh)
        return self

    def push(self):
        warchest.set_weights(self.flat())
