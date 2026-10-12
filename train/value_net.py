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
LAYERS = 10
HEADS = 6
TOKENS = N_HEXES + NTYPE + 1
STATE = TOKENS - 1
NONE = TOKENS
INDEX = torch.tensor(warchest.ATTENTION_INDEX).view(TOKENS, TOKENS)


def gelu(x):
    return F.gelu(x, approximate="tanh")


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.bias = nn.Parameter(torch.zeros(HEADS, int(INDEX.max()) + 1))
        self.ln2 = nn.LayerNorm(D)
        self.up = nn.Linear(D, 2 * D)
        self.down = nn.Linear(2 * D, D)

    def forward(self, x, index):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, index].to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + self.down(gelu(self.up(self.ln2(x))))


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.card1 = nn.Linear(CARD_FEATS, TYPE)
        self.card2 = nn.Linear(TYPE, D)
        self.pile = nn.Linear(PILE_COUNTS, D, bias=False)
        self.seat = nn.Embedding(2, D)
        self.hex_stem = nn.Linear(HEX_FACTS, D)
        self.pos = nn.Embedding(N_HEXES, D)
        self.state_stem = nn.Linear(LOOSE, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D)

        self.value_hidden = nn.Linear(D, D)
        self.value_out = nn.Linear(D, 1)
        self.short_out = nn.Linear(D, SHORT)
        for head in (self.value_out, self.short_out):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.kind = nn.Embedding(N_KINDS, D)
        self.none = nn.Parameter(torch.randn(D))
        self.move1 = nn.Linear(7 * D, D)
        self.move2 = nn.Linear(D, 1)

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
        tokens = torch.cat([x, self.none.expand(x.shape[0], 1, D)], 1)
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
        depth = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
        fresh = {f"blocks.{i}.{k}": torch.zeros_like(v) if k.startswith(("proj.", "down.")) else v
                 for i in range(depth, LAYERS) for k, v in self.blocks[i].state_dict().items()}
        self.load_state_dict(state | fresh)
        return self

    def push(self):
        warchest.set_weights(self.flat())
