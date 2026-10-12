import numpy as np
import torch

import mirror
import warchest
from test_parity import positions, random_net
from value_net import LAYERS, Net


def outputs(net, x, desc, parow):
    win, short, tokens = net.evaluate(x)
    return win, short, tokens, net.actions(desc, tokens, parow), *net.aux_heads(tokens)


@torch.no_grad()
def main():
    rows, desc, off = positions(4, 11)
    parow = np.repeat(np.arange(len(rows)), np.diff(off))
    rows, desc = mirror.to_mover(rows, desc, parow)
    x = torch.as_tensor(np.asarray(warchest.expand_rows(rows.ravel()), np.float32).reshape(len(rows), -1))
    batch = x, torch.as_tensor(desc), torch.as_tensor(parow)

    small = random_net(3)
    small.blocks = small.blocks[:LAYERS // 2]
    grown = Net().grow(small.state_dict())
    assert len(grown.blocks) == LAYERS
    want, got = outputs(small, *batch), outputs(grown, *batch)
    assert want[0].std() > 0.1
    for a, b in zip(want, got, strict=True):
        assert torch.equal(a, b)

    try:
        Net().grow(small.state_dict() | {"ln.weight": torch.ones(small.ln.weight.numel() - 1)})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a narrower net was accepted")
    print("depth growth OK")


if __name__ == "__main__":
    main()
