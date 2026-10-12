import numpy as np
import torch

import mirror
import warchest
from test_parity import positions, random_net
from value_net import D, DH, LAYERS, Net


def outputs(net, x, desc, parow):
    win, short, tokens = net.evaluate(x)
    return win, short, net.actions(desc, tokens, parow)


@torch.no_grad()
def main():
    rows, desc, off = positions(4, 11)
    parow = np.repeat(np.arange(len(rows)), np.diff(off))
    rows, desc = mirror.to_mover(rows, desc, parow)
    x = torch.as_tensor(np.asarray(warchest.expand_rows(rows.ravel()), np.float64).reshape(len(rows), -1))
    batch = x, torch.as_tensor(desc), torch.as_tensor(parow)

    for width, depth in ((D, LAYERS // 2), (D // 2, LAYERS), (D // 2, LAYERS // 2)):
        small = random_net(3, width).double()
        small.blocks = small.blocks[:depth]
        grown = Net().double().grow(small.state_dict())
        want, got = outputs(small, *batch), outputs(grown, *batch)
        assert want[0].std() > 0.1
        assert not torch.equal(*grown.move2.weight.chunk(2, 1))
        for a, b in zip(want, got, strict=True):
            torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-9)

    try:
        Net().grow(Net(D - DH).state_dict())
    except RuntimeError:
        pass
    else:
        raise AssertionError("a width that is not a whole multiple was accepted")
    print("growth OK")


if __name__ == "__main__":
    main()
