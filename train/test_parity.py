import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest
from gpu_batch import expand
from value_net import Net

ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES


def random_net(seed):
    torch.manual_seed(seed)
    net = Net()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.08)
        for module in net.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.normal_(1, 0.2)
                module.bias.normal_(0, 0.1)
    return net


def positions(games, seed):
    rows, desc, off = warchest.sample_positions(games, seed)
    return (np.frombuffer(bytes(rows), np.uint8).reshape(-1, ROW_BYTES).copy(),
            np.frombuffer(bytes(desc), np.uint8).reshape(-1, ACT_BYTES).copy(),
            np.asarray(off, np.int64))


def packed_row_cuda_parity(device):
    rows = positions(128, 19)[0][:2048]
    mirrored = warchest.mirror_rows(rows.ravel()).reshape(rows.shape)
    rows = np.concatenate([rows, mirrored]).copy()
    want = np.asarray(warchest.expand_rows(rows.ravel()), np.float32).reshape(len(rows), warchest.PUBFEAT)
    got = expand(rows, device).cpu().numpy()
    exact = (want == 0.0) | (want == 1.0)
    assert np.array_equal(got[exact], want[exact]), "one-hot expansion drift"
    np.testing.assert_allclose(got[~exact], want[~exact], rtol=0, atol=1e-6)
    print(f"packed-row CUDA parity ok: {len(rows)} rows, mirrors included")


@torch.no_grad()
def evaluation_parity(device):
    net = random_net(7)
    net.push()
    rows, desc, off = positions(12, 5)
    value, prior = warchest.evaluate(rows.ravel().tolist(), desc.ravel().tolist(), off.tolist(), device.index)
    net = net.to(device).eval()
    logits, pieces = net.evaluate(expand(rows, device))
    parow = torch.as_tensor(np.repeat(np.arange(len(rows)), np.diff(off)), device=device)
    logit = net.actions(torch.as_tensor(desc, device=device), pieces, parow)
    top = torch.full((len(rows),), -1e30, device=device).scatter_reduce(0, parow, logit, reduce="amax")
    e = (logit - top[parow]).exp()
    want_prior = e / torch.zeros(len(rows), device=device).index_add_(0, parow, e)[parow]
    for name, got, want in (("value", value, net.value(logits)), ("prior", prior, want_prior)):
        err = np.abs(np.asarray(got) - want.cpu().numpy())
        assert err.max() < 2e-2 and err.mean() < 2e-3, f"{name} drift: max {err.max():.4f} mean {err.mean():.5f}"
        print(f"{name} CUDA parity ok over {len(err)}: max {err.max():.2e} mean {err.mean():.2e}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requires a working GPU")
    device = torch.device("cuda:0")
    packed_row_cuda_parity(device)
    evaluation_parity(device)


if __name__ == "__main__":
    main()
