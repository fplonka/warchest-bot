import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import export_weights
import mirror
import warchest
from gpu_batch import expand
from value_net import D, Net

ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES


def random_net(seed, d=D):
    torch.manual_seed(seed)
    net = Net(d)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.08)
        for module in net.modules():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0, 1.6 * module.in_features ** -0.5)
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


def reference(net, rows, desc, parow, dtype):
    with torch.autocast("cuda", dtype, enabled=dtype is not None):
        win, _, x = net.evaluate(expand(rows, parow.device))
        logit = net.actions(desc, x, parow).float()
    top = torch.full((len(rows),), -1e30, device=parow.device).scatter_reduce(0, parow, logit, reduce="amax")
    e = (logit - top[parow]).exp()
    prior = e / torch.zeros(len(rows), device=parow.device).index_add_(0, parow, e)[parow]
    return net.value(win.float()).cpu().numpy(), prior.cpu().numpy()


@torch.no_grad()
def evaluation_parity(name, net, device):
    net.push()
    rows, desc, off = positions(12, 5)
    got = warchest.evaluate(rows.ravel(), desc.ravel(), off.astype(np.uint32), device.index)
    parow = np.repeat(np.arange(len(rows)), np.diff(off))
    rows, desc = mirror.to_mover(rows, desc, parow)
    net = net.to(device).eval()
    desc, parow = torch.as_tensor(desc, device=device), torch.as_tensor(parow, device=device)
    want = reference(net, rows, desc, parow, None)
    trained = reference(net, rows, desc, parow, torch.bfloat16)
    for head, cuda, bf16, exact in zip(("value", "prior"), got, trained, want):
        err, slack = np.abs(np.asarray(cuda) - exact), np.abs(bf16 - exact)
        assert err.max() < slack.max() and err.mean() < slack.mean(), \
            f"{name} {head} drifts more than bf16: max {err.max():.2e} vs {slack.max():.2e}, mean {err.mean():.2e} vs {slack.mean():.2e}"
        print(f"{name} {head} CUDA parity ok over {len(err)}: max {err.max():.2e} mean {err.mean():.2e} "
              f"(bf16: {slack.max():.2e} {slack.mean():.2e})")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requires a working GPU")
    device = torch.device("cuda:0")
    packed_row_cuda_parity(device)
    evaluation_parity("random", random_net(7), device)
    for path in sys.argv[1:]:
        evaluation_parity(path, export_weights.load(path), device)


if __name__ == "__main__":
    main()
