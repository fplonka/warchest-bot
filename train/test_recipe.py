import importlib.util
import io
import os
import pathlib
import sys
import types

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch

import config

HOUR = 3600.0
HORIZONS = [float("inf"), 6.0, 16.0, 50.0]
Z = len(HORIZONS) + 2 + 3


def load_train_module():
    warchest = types.ModuleType("warchest")
    warchest.ROW_BYTES, warchest.ACT_BYTES, warchest.HORIZONS, warchest.N_LOCATIONS, warchest.NSLOT = 7, 6, HORIZONS, 2, 3
    value_net = types.ModuleType("value_net")
    value_net.Net = object
    sys.modules.update({"warchest": warchest, "value_net": value_net,
                        "config": types.ModuleType("config"), "export_weights": types.ModuleType("export_weights")})
    path = pathlib.Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("train_recipe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lr_warms_up_holds_the_peak_then_cools_down():
    cfg = config.Cfg(minutes=20160.0)
    assert cfg.lr_at(0.0) == 0.0
    assert abs(cfg.lr_at(5 * 60.0) - cfg.lr / 2) < 1e-12
    for hours in (1, 100, 0.9 * 336):
        assert cfg.lr_at(hours * HOUR) == cfg.lr
    assert abs(cfg.lr_at(0.95 * 336 * HOUR) - (cfg.lr + cfg.lr_final) / 2) < 1e-12
    assert cfg.lr_at(336 * HOUR) == cfg.lr_final
    assert cfg.lr_at(400 * HOUR) == cfg.lr_final


def test_playout_switches_at_half_the_run():
    cfg = config.Cfg(minutes=20160.0)
    assert cfg.playout_at(0.0) == cfg.playout_at(167.9 * HOUR) == (32, 16, 1.0)
    assert cfg.playout_at(168 * HOUR) == cfg.playout_at(400 * HOUR) == (128, 16, 0.5)
    short = config.Cfg(minutes=20.0)
    assert short.playout_at(599.0)[2] == 1.0 and short.playout_at(600.0)[2] == 0.5


class Net:
    def evaluate(self, x):
        return x[:, 0], x[:, 1:len(HORIZONS)].tanh(), x

    def actions(self, desc, pieces, parow):
        return desc.float().sum(1)

    def aux_heads(self, x):
        return x[:, len(HORIZONS):Z - 3], x[:, Z - 3:Z]

    def value(self, logit):
        return logit


def test_value_only_rows_leave_the_policy_loss_alone():
    train = load_train_module()
    torch.manual_seed(0)
    x, z = torch.randn(5, 16), torch.rand(5, Z) * 2 - 1
    desc = torch.randint(0, 9, (9, 6), dtype=torch.uint8)
    parow = torch.tensor([0, 0, 0, 1, 1, 3, 3, 3, 3])
    target = torch.tensor([0.2, 0.5, 0.3, 0.9, 0.1, 0.25, 0.25, 0.25, 0.25])
    _, mixed = train.losses(Net(), x, desc, parow, target, z, 1.0)
    taught = [0, 1, 3]
    _, alone = train.losses(Net(), x[taught], desc, torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]), target, z[taught], 1.0)
    for key in ("policy_loss", "policy_target_entropy", "policy_kl"):
        assert torch.allclose(mixed[key], alone[key]), key


def test_a_saved_buffer_samples_as_before():
    train = load_train_module()
    rng = np.random.default_rng(0)
    buf = train.Buffer(16)
    for n in rng.integers(1, 6, 9):
        off = np.concatenate([[0], np.cumsum(rng.integers(0, 120, n))])
        buf.add(rng.integers(0, 256, (n, 7), np.uint8), rng.integers(0, 256, (off[-1], 6), np.uint8), off,
                rng.random(off[-1]).astype(np.float16), rng.random((n, Z), np.float32))
    saved = io.BytesIO()
    buf.save(saved)
    saved.seek(0)
    back = train.Buffer(16)
    back.load(saved)
    assert buf.lo > 0 and buf.acts > buf.acap
    assert (back.lo, back.rows, back.acts) == (buf.lo, buf.rows, buf.acts)
    ids = np.arange(buf.lo, buf.rows)
    for a, b in zip(buf.gather(ids), back.gather(ids)):
        assert np.array_equal(a, b)
    buf.z = buf.z[:, :len(HORIZONS)]
    saved = io.BytesIO()
    buf.save(saved)
    saved.seek(0)
    back = train.Buffer(16)
    back.load(saved)
    z = back.gather(ids)[4]
    assert np.array_equal(z[:, :len(HORIZONS)], buf.gather(ids)[4]) and not z[:, len(HORIZONS):].any()


if __name__ == "__main__":
    test_lr_warms_up_holds_the_peak_then_cools_down()
    test_playout_switches_at_half_the_run()
    test_value_only_rows_leave_the_policy_loss_alone()
    test_a_saved_buffer_samples_as_before()
    print("recipe test OK")
