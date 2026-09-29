import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

import config
from test_buffer_snapshot import load_train_module

HOUR = 3600.0


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
        return x, None

    def actions(self, desc, pieces, parow):
        return desc.float().sum(1)

    def value(self, logits):
        return logits.mean(1)


def test_value_only_rows_leave_the_policy_loss_alone():
    train = load_train_module()
    torch.manual_seed(0)
    x, z = torch.randn(5, 16), torch.rand(5) * 2 - 1
    desc = torch.randint(0, 9, (9, 6), dtype=torch.uint8)
    parow = torch.tensor([0, 0, 0, 1, 1, 3, 3, 3, 3])
    target = torch.tensor([0.2, 0.5, 0.3, 0.9, 0.1, 0.25, 0.25, 0.25, 0.25])
    _, mixed = train.losses(Net(), x, desc, parow, target, z, 1.0)
    taught = [0, 1, 3]
    _, alone = train.losses(Net(), x[taught], desc, torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]), target, z[taught], 1.0)
    for key in ("policy_loss", "policy_target_entropy", "policy_kl"):
        assert torch.allclose(mixed[key], alone[key]), key


if __name__ == "__main__":
    test_lr_warms_up_holds_the_peak_then_cools_down()
    test_playout_switches_at_half_the_run()
    test_value_only_rows_leave_the_policy_loss_alone()
    print("recipe test OK")
