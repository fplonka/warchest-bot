import config

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


if __name__ == "__main__":
    test_lr_warms_up_holds_the_peak_then_cools_down()
    print("recipe test OK")
