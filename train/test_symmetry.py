import numpy as np

import mirror
import warchest


def main():
    rows, desc, off = warchest.sample_positions(2, 23)
    rows = np.frombuffer(bytes(rows), np.uint8).reshape(-1, warchest.ROW_BYTES)[:3].copy()
    off = np.asarray(off, np.int64)[:4]
    desc = np.frombuffer(bytes(desc), np.uint8).reshape(-1, warchest.ACT_BYTES)[:off[-1]].copy()
    parow = np.repeat(np.arange(3), np.diff(off))
    target = np.random.default_rng(0).random(len(desc)).astype(np.float32)
    z = np.random.default_rng(1).uniform(-1, 1, (3, len(warchest.HORIZONS))).astype(np.float32)
    parts = (rows, desc, parow, target, z)
    flipped = np.asarray([True, False, True])

    once = mirror.mirror_batch(parts, flipped)
    assert np.array_equal(once[0][0], mirror.mirror_rows(rows[:1])[0])
    assert np.array_equal(once[0][1], rows[1])
    assert np.array_equal(once[4], z)
    kept = parow == 1
    assert np.array_equal(once[1][kept], desc[kept])
    turned = once[1][parow == 0]
    coins = desc[parow == 0][:, 1:3]
    assert np.array_equal(turned[:, 1:3][coins < warchest.NTYPE],
                          (coins[coins < warchest.NTYPE] + warchest.NSLOT) % warchest.NTYPE)
    twice = mirror.mirror_batch(once, flipped)
    for got, want in zip(twice, parts, strict=True):
        assert np.array_equal(got, want)
    print("replay symmetry OK")


if __name__ == "__main__":
    main()
