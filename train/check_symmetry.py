import numpy as np

import mirror
import warchest


def main():
    rows, desc, off = warchest.sample_positions(2, 23)
    rows = np.frombuffer(bytes(rows), np.uint8).reshape(-1, warchest.ROW_BYTES)[:3].copy()
    off = np.asarray(off, np.int64)[:4]
    desc = np.frombuffer(bytes(desc), np.uint8).reshape(-1, warchest.ACT_BYTES)[:off[-1]].copy()
    parow = np.repeat(np.arange(3), np.diff(off))
    flipped = np.asarray([True, False, True])

    once = mirror.mirror(warchest.mirror_descs, desc, flipped[parow])
    kept = parow == 1
    assert np.array_equal(once[kept], desc[kept])
    coins = desc[parow == 0][:, 1:3]
    assert np.array_equal(once[parow == 0][:, 1:3][coins < warchest.NTYPE],
                          (coins[coins < warchest.NTYPE] + warchest.NSLOT) % warchest.NTYPE)
    assert np.array_equal(mirror.mirror(warchest.mirror_descs, once, flipped[parow]), desc)
    turned = mirror.mirror(warchest.mirror_rows, rows, flipped)
    assert np.array_equal(turned[1], rows[1])
    assert np.array_equal(mirror.mirror(warchest.mirror_rows, turned, flipped), rows)

    framed = mirror.to_mover(rows, desc, parow)
    assert not framed[0][:, warchest.ROW_TO_ACT].any()
    coins = framed[1][:, 1:3]
    assert (coins[coins < warchest.NTYPE] < warchest.NSLOT).all()
    every = np.ones(3, bool)
    swapped = mirror.mirror(warchest.mirror_rows, rows, every), mirror.mirror(warchest.mirror_descs, desc, every[parow])
    for got, want in zip(mirror.to_mover(*swapped, parow), framed, strict=True):
        assert np.array_equal(got, want)
    print("replay symmetry OK")


if __name__ == "__main__":
    main()
