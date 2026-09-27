import importlib.util
import io
import pathlib
import sys
import types

import numpy as np
import torch


def load_train_module():
    warchest = types.ModuleType("warchest")
    warchest.ROW_BYTES, warchest.ACT_BYTES = 7, 6
    value_net = types.ModuleType("value_net")
    value_net.BINS, value_net.Net = 16, object
    export_weights = types.ModuleType("export_weights")
    sys.modules.update({"warchest": warchest, "value_net": value_net,
                        "config": types.ModuleType("config"), "export_weights": export_weights})
    path = pathlib.Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("train_buffer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(train, start, n):
    ids = np.arange(start, start + n)
    lens = 2 + ids % 5
    off = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    rows = (np.arange(n * train.ROW_BYTES) + start).astype(np.uint8).reshape(n, train.ROW_BYTES)
    desc = (np.arange(off[-1] * train.ACT_BYTES) + start).astype(np.uint8).reshape(-1, train.ACT_BYTES)
    target = np.concatenate([np.full(k, 1.0 / k) for k in lens]).astype(np.float16)
    z = np.where(ids % 3 == 0, 1.0, -1.0).astype(np.float32)
    return rows, desc, off, target, z


def assert_same(left, right):
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


def main():
    train = load_train_module()
    train.time.time = lambda: 1_000_000.0
    buf = train.Buffer(4096)
    for start in range(0, 6000, 1000):
        buf.add(*chunk(train, start, 1000))
    assert len(buf) == 4096 and buf.lo == 6000 - 4096

    saved = io.BytesIO()
    torch.save(buf.state_dict(), saved)
    saved.seek(0)
    restored = train.Buffer(2 * buf.cap)
    restored.add(*chunk(train, 9000, 100))
    restored.load_state_dict(torch.load(saved))
    ids = np.arange(buf.lo, buf.rows)
    assert_same(buf.gather(ids), restored.gather(ids))
    assert (restored.lo, restored.rows) == (buf.lo, buf.rows)

    extra = chunk(train, 6000, 1000)
    buf.add(*extra)
    restored.add(*extra)
    ids = np.arange(buf.lo, buf.rows)
    assert_same(buf.gather(ids), restored.gather(ids))

    rows, desc, parow, target, z = buf.gather(np.array([buf.rows - 1]))
    last = slice(*extra[2][-2:])
    assert_same((rows, desc, target, z),
                (extra[0][-1:], extra[1][last], extra[3][last].astype(np.float32), extra[4][-1:]))
    assert np.all(parow == 0)
    print("buffer snapshot test OK")


if __name__ == "__main__":
    main()
