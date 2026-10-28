import os
import queue
import threading
from pathlib import Path

import numpy as np

import warchest

HELD_EVERY = 50


def write(path, d):
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rows=np.asarray(d["rows"], np.uint8), desc=np.asarray(d["desc"], np.uint8),
                        off=np.asarray(d["off"], np.int64), value=np.asarray(d["value"], np.float16),
                        prior=np.asarray(d["prior"], np.float16))
    os.replace(tmp, path)


def merge(parts):
    lens = np.concatenate([np.diff(p["off"]) for p in parts])
    return {"rows": np.concatenate([p["rows"] for p in parts]), "desc": np.concatenate([p["desc"] for p in parts]),
            "off": np.concatenate([[0], np.cumsum(lens)]), "value": np.concatenate([p["value"] for p in parts]),
            "prior": np.concatenate([p["prior"] for p in parts])}


def held(path):
    return int(Path(path).stem.rsplit("-", 1)[-1]) % HELD_EVERY == 0


def listed(directory, want_held):
    return sorted(p for p in Path(directory).glob("*.npz") if not p.name.endswith(".tmp.npz") and held(p) == want_held)


class Shard:

    def __init__(self, path):
        with np.load(path) as f:
            self.rows = f["rows"].reshape(-1, warchest.ROW_BYTES)
            self.desc = f["desc"].reshape(-1, warchest.ACT_BYTES)
            self.off, self.value, self.prior = f["off"], f["value"], f["prior"]

    def __len__(self):
        return len(self.rows)

    def gather(self, ids):
        lens = np.diff(self.off)[ids]
        at = np.repeat(self.off[ids], lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        return self.rows[ids], self.desc[at], np.repeat(np.arange(len(ids)), lens), self.prior[at].astype(np.float32), self.value[ids].astype(np.float32)


def concat(parts):
    rows, desc, _, prior, value = (np.concatenate(a) for a in zip(*parts))
    bases = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    return rows, desc, np.concatenate([p[2] + b for p, b in zip(parts, bases)]), prior, value


class Stream:

    def __init__(self, directories, pool, rng):
        self.directories, self.rng, self.pool, self.size = directories, rng, [], pool
        self.order = np.random.default_rng(rng.integers(1 << 62))
        self.loaded = queue.Queue(maxsize=2)
        threading.Thread(target=self.load, daemon=True).start()

    def prefetch(self, n, make):
        ready = queue.Queue(maxsize=4)

        def fill():
            while True:
                ready.put(make(self.batch(n)))

        threading.Thread(target=fill, daemon=True).start()
        return ready

    def load(self):
        while True:
            files = [p for d in self.directories for p in listed(d, False)]
            for i in self.order.permutation(len(files)):
                self.loaded.put(Shard(files[i]))

    def batch(self, n):
        while len(self.pool) < self.size or not self.loaded.empty():
            self.pool.append(self.loaded.get())
            self.pool = self.pool[-self.size:]
        sizes = np.array([len(s) for s in self.pool])
        counts = self.rng.multinomial(n, sizes / sizes.sum())
        return concat([s.gather(self.rng.integers(len(s), size=k)) for s, k in zip(self.pool, counts) if k])
