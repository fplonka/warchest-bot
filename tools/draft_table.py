import sys
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import warchest

from draft import key

SLOTS = [(1,), (2, 52), (3,), (4,), (5,), (6, 53), (7,), (8,), (9,), (10,), (11,), (12,), (16,), (17,), (18, 54), (19,)]
CHUNK = 4096


def leaves():
    for slots in combinations(SLOTS, 8):
        for units in product(*slots):
            for first in combinations(units, 4):
                yield first, tuple(u for u in units if u not in first)


def main(bot, out, shard, shards):
    player = warchest.Player(bot)
    mine = [leaf for i, leaf in enumerate(leaves()) if i % shards == shard]
    values, t = np.empty(len(mine), np.float32), time.time()
    for at in range(0, len(mine), CHUNK):
        part = mine[at:at + CHUNK]
        values[at:at + len(part)] = player.values([(list(a), list(b), 0) for a, b in part])
        done = at + len(part)
        print(f"{done}/{len(mine)} leaves, {(time.time() - t) / done * (len(mine) - done) / 60:.0f} min left", flush=True)
    keys = np.array([key(a, b) for a, b in mine], np.uint64)
    np.savez(out, keys=keys, values=values)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
