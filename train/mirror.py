import numpy as np

import warchest

N_HEXES = warchest.N_HEXES
NSLOT = warchest.NSLOT
NTYPE = warchest.NTYPE
HEXMAP = np.asarray(warchest.hex_mirror(), np.int64)


def mirror_rows(rows):
    return warchest.mirror_rows(np.ascontiguousarray(rows).ravel()).reshape(len(rows), -1)


def mirror_batch(parts, flipped):
    rows, desc, parow, target, z = parts
    if not flipped.any():
        return parts
    rows = rows.copy()
    rows[flipped] = mirror_rows(rows[flipped])
    desc = desc.copy()
    moved = flipped[parow]
    actions = desc[moved]
    for field in (1, 2):
        here = actions[:, field] < NTYPE
        actions[here, field] = (actions[here, field] + NSLOT) % NTYPE
    for field in (3, 4, 5):
        here = actions[:, field] < N_HEXES
        actions[here, field] = HEXMAP[actions[here, field]]
    desc[moved] = actions
    return rows, desc, parow, target, z
