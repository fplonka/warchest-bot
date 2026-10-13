import numpy as np

import warchest


def mirror(f, a, which):
    a = a.copy()
    a[which] = f(a[which].ravel()).reshape(-1, a.shape[1])
    return a


def to_mover(rows, desc, parow):
    turned = rows[:, warchest.ROW_TO_ACT] == 1
    return mirror(warchest.mirror_rows, rows, turned), mirror(warchest.mirror_descs, desc, turned[parow])
