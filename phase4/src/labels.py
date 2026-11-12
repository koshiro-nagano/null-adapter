"""Labelled set of Phase 4: every grid pixel of the first N train images of the seed's image order.

labelled(seed) returns the arrays of the Phase 2 labelled pool (l_img, l_pix, l_y, l_x) with l_rank = the position
of the pixel's image in the seed's order, so `l_rank < N` is the N-image subset and the Phase 2 / Phase 3 loops
(which select `l_rank < K`) run unchanged with N in place of K.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE4 = Path("phase4/cache")
NS = [1, 5, 20, 100]


def labelled(seed, cache=CACHE4):
    g = np.load(Path(cache) / "train_grid.npz")
    order = np.load(Path(cache) / "image_order.npz")[f"seed{seed}"]
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    r = rank[g["img"]]
    keep = r < max(NS)
    return dict(l_img=g["img"][keep], l_pix=g["pix"][keep], l_y=g["y"][keep].astype(np.int64),
                l_x=g["x"][keep], l_rank=r[keep])
