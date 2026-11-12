"""One pass over the LIB-HSI cubes: 4 x 4 average-pooled reflectance maps x̄ for every split (Phase 3 PLAN.md).

Usage (from null-adapter/):  python -m phase3.scripts.prepare <root> <cache3>
Writes <cache3>/xbar/<split>.npy, float16 (n_img, 128, 128, 204), image order = phase2/cache/ids.json.
x̄[i, a, b] is the mean of x over pixels (4a..4a+3, 4b..4b+3); since A is linear, A x̄ is the pooled RGB.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from phase2.src.libhsi import H, N_BANDS, W, load_cube

POOL = 4


def pool(cube: np.ndarray) -> np.ndarray:
    return cube.reshape(H // POOL, POOL, W // POOL, POOL, -1).mean((1, 3), dtype=np.float64)


def main(root, cache3, ids_file="phase2/cache/ids.json"):
    out = Path(cache3) / "xbar"
    out.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(ids_file))
    t0 = time.time()
    for split in ("validation", "test", "train"):
        dst = out / f"{split}.npy"
        tmp = out / f"{split}.partial.npy"
        mm = np.lib.format.open_memmap(tmp, "w+", np.float16, (len(ids[split]), H // POOL, W // POOL, N_BANDS))
        for i, sid in enumerate(ids[split]):
            mm[i] = pool(load_cube(root, split, sid))
            if i % 50 == 0:
                print(f"{split} {i}/{len(ids[split])} {time.time() - t0:.0f}s", flush=True)
        mm.flush(); del mm
        tmp.rename(dst)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
