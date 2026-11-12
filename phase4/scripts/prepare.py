"""One pass over the train cubes for Phase 4 (whole-image labels, PREREGISTRATION.md).

Usage (from null-adapter/):  python -m phase4.scripts.prepare <root> phase4/cache
Writes
  <cache>/train_grid.npz    img, pix, y, x   every train image on the stride-16 grid (offset 8): 1024 pixels per
                                             image, labels and spectra (float32); image index = phase2/cache/ids.json
  <cache>/image_order.npz   seed<s>          a random permutation of the train images (numpy rng 4000 + s); the
                                             labelled images of N are the first N of it, so the sets are nested
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from phase2.src.libhsi import H, N_BANDS, W, load_classes, load_cube, load_labels

STRIDE, OFFSET = 16, 8
SEEDS = range(10)


def main(root, cache, ids_file="phase2/cache/ids.json"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(ids_file))["train"]
    classes = load_classes(root)
    gy, gx = np.meshgrid(np.arange(OFFSET, H, STRIDE), np.arange(OFFSET, W, STRIDE), indexing="ij")
    grid = (gy * W + gx).ravel()
    out = {k: [] for k in ("img", "pix", "y", "x")}
    t0 = time.time()
    for i, sid in enumerate(ids):
        flat = load_cube(root, "train", sid).reshape(-1, N_BANDS)
        y = load_labels(root, "train", sid, classes).ravel()[grid]
        keep = y >= 0
        out["img"].append(np.full(keep.sum(), i)); out["pix"].append(grid[keep])
        out["y"].append(y[keep]); out["x"].append(flat[grid[keep]].astype(np.float32))
        if i % 50 == 0:
            print(f"train {i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    np.savez(cache / "train_grid.npz", **{k: np.concatenate(v) for k, v in out.items()})
    np.savez(cache / "image_order.npz", **{f"seed{s}": np.random.default_rng(4000 + s).permutation(len(ids)) for s in SEEDS})
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
