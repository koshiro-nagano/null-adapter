"""Evaluation grid (stride 8, offset 4) of one split, as in prepare.py. Used for validation sanity runs only.

Usage: python -I phase2/scripts/eval_grid.py <root> <cache> <split>   -> <cache>/<split>.npz (img, pix, y, x)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase2.src.libhsi import H, N_BANDS, W, load_classes, load_cube, load_labels  # noqa: E402
from phase2.scripts.prepare import OFFSET, STRIDE  # noqa: E402

root, cache, split = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ids, classes = json.load(open(cache / "ids.json"))[split], load_classes(root)
gy, gx = np.meshgrid(np.arange(OFFSET, H, STRIDE), np.arange(OFFSET, W, STRIDE), indexing="ij")
grid = (gy * W + gx).ravel()
out = {k: [] for k in ("img", "pix", "y", "x")}
for i, sid in enumerate(ids):
    out["img"].append(np.full(len(grid), i)); out["pix"].append(grid)
    out["y"].append(load_labels(root, split, sid, classes).ravel()[grid])
    out["x"].append(load_cube(root, split, sid).reshape(-1, N_BANDS)[grid])
np.savez(cache / f"{split}.npz", **{k: np.concatenate(v) for k, v in out.items()})
print(split, len(ids), "images")
