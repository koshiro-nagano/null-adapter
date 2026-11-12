"""Alignment of every LIB-HSI cube with its shipped RGB PNG (band nearest 540 nm vs PNG green channel).

Usage: python -I phase2/scripts/check_alignment.py <root> <out.json>
For each image: Pearson r in the orientation returned by load_cube and in the other three 90-degree rotations.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase2.src.libhsi import WAVELENGTHS, list_ids, load_cube, load_rgb  # noqa: E402

root, out = sys.argv[1:3]
b = int(np.argmin(abs(WAVELENGTHS - 540)))
rows = []
for split in ("train", "validation", "test"):
    for sid in list_ids(root, split):
        band, g = load_cube(root, split, sid)[..., b], load_rgb(root, split, sid)[..., 1].astype(float).ravel()
        r = [float(np.corrcoef(np.rot90(band, k).ravel(), g)[0, 1]) for k in range(4)]
        rows.append({"split": split, "id": sid, "r": r, "best_k": int(np.nanargmax(r))})
json.dump(rows, open(out, "w"), indent=0)
r0 = np.array([x["r"][0] for x in rows])
print("images", len(rows), "best rotation != 0:", [(x["split"], x["id"]) for x in rows if x["best_k"] != 0])
print("r in returned orientation: min %.3f  5%% %.3f  median %.3f" % (r0.min(), np.percentile(r0, 5), np.median(r0)))
