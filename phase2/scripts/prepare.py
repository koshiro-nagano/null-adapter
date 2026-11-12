"""One pass over the LIB-HSI cubes: rendered RGB images and the pixel pools of PREREGISTRATION.md.

Usage (from null-adapter/):  python -I phase2/scripts/prepare.py <root> <cache>
Writes
  <cache>/render/<split>/<id>.png        rendered RGB (render.to_display(A x)), input of DINOv3 / SAM 3
  <cache>/pools/seed<s>.npz              u_img, u_pix, u_x   unlabeled pool (1M train pixels, labels never read)
                                         l_img, l_pix, l_y, l_x, l_rank   labeled pool (<= 1000 per class; the
                                         first K of each class in l_rank order is the K-subset)
  <cache>/test.npz                       img, pix, y, x      test grid (stride 8, offset 4)
  <cache>/ids.json                       image ids per split (index = *_img above)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase2.src.libhsi import H, N_BANDS, W, list_ids, load_classes, load_cube, load_labels  # noqa: E402
from phase2.src.render import build_operator, to_display  # noqa: E402

SEEDS = range(10)
N_UNLAB = 1_000_000
K_MAX = 1000
STRIDE, OFFSET = 8, 4


def main(root, cache):
    cache = Path(cache)
    (cache / "pools").mkdir(parents=True, exist_ok=True)
    A = build_operator()
    classes = load_classes(root)
    ids = {s: list_ids(root, s) for s in ("train", "validation", "test")}
    json.dump(ids, open(cache / "ids.json", "w"))

    # --- sample coordinates (labels are read only for the labeled pool and the test grid) ---
    n_tr = len(ids["train"])
    lab_tr = np.stack([load_labels(root, "train", s, classes).ravel() for s in ids["train"]])  # (n_img, H*W)
    lab_flat = lab_tr.ravel()
    order = np.argsort(lab_flat, kind="stable")
    bounds = np.searchsorted(lab_flat[order], np.arange(len(classes) + 1))
    class_flat = [order[bounds[k]:bounds[k + 1]] for k in range(len(classes))]  # sorted flat indices per class
    assert lab_flat.min() >= 0
    plan = {}
    for s in SEEDS:
        ru = np.random.default_rng(1000 + s)
        u_img, u_pix = ru.integers(0, n_tr, N_UNLAB), ru.integers(0, H * W, N_UNLAB)
        rl = np.random.default_rng(2000 + s)
        l_img, l_pix, l_y, l_rank = [], [], [], []
        for k in range(len(classes)):
            flat = class_flat[k]
            pick = rl.choice(flat, min(K_MAX, len(flat)), replace=False)  # random order -> first K is uniform
            l_img.append(pick // (H * W)); l_pix.append(pick % (H * W))
            l_y.append(np.full(len(pick), k)); l_rank.append(np.arange(len(pick)))
        plan[s] = dict(u_img=u_img, u_pix=u_pix, l_img=np.concatenate(l_img), l_pix=np.concatenate(l_pix),
                       l_y=np.concatenate(l_y), l_rank=np.concatenate(l_rank))
        for key in ("u", "l"):
            plan[s][f"{key}_x"] = np.empty((len(plan[s][f"{key}_img"]), N_BANDS), np.float32)
    del lab_tr, lab_flat, order, class_flat

    gy, gx = np.meshgrid(np.arange(OFFSET, H, STRIDE), np.arange(OFFSET, W, STRIDE), indexing="ij")
    grid = (gy * W + gx).ravel()
    test = {k: [] for k in ("img", "pix", "y", "x")}

    t0 = time.time()
    for split in ("train", "validation", "test"):
        (cache / "render" / split).mkdir(parents=True, exist_ok=True)
        for i, sid in enumerate(ids[split]):
            cube = load_cube(root, split, sid)
            Image.fromarray(to_display(cube @ A.T)).save(cache / "render" / split / f"{sid}.png")
            flat = cube.reshape(-1, N_BANDS)
            if split == "train":
                for s in SEEDS:
                    for key in ("u", "l"):
                        m = plan[s][f"{key}_img"] == i
                        plan[s][f"{key}_x"][m] = flat[plan[s][f"{key}_pix"][m]]
            elif split == "test":
                test["img"].append(np.full(len(grid), i)); test["pix"].append(grid)
                test["y"].append(load_labels(root, split, sid, classes).ravel()[grid]); test["x"].append(flat[grid])
            if i % 50 == 0:
                print(f"{split} {i}/{len(ids[split])} {time.time() - t0:.0f}s", flush=True)

    for s in SEEDS:
        np.savez(cache / "pools" / f"seed{s}.npz", **plan[s])
    np.savez(cache / "test.npz", **{k: np.concatenate(v) for k, v in test.items()})
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
