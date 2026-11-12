"""Phase 4, part A (PREREGISTRATION.md): the Phase 2 methods (pixel features -> linear head, and B_HSIA) with
whole-image labels. Components are trained exactly as in Phase 2 (same unlabelled pool, seeds and recipe); only
the labelled set and the head solver change (GPU port of the head, tol 1e-8, max_iter 5000).

Usage (from null-adapter/):
  python -m phase4.src.run_linear --mode sanity --seeds 0 --device cuda:0 --out phase4/results/linear_sanity
  python -m phase4.src.run_linear --mode main --seeds 0 2 4 6 8 --device cuda:0 --out phase4/results/linear
sanity: validation grid, N = 1 and 5 only. Writes <out>/seed<s>/{runs.csv, confusion.npz, losses.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase2.src import run as R2
from phase2.src.gpu_head import GPULogReg
from phase2.src.run import CHUNK, E, METHODS, MIN_TEST_PIXELS, N_CLASSES, confusion, sample_patch_features, scores
from .labels import NS, labelled

HEAD = dict(C=1.0, tol=1e-8, max_iter=5000)
SANITY_NS = [1, 5]


def run_seed(s, mode, out, dev, cache=Path("phase2/cache")):
    t0 = time.time()
    E.set_device(dev)
    split = "test" if mode == "main" else "validation"
    ns = NS if mode == "main" else SANITY_NS
    out = Path(out) / f"seed{s}"
    out.mkdir(parents=True, exist_ok=True)
    A = R2.build_operator()
    feats = {k: {sp: np.load(cache / "feat" / d / f"{sp}.npy", mmap_mode="r") for sp in ("train", split)}
             for k, d in [("B", "B_render"), ("S", "S_render"), ("L", "L_render"), ("png", "B_png")]}
    pool, grid = np.load(cache / "pools" / f"seed{s}.npz"), np.load(cache / f"{split}.npz")
    xu = pool["u_x"].astype(np.float64)
    fu = sample_patch_features(feats["B"]["train"], pool["u_img"], pool["u_pix"])
    comp = R2.Components(xu, xu @ A.T, fu, A, s)
    del xu, fu
    print(f"seed {s}: components trained {time.time() - t0:.0f}s", flush=True)

    n = np.bincount(grid["y"][grid["y"] >= 0], minlength=N_CLASSES)
    scored = np.flatnonzero(n >= MIN_TEST_PIXELS)
    tm = np.isin(grid["y"], scored)
    tx, ty, timg, tpix = grid["x"][tm].astype(np.float64), grid["y"][tm], grid["img"][tm], grid["pix"][tm]
    tf = {k: sample_patch_features(v[split], timg, tpix) for k, v in feats.items()}
    lab = labelled(s)
    lx, ly, lrank = lab["l_x"].astype(np.float64), lab["l_y"], lab["l_rank"]
    lf = {k: sample_patch_features(v["train"], lab["l_img"], lab["l_pix"]) for k, v in feats.items()}

    rows, cms = [], {}

    def record(m, N, cm, it, dt):
        bacc, miou = scores(cm, scored)
        sel = lrank < N
        rows.append(dict(seed=s, N=N, method=m, bacc=bacc, miou=miou, n_iter=int(it), fit_s=round(dt, 1),
                         n_lab=int(sel.sum()), n_cls=int(len(np.unique(ly[sel])))))
        cms[f"{m}/N{N}"] = cm
        print(f"seed {s} {m:9s} N={N:3d} bacc {bacc:.3f} miou {miou:.3f} it {it} ({dt:.0f}s)", flush=True)

    for m in METHODS:
        Ftr = comp.feature(m, lx, lf)
        Fte = np.concatenate([comp.feature(m, tx[a:a + CHUNK], {k: v[a:a + CHUNK] for k, v in tf.items()})
                              for a in range(0, len(ty), CHUNK)])
        for N in ns:
            sel = lrank < N
            t1 = time.time()
            head = GPULogReg(**HEAD, device=dev).fit(Ftr[sel], ly[sel])
            record(m, N, confusion(ty, head.predict(Fte)), head.n_iter_, time.time() - t1)
            del head
            torch.cuda.empty_cache()
        del Ftr, Fte

    for N in ns:
        sel = lrank < N
        t1 = time.time()
        pred = R2.fit_hsia(comp.sf, comp.sx, lf["B"][sel], lx[sel], ly[sel], s, dev)
        p = np.concatenate([pred(tf["B"][a:a + CHUNK], tx[a:a + CHUNK]) for a in range(0, len(ty), CHUNK)])
        record("B_HSIA", N, confusion(ty, p), R2.HSIA["steps"], time.time() - t1)

    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "confusion.npz", scored=scored, **cms)
    json.dump(comp.losses, open(out / "losses.json", "w"))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "main"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for s in a.seeds:
        run_seed(s, a.mode, a.out, a.device)


if __name__ == "__main__":
    main()
