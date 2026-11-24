"""Stratified comparison of null_p vs fuse_p (and vs rgb): by ZJU semantic class (incl. how rare it is in the
labelled subset) and by how hard an instance is for the RGB-only model.

Hardness is leave-one-seed-out: for seed s, an instance's RGB difficulty is the mean rgb IoU over the *other* seeds,
so the comparison at seed s does not reuse the noise it was selected on (no regression-to-the-mean artefact).
CIs: two-level bootstrap (resample seeds, then instances within the stratum), 95 %.

  python -m polar.scripts.strata   -> polar/results/strata.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from polar.src.data import CNT_THRESHOLD, eligible, load_split, mask1024, regions

ROOT = Path("polar/results")
RAW = Path("polar/data/raw/zju/zju-rgbp")
CLASSES = ["background", "building", "glass", "car", "road", "tree", "sky", "pedestrian", "bicycle"]
SETTINGS = [("zju", "all"), ("zju", "20"), ("pgsnet", "all"), ("pgsnet", "100")]
METHODS = ("simcmf", "rgb", "fuse_p", "null_p")
SEEDS = (0, 1, 2)
BINS = [(0, .5, "RGB IoU < 0.5"), (.5, .8, "0.5–0.8"), (.8, 1.01, "≥ 0.8")]
NB = 4000


def load_iou(ds, n, m, s):
    with open(ROOT / ds / f"{m}_n{n}_s{s}" / "instances.csv") as f:
        rows = list(csv.DictReader(f))
    return {(int(r["image"]), int(r["region"])): float(r["iou"]) for r in rows}


def instance_class(split):
    """(image index, label_sam region) -> majority semantic class over the region's native-resolution pixels."""
    names = (Path("polar/cache/zju") / f"{split}_names.txt").read_text().split()
    _, mask, _ = load_split("zju", split)
    out = {}
    for i, nm in enumerate(names):
        lab = np.asarray(Image.open(RAW / split / "label" / f"{nm}.png"))
        mk = np.asarray(mask[i])
        for r in np.unique(mk):
            v = lab[mk == r]
            out[(i, int(r))] = int(np.bincount(v, minlength=9).argmax())
    return out


def boot(diff_by_seed, rng):
    """diff_by_seed: list (per seed) of per-instance difference arrays -> mean, lo, hi (pt)."""
    est = np.mean([d.mean() for d in diff_by_seed])
    k = len(diff_by_seed)
    bs = np.empty(NB)
    for b in range(NB):
        ss = rng.integers(0, k, k)
        bs[b] = np.mean([diff_by_seed[s][rng.integers(0, len(diff_by_seed[s]), len(diff_by_seed[s]))].mean()
                         for s in ss])
    return [100 * est, 100 * np.quantile(bs, .025), 100 * np.quantile(bs, .975)]


def main():
    rng = np.random.default_rng(0)
    out = {"class": {}, "hard": {}, "train_class_counts": {}}
    cls_val = instance_class("val")
    cls_tr = instance_class("train")
    pool = eligible("zju", "train")
    _, trmask, _ = load_split("zju", "train")
    # instances (>= threshold, as trained on) per class in each seed's labelled subset
    for n in ("all", "20"):
        for s in SEEDS:
            perm = np.random.default_rng(1000 + s).permutation(pool)
            lab = perm if n == "all" else perm[: int(n)]
            cnt = np.zeros(9, int)
            for i in lab:
                ids, _ = regions(mask1024(trmask[i]), CNT_THRESHOLD["zju"])
                for r in ids:
                    cnt[cls_tr[(int(i), int(r))]] += 1
            out["train_class_counts"][f"{n}-{s}"] = dict(zip(CLASSES, cnt.tolist()))

    for ds, n in SETTINGS:
        iou = {(m, s): load_iou(ds, n, m, s) for m in METHODS for s in SEEDS}
        keys = sorted(iou[("rgb", 0)])
        A = {(m, s): np.array([iou[(m, s)][k] for k in keys]) for m in METHODS for s in SEEDS}
        tag = f"{ds}-{n}"
        pairs = [("null_p", "fuse_p"), ("null_p", "rgb"), ("fuse_p", "rgb")]

        if ds == "zju":
            c = np.array([cls_val[k] for k in keys])
            res = {}
            for ci, cn in enumerate(CLASSES):
                sel = c == ci
                if sel.sum() == 0:
                    continue
                r = {"n": int(sel.sum()),
                     "miou": {m: 100 * float(np.mean([A[(m, s)][sel].mean() for s in SEEDS])) for m in METHODS}}
                for a, b in pairs:
                    r[f"{a}-{b}"] = boot([A[(a, s)][sel] - A[(b, s)][sel] for s in SEEDS], rng)
                res[cn] = r
            out["class"][tag] = res

        res = {}
        for lo, hi, name in BINS:
            r = {"n_per_seed": []}
            sels = []
            for s in SEEDS:
                h = np.mean([A[("rgb", t)] for t in SEEDS if t != s], 0)
                sel = (h >= lo) & (h < hi)
                sels.append(sel)
                r["n_per_seed"].append(int(sel.sum()))
            r["miou"] = {m: 100 * float(np.mean([A[(m, s)][sels[i]].mean() for i, s in enumerate(SEEDS)]))
                         for m in METHODS}
            for a, b in pairs:
                r[f"{a}-{b}"] = boot([A[(a, s)][sels[i]] - A[(b, s)][sels[i]] for i, s in enumerate(SEEDS)], rng)
            res[name] = r
        out["hard"][tag] = res
        print(tag, "done", flush=True)

    (ROOT / "strata.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    f = lambda v: f"{v[0]:+5.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]"
    print("\n# train instances per class in labelled subset (seed 0/1/2)")
    for n in ("all", "20"):
        print(n, {c: [out["train_class_counts"][f"{n}-{s}"][c] for s in SEEDS] for c in CLASSES})
    for tag, res in out["class"].items():
        print(f"\n# {tag} by class   n | rgb fuse null | null-fuse | null-rgb | fuse-rgb")
        for cn, r in res.items():
            mi = r["miou"]
            print(f"{cn:11s}{r['n']:4d} | {mi['rgb']:5.1f} {mi['fuse_p']:5.1f} {mi['null_p']:5.1f} | "
                  f"{f(r['null_p-fuse_p'])} | {f(r['null_p-rgb'])} | {f(r['fuse_p-rgb'])}")
    for tag, res in out["hard"].items():
        print(f"\n# {tag} by RGB hardness (LOSO)   n/seed | rgb fuse null | null-fuse | null-rgb | fuse-rgb")
        for name, r in res.items():
            mi = r["miou"]
            print(f"{name:14s}{str(r['n_per_seed']):16s} | {mi['rgb']:5.1f} {mi['fuse_p']:5.1f} {mi['null_p']:5.1f} | "
                  f"{f(r['null_p-fuse_p'])} | {f(r['null_p-rgb'])} | {f(r['fuse_p-rgb'])}")


if __name__ == "__main__":
    main()
