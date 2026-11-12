"""Exploratory E3 (PREREGISTRATION.md): linear head (GPU port of the Phase 2 head, tol 1e-8) on f ⊕ r̄ (4 x 4
averaged residual, read at the pixel) vs f ⊕ r (Phase 2 per-pixel residual), K = 5, 20, 100, test.

Usage (from null-adapter/):
  python -m phase3.scripts.explore_e3 --seeds 0 2 4 6 8 --device cuda:0 --out phase3/results/explore_e3
  python -m phase3.scripts.explore_e3 --summarize --out phase3/results/explore_e3
The per-pixel m is trained exactly as Phase 2 M5_mlp (same pool, standardisers, torch seed, recipe); M0 is re-fitted
with the same head so all three rows share one optimiser.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase2.scripts.analyze import paired
from phase2.src import run as R2
from phase2.src.gpu_head import GPULogReg
from phase3.src import run as R3
from phase3.src.head import read_at

KS = [5, 20, 100]
HEAD = dict(C=1.0, tol=1e-8, max_iter=5000)
METHODS = ["M0", "M5_mlp", "M5_bar"]


def pixel_m(xu, A, seed):
    """Phase 2 Components.m5, reproduced (torch seed seed * 100 + 12 before creation, same fit_regression seed)."""
    cu = xu @ A.T
    sc, sx = R2.E.Standardizer.fit(cu), R2.E.Standardizer.fit(xu)
    torch.manual_seed(seed * 100 + 12)
    m = R2.E.mlp(3, xu.shape[1], **R2.NET)
    R2.E.fit_regression(m, sc(cu), sx(xu), R2.TR, seed * 100 + 12)
    m.requires_grad_(False).eval()
    return lambda x: x - sx.inv(R2.E.apply(m, sc(x @ A.T)))


@torch.no_grad()
def r_bar_at(comp, sp, img, pix, dev):
    """r̄ map (unstandardised) of each image, read at pixels by the bilinear read-out of the head."""
    out = np.empty((len(img), 204))
    for i in np.unique(img):
        m = img == i
        r = comp._u_raw("r", sp.x[[int(i)]])
        out[m] = read_at(r, torch.zeros(m.sum(), dtype=torch.long, device=dev), torch.from_numpy(pix[m]).to(dev)).double().cpu().numpy()
    return out


def run_seed(s, out, dev):
    t0 = time.time()
    R2.E.set_device(dev)
    A = R2.build_operator()
    fB = {sp: np.load(R3.CACHE2 / "feat" / "B_render" / f"{sp}.npy", mmap_mode="r") for sp in ("train", "test")}
    pool, test = np.load(R3.CACHE2 / "pools" / f"seed{s}.npz"), np.load(R3.CACHE2 / "test.npz")
    resid = pixel_m(pool["u_x"].astype(np.float64), A, s)
    tr, te = R3.Split("train", dev), R3.Split("test", dev)
    comp = R3.Components(tr, s, dev)
    print(f"seed {s}: components {time.time() - t0:.0f}s", flush=True)
    n = np.bincount(test["y"][test["y"] >= 0], minlength=R2.N_CLASSES)
    scored = np.flatnonzero(n >= R2.MIN_TEST_PIXELS)
    km = np.isin(test["y"], scored)
    ti, tp, ty, tx = test["img"][km], test["pix"][km], test["y"][km], test["x"][km].astype(np.float64)
    tf = R2.sample_patch_features(fB["test"], ti, tp)
    lx, ly, lrank = pool["l_x"].astype(np.float64), pool["l_y"], pool["l_rank"]
    lf = R2.sample_patch_features(fB["train"], pool["l_img"], pool["l_pix"])
    feats = {"M0": (lf, tf), "M5_mlp": (np.c_[lf, resid(lx)], np.c_[tf, resid(tx)]),
             "M5_bar": (np.c_[lf, r_bar_at(comp, tr, pool["l_img"], pool["l_pix"], dev)],
                        np.c_[tf, r_bar_at(comp, te, ti, tp, dev)])}
    del tr, te, comp
    torch.cuda.empty_cache()
    rows = []
    for m, (Ftr, Fte) in feats.items():
        for K in KS:
            sel = lrank < K
            t1 = time.time()
            g = GPULogReg(**HEAD, device=dev).fit(Ftr[sel], ly[sel])
            bacc, miou = R2.scores(R2.confusion(ty, g.predict(Fte)), scored)
            rows.append(dict(seed=s, K=K, method=m, bacc=bacc, miou=miou, n_iter=g.n_iter_, fit_s=round(time.time() - t1, 1)))
            print(f"seed {s} {m:7s} K={K:3d} bacc {bacc:.4f} it {g.n_iter_}", flush=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"seed{s}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def summarize(out):
    out = Path(out)
    rows = [r for f in sorted(out.glob("seed*.csv")) for r in csv.DictReader(open(f))]
    v = {}
    for r in rows:
        v.setdefault((r["method"], int(r["K"])), {})[int(r["seed"])] = float(r["bacc"])
    seeds = sorted({int(r["seed"]) for r in rows})
    get = lambda m, K: [v[(m, K)][s] for s in seeds]
    res = {m: {K: dict(mean=float(np.mean(get(m, K))), vs_M0=paired(get(m, K), get("M0", K)),
                       vs_M5_mlp=paired(get(m, K), get("M5_mlp", K))) for K in KS} for m in METHODS}
    json.dump(dict(seeds=seeds, results=res), open(out / "summary.json", "w"), indent=1)
    fmt = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    L = [f"# E3: linear head on the 4 x 4 averaged residual (test, seeds {seeds}, GPU head tol 1e-8)", "",
         "| method | " + " | ".join(f"K={K} bacc | vs M0 | vs M5_mlp" for K in KS) + " |", "|---|" + "---|---|---|" * len(KS)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{res[m][K]['mean']:.4f} | {fmt(res[m][K]['vs_M0'])} | {fmt(res[m][K]['vs_M5_mlp'])}"
                                          for K in KS) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase3/results/explore_e3")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.out)
    for s in a.seeds:
        run_seed(s, a.out, a.device)


if __name__ == "__main__":
    main()
