"""Exploratory follow-up of P2 on the validation split (EXPLORATORY_P2.md). Never reads the test split.

Usage (from null-adapter/):
  python -m phase2.scripts.explore_p2 --seeds 0 2 4 --device cuda:0 --out phase2/results/explore_p2
  python -m phase2.scripts.explore_p2 --summarize --out phase2/results/explore_p2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from phase2.src import run as R
from phase2.src.gpu_head import GPULogReg
from phase2.scripts.analyze import paired

KS = [5, 20, 100]
HEAD = dict(C=1.0, tol=1e-8, max_iter=5000)
N_PCA = 200_000
PCA_K = [4, 16, 64]
WEIGHTS = [0.25, 0.5, 2.0]


def run_seed(s, cache, out, dev):
    t0 = time.time()
    R.E.set_device(dev)
    cache = Path(cache)
    A = R.build_operator()
    fB = {sp: np.load(cache / "feat" / "B_render" / f"{sp}.npy", mmap_mode="r") for sp in ("train", "validation")}
    pool, val = np.load(cache / "pools" / f"seed{s}.npz"), np.load(cache / "validation.npz")
    xu = pool["u_x"].astype(np.float64)
    fu = R.sample_patch_features(fB["train"], pool["u_img"], pool["u_pix"])
    comp = R.Components(xu, xu @ A.T, fu, A, s)
    resid = lambda x, f: comp.feature("M5_mlp", x, {"B": f})[:, f.shape[1]:]
    ru = resid(xu[:N_PCA], fu[:N_PCA])
    r_mean = ru.mean(0)
    V = np.linalg.svd(ru - r_mean, full_matrices=False)[2]  # rows = principal directions
    sr = R.E.Standardizer.fit(ru)
    del xu, fu, ru
    print(f"seed {s}: components {time.time() - t0:.0f}s", flush=True)

    n_val = np.bincount(val["y"][val["y"] >= 0], minlength=R.N_CLASSES)
    scored = np.flatnonzero(n_val >= R.MIN_TEST_PIXELS)
    vm = np.isin(val["y"], scored)
    vx, vy = val["x"][vm].astype(np.float64), val["y"][vm]
    vf = R.sample_patch_features(fB["validation"], val["img"][vm], val["pix"][vm])
    lx, ly, lrank = pool["l_x"].astype(np.float64), pool["l_y"], pool["l_rank"]
    lf = R.sample_patch_features(fB["train"], pool["l_img"], pool["l_pix"])
    lr, vr = resid(lx, lf), resid(vx, vf)
    D = lf.shape[1]

    feats = {"M0": (lf, vf), "M1": (lx, vx), "M2": (np.c_[lf, lx], np.c_[vf, vx]), "M5_mlp": (np.c_[lf, lr], np.c_[vf, vr]),
             "M6": (comp.feature("M6", lx, {"B": lf}), comp.feature("M6", vx, {"B": vf})), "R_only": (lr, vr)}
    for k in PCA_K:
        feats[f"PCA_{k}"] = (np.c_[lf, (lr - r_mean) @ V[:k].T], np.c_[vf, (vr - r_mean) @ V[:k].T])
    weights = {f"W_{a:g}": np.r_[np.ones(D), np.full(lr.shape[1], a)] for a in WEIGHTS}

    rows = []

    def add(m, K, pred, n_iter, dt):
        bacc = R.scores(R.confusion(vy, pred), scored)[0]
        rows.append(dict(seed=s, K=K, method=m, bacc=bacc, n_iter=n_iter, fit_s=round(dt, 1)))
        print(f"seed {s} {m:8s} K={K:3d} bacc {bacc:.4f} it {n_iter} ({dt:.0f}s)", flush=True)

    jobs = [(m, Ftr, Fte, None) for m, (Ftr, Fte) in feats.items()]
    jobs += [(m, feats["M5_mlp"][0], feats["M5_mlp"][1], w) for m, w in weights.items()]
    for m, Ftr, Fte, w in jobs:
        for K in KS:
            sel = lrank < K
            t1 = time.time()
            g = GPULogReg(**HEAD, device=dev, col_weight=w).fit(Ftr[sel], ly[sel])
            add(m, K, g.predict(Fte), g.n_iter_, time.time() - t1)
    for m, sx_, xtr, xte in [("B_HSIA", comp.sx, lx, vx), ("HSIA_r", sr, lr, vr)]:
        for K in KS:
            sel = lrank < K
            t1 = time.time()
            pred = R.fit_hsia(comp.sf, sx_, lf[sel], xtr[sel], ly[sel], s, dev)
            p = np.concatenate([pred(vf[a:a + R.CHUNK], xte[a:a + R.CHUNK]) for a in range(0, len(vy), R.CHUNK)])
            add(m, K, p, R.HSIA["steps"], time.time() - t1)

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"seed{s}.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def summarize(out):
    out = Path(out)
    rows = [r for f in sorted(out.glob("seed*.csv")) for r in csv.DictReader(open(f))]
    v = {}
    for r in rows:
        v.setdefault((r["method"], int(r["K"])), {})[int(r["seed"])] = float(r["bacc"])
    seeds = sorted({int(r["seed"]) for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    get = lambda m, K: [v[(m, K)][s] for s in seeds]
    res = {m: {K: dict(mean=float(np.mean(get(m, K))), vs_M0=paired(get(m, K), get("M0", K)),
                       vs_M5_mlp=paired(get(m, K), get("M5_mlp", K))) for K in KS} for m in methods}
    json.dump(dict(seeds=seeds, results=res), open(out / "summary.json", "w"), indent=1)
    fmt = lambda p: f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    L = [f"# Exploratory P2 follow-up (val, seeds {seeds}, GPU head tol 1e-8)", "",
         "| method | " + " | ".join(f"K={K} bacc | vs M0" for K in KS) + " |", "|---|" + "---|---|" * len(KS)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{res[m][K]['mean']:.4f} | {fmt(res[m][K]['vs_M0'])}" for K in KS) + " |")
    L += ["", "| method | " + " | ".join(f"K={K} vs M5_mlp" for K in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(fmt(res[m][K]["vs_M5_mlp"]) for K in KS) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="phase2/cache")
    ap.add_argument("--out", default="phase2/results/explore_p2")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.out)
    for s in a.seeds:
        run_seed(s, a.cache, a.out, a.device)


if __name__ == "__main__":
    main()
