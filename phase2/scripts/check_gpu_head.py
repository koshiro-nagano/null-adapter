"""Is src/gpu_head.GPULogReg equivalent to the pre-registered sklearn head? Checked on val only (seed 0).

Usage (from null-adapter/):  python -m phase2.scripts.check_gpu_head --device cuda:0 --out phase2/results/gpu_head_check
Both heads are fitted on the same feature matrices. The label-free components are trained on the first
N_POOL pixels of the seed-0 unlabeled pool (not the full pool: this check is about the head, not the features,
and it must fit next to the running main jobs). sklearn is fitted only for SKL_METHODS (CPU cost); the GPU head
is fitted for every method to measure its time.

Pass criteria, fixed before the first run. For every (method in SKL_METHODS, K):
  - objective: f(GPU solution) <= f(sklearn solution) + 1e-6 (the problem is strictly convex, so both should
    reach the same optimum; f as in gpu_head.py, evaluated in float64 on the same standardised data);
  - val predictions agree on >= 99.5 % of the scored grid pixels;
  - |bacc(GPU) - bacc(sklearn)| <= 0.002.
Writes <out>/check.csv and <out>/summary.json.

--tol 1e-8 (exploratory, run after the main test run; criteria fixed before running it): both heads use tol=1e-8,
so both should reach the exact optimum. Pass per (method, K): |f(GPU) - f(sklearn)| <= 1e-6 f(sklearn), agreement
>= 99.9 %, |d_bacc| <= 0.0005; a head that stops at max_iter is reported as not converged. The sklearn head at the
pre-registered tol (1e-4) is fitted too, on the same features, to measure how far the pre-registered stopping rule
moves bacc from the exact optimum (skl_bacc - skl_pre_bacc).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase2.src import run as R
from phase2.src.gpu_head import GPULogReg

N_POOL = 200_000
SKL_METHODS = ["M0c", "M0", "M1", "M2", "M5_mlp", "B_RND"]
CRIT = dict(obj=1e-6, agree=0.995, bacc=0.002)
CRIT_TIGHT = dict(obj_rel=1e-6, agree=0.999, bacc=0.0005)


def objective(Z, y, W, b, C):
    """mean CE + ||W||^2 / (2 C n) on standardised Z; W (d, k), b (k,)."""
    ce = torch.nn.functional.cross_entropy(Z @ W + b, y)
    return float(ce + 0.5 / (C * len(y)) * (W * W).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="phase2/cache")
    ap.add_argument("--out", default="phase2/results/gpu_head_check")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=1e-4, help="1e-4 = pre-registered; 1e-8 = exploratory check")
    a = ap.parse_args()
    tight = a.tol != 1e-4
    head = dict(R.HEAD, tol=a.tol)
    out, cache, dev = Path(a.out), Path(a.cache), a.device
    out.mkdir(parents=True, exist_ok=True)
    R.E.set_device(dev)
    t0 = time.time()
    A = R.build_operator()
    feats = {k: {sp: np.load(cache / "feat" / d / f"{sp}.npy", mmap_mode="r") for sp in ("train", "validation")}
             for k, d in [("B", "B_render"), ("S", "S_render"), ("L", "L_render"), ("png", "B_png")]}
    pool, val = np.load(cache / "pools" / "seed0.npz"), np.load(cache / "validation.npz")
    xu = pool["u_x"][:N_POOL].astype(np.float64)
    fu = R.sample_patch_features(feats["B"]["train"], pool["u_img"][:N_POOL], pool["u_pix"][:N_POOL])
    comp = R.Components(xu, xu @ A.T, fu, A, 0)
    del xu, fu
    print(f"components {time.time() - t0:.0f}s", flush=True)

    n_val = np.bincount(val["y"][val["y"] >= 0], minlength=R.N_CLASSES)
    scored = np.flatnonzero(n_val >= R.MIN_TEST_PIXELS)
    vm = np.isin(val["y"], scored)
    vx, vy = val["x"][vm].astype(np.float64), val["y"][vm]
    vf = {k: R.sample_patch_features(v["validation"], val["img"][vm], val["pix"][vm]) for k, v in feats.items()}
    lx = pool["l_x"].astype(np.float64)
    lf = {k: R.sample_patch_features(v["train"], pool["l_img"], pool["l_pix"]) for k, v in feats.items()}
    ly, lrank = pool["l_y"], pool["l_rank"]

    rows = []
    for m in R.METHODS:
        Ftr = comp.feature(m, lx, lf)
        Fte = np.concatenate([comp.feature(m, vx[i:i + R.CHUNK], {k: v[i:i + R.CHUNK] for k, v in vf.items()})
                              for i in range(0, len(vy), R.CHUNK)])
        for K in R.KS:
            sel = lrank < K
            torch.cuda.synchronize(dev); t1 = time.time()
            g = GPULogReg(**head, device=dev).fit(Ftr[sel], ly[sel])
            pg = g.predict(Fte); torch.cuda.synchronize(dev)
            row = dict(method=m, K=K, d=Ftr.shape[1], n=int(sel.sum()), gpu_s=round(time.time() - t1, 2),
                       gpu_iter=g.n_iter_, gpu_bacc=R.scores(R.confusion(vy, pg), scored)[0])
            if m in SKL_METHODS:
                t1 = time.time()
                s = make_pipeline(StandardScaler(), LogisticRegression(**head)).fit(Ftr[sel], ly[sel])
                ps = s.predict(Fte)
                Z = g._z(Ftr[sel])
                y = torch.as_tensor(np.searchsorted(g.classes_, ly[sel]), device=dev)
                Ws, bs = g._t(s[-1].coef_.T), g._t(s[-1].intercept_)
                row.update(skl_s=round(time.time() - t1, 2), skl_iter=int(s[-1].n_iter_.max()),
                           skl_bacc=R.scores(R.confusion(vy, ps), scored)[0],
                           obj_gpu=objective(Z, y, g.W_, g.b_, R.HEAD["C"]), obj_skl=objective(Z, y, Ws, bs, R.HEAD["C"]),
                           agree=float((pg == ps).mean()),
                           coef_rel=float((g.W_ - Ws).norm() / Ws.norm()))
                row["d_bacc"] = row["gpu_bacc"] - row["skl_bacc"]
                if tight:
                    s0 = make_pipeline(StandardScaler(), LogisticRegression(**R.HEAD)).fit(Ftr[sel], ly[sel])
                    row["skl_pre_bacc"] = R.scores(R.confusion(vy, s0.predict(Fte)), scored)[0]
                    row["d_bacc_pre"] = row["skl_bacc"] - row["skl_pre_bacc"]
                    row["converged"] = bool(g.n_iter_ < R.HEAD["max_iter"] and row["skl_iter"] < R.HEAD["max_iter"])
                    row["pass"] = bool(abs(row["obj_gpu"] - row["obj_skl"]) <= CRIT_TIGHT["obj_rel"] * row["obj_skl"]
                                       and row["agree"] >= CRIT_TIGHT["agree"] and abs(row["d_bacc"]) <= CRIT_TIGHT["bacc"])
                else:
                    row["pass"] = bool(row["obj_gpu"] <= row["obj_skl"] + CRIT["obj"] and row["agree"] >= CRIT["agree"]
                                       and abs(row["d_bacc"]) <= CRIT["bacc"])
                del Z
            rows.append(row)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        del Ftr, Fte

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    cmp = [r for r in rows if "pass" in r]
    summary = dict(tol=a.tol, criteria=CRIT_TIGHT if tight else CRIT, n_compared=len(cmp), all_pass=all(r["pass"] for r in cmp),
                   failed=[(r["method"], r["K"]) for r in cmp if not r["pass"]],
                   max_abs_d_bacc=max(abs(r["d_bacc"]) for r in cmp), min_agree=min(r["agree"] for r in cmp),
                   max_obj_excess=max(r["obj_gpu"] - r["obj_skl"] for r in cmp),
                   time_skl_s=sum(r["skl_s"] for r in cmp), time_gpu_same_s=sum(r["gpu_s"] for r in cmp),
                   time_gpu_all_s=sum(r["gpu_s"] for r in rows), n_pool=N_POOL, total_s=round(time.time() - t0))
    if tight:
        summary.update(not_converged=[(r["method"], r["K"]) for r in cmp if not r["converged"]],
                       max_abs_d_bacc_pre=max(abs(r["d_bacc_pre"]) for r in cmp))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
