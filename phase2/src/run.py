"""Phase 2 experiment loop (PREREGISTRATION.md): per seed, train the label-free components on the unlabeled
pool, then fit the shared linear head on K labeled pixels per class and score the test grid.

Usage (from null-adapter/):
  python -I -m phase2.src.run --cache phase2/cache --out phase2/results/main --seeds 0 1 --device cuda:0
Writes <out>/seed<s>/{runs.csv, confusion.npz, losses.json, diagnostics.json}.
Networks and their training loop are the Phase 1 ones (toy1/src/encoders.py), unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "toy1"))
from src import encoders as E  # noqa: E402  (Phase 1 networks)

from .libhsi import H, W, WAVELENGTHS  # noqa: E402
from .render import build_operator, projectors  # noqa: E402

N_CLASSES = 44
KS = [5, 20, 100, 1000]
MIN_TEST_PIXELS = 50
PATCH = 16
HEAD = dict(C=1.0, max_iter=5000)
TR = dict(epochs=30, batch=1024, lr=1e-3, weight_decay=0.0)
NET = dict(width=256, depth=3)
G_LATENT = 16
HSIA = dict(steps=300, lr=1e-3, weight_decay=1e-4, width=64)
METHODS = ["M0c", "M0", "M0_S", "M0_L", "M0_png", "M1", "M2", "M3", "M5_ridge", "M5_mlp", "M5_ctx", "M6",
           "B_RND", "B_RNDres"]  # + B_HSIA (end to end, below)
CHUNK = 50_000


def sample_patch_features(grid: np.ndarray, img: np.ndarray, pix: np.ndarray) -> np.ndarray:
    """Bilinear upsampling of (n_img, g, g, D) patch tokens to 512 x 512, read at (img, pix);
    identical to F.interpolate(mode="bilinear", align_corners=False) (tests/test_run.py)."""
    g = grid.shape[1]
    y, x = pix // W, pix % W

    def axis(p):
        s = np.clip((p + 0.5) / PATCH - 0.5, 0, None)
        i0 = np.minimum(np.floor(s).astype(int), g - 1)
        return i0, np.minimum(i0 + 1, g - 1), (s - i0).astype(np.float32)[:, None]

    y0, y1, wy = axis(y)
    x0, x1, wx = axis(x)
    out = np.empty((len(img), grid.shape[3]), np.float32)
    for a in range(0, len(img), CHUNK):
        sl = slice(a, a + CHUNK)
        q = lambda yy, xx: grid[img[sl], yy[sl], xx[sl]].astype(np.float32)
        out[sl] = ((1 - wy[sl]) * ((1 - wx[sl]) * q(y0, x0) + wx[sl] * q(y0, x1))
                   + wy[sl] * ((1 - wx[sl]) * q(y1, x0) + wx[sl] * q(y1, x1)))
    return out


class Components:
    """Everything learned without labels for one seed, plus the per-pixel feature builder."""

    def __init__(self, xu, cu, fu, A, seed):
        torch.manual_seed(seed)
        self.A = A
        self.A_pinv = np.linalg.pinv(A)
        _, self.P_N = projectors(A)
        self.sc, self.sx, self.sf = E.Standardizer.fit(cu), E.Standardizer.fit(xu), E.Standardizer.fit(fu)
        C, X, F = self.sc(cu), self.sx(xu), self.sf(fu)
        self.losses = {}
        self.ridge = make_pipeline(StandardScaler(), Ridge(alpha=1e-3)).fit(cu, xu)

        def fit(name, model, Xin, Y, k):
            torch.manual_seed(seed * 100 + k)
            self.losses[name] = E.fit_regression(model, Xin, Y, TR, seed * 100 + k)
            return model.requires_grad_(False).eval()

        self.m5 = fit("m5_mlp", E.mlp(3, X.shape[1], **NET), C, X, 12)
        gae = fit("g_ae", E.AE(X.shape[1], G_LATENT, **NET), X, X, 10)
        self.g = gae.enc
        Z = E.apply(self.g, X)
        self.p = fit("p", E.mlp(F.shape[1], G_LATENT, **NET), F, Z, 11)
        del Z
        self.rnd = fit("rnd", E.RNDReconstructor(self.A_pinv, self.P_N, self.sc, self.sx, **NET), C, X, 13)
        CF = np.empty((len(C), 3 + F.shape[1]), np.float32)
        CF[:, :3], CF[:, 3:] = C, F
        del F
        self.mctx = fit("m5_ctx", E.mlp(CF.shape[1], X.shape[1], **NET), CF, X, 14)
        del C, X, CF

    def feature(self, m, x, f):
        """f: dict of raw DINOv3 features (B, S, L, png) for the same pixels."""
        c = x @ self.A.T
        cs, xs, fb = self.sc(c), self.sx(x), f["B"]
        if m == "M0c":
            return c
        if m == "M0":
            return fb
        if m in ("M0_S", "M0_L"):
            return f[m[-1]]
        if m == "M0_png":
            return f["png"]
        if m == "M1":
            return x
        if m == "M2":
            return np.c_[fb, x]
        if m == "M3":
            return np.c_[fb, x @ self.P_N.T]
        return np.c_[fb, self.extra(m, x, c, cs, xs, fb)]

    def extra(self, m, x, c, cs, xs, fb):
        if m == "M5_ridge":
            return x - self.ridge.predict(c)
        if m == "M5_mlp":
            return x - self.sx.inv(E.apply(self.m5, cs))
        if m == "M5_ctx":
            return x - self.sx.inv(E.apply(self.mctx, np.c_[cs, self.sf(fb)]))
        if m == "M6":
            return E.apply(self.g, xs) - E.apply(self.p, self.sf(fb))
        if m == "B_RND":
            return self.sx.inv(E.apply(self.rnd, cs))
        if m == "B_RNDres":
            return x - self.sx.inv(E.apply(self.rnd, cs))
        raise KeyError(m)

    def diagnostics(self, x, fb):
        """Share of the null-space energy of x (about the pool mean) explained by each E[x|c] estimate."""
        c = x @ self.A.T
        cs = self.sc(c)
        xn = x @ self.P_N.T
        xn0 = xn - self.sx.mean @ self.P_N.T
        out = {}
        for name, est in [("M5_ridge", self.ridge.predict(c)), ("M5_mlp", self.sx.inv(E.apply(self.m5, cs))),
                          ("M5_ctx", self.sx.inv(E.apply(self.mctx, np.c_[cs, self.sf(fb)]))),
                          ("B_RNDres", self.sx.inv(E.apply(self.rnd, cs)))]:
            res = xn - est @ self.P_N.T
            for reg, mask in [("all", WAVELENGTHS > 0), ("vis_le780", WAVELENGTHS <= 780),
                              ("nir_gt780", WAVELENGTHS > 780)]:
                out[f"{name}/{reg}"] = float(1 - (res[:, mask] ** 2).sum() / (xn0[:, mask] ** 2).sum())
        out["norm_PNx_median"] = float(np.median(np.linalg.norm(xn, axis=1)))
        out["norm_r_M5_mlp_median"] = float(np.median(np.linalg.norm(x - self.sx.inv(E.apply(self.m5, cs)), axis=1)))
        return out


def fit_hsia(sf, sx, f_tr, x_tr, y_tr, seed, device):
    """HSI-Adapter-lite port: head(f + gamma * adapter(x)), zero-initialised gate, phi frozen (PREREGISTRATION)."""
    torch.manual_seed(seed)
    adapter = E.mlp(x_tr.shape[1], f_tr.shape[1], HSIA["width"], 1).to(device)
    gamma = nn.Parameter(torch.zeros(f_tr.shape[1], device=device))
    head = nn.Linear(f_tr.shape[1], N_CLASSES).to(device)
    params = list(adapter.parameters()) + [gamma] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=HSIA["lr"], weight_decay=HSIA["weight_decay"])
    F_, X_ = torch.as_tensor(sf(f_tr), dtype=torch.float32, device=device), torch.as_tensor(sx(x_tr), dtype=torch.float32, device=device)
    Y_ = torch.as_tensor(y_tr, device=device)
    for _ in range(HSIA["steps"]):
        loss = nn.functional.cross_entropy(head(F_ + gamma * adapter(X_)), Y_)
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def predict(f, x):
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        return head(t(sf(f)) + gamma * adapter(t(sx(x)))).argmax(1).cpu().numpy()
    return predict


def confusion(y, p):
    return np.bincount(y * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def scores(cm, scored):
    tp = np.diag(cm)[scored].astype(float)
    recall = tp / cm[scored].sum(1)
    # pixels of unscored classes are dropped before cm is built, so column sums are over scored rows only
    iou = tp / (cm[scored].sum(1) + cm[:, scored].sum(0) - tp)
    return float(recall.mean()), float(iou.mean())


def run_seed(s, cache, out, device, n_threads, eval_file="test.npz"):
    t0 = time.time()
    E.set_device(device)
    cache, out = Path(cache), Path(out) / f"seed{s}"
    out.mkdir(parents=True, exist_ok=True)
    A = build_operator()
    feats = {k: {sp: np.load(cache / "feat" / f"{d}" / f"{sp}.npy", mmap_mode="r")
                 for sp in ("train", "validation", "test")}
             for k, d in [("B", "B_render"), ("S", "S_render"), ("L", "L_render"), ("png", "B_png")]}
    pool, test = np.load(cache / "pools" / f"seed{s}.npz"), np.load(cache / eval_file)
    split = "validation" if eval_file.startswith("validation") else "test"
    xu = pool["u_x"].astype(np.float64)
    fu = sample_patch_features(feats["B"]["train"], pool["u_img"], pool["u_pix"])
    comp = Components(xu, xu @ A.T, fu, A, s)
    del xu, fu
    print(f"seed {s}: components trained {time.time() - t0:.0f}s", flush=True)

    keep = test["y"] >= 0
    n_test = np.bincount(test["y"][keep], minlength=N_CLASSES)
    scored = np.flatnonzero(n_test >= MIN_TEST_PIXELS)
    tmask = np.isin(test["y"], scored)
    tx, ty, timg, tpix = test["x"][tmask].astype(np.float64), test["y"][tmask], test["img"][tmask], test["pix"][tmask]
    tf = {k: sample_patch_features(v[split], timg, tpix) for k, v in feats.items()}
    lx = pool["l_x"].astype(np.float64)
    lf = {k: sample_patch_features(v["train"], pool["l_img"], pool["l_pix"]) for k, v in feats.items()}
    ly, lrank = pool["l_y"], pool["l_rank"]

    diag = comp.diagnostics(tx, tf["B"])
    diag.update(n_scored=int(len(scored)), n_test_pixels=int(len(ty)))
    json.dump(diag, open(out / "diagnostics.json", "w"), indent=1)

    rows, cms = [], {}
    for m in METHODS:
        Ftr = comp.feature(m, lx, lf)
        Fte = np.concatenate([comp.feature(m, tx[a:a + CHUNK], {k: v[a:a + CHUNK] for k, v in tf.items()})
                              for a in range(0, len(ty), CHUNK)])

        def job(K, m=m, Ftr=Ftr, Fte=Fte):
            sel = lrank < K
            t1 = time.time()
            head = make_pipeline(StandardScaler(), LogisticRegression(**HEAD)).fit(Ftr[sel], ly[sel])
            cm = confusion(ty, head.predict(Fte))
            return K, cm, head[-1].n_iter_.max(), time.time() - t1

        with ThreadPoolExecutor(n_threads) as ex:
            for K, cm, it, dt in ex.map(job, KS):
                bacc, miou = scores(cm, scored)
                rows.append(dict(seed=s, K=K, method=m, bacc=bacc, miou=miou, n_iter=int(it), fit_s=round(dt, 1)))
                cms[f"{m}/K{K}"] = cm
                print(f"seed {s} {m:9s} K={K:4d} bacc {bacc:.3f} miou {miou:.3f} ({dt:.0f}s)", flush=True)
        del Ftr, Fte

    for K in KS:
        sel = lrank < K
        pred = fit_hsia(comp.sf, comp.sx, lf["B"][sel], lx[sel], ly[sel], s, device)
        p = np.concatenate([pred(tf["B"][a:a + CHUNK], tx[a:a + CHUNK]) for a in range(0, len(ty), CHUNK)])
        cm = confusion(ty, p)
        bacc, miou = scores(cm, scored)
        rows.append(dict(seed=s, K=K, method="B_HSIA", bacc=bacc, miou=miou, n_iter=HSIA["steps"], fit_s=0.0))
        cms[f"B_HSIA/K{K}"] = cm
        print(f"seed {s} B_HSIA    K={K:4d} bacc {bacc:.3f} miou {miou:.3f}", flush=True)

    import csv
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "confusion.npz", scored=scored, **cms)
    json.dump(comp.losses, open(out / "losses.json", "w"))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="phase2/cache")
    ap.add_argument("--out", default="phase2/results/main")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval", default="test.npz", help="evaluation grid in <cache>; validation.npz for sanity runs")
    a = ap.parse_args()
    for s in a.seeds:
        run_seed(s, a.cache, a.out, a.device, a.threads, a.eval)


if __name__ == "__main__":
    main()
