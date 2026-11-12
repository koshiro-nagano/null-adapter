"""Phase 3 experiment loop (PLAN.md / PREREGISTRATION.md): per seed, train the label-free components on stride-4
cells of the train maps, then train the shared spatial head (src/head.py) on K labelled pixels per class for each
method and score the evaluation grid.

Usage (from null-adapter/):
  python -m phase3.src.run --mode select --seeds 0 1 2 --device cuda:0 --out phase3/results/select   (val, S0 only)
  python -m phase3.src.run --mode sanity --seeds 0 --device cuda:0 --out phase3/results/val_sanity
  python -m phase3.src.run --mode main --seeds 0 2 4 6 8 --device cuda:0 --out phase3/results/main
select writes <out>/steps.json (the S_K rule); sanity and main read phase3/results/select/steps.json.
Writes <out>/seed<s>/{runs.csv, confusion.npz, losses.json, curves.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase2.src import run as R2
from phase2.src.run import E, N_CLASSES, MIN_TEST_PIXELS, confusion, scores, build_operator, projectors
from .head import MAP, FusionHead, neighbourhood, read_at

CACHE2, CACHE3 = Path("phase2/cache"), Path("phase3/cache")
KS = [5, 20, 100, 1000]
METHODS = ["S0", "S1", "S2", "S5", "S5_sp", "S_RNDres"]
U_OF = {"S1": "x", "S2": "x", "S5": "r", "S5_sp": "r_sp", "S_RNDres": "r_rnd"}
STEP_GRID = [300, 1000, 3000]
HEAD = dict(width=128, adapter=64)
OPT = dict(lr=1e-3, weight_decay=1e-4, batch_images=16)
N_UNLAB = 1_000_000
NB = 2  # neighbourhood radius of m_sp (5 x 5)
LOG_EVERY = 25
EVAL_BATCH = 16


class Split:
    """x̄ maps and DINOv3 patch grids of one split, resident on the GPU as float16."""

    def __init__(self, split, dev):
        self.x = torch.from_numpy(np.load(CACHE3 / "xbar" / f"{split}.npy")).to(dev)          # (n, 128, 128, 204)
        self.f = torch.from_numpy(np.load(CACHE2 / "feat" / "B_render" / f"{split}.npy")).to(dev)  # (n, 32, 32, 768)
        self.n = len(self.x)


class Components:
    """Label-free parts for one seed: m (c̄ -> x̄), m_sp (5 x 5 c̄ neighbourhood -> x̄), the RND-SCI generator,
    and the standardisation of f and of every u, computed over all train cells."""

    def __init__(self, tr: Split, seed: int, dev):
        self.dev = dev
        A = build_operator()
        self.A = torch.tensor(A, dtype=torch.float32, device=dev)
        rng = np.random.default_rng(3000 + seed)
        img = torch.from_numpy(rng.integers(0, tr.n, N_UNLAB)).to(dev)
        cell = torch.from_numpy(rng.integers(0, MAP * MAP, N_UNLAB)).to(dev)
        xu = tr.x[img, cell // MAP, cell % MAP].double().cpu().numpy()
        cu = xu @ A.T
        cpad = F.pad(torch.cat([(tr.x[a:a + EVAL_BATCH].float() @ self.A.T).permute(0, 3, 1, 2)
                                for a in range(0, tr.n, EVAL_BATCH)]), (NB, NB, NB, NB), mode="replicate")
        nb = torch.cat([self._nb_raw(cpad, img[a:a + 50000], cell[a:a + 50000])
                        for a in range(0, N_UNLAB, 50000)]).double().cpu().numpy()
        del cpad
        self.sc, self.sx = E.Standardizer.fit(cu), E.Standardizer.fit(xu)
        C, X = self.sc(cu), self.sx(xu)
        NBs = (nb - np.repeat(self.sc.mean, (2 * NB + 1) ** 2)) / np.repeat(self.sc.std, (2 * NB + 1) ** 2)
        self.losses = {}

        def fit(name, model, Xin, Y, k):
            torch.manual_seed(seed * 100 + k)
            self.losses[name] = E.fit_regression(model, Xin, Y, R2.TR, seed * 100 + k)
            return model.requires_grad_(False).eval()

        A_pinv = np.linalg.pinv(A)
        _, P_N = projectors(A)
        self.m = fit("m", E.mlp(3, X.shape[1], **R2.NET), C, X, 12)
        self.rnd = fit("rnd", E.RNDReconstructor(A_pinv, P_N, self.sc, self.sx, **R2.NET), C, X, 13)
        self.m_sp = fit("m_sp", E.mlp(NBs.shape[1], X.shape[1], **R2.NET), NBs, X, 15)
        del xu, cu, nb, C, X, NBs
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        self.c_mu, self.c_sd, self.x_mu, self.x_sd = t(self.sc.mean), t(self.sc.std), t(self.sx.mean), t(self.sx.std)
        self.u_stats = {}
        self.f_stats = self._stats(tr, lambda b: F.interpolate(tr.f[b].float().permute(0, 3, 1, 2), size=(MAP, MAP),
                                                               mode="bilinear", align_corners=False))
        for u in sorted(set(U_OF.values())):
            self.u_stats[u] = self._stats(tr, lambda b, u=u: self._u_raw(u, tr.x[b]))

    @staticmethod
    def _nb_raw(cpad, img, cell):
        """5 x 5 neighbourhood of c̄ around `cell` of image `img`; cpad = replicate-padded c̄ maps (n, 3, 132, 132).
        Returns (len(img), 75), channel-major, then row, then column (= F.unfold order, head.neighbourhood)."""
        k = 2 * NB + 1
        dy, dx = torch.meshgrid(torch.arange(k, device=cpad.device), torch.arange(k, device=cpad.device), indexing="ij")
        g = cpad[img[:, None], :, (cell // MAP)[:, None] + dy.reshape(1, -1), (cell % MAP)[:, None] + dx.reshape(1, -1)]
        return g.permute(0, 2, 1).reshape(len(img), -1)  # (n, k*k, 3) -> (n, 3*k*k)

    @torch.no_grad()
    def _u_raw(self, u, xb):
        """Unstandardised u map (B, 204, 128, 128) from x̄ maps (B, 128, 128, 204)."""
        x = xb.float()
        if u == "x":
            out = x
        else:
            cs = (x @ self.A.T - self.c_mu) / self.c_sd
            if u == "r":
                out = x - (self.m(cs) * self.x_sd + self.x_mu)
            elif u == "r_rnd":
                out = x - (self.rnd(cs) * self.x_sd + self.x_mu)
            elif u == "r_sp":
                nb = neighbourhood(cs.permute(0, 3, 1, 2), NB).permute(0, 2, 3, 1)
                out = x - (self.m_sp(nb) * self.x_sd + self.x_mu)
            else:
                raise KeyError(u)
        return out.permute(0, 3, 1, 2)

    @torch.no_grad()
    def _stats(self, tr, fn):
        s = s2 = n = 0
        for a in range(0, tr.n, EVAL_BATCH):
            v = fn(torch.arange(a, min(a + EVAL_BATCH, tr.n), device=self.dev)).double()
            s = s + v.sum((0, 2, 3)); s2 = s2 + (v ** 2).sum((0, 2, 3)); n += v.shape[0] * v.shape[2] * v.shape[3]
        mu = s / n
        return mu.float().view(1, -1, 1, 1), (s2 / n - mu ** 2).clamp(min=0).sqrt().add(1e-12).float().view(1, -1, 1, 1)

    @torch.no_grad()
    def inputs(self, method, sp: Split, b):
        f32 = None
        if method != "S1":
            mu, sd = self.f_stats  # f is standardised on the 32 x 32 grid; the affine map commutes with upsampling
            f32 = (sp.f[b].float().permute(0, 3, 1, 2) - mu) / sd
        u = None
        if method != "S0":
            mu, sd = self.u_stats[U_OF[method]]
            u = (self._u_raw(U_OF[method], sp.x[b]) - mu) / sd
        return f32, u


def gamma_size(head):
    return float(head.gamma.abs().mean()) if head.gamma is not None else float("nan")


def train_eval(comp, tr, ev, method, K, seed, pool, ev_pts, checkpoints, dev):
    """Train one head for max(checkpoints) steps; score the evaluation points at every checkpoint."""
    sel = pool["l_rank"] < K
    li = torch.from_numpy(pool["l_img"][sel]).to(dev)
    lp = torch.from_numpy(pool["l_pix"][sel]).to(dev)
    ly = torch.from_numpy(pool["l_y"][sel].astype(np.int64)).to(dev)
    imgs = torch.unique(li)
    torch.manual_seed(seed * 1000 + K)
    head = FusionHead(None if method == "S1" else 768, None if method == "S0" else 204, N_CLASSES, **HEAD).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    gen = torch.Generator().manual_seed(seed * 1000 + K)  # same batches for every method
    curve, results, run_loss = [], {}, []
    t0 = time.time()
    for step in range(1, max(checkpoints) + 1):
        head.train()
        b = imgs[torch.randperm(len(imgs), generator=gen)[:OPT["batch_images"]].to(dev)]
        bs = torch.sort(b).values
        m = torch.isin(li, bs)
        pos = torch.searchsorted(bs, li[m])
        f32, u = comp.inputs(method, tr, bs)
        logits = read_at(head(f32, u), pos, lp[m])
        loss = F.cross_entropy(logits, ly[m])
        opt.zero_grad(); loss.backward(); opt.step()
        run_loss.append(loss.detach())
        if step % LOG_EVERY == 0:
            curve.append((step, float(torch.stack(run_loss).mean()))); run_loss = []
        if step in checkpoints:
            cm = evaluate(comp, ev, method, head, ev_pts, dev)
            results[step] = (cm, gamma_size(head), time.time() - t0)
    return results, curve


@torch.no_grad()
def evaluate(comp, ev, method, head, pts, dev):
    head.eval()
    img, pix, y = pts
    pred = np.empty(len(y), np.int64)
    for a in range(0, ev.n, EVAL_BATCH):
        b = torch.arange(a, min(a + EVAL_BATCH, ev.n), device=dev)
        m = (img >= a) & (img < a + len(b))
        if not m.any():
            continue
        f32, u = comp.inputs(method, ev, b)
        out = head(f32, u)
        pred[m] = read_at(out, torch.from_numpy(img[m] - a).to(dev), torch.from_numpy(pix[m]).to(dev)).argmax(1).cpu().numpy()
    return confusion(y, pred)


def run_seed(s, mode, out, dev, steps_file):
    t0 = time.time()
    E.set_device(dev)
    split = "test" if mode == "main" else "validation"
    out = Path(out) / f"seed{s}"
    out.mkdir(parents=True, exist_ok=True)
    tr, ev = Split("train", dev), Split(split, dev)
    comp = Components(tr, s, dev)
    print(f"seed {s}: components {time.time() - t0:.0f}s", flush=True)
    pool = np.load(CACHE2 / "pools" / f"seed{s}.npz")
    grid = np.load(CACHE2 / f"{split}.npz")
    n = np.bincount(grid["y"][grid["y"] >= 0], minlength=N_CLASSES)
    scored = np.flatnonzero(n >= MIN_TEST_PIXELS)
    keep = np.isin(grid["y"], scored)
    pts = (grid["img"][keep], grid["pix"][keep], grid["y"][keep].astype(np.int64))

    if mode == "select":
        methods = ["S0"]
    else:
        chosen = {int(k): v for k, v in json.load(open(steps_file))["steps"].items()}
        methods = METHODS
    rows, cms, curves = [], {}, {}
    for method in methods:
        for K in KS:
            cps = STEP_GRID if mode == "select" else [chosen[K]]
            res, curve = train_eval(comp, tr, ev, method, K, s, pool, pts, cps, dev)
            curves[f"{method}/K{K}"] = curve
            for step, (cm, g, dt) in res.items():
                bacc, miou = scores(cm, scored)
                rows.append(dict(seed=s, K=K, method=method, steps=step, bacc=bacc, miou=miou, gamma=g, train_s=round(dt, 1)))
                cms[f"{method}/K{K}/S{step}"] = cm
                print(f"seed {s} {method:8s} K={K:4d} steps {step:4d} bacc {bacc:.3f} miou {miou:.3f} |γ| {g:.4f} ({dt:.0f}s)", flush=True)
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "confusion.npz", scored=scored, **cms)
    json.dump(comp.losses, open(out / "losses.json", "w"))
    json.dump(curves, open(out / "curves.json", "w"))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def choose_steps(out):
    """The S_K rule: per K, the step count with the highest mean val bacc of S0 over the select seeds
    (ties: fewer steps)."""
    rows = [r for p in sorted(Path(out).glob("seed*/runs.csv")) for r in csv.DictReader(open(p))]
    seeds = sorted({int(r["seed"]) for r in rows})
    mean = {(int(r["K"]), int(r["steps"])): [] for r in rows}
    for r in rows:
        mean[(int(r["K"]), int(r["steps"]))].append(float(r["bacc"]))
    table = {f"{K}/{S}": float(np.mean(v)) for (K, S), v in mean.items()}
    steps = {K: max(STEP_GRID, key=lambda S: (round(np.mean(mean[(K, S)]), 6), -S)) for K in KS}
    json.dump(dict(seeds=seeds, steps=steps, mean_bacc=table), open(Path(out) / "steps.json", "w"), indent=1)
    print("steps", steps, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["select", "sanity", "main"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps-file", default="phase3/results/select/steps.json")
    ap.add_argument("--choose", action="store_true", help="select mode: only apply the S_K rule to finished seeds")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.choose:
        return choose_steps(a.out)
    for s in a.seeds:
        run_seed(s, a.mode, a.out, a.device, a.steps_file)


if __name__ == "__main__":
    main()
