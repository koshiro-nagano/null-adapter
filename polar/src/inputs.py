"""Network input of each method from a raw uint8 batch. Everything fitted here is label-free and uses all train
images (the RGB-polarisation pairs, never the masks): the standardisation of p and r, and the predictor m(c)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import CACHE, features, load_split, sam_norm
from .projectors import ResidualPredictor

METHODS = ("simcmf", "rgb", "fuse_p", "null_p")
N_EXTRA = {"rgb": 0, "fuse_p": 6, "null_p": 6}
PRED = dict(steps=3000, batch=8, crop=256, lr=1e-3, wd=1e-4)
LOW = 512  # resolution of the predictor
N_STATS = 200


def down(x):
    return F.avg_pool2d(x, x.shape[-1] // LOW)


class Inputs:
    def __init__(self, ds: str, method: str, seed: int, dev, log=print):
        self.ds, self.method, self.dev = ds, method, dev
        self.raw_tr, _, _ = load_split(ds, "train")
        self.info = {}
        if method in ("fuse_p", "null_p"):
            self.p_mu, self.p_sd = self._stats(lambda f: f["p"])
        if method == "null_p":
            self.m = self._predictor(seed, log)
            self.r_mu, self.r_sd = self._stats(lambda f: self._resid(f))

    def _batch(self, raw, idx):
        return features(self.ds, torch.from_numpy(np.stack([raw[i] for i in idx])).to(self.dev))

    @torch.no_grad()
    def _stats(self, fn):
        idx = np.random.default_rng(11).permutation(len(self.raw_tr))[:N_STATS]
        s = s2 = n = 0
        for a in range(0, len(idx), 8):
            v = fn(self._batch(self.raw_tr, idx[a:a + 8])).double()
            s, s2, n = s + v.sum((0, 2, 3)), s2 + (v ** 2).sum((0, 2, 3)), n + v[:, 0].numel()
        mu = s / n
        return mu.float().view(1, -1, 1, 1), (s2 / n - mu ** 2).clamp(min=1e-12).sqrt().float().view(1, -1, 1, 1)

    def _pstd(self, f):
        return (f["p"] - self.p_mu) / self.p_sd

    def _resid(self, f):
        """r = p_std - m(c), m evaluated at 512 and bilinearly upsampled; p keeps its full resolution."""
        pred = F.interpolate(self.m(down(sam_norm(f["c"]))), size=f["p"].shape[-2:], mode="bilinear", align_corners=False)
        return self._pstd(f) - pred

    def _predictor(self, seed, log):
        path = CACHE / self.ds / f"predictor_seed{seed}.pt"
        m = ResidualPredictor().to(self.dev)
        if path.exists():
            ck = torch.load(path, map_location=self.dev)
            m.load_state_dict(ck["state"]); self.info["predictor"] = ck["info"]
            return m.eval().requires_grad_(False)
        torch.manual_seed(5000 + seed)
        g = np.random.default_rng(5000 + seed)
        opt = torch.optim.AdamW(m.parameters(), lr=PRED["lr"], weight_decay=PRED["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PRED["steps"])
        t0, losses = time.time(), []
        for step in range(PRED["steps"]):
            with torch.no_grad():
                f = self._batch(self.raw_tr, g.integers(0, len(self.raw_tr), PRED["batch"]))
                c, p = down(sam_norm(f["c"])), down(self._pstd(f))
                y0, x0 = g.integers(0, LOW - PRED["crop"] + 1, 2)
                c, p = (t[..., y0:y0 + PRED["crop"], x0:x0 + PRED["crop"]] for t in (c, p))
            loss = F.mse_loss(m(c), p)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(float(loss))
            if (step + 1) % 500 == 0:
                log(f"  predictor step {step + 1}: mse {np.mean(losses[-500:]):.4f}")
        m.eval().requires_grad_(False)
        info = {"train_mse_last500": float(np.mean(losses[-500:])), "seconds": time.time() - t0,
                "r2_eval": self._r2(m)}
        log(f"  predictor R^2 on the evaluation split (p from c, per channel): {np.round(info['r2_eval'], 3).tolist()}")
        torch.save({"state": m.state_dict(), "info": info}, path)
        self.info["predictor"] = info
        return m

    @torch.no_grad()
    def _r2(self, m):
        """Fraction of the variance of standardised p (at 512) explained by m(c) on the evaluation split."""
        raw_ev, _, _ = load_split(self.ds, "test" if self.ds == "pgsnet" else "val")
        se = ss = s = n = 0
        for a in range(0, len(raw_ev), 8):
            f = self._batch(raw_ev, range(a, min(a + 8, len(raw_ev))))
            c, p = down(sam_norm(f["c"])), down(self._pstd(f)).double()
            se = se + ((p - m(c).double()) ** 2).sum((0, 2, 3)); ss = ss + (p ** 2).sum((0, 2, 3))
            s = s + p.sum((0, 2, 3)); n += p[:, 0].numel()
        var = ss / n - (s / n) ** 2
        return (1 - (se / n) / var).cpu().tolist()

    @torch.no_grad()
    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        f = features(self.ds, raw)
        if self.method == "simcmf":
            return f["simcmf"]
        c = sam_norm(f["c"])
        if self.method == "rgb":
            return c
        if self.method == "fuse_p":
            return torch.cat([c, self._pstd(f)], 1)
        return torch.cat([c, (self._resid(f) - self.r_mu) / self.r_sd], 1)
