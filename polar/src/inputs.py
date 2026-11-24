"""Network input of each method from a raw uint8 batch. Everything fitted here is label-free and uses all train
images (the RGB-polarisation pairs, never the masks): the standardisation of p and r, and the predictor m(c)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import CACHE, SAM_MEAN, SAM_STD, features, load_split, sam_norm
from .projectors import ResidualPredictor

METHODS = ("simcmf", "rgb", "fuse_p", "null_p", "null_phys", "null_orth", "null_orthf", "null_fres")
N_EXTRA = {"rgb": 0, "fuse_p": 6, "null_p": 6, "null_phys": 6, "null_orth": 6, "null_orthf": 6, "null_fres": 6}
# Non-neural m(c) (same module, same r standardisation, only how m(c) is formed differs):
#   null_phys  m(c) = closed-form physics operator: AoLP parallel / perpendicular to the intensity-gradient azimuth
#              (diffuse / specular reflection), magnitude modulated by edge strength, specularity (min/mean colour)
#              and brightness; 8 linear coefficients per colour fitted by least squares on train (label-free).
#   null_orth  m(c) = projection of p onto the span of c inside each 16 x 16 SAM patch (Gram-Schmidt-style
#              u = g - proj_c g, cf. Han et al. 2025 orthogonal gradients, applied to the input): r is p minus its
#              least-squares fit by the patch-centred c, so r is orthogonal to c's within-patch variation while the
#              patch mean of p is kept. No fitted parameters.
#   null_fres  m(c) = Fresnel operator on monocular surface normals n(c) (precomputed, polar/scripts/normals.py):
#              zenith theta = acos|n_z| (orthographic view), azimuth alpha = atan2(n_y, n_x); refractive index 1.5;
#              diffuse  DoLP rho_d(theta), AoLP = alpha;  specular DoLP rho_s(theta), AoLP = alpha + 90 deg.
#              Bases rho_d u, rho_s u, spec * rho_s u (u = (cos 2alpha, sin 2alpha), both image handednesses) and
#              a constant; 8 coefficients per colour by least squares on train (label-free). Also logs the R^2 of the
#              unfitted (pure) diffuse / specular Fresnel predictions.
#   null_orthf input = p (as fuse_p); the orthogonalisation is done on features, token by token, in the projector:
#              the polarisation token sum_i w_i E_i(p_i) minus its projection on the RGB token E(c).
PATCH = 16
ORTH_RIDGE = 1e-6
ETA = 1.5


def fingerprint(raw: torch.Tensor) -> list[str]:
    """Identify an image of the cached splits from its raw uint8 stack (the training data is not augmented)."""
    x = raw[:, :, ::53, ::59].contiguous().cpu().numpy()
    return [hashlib.sha1(v.tobytes()).hexdigest() for v in x]


def fresnel(cos_t):
    """Degree of linear polarisation of diffuse and specular reflection (refractive index ETA) for cos(zenith)."""
    s2 = (1 - cos_t ** 2).clamp(min=0)
    root = (ETA ** 2 - s2).sqrt()
    rd = (ETA - 1 / ETA) ** 2 * s2 / (2 + 2 * ETA ** 2 - (ETA + 1 / ETA) ** 2 * s2 + 4 * cos_t * root)
    rs = 2 * s2 * cos_t * root / (ETA ** 2 - s2 - ETA ** 2 * s2 + 2 * s2 ** 2)
    return rd, rs
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
        if method == "null_fres":
            self._load_normals()  # _batch attaches normals, so they must be indexed before any statistics
        if method in ("fuse_p", "null_orthf") or method.startswith("null_"):
            self.p_mu, self.p_sd = self._stats(lambda f: f["p"])
        if method == "null_p":
            self.m = self._predictor(seed, log)
        elif method == "null_phys":
            self.m = self._phys_fit(log)
        elif method == "null_orth":
            self.m = None
        elif method == "null_fres":
            self.m = self._fres_fit(log)
        if method.startswith("null_") and method != "null_orthf":
            if method != "null_p":
                self.info["predictor"] = {"r2_eval": self._r2_fres() if method == "null_fres" else
                                          self._r2(self.m) if self.m is not None else self._r2_orth()}
                log(f"  {method} R^2 on the evaluation split (variance of p_std removed, per channel): "
                    f"{np.round(self.info['predictor']['r2_eval'], 3).tolist()}")
            self.r_mu, self.r_sd = self._stats(lambda f: self._resid(f))

    def _batch(self, raw, idx):
        x = torch.from_numpy(np.stack([raw[i] for i in idx])).to(self.dev)
        f = features(self.ds, x)
        if self.method == "null_fres":
            f["nrm"] = self._normals(x)
        return f

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
        if self.method == "null_orth":
            return self._orth(self._pstd(f), sam_norm(f["c"]))
        m = self.m(down(sam_norm(f["c"])), f["nrm"]) if self.method == "null_fres" else self.m(down(sam_norm(f["c"])))
        pred = F.interpolate(m, size=f["p"].shape[-2:], mode="bilinear", align_corners=False)
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

    # ---- non-neural m(c) -------------------------------------------------------------------------------------
    @staticmethod
    def _orth(p, c):
        """Per 16x16 patch: p - sum_k a_k (c_k - mean c_k), a = ridge least squares of p on [1, c] in the patch."""
        B, P, H, W = p.shape
        unf = lambda t: t.view(B, t.shape[1], H // PATCH, PATCH, W // PATCH, PATCH).permute(0, 2, 4, 3, 5, 1) \
            .reshape(B, H // PATCH, W // PATCH, PATCH * PATCH, t.shape[1])
        X, Y = unf(c.double()), unf(p.double())
        Xc = X - X.mean(-2, keepdim=True)
        G = Xc.transpose(-1, -2) @ Xc + ORTH_RIDGE * PATCH * PATCH * torch.eye(3, device=p.device, dtype=X.dtype)
        a = torch.linalg.solve(G, Xc.transpose(-1, -2) @ Y)                        # (B, h, w, 3, P)
        R = (Y - Xc @ a).float()
        return R.view(B, H // PATCH, W // PATCH, PATCH, PATCH, P).permute(0, 5, 1, 3, 2, 4).reshape(B, P, H, W)

    # ---- Fresnel operator on estimated normals ---------------------------------------------------------------------
    def _load_normals(self):
        self.nrm_arr, self.nrm_idx = {}, {}
        for sp in ("train", "test" if self.ds == "pgsnet" else "val"):
            f = CACHE / self.ds / f"{sp}_normal.npy"
            if not f.exists():
                raise FileNotFoundError(f"{f}: run python -m polar.scripts.normals {self.ds} first")
            self.nrm_arr[sp] = np.load(f, mmap_mode="r")
            raw, _, _ = load_split(self.ds, sp)
            for i in range(len(raw)):
                self.nrm_idx[fingerprint(torch.from_numpy(np.asarray(raw[i:i + 1])))[0]] = (sp, i)

    def _normals(self, raw):
        """Unit camera-frame normals (x right, y down, z forward) at 512 x 512 for a raw batch."""
        keys = [self.nrm_idx[k] for k in fingerprint(raw)]
        n = torch.from_numpy(np.stack([np.asarray(self.nrm_arr[sp][i]) for sp, i in keys])).to(self.dev).float()
        return n / n.norm(dim=1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _fres_basis(c, n):
        """c (B,3,512,512) SAM-normalised, n (B,3,512,512) -> (B, 3 colours, 2 Stokes comps, 8 bases, H, W) and the
        pure Fresnel predictions (diffuse, specular) as (B, 2 comps, H, W) each."""
        c = c * SAM_STD.to(c.device) + SAM_MEAN.to(c.device)
        spec = c.min(1, keepdim=True)[0] / (c.mean(1, keepdim=True) + 1.0)
        nx, ny, nz = n[:, :1], n[:, 1:2], n[:, 2:3]
        rd, rs = fresnel(nz.abs().clamp(max=1))
        h2 = (nx ** 2 + ny ** 2).clamp(min=1e-8)
        cos2, sin2 = (nx ** 2 - ny ** 2) / h2, 2 * nx * ny / h2                      # 2 x azimuth
        z, o = torch.zeros_like(rd), torch.ones_like(rd)
        bq, bu = [o, z], [z, o]
        for w in (rd, rs, spec * rs):
            bq += [w * cos2, w * cos2]; bu += [w * sin2, -w * sin2]
        b = torch.stack([torch.cat(bq, 1), torch.cat(bu, 1)], 1)
        pure_d = torch.cat([rd * cos2, rd * sin2], 1)
        pure_s = -torch.cat([rs * cos2, rs * sin2], 1)                              # AoLP + 90 deg
        return b.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1), pure_d, pure_s

    def _fres_fit(self, log):
        idx = np.random.default_rng(11).permutation(len(self.raw_tr))[:N_STATS]
        XtX = torch.zeros(3, 8, 8, dtype=torch.float64, device=self.dev); XtY = torch.zeros(3, 8, dtype=torch.float64, device=self.dev)
        with torch.no_grad():
            for a in range(0, len(idx), 8):
                f = self._batch(self.raw_tr, idx[a:a + 8])
                X = self._fres_basis(down(sam_norm(f["c"])), f["nrm"])[0].double()
                p = down(self._pstd(f)).double()
                Y = torch.stack([p[:, :3], p[:, 3:]], 2)
                XtX += torch.einsum("bcsihw,bcsjhw->cij", X, X); XtY += torch.einsum("bcsihw,bcshw->ci", X, Y)
        coef = torch.linalg.solve(XtX + 1e-6 * XtX.diagonal(dim1=-2, dim2=-1).mean() * torch.eye(8, device=self.dev, dtype=torch.float64), XtY).float()
        self.info["fres_coef"] = coef.cpu().tolist()
        log(f"  Fresnel operator coefficients (per colour; const q,u, rho_d, rho_d', rho_s, rho_s', spec rho_s, spec rho_s'): "
            f"{np.round(coef.cpu().numpy(), 3).tolist()}")

        def m(c, n):
            out = torch.einsum("bcsihw,ci->bcshw", self._fres_basis(c, n)[0], coef)
            return torch.cat([out[:, :, 0], out[:, :, 1]], 1)
        return m

    @torch.no_grad()
    def _r2_fres(self):
        """R^2 at 512 of the fitted operator, and of the unfitted Fresnel predictions in raw p units
        (diffuse / specular, each with both image handednesses), on the evaluation split."""
        raw_ev, _, _ = load_split(self.ds, "test" if self.ds == "pgsnet" else "val")
        acc = {k: 0 for k in ("fit", "d+", "d-", "s+", "s-")}
        ss = s = n = 0
        for a in range(0, len(raw_ev), 8):
            f = self._batch(raw_ev, range(a, min(a + 8, len(raw_ev))))
            c, nrm = down(sam_norm(f["c"])), f["nrm"]
            pstd, praw = down(self._pstd(f)).double(), down(f["p"]).double()
            _, pd, ps = self._fres_basis(c, nrm)
            flip = lambda q: torch.cat([q[:, :1], -q[:, 1:]], 1)
            rgb2 = lambda q: torch.cat([q[:, :1].expand(-1, 3, -1, -1), q[:, 1:].expand(-1, 3, -1, -1)], 1).double()
            acc["fit"] = acc["fit"] + ((pstd - self.m(c, nrm).double()) ** 2).sum((0, 2, 3))
            for k, q in (("d+", pd), ("d-", flip(pd)), ("s+", ps), ("s-", flip(ps))):
                acc[k] = acc[k] + ((praw - rgb2(q)) ** 2).sum((0, 2, 3))
            ss = ss + (praw ** 2).sum((0, 2, 3)); s = s + praw.sum((0, 2, 3)); n += praw[:, 0].numel()
            if a == 0:
                ss_std = s_std = 0
            ss_std = ss_std + (pstd ** 2).sum((0, 2, 3)); s_std = s_std + pstd.sum((0, 2, 3))
        var_raw = ss / n - (s / n) ** 2
        var_std = ss_std / n - (s_std / n) ** 2
        self.info["fres_pure_r2"] = {k: (1 - (acc[k] / n) / var_raw).mean().item() for k in ("d+", "d-", "s+", "s-")}
        return (1 - (acc["fit"] / n) / var_std).cpu().tolist()

    @staticmethod
    def _phys_basis(c):
        """c: (B,3,H,W) SAM-normalised intensity at 512 -> (B, 3 colours, 2 Stokes comps, 8 bases, H, W)."""
        c = c * SAM_STD.to(c.device) + SAM_MEAN.to(c.device)                        # back to 0..255
        g = c.mean(1, keepdim=True)
        k = torch.exp(-torch.arange(-6, 7, device=c.device).float() ** 2 / 8); k = k / k.sum()  # Gaussian sigma 2
        g = F.conv2d(F.conv2d(F.pad(g, (6, 6, 6, 6), mode="replicate"), k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=c.device).view(1, 1, 3, 3)
        gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
        gx, gy = F.conv2d(gp, sx), F.conv2d(gp, sx.transpose(-1, -2))
        m2 = gx ** 2 + gy ** 2
        cos2, sin2 = (gx ** 2 - gy ** 2) / (m2 + 1e-6), 2 * gx * gy / (m2 + 1e-6)  # gradient azimuth, doubled
        mag = m2.sqrt()
        t = mag / (mag + 4.0)                                                       # edge strength in [0,1)
        spec = c.min(1, keepdim=True)[0] / (c.mean(1, keepdim=True) + 1.0)          # specular-free ratio
        L = c.mean(1, keepdim=True) / 255
        z, o = torch.zeros_like(t), torch.ones_like(t)
        bq, bu = [o, z], [z, o]                                                     # constant Stokes offset
        for w in (t, t * spec, t * L):
            bq += [w * cos2, w * cos2]; bu += [w * sin2, -w * sin2]               # both image-axis handednesses
        b = torch.stack([torch.cat(bq, 1), torch.cat(bu, 1)], 1)                   # (B, 2, 8, H, W)
        return b.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1)

    def _phys_fit(self, log):
        idx = np.random.default_rng(11).permutation(len(self.raw_tr))[:N_STATS]
        XtX = torch.zeros(3, 8, 8, dtype=torch.float64, device=self.dev); XtY = torch.zeros(3, 8, dtype=torch.float64, device=self.dev)
        with torch.no_grad():
            for a in range(0, len(idx), 8):
                f = self._batch(self.raw_tr, idx[a:a + 8])
                X = self._phys_basis(down(sam_norm(f["c"]))).double()               # B,3,2,8,H,W
                p = down(self._pstd(f)).double()
                Y = torch.stack([p[:, :3], p[:, 3:]], 2)                             # B,3,2,H,W
                XtX += torch.einsum("bcsihw,bcsjhw->cij", X, X); XtY += torch.einsum("bcsihw,bcshw->ci", X, Y)
        coef = torch.linalg.solve(XtX + 1e-6 * XtX.diagonal(dim1=-2, dim2=-1).mean() * torch.eye(8, device=self.dev, dtype=torch.float64), XtY).float()
        self.info["phys_coef"] = coef.cpu().tolist()
        log(f"  physics operator coefficients (per colour): {np.round(coef.cpu().numpy(), 3).tolist()}")

        def m(c):
            X = self._phys_basis(c)                                                 # B,3,2,8,H,W
            out = torch.einsum("bcsihw,ci->bcshw", X, coef)
            return torch.cat([out[:, :, 0], out[:, :, 1]], 1)
        return m

    @torch.no_grad()
    def _r2_orth(self):
        """Same definition as _r2 (1 - mean (p - m)^2 / var p, here m = p - r), at full resolution."""
        raw_ev, _, _ = load_split(self.ds, "test" if self.ds == "pgsnet" else "val")
        se = ss = s = n = 0
        for a in range(0, len(raw_ev), 4):
            f = self._batch(raw_ev, range(a, min(a + 4, len(raw_ev))))
            p = self._pstd(f)
            r = self._orth(p, sam_norm(f["c"])).double(); p = p.double()
            se = se + (r ** 2).sum((0, 2, 3)); ss = ss + (p ** 2).sum((0, 2, 3))
            s = s + p.sum((0, 2, 3)); n += p[:, 0].numel()
        var = ss / n - (s / n) ** 2
        return (1 - (se / n) / var).cpu().tolist()

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
        if self.method == "null_fres":
            f["nrm"] = self._normals(raw)
        if self.method == "simcmf":
            return f["simcmf"]
        c = sam_norm(f["c"])
        if self.method == "rgb":
            return c
        if self.method in ("fuse_p", "null_orthf"):
            return torch.cat([c, self._pstd(f)], 1)
        return torch.cat([c, (self._resid(f) - self.r_mu) / self.r_sd], 1)
