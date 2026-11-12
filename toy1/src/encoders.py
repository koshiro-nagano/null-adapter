"""Self-supervised encoders trained on the unlabeled pool only (never on labels).

phi(c)      : "RGB foundation model" stand-in, denoising AE on c, frozen after training.
g(x)        : target encoder, AE on x (M6), or trained through the predictive loss (M6b).
p           : predictor phi(c) -> g(x).
m5_mlp      : c -> x regressor (estimate of E[x|c]).
rnd_h       : RND-SCI-style generator, x_hat = A^+ c + P_N h(c).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cpu")


def set_device(device: str) -> torch.device:
    global DEVICE
    DEVICE = torch.device(device)
    return DEVICE


def mlp(d_in: int, d_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> "Standardizer":
        return cls(a.mean(0), a.std(0) + 1e-12)

    def __call__(self, a):
        return (a - self.mean) / self.std

    def inv(self, a):
        return a * self.std + self.mean


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def _batches(n: int, bs: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen).to(DEVICE)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


@torch.no_grad()
def apply(model: nn.Module, a: np.ndarray, bs: int = 20000) -> np.ndarray:
    model.eval()
    out = [model(_t(a[i:i + bs])).cpu().numpy() for i in range(0, len(a), bs)]
    return np.concatenate(out).astype(np.float64)


def fit_regression(model: nn.Module, X: np.ndarray, Y: np.ndarray, tr: dict, seed: int,
                   input_noise: float = 0.0, loss_fn=None, epochs: int | None = None,
                   on_epoch=None) -> list:
    """Generic MSE regression loop (Adam, mini-batch). Returns per-epoch mean loss."""
    gen = torch.Generator().manual_seed(seed)
    Xt, Yt = _t(X), _t(Y)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    loss_fn = loss_fn or (lambda pred, y: ((pred - y) ** 2).mean())
    hist = []
    for _ in range(epochs or tr["epochs"]):
        model.train()
        tot, nb = torch.zeros((), device=DEVICE), 0
        for idx in _batches(len(Xt), tr["batch"], gen):
            xb = Xt[idx]
            if input_noise > 0:
                xb = xb + input_noise * torch.randn(xb.shape, generator=gen).to(DEVICE)
            loss = loss_fn(model(xb), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach(); nb += 1
        hist.append(float(tot) / nb)
        if on_epoch is not None:
            on_epoch(len(hist), len(hist) * nb)
    return hist


class AE(nn.Module):
    def __init__(self, d_in, latent, width, depth):
        super().__init__()
        self.enc = mlp(d_in, latent, width, depth)
        self.dec = mlp(latent, d_in, width, depth)

    def forward(self, a):
        return self.dec(self.enc(a))


class RNDReconstructor(nn.Module):
    """x_hat = A^+ c + P_N q(c), in standardized-x units (RND-SCI, Wang et al. 2023)."""

    def __init__(self, A_pinv, P_N, sc: Standardizer, sx: Standardizer, width, depth):
        super().__init__()
        self.h = mlp(len(sc.mean), len(sx.mean), width, depth)
        self.register_buffer("A_pinv", _t(A_pinv))
        self.register_buffer("P_N", _t(P_N))
        for k, v in dict(c_mu=sc.mean, c_sd=sc.std, x_mu=sx.mean, x_sd=sx.std).items():
            self.register_buffer(k, _t(v))

    def forward(self, c_std):
        c = c_std * self.c_sd + self.c_mu
        xr = c @ self.A_pinv.T                    # range part, fixed by c
        q = self.h(c_std) * self.x_sd + self.x_mu  # unconstrained guess
        xn = q @ self.P_N.T                       # projected to null(A)
        return (xr + xn - self.x_mu) / self.x_sd


@dataclass
class EncoderBundle:
    sc: Standardizer
    sx: Standardizer
    phi: dict           # scale -> encoder nn.Module (c_std -> latent)
    g: nn.Module        # x_std -> latent (frozen AE encoder)
    p: nn.Module        # phi_main(c) -> g(x)
    m5_mlp: nn.Module   # c_std -> x_std
    rnd: nn.Module      # c_std -> x_std (range-null constrained)
    m6b: dict           # variant -> (g, p)
    dynamics: list      # rows for M6 / M6b training dynamics
    losses: dict


def _log_dyn(rows, variant, step, g, p, F_eval, X_eval):
    with torch.no_grad():
        g.eval(); p.eval()
        z = g(X_eval)
        r = z - p(F_eval)
        var = z.var(0)
        rows.append(dict(variant=variant, step=step,
                         r_norm=float(r.norm(dim=1).mean()),
                         r_var_total=float(r.var(0).sum()),
                         g_var_total=float(var.sum()),
                         g_var_min=float(var.min()),
                         g_var_dims=var.cpu().numpy().tolist(),
                         pred_r2=float(1 - r.var(0).sum() / var.sum().clamp_min(1e-12))))
        g.train(); p.train()


def train_m6b(variant: str, g_init: nn.Module, phi_feat: np.ndarray, X: np.ndarray, cfg: dict,
              seed: int, rows: list, eval_idx: np.ndarray, dec_init: nn.Module | None = None):
    """Jointly train target g and predictor p on || g(x) - p(phi(c)) ||^2.

    direct : gradient flows into g (no anti-collapse term).
    ema    : online g_o gets the gradient; residual target is g_t = EMA(g_o).
    vic    : direct + VICReg-style variance hinge  sum_d relu(target_std - std_d(g(x))).
    rec*   : direct + lambda * || dec(g(x)) - x ||^2 ; dec starts from the AE decoder of g and is
             trained jointly (lambda from cfg encoders.m6b.recon_weights[variant]).
    """
    e = cfg["encoders"]; m = e["m6b"]; tr = e["train"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    g_on = copy.deepcopy(g_init).requires_grad_(True).train()
    p = mlp(phi_feat.shape[1], e["g"]["latent"], e["predictor"]["width"], e["predictor"]["depth"]).to(DEVICE)
    g_tgt = copy.deepcopy(g_on).requires_grad_(False) if variant == "ema" else g_on
    params = list(g_on.parameters()) + list(p.parameters())
    dec = None
    if variant.startswith("rec"):
        lam = m["recon_weights"][variant]
        dec = copy.deepcopy(dec_init).requires_grad_(True).train()
        params += list(dec.parameters())
    opt = torch.optim.Adam(params, lr=tr["lr"])
    Ft, Xt = _t(phi_feat), _t(X)
    ev = torch.as_tensor(eval_idx, device=DEVICE)
    F_eval, X_eval = Ft[ev], Xt[ev]
    step = 0
    _log_dyn(rows, f"M6b_{variant}", step, g_tgt, p, F_eval, X_eval)
    for _ in range(m["epochs"]):
        for idx in _batches(len(Xt), tr["batch"], gen):
            z = g_on(Xt[idx])
            loss = ((z - p(Ft[idx])) ** 2).sum(1).mean()
            if variant == "vic":
                std = torch.sqrt(z.var(0) + 1e-4)
                loss = loss + m["vic_weight"] * torch.relu(m["vic_target_std"] - std).sum()
            if dec is not None:
                loss = loss + lam * ((dec(z) - Xt[idx]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if variant == "ema":
                with torch.no_grad():
                    for pt, po in zip(g_tgt.parameters(), g_on.parameters()):
                        pt.mul_(m["ema_momentum"]).add_(po, alpha=1 - m["ema_momentum"])
            step += 1
            if step % m["log_every"] == 0:
                _log_dyn(rows, f"M6b_{variant}", step, g_tgt, p, F_eval, X_eval)
    return g_tgt, p


def train_encoders(data, cfg: dict, seed: int, scales=None) -> EncoderBundle:
    torch.manual_seed(seed)
    e = cfg["encoders"]; tr = e["train"]
    U = data.unlabeled
    sc, sx = Standardizer.fit(U.c), Standardizer.fit(U.x)
    C, X = sc(U.c), sx(U.x)
    losses = {}

    phi = {}
    for i, (name, sz) in enumerate(e["phi"]["scales"].items()):
        if scales is not None and name not in scales:
            continue
        torch.manual_seed(seed * 100 + i)
        ae = AE(C.shape[1], e["phi"]["latent"], sz["width"], sz["depth"])
        losses[f"phi_{name}"] = fit_regression(ae, C, C, tr, seed * 100 + i, e["phi"]["input_noise"])
        phi[name] = ae.enc.requires_grad_(False).eval()
    F = apply(phi[e["phi"]["main_scale"]], C)

    torch.manual_seed(seed * 100 + 10)
    gae = AE(X.shape[1], e["g"]["latent"], e["g"]["width"], e["g"]["depth"])
    losses["g_ae"] = fit_regression(gae, X, X, tr, seed * 100 + 10)
    g = gae.enc.requires_grad_(False).eval()

    # M6: predictor with frozen target; log the same dynamics for comparison.
    rows = []
    eval_idx = np.random.default_rng(seed).choice(len(X), 5000, replace=False)
    Z = apply(g, X)
    torch.manual_seed(seed * 100 + 11)
    p = mlp(F.shape[1], e["g"]["latent"], e["predictor"]["width"], e["predictor"]["depth"]).to(DEVICE)
    F_ev, X_ev = _t(F[eval_idx]), _t(X[eval_idx])
    _log_dyn(rows, "M6", 0, g, p, F_ev, X_ev)
    losses["p"] = fit_regression(p, F, Z, tr, seed * 100 + 11, epochs=e["m6b"]["epochs"],
                                 on_epoch=lambda ep, step: _log_dyn(rows, "M6", step, g, p, F_ev, X_ev))
    p.requires_grad_(False).eval()

    torch.manual_seed(seed * 100 + 12)
    m5 = mlp(C.shape[1], X.shape[1], e["m5_mlp"]["width"], e["m5_mlp"]["depth"])
    losses["m5_mlp"] = fit_regression(m5, C, X, tr, seed * 100 + 12)

    torch.manual_seed(seed * 100 + 13)
    w = data.world
    rnd = RNDReconstructor(w.proj.A_pinv, w.proj.P_N, sc, sx, e["rnd_h"]["width"], e["rnd_h"]["depth"])
    losses["rnd"] = fit_regression(rnd, C, X, tr, seed * 100 + 13)

    m6b = {}
    for j, v in enumerate(e["m6b"].get("variants", ["direct", "ema", "vic"])):
        gv, pv = train_m6b(v, g, F, X, cfg, seed * 100 + 20 + j, rows, eval_idx, dec_init=gae.dec)
        m6b[v] = (gv.requires_grad_(False).eval(), pv.requires_grad_(False).eval())

    return EncoderBundle(sc, sx, phi, g, p, m5.eval(), rnd.eval(), m6b, rows, losses)
