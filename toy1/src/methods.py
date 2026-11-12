"""Feature extractors for M0-M6b and the reconstruction baselines (linear-probe family).

Every method maps (c, x) -> feature matrix; the head is the same for all
(StandardScaler + L2 logistic regression with fixed C).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .encoders import EncoderBundle, apply

# Pre-registered methods (used for H1-H4) and comparison baselines (reported separately).
PREREGISTERED = ["M0", "M1", "M2", "M3", "M4", "M5_ridge", "M5_mlp", "M6",
                 "M6b_direct", "M6b_ema", "M6b_vic"]
SCALE_ONLY = ["M0_S", "M0_L"]  # M0 with other phi sizes (M0 itself uses the main scale)
BASELINES_LINEAR = ["B_RND", "B_RNDres"]
LINEAR_METHODS = PREREGISTERED + SCALE_ONLY + BASELINES_LINEAR

RGB_ONLY = {"M0", "M0_S", "M0_L", "B_RND"}  # features are a function of c alone


class FeatureFactory:
    def __init__(self, data, enc: EncoderBundle, cfg: dict):
        self.w = data.world
        self.enc = enc
        self.main = cfg["encoders"]["phi"]["main_scale"]
        U = data.unlabeled
        # M5 (ridge): E[x|c] estimated by ridge regression of x on c over the unlabeled pool.
        self.ridge = make_pipeline(StandardScaler(), Ridge(alpha=1e-3)).fit(U.c, U.x)

    def phi(self, c, scale=None):
        return apply(self.enc.phi[scale or self.main], self.enc.sc(c))

    def all(self, c: np.ndarray, x: np.ndarray, methods=LINEAR_METHODS) -> dict:
        e, w = self.enc, self.w
        f = self.phi(c)
        cs, xs = e.sc(c), e.sx(x)
        out = {}
        for m in methods:
            if m == "M0":
                out[m] = f
            elif m in ("M0_S", "M0_L"):
                scale = m.split("_")[1]
                if scale in e.phi:
                    out[m] = self.phi(c, scale)
            elif m == "M1":
                out[m] = x
            elif m == "M2":
                out[m] = np.c_[f, x]
            elif m == "M3":
                out[m] = np.c_[f, x @ w.proj.P_N.T]
            elif m == "M4":
                out[m] = np.c_[f, w.post.residual(x, c)]
            elif m == "M5_ridge":
                out[m] = np.c_[f, x - self.ridge.predict(c)]
            elif m == "M5_mlp":
                out[m] = np.c_[f, x - e.sx.inv(apply(e.m5_mlp, cs))]
            elif m == "M6":
                out[m] = np.c_[f, apply(e.g, xs) - apply(e.p, f)]
            elif m.startswith("M6b_"):
                g, p = e.m6b[m.split("_", 1)[1]]
                out[m] = np.c_[f, apply(g, xs) - apply(p, f)]
            elif m == "B_RND":
                out[m] = np.c_[f, e.sx.inv(apply(e.rnd, cs))]
            elif m == "B_RNDres":
                out[m] = np.c_[f, x - e.sx.inv(apply(e.rnd, cs))]
            else:
                raise KeyError(m)
        return out


def make_head(cfg: dict, seed: int = 0):
    """Shared head. type=logreg (default): L2 logistic regression.
    type=mlp: small ReLU MLP (sklearn, L-BFGS, fixed L2), used for the head-capacity control."""
    h = cfg["head"]
    if h.get("type", "logreg") == "mlp":
        m = h["mlp"]
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(m["hidden"]), alpha=m["alpha"], solver="lbfgs",
            max_iter=m["max_iter"], random_state=seed))
    return make_pipeline(StandardScaler(), LogisticRegression(C=h["C"], max_iter=h["max_iter"]))
