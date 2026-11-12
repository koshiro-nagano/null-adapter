"""GPU port of the pre-registered head, make_pipeline(StandardScaler(), LogisticRegression(C, max_iter)).

Same problem as scikit-learn 1.5 (lbfgs, multinomial, L2): minimise
  mean_i CE(W z_i + b, y_i) + 1 / (2 C n) ||W||^2      (intercept not penalised)
on z = standardised features (population std, zero-variance columns left unscaled), in float64, with L-BFGS
(history 10, strong Wolfe) and the same stopping rule as sklearn's scipy call: max |grad| <= tol (1e-4), or a
relative objective decrease <= 64 eps, or max_iter iterations. Not used by the frozen main run; equivalence with
sklearn is checked on val by scripts/check_gpu_head.py before any use.
"""
from __future__ import annotations

import numpy as np
import torch


class GPULogReg:
    def __init__(self, C=1.0, max_iter=5000, tol=1e-4, device="cuda", dtype=torch.float64, col_weight=None):
        """col_weight: optional (d,) multiplier applied after standardisation (exploratory block weighting)."""
        self.C, self.max_iter, self.tol, self.device, self.dtype = C, max_iter, tol, device, dtype
        self.col_weight = col_weight

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    def _z(self, X):
        return (self._t(X) - self.mean_) / self.scale_

    def fit(self, X, y):
        X = self._t(X)
        self.mean_ = X.mean(0)
        sd = X.std(0, unbiased=False)
        self.scale_ = torch.where(sd < 10 * torch.finfo(self.dtype).eps, torch.ones_like(sd), sd)
        if self.col_weight is not None:
            self.scale_ = self.scale_ / self._t(self.col_weight)
        Z = (X - self.mean_) / self.scale_
        del X
        self.classes_, yi = np.unique(y, return_inverse=True)
        Y = torch.as_tensor(yi, device=self.device)
        n, d, k = Z.shape[0], Z.shape[1], len(self.classes_)
        W = torch.zeros(d, k, dtype=self.dtype, device=self.device, requires_grad=True)
        b = torch.zeros(k, dtype=self.dtype, device=self.device, requires_grad=True)
        l2 = 1.0 / (self.C * n)
        # torch's own stopping tests are switched off: only the sklearn-equivalent rule below decides
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=1, history_size=10, line_search_fn="strong_wolfe",
                                tolerance_grad=0.0, tolerance_change=0.0)

        def closure():
            opt.zero_grad()
            f = torch.nn.functional.cross_entropy(Z @ W + b, Y) + 0.5 * l2 * (W * W).sum()
            f.backward()
            return f

        f_prev = None
        for it in range(1, self.max_iter + 1):
            f = float(opt.step(closure))
            g = max(float(W.grad.abs().max()), float(b.grad.abs().max()))
            if g <= self.tol:
                break
            if f_prev is not None and (f_prev - f) / max(abs(f_prev), abs(f), 1.0) <= 64 * np.finfo(float).eps:
                break
            f_prev = f
        self.n_iter_, self.W_, self.b_ = it, W.detach(), b.detach()
        return self

    @torch.no_grad()
    def decision_function(self, X, chunk=50_000):
        return torch.cat([self._z(X[a:a + chunk]) @ self.W_ + self.b_ for a in range(0, len(X), chunk)])

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1).cpu().numpy()]
