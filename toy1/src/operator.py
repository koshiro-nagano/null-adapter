"""Linear sensing operator A, range/null projectors and Gaussian posterior.

Everything here is generic in A (m x n), so Phase 2 can swap in a camera
spectral-sensitivity matrix without touching the rest of the code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wavelengths(cfg: dict) -> np.ndarray:
    w = cfg["wavelengths"]
    return np.arange(w["start"], w["end"] + 1e-9, w["step"], dtype=float)


def build_cie_operator(cfg: dict) -> np.ndarray:
    """A (3 x n) = CMF(lambda) * D65(lambda) * dlambda, optionally row-normalized."""
    import colour

    w = cfg["wavelengths"]
    shape = colour.SpectralShape(w["start"], w["end"], w["step"])
    cmfs = colour.MSDS_CMFS[cfg["operator"]["cmfs"]].copy().align(shape)
    illum = colour.SDS_ILLUMINANTS[cfg["operator"]["illuminant"]].copy().align(shape)
    A = (cmfs.values * illum.values[:, None] * w["step"]).T  # (3, n)
    if cfg["operator"].get("row_normalize", True):
        A = A / A.sum(axis=1, keepdims=True)
    return A


@dataclass
class Projectors:
    A_pinv: np.ndarray  # (n, m)
    P_R: np.ndarray     # (n, n) projector onto range(A^T) = row space
    P_N: np.ndarray     # (n, n) projector onto null(A)
    N_basis: np.ndarray  # (n, n - rank) orthonormal basis of null(A)


def projectors(A: np.ndarray) -> Projectors:
    A_pinv = np.linalg.pinv(A)
    P_R = A_pinv @ A
    n = A.shape[1]
    P_N = np.eye(n) - P_R
    _, s, Vt = np.linalg.svd(A)
    rank = int((s > s.max() * 1e-10).sum())
    return Projectors(A_pinv, P_R, P_N, Vt[rank:].T)


@dataclass
class GaussianPosterior:
    """x ~ N(mu, Sigma), c = A x + eps, eps ~ N(0, sigma^2 I)."""

    A: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray
    sigma: float

    def __post_init__(self):
        m = self.A.shape[0]
        S_c = self.A @ self.Sigma @ self.A.T + self.sigma**2 * np.eye(m)
        self.gain = self.Sigma @ self.A.T @ np.linalg.inv(S_c)  # (n, m)
        self.Sigma_post = self.Sigma - self.gain @ self.A @ self.Sigma
        self.Sigma_post = 0.5 * (self.Sigma_post + self.Sigma_post.T)
        self.S_c = S_c

    def mean(self, c: np.ndarray) -> np.ndarray:
        """E[x | c] for rows of c (N, m) -> (N, n)."""
        return self.mu + (c - self.A @ self.mu) @ self.gain.T

    def residual(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return x - self.mean(c)

    def cond_median(self, v: np.ndarray, c: np.ndarray) -> np.ndarray:
        """median(v^T x | c); equals v^T E[x|c] for a Gaussian posterior."""
        return self.mean(c) @ v

    def r2_of_direction(self, v: np.ndarray) -> float:
        """Fraction of Var(v^T x) explained by c."""
        v = np.asarray(v, float)
        return float(1.0 - v @ self.Sigma_post @ v / (v @ self.Sigma @ v))

    def sample_metamer(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """x' ~ p(x' | A x' = A x) under the prior. Exact metamer: A x' == A x."""
        z = rng.multivariate_normal(self.mu, self.Sigma, size=x.shape[0])
        G0 = self.Sigma @ self.A.T @ np.linalg.inv(self.A @ self.Sigma @ self.A.T)
        return z - (z - x) @ (G0 @ self.A).T


def _gauss_logpdf(r: np.ndarray, S: np.ndarray) -> np.ndarray:
    """log N(r; 0, S) for rows of r."""
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, r.T)
    return -0.5 * (z**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * len(S) * np.log(2 * np.pi)


class MixturePosterior:
    """x ~ sum_k pi_k N(mu_k, Sigma_k), c = A x + eps.  E[x|c] is non-linear in c but closed form.

    p(k | c) ∝ pi_k N(c; A mu_k, A Sigma_k A^T + sigma^2 I);  E[x | c, k] is the per-component
    Gaussian posterior mean, Cov(x | c, k) = Sigma_post_k does not depend on c.
    """

    def __init__(self, A, pis, mus, Sigmas, sigma):
        self.A, self.sigma = A, sigma
        self.pis = np.asarray(pis, float)
        self.comps = [GaussianPosterior(A, m, S, sigma) for m, S in zip(mus, Sigmas)]
        self.mu = sum(p * g.mu for p, g in zip(self.pis, self.comps))
        self.Sigma = sum(p * (g.Sigma + np.outer(g.mu - self.mu, g.mu - self.mu)) for p, g in zip(self.pis, self.comps))

    def weights(self, c: np.ndarray, noiseless: bool = False) -> np.ndarray:
        lw = []
        for p, g in zip(self.pis, self.comps):
            S = self.A @ g.Sigma @ self.A.T + (0 if noiseless else self.sigma**2) * np.eye(len(self.A))
            lw.append(np.log(p) + _gauss_logpdf(c - self.A @ g.mu, S))
        lw = np.stack(lw, 1)
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        return w / w.sum(1, keepdims=True)

    def comp_means(self, c):
        return np.stack([g.mean(c) for g in self.comps], 1)  # (N, K, n)

    def mean(self, c: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nkd->nd", self.weights(c), self.comp_means(c))

    def residual(self, x, c):
        return x - self.mean(c)

    def cond_cov(self, c: np.ndarray) -> np.ndarray:
        """Cov(x | c) for rows of c -> (N, n, n)."""
        w, M = self.weights(c), self.comp_means(c)
        m = np.einsum("nk,nkd->nd", w, M)
        P = np.stack([g.Sigma_post for g in self.comps])
        second = np.einsum("nk,kde->nde", w, P) + np.einsum("nk,nkd,nke->nde", w, M, M)
        return second - np.einsum("nd,ne->nde", m, m)

    def cond_median(self, v: np.ndarray, c: np.ndarray, iters: int = 60) -> np.ndarray:
        """median(v^T x | c): 1-D Gaussian mixture, solved by vectorized bisection."""
        from scipy.special import ndtr
        w = self.weights(c)
        loc = self.comp_means(c) @ v                                   # (N, K)
        sd = np.sqrt([v @ g.Sigma_post @ v for g in self.comps])[None]  # (1, K)
        lo, hi = (loc - 8 * sd).min(1), (loc + 8 * sd).max(1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F = (w * ndtr((mid[:, None] - loc) / sd)).sum(1)
            lo, hi = np.where(F < 0.5, mid, lo), np.where(F < 0.5, hi, mid)
        return 0.5 * (lo + hi)

    def sample_metamer(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """x' ~ p(x' | A x' = A x): pick component by noiseless evidence, then condition."""
        w = self.weights(x @ self.A.T, noiseless=True)
        k = (rng.random((len(x), 1)) > np.cumsum(w, 1)).sum(1)
        out = np.empty_like(x)
        for j in np.unique(k):
            idx = np.flatnonzero(k == j)
            out[idx] = self.comps[j].sample_metamer(x[idx], rng)
        return out
