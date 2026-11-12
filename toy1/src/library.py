"""Real spectra and real cameras.

Reflectances : SFU surface reflectance database (Barnard, Martin, Funt, Coath, Color Res. Appl. 27(3), 2002):
               1995 spectra (24 Macbeth, 1269 Munsell, 120 Dupont, 170 natural objects, 350 Krinov, 57 extra),
               380-780 nm at 4 nm, resampled to the experiment grid.
Cameras      : Nikon D5100 (NPL measurement, shipped with colour-science) and Sony DXC-930 (SFU calibration).
Pixels       : sub-pixel linear mixtures of 1-3 library spectra (Dirichlet weights) times a shading factor.

E[x|c] has no closed form here. A *reference* posterior is estimated once per data seed from a large
calibration sample (1M pixels, disjoint from every split): an MLP trend plus a k-NN residual correction in c. It defines
the T_resid labels and plays the role of M4; it is reported as a reference, not an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .operator import wavelengths


def _read_sfu(path: str) -> np.ndarray:
    vals = np.loadtxt(path)  # comment lines (#) are skipped; one value per line
    return vals.reshape(-1, 101)  # 380:4:780


def load_reflectances(cfg: dict) -> np.ndarray:
    lam = wavelengths(cfg)
    R = _read_sfu(cfg["spectra"]["library"]["path"])
    src = np.arange(380, 781, 4, dtype=float)
    return np.stack([np.interp(lam, src, r) for r in R])


def build_camera_operator(cfg: dict) -> np.ndarray:
    """A (3 x n) = camera SSF (R, G, B) x illuminant x dlambda, optionally row-normalised."""
    import colour

    o, w = cfg["operator"], cfg["wavelengths"]
    lam = wavelengths(cfg)
    shape = colour.SpectralShape(w["start"], w["end"], w["step"])
    illum = colour.SDS_ILLUMINANTS[o["illuminant"]].copy().align(shape).values
    if o["source"] == "colour_camera":
        ssf = colour.characterisation.MSDS_CAMERA_SENSITIVITIES[o["camera"]].copy().align(shape).values  # (n, 3)
    elif o["source"] == "sfu_file":
        raw = np.loadtxt(o["path"]).reshape(3, 101)  # red, green, blue at 380:4:780
        src = np.arange(380, 781, 4, dtype=float)
        ssf = np.stack([np.interp(lam, src, r) for r in raw], 1)
    else:
        raise KeyError(o["source"])
    A = (ssf * illum[:, None] * w["step"]).T
    if o.get("row_normalize", True):
        A = A / A.sum(axis=1, keepdims=True)
    return A


@dataclass
class LibraryPrior:
    R: np.ndarray        # (n_lib, n) reflectances
    n_mix: list          # allowed numbers of mixed spectra
    shading: tuple       # uniform range of the shading factor

    @property
    def mu(self):
        return self.R.mean(0) * float(np.mean(self.shading))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = rng.choice(self.n_mix, size=n)
        x = np.zeros((n, self.R.shape[1]))
        for kk in np.unique(k):
            idx = np.flatnonzero(k == kk)
            sel = rng.integers(0, len(self.R), size=(len(idx), kk))
            wts = rng.dirichlet(np.ones(kk), size=len(idx))
            x[idx] = np.einsum("ij,ijd->id", wts, self.R[sel])
        return x * rng.uniform(*self.shading, size=(n, 1))


def _fit_mlp(X, Y, rc, seed, device):
    """MSE regression MLP (GELU), Adam + cosine schedule; returns a CPU predictor function."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    layers, d = [], X.shape[1]
    for _ in range(rc["depth"]):
        layers += [nn.Linear(d, rc["width"]), nn.GELU()]
        d = rc["width"]
    net = nn.Sequential(*layers, nn.Linear(d, Y.shape[1])).to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=rc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, rc["epochs"])
    for _ in range(rc["epochs"]):
        perm = torch.randperm(len(Xt), generator=gen).to(device)
        for i in range(0, len(Xt), rc["batch"]):
            b = perm[i:i + rc["batch"]]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net = net.cpu().eval()

    @torch.no_grad()
    def predict(Z, bs=200000):
        return np.concatenate([net(torch.as_tensor(Z[i:i + bs], dtype=torch.float32)).numpy()
                               for i in range(0, len(Z), bs)]).astype(np.float64)
    return predict


class LibraryPosterior:
    """Reference posterior from a calibration sample: smooth MLP trend + k-NN residual correction.

    m0(c)             : MLP regression of x on c (trained on the calibration sample),
    E[x|c]            ~ m0(c) + mean of (x_j - m0(c_j)) over the k calibration pixels nearest to c,
    median(u^T x | c) ~ u^T m0(c) + median of u^T (x_j - m0(c_j)) over the same neighbours.
    The k-NN step only has to estimate the weakly c-dependent residual, which keeps its smoothing bias small.
    Predictions for the experiment splits are cached at build time; the calibration sample, the MLP and the
    k-NN index are then dropped, so workers never need them.
    """

    def __init__(self, A, prior: LibraryPrior, sigma, cfg, rng, device="cpu"):
        from sklearn.neighbors import NearestNeighbors

        lc = cfg["spectra"]["library"]
        self.A, self.sigma, self.k = A, sigma, lc["ref_knn"]
        self.A_pinv = np.linalg.pinv(A)
        xs = prior.sample(lc["n_calib"], rng)
        cs = xs @ A.T + sigma * rng.standard_normal((len(xs), A.shape[0]))
        self.c_mu, self.c_sd = cs.mean(0), cs.std(0)
        self.x_mu, self.x_sd = xs.mean(0), xs.std(0)
        f = _fit_mlp((cs - self.c_mu) / self.c_sd, (xs - self.x_mu) / self.x_sd, lc["ref_net"],
                     int(rng.integers(1 << 30)), device)
        self.m0 = lambda c: f((np.asarray(c) - self.c_mu) / self.c_sd) * self.x_sd + self.x_mu
        self.bank_x, self.bank_c = xs, cs
        self.bank_r = xs - self.m0(cs)                 # calibration residuals of the smooth trend
        self.nn = NearestNeighbors(n_neighbors=self.k).fit((cs - self.c_mu) / self.c_sd)
        self.bank_nn = NearestNeighbors(n_neighbors=lc["metamer_knn"]).fit(xs @ A.T)  # noiseless, for metamers
        self._cache = {}
        # moments from an independent estimation sample (not in the bank)
        xe = prior.sample(lc["n_estimate"], rng)
        ce = xe @ A.T + sigma * rng.standard_normal((len(xe), A.shape[0]))
        me = self.mean(ce)
        self.Sigma, self.mu = np.cov(xe.T), xe.mean(0)
        self.E_cov = np.cov((xe - me).T)    # reference E_c[Cov(x|c)]
        self.cov_mean = np.cov(me.T)        # Cov(E[x|c])
        pe, pv = np.linalg.eigh(self.E_cov)
        self.u = pv[:, -1] / np.linalg.norm(pv[:, -1])
        self.E_cov_eigs = pe[::-1]
        self.calib = (xe, ce)  # for task thresholds; dropped after build

    def _key(self, c):
        import hashlib  # stable across processes (built-in hash() of bytes is salted per process)
        return hashlib.sha1(np.ascontiguousarray(c, dtype=np.float64).tobytes()).hexdigest()

    def _neighbours(self, c):
        key = self._key(c)
        if key not in self._cache:
            if self.nn is None:
                raise RuntimeError("reference posterior was finalised; c not in the cached splits")
            _, nb = self.nn.kneighbors((np.asarray(c) - self.c_mu) / self.c_sd)
            m0 = self.m0(c)
            Rn = self.bank_r[nb]                       # (N, k, n) neighbour residuals
            med = m0 @ self.u + np.median(Rn @ self.u, 1) if hasattr(self, "u") else None
            self._cache[key] = (m0 + Rn.mean(1), med, nb if not hasattr(self, "u") else None)
        m, med, nb = self._cache[key]
        if med is None and hasattr(self, "u"):
            med = (m - self.bank_r[nb].mean(1)) @ self.u + np.median(self.bank_r[nb] @ self.u, 1)
            self._cache[key] = (m, med, None)
        return m, med

    def mean(self, c):
        return self._neighbours(c)[0]

    def residual(self, x, c):
        return x - self.mean(c)

    def cond_median(self, v, c):
        assert np.allclose(v, self.u), "reference median is only defined for the T_resid direction"
        return self._neighbours(c)[1]

    def r2_of_direction(self, v):
        return float(1 - v @ self.E_cov @ v / (v @ self.Sigma @ v))

    def sample_metamer(self, x, rng):
        """Approximate draw from p(x' | A x' = A x): a random one of the k nearest calibration pixels
        in noiseless camera space, then an exact range correction so that A x' == A x."""
        c0 = x @ self.A.T
        _, nb = self.bank_nn.kneighbors(c0)
        pick = nb[np.arange(len(x)), rng.integers(0, nb.shape[1], size=len(x))]
        xp = self.bank_x[pick]
        return xp + (c0 - xp @ self.A.T) @ self.A_pinv.T

    def finalise(self, keep_cs):
        """Keep only cached predictions for the given c arrays; drop the calibration sample."""
        keep = {self._key(c) for c in keep_cs}
        for c in keep_cs:
            self._neighbours(c)
        self._cache = {k: v for k, v in self._cache.items() if k in keep}
        self.bank_x = self.bank_c = self.bank_r = self.nn = self.bank_nn = self.calib = self.m0 = None
