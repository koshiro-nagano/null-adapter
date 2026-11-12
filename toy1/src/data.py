"""Synthetic spectra, tasks and metameric pairs (all from a single data seed).

Two priors:
  gaussian : x = mu + B w + eps                      (linear E[x|c], closed form)
  mog      : mixture of such Gaussians (own basis, mean and Lambda per component)
             -> E[x|c] non-linear in c, still closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .operator import GaussianPosterior, MixturePosterior, Projectors, build_cie_operator, projectors, wavelengths

TASKS = ["T_range", "T_null_pred", "T_resid", "T_mixed"]


@dataclass
class SpectralPrior:
    mu: np.ndarray
    B: np.ndarray
    Lam: np.ndarray  # diag entries
    tau: float

    @property
    def Sigma(self) -> np.ndarray:
        return self.B @ np.diag(self.Lam) @ self.B.T + self.tau**2 * np.eye(len(self.mu))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        w = rng.standard_normal((n, len(self.Lam))) * np.sqrt(self.Lam)
        eps = rng.standard_normal((n, len(self.mu))) * self.tau
        return self.mu + w @ self.B.T + eps


@dataclass
class MixturePrior:
    pis: np.ndarray
    comps: list  # of SpectralPrior

    @property
    def mu(self):
        return sum(p * c.mu for p, c in zip(self.pis, self.comps))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = rng.choice(len(self.pis), size=n, p=self.pis)
        x = np.empty((n, len(self.comps[0].mu)))
        for j, comp in enumerate(self.comps):
            idx = np.flatnonzero(k == j)
            x[idx] = comp.sample(len(idx), rng)
        return x


def _gauss_basis(lam, k, s, rng):
    centers = np.linspace(lam[0], lam[-1], k) + rng.uniform(-s["center_jitter"], s["center_jitter"], k)
    widths = rng.uniform(*s["width_range"], k)
    return np.exp(-0.5 * ((lam[:, None] - centers[None]) / widths[None]) ** 2)


def build_prior(cfg: dict, rng: np.random.Generator) -> SpectralPrior:
    s = cfg["spectra"]
    lam = wavelengths(cfg)
    k = s["n_basis"]
    B = _gauss_basis(lam, k, s, rng)
    mu = s["mu_level"] + s["mu_slope"] * (lam - 550.0) / 150.0
    Lam = rng.uniform(*s["lambda_range"], k)
    return SpectralPrior(mu=mu, B=B, Lam=Lam, tau=s["tau"])


def build_mixture_prior(cfg: dict, A: np.ndarray, proj: Projectors, rng: np.random.Generator) -> MixturePrior:
    """Components differ in basis (covariance orientation), Lambda and mean.

    Mean offsets are split into a range part (identifiable from c) and a null part
    (invisible to c), so the component is only partly identifiable from RGB.
    """
    s, m = cfg["spectra"], cfg["spectra"]["mixture"]
    lam = wavelengths(cfg)
    base = s["mu_level"] + s["mu_slope"] * (lam - 550.0) / 150.0
    comps = []
    for _ in range(m["n_components"]):
        B = _gauss_basis(lam, s["n_basis"], s, rng)
        Lam = rng.uniform(*s["lambda_range"], s["n_basis"]) * rng.uniform(*m["lambda_scale_range"])
        smooth = _gauss_basis(lam, 8, s, rng) @ rng.standard_normal(8)
        d_R, d_N = smooth @ proj.P_R.T, smooth @ proj.P_N.T
        mu = (base + m["range_offset"] * d_R / (np.linalg.norm(d_R) + 1e-12)
              + m["null_offset"] * d_N / (np.linalg.norm(d_N) + 1e-12))
        comps.append(SpectralPrior(mu=mu, B=B, Lam=Lam, tau=s["tau"]))
    pis = np.full(m["n_components"], 1.0 / m["n_components"])
    return MixturePrior(pis=pis, comps=comps)


@dataclass
class TaskDef:
    """kind = linear : y = 1[a^T x > t]
              resid  : y = 1[u^T x > median(u^T x | c)]            (P(y=1|c) = 1/2 exactly)
              mixed  : y = 1[a^T x / sd_R + (u^T x - median(u^T x|c)) / sd_r > t]
    For a Gaussian prior median(u^T x | c) = u^T E[x|c], i.e. resid == 1[u^T (x - E[x|c]) > 0].
    """

    name: str
    kind: str
    a: np.ndarray
    t: float = 0.0
    u: np.ndarray | None = None
    sd_R: float = 1.0
    sd_r: float = 1.0

    def score(self, x: np.ndarray, med: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            return x @ self.a
        if self.kind == "resid":
            return x @ self.a - med
        return x @ self.a / self.sd_R + (x @ self.u - med) / self.sd_r

    def label(self, x: np.ndarray, med: np.ndarray) -> np.ndarray:
        return (self.score(x, med) > self.t).astype(int)


@dataclass
class World:
    cfg: dict
    A: np.ndarray
    proj: Projectors
    prior: object
    post: object
    tasks: dict
    info: dict = field(default_factory=dict)

    @property
    def u(self):
        return self.tasks["T_resid"].a

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return x @ self.A.T + self.post.sigma * rng.standard_normal((x.shape[0], self.A.shape[0]))

    def labels(self, x: np.ndarray, c: np.ndarray) -> dict:
        med = self.post.cond_median(self.u, c)
        return {k: t.label(x, med) for k, t in self.tasks.items()}


def _unit(v):
    return v / np.linalg.norm(v)


def _null_pred_direction(N, Sigma, M):
    evals, evecs = eigh(N.T @ M @ N, N.T @ Sigma @ N)
    return _unit(N @ evecs[:, -1])


def build_operator(cfg: dict) -> np.ndarray:
    if cfg["operator"].get("source", "cie") == "cie":
        return build_cie_operator(cfg)
    from .library import build_camera_operator
    return build_camera_operator(cfg)


def build_world(cfg: dict) -> World:
    rng = np.random.default_rng(cfg["data_seed"])
    A = build_operator(cfg)
    proj = projectors(A)
    kind = cfg["spectra"].get("prior", "gaussian")
    if kind == "gaussian":
        return _gaussian_world(cfg, A, proj, rng)
    if kind == "mog":
        return _mixture_world(cfg, A, proj, rng)
    if kind == "library":
        return _library_world(cfg, A, proj, rng)
    raise KeyError(kind)


def _library_world(cfg, A, proj, rng) -> World:
    """Real reflectances x real camera; reference posterior fitted on a calibration sample."""
    from .library import LibraryPosterior, LibraryPrior, load_reflectances

    lc = cfg["spectra"]["library"]
    prior = LibraryPrior(load_reflectances(cfg), list(lc["n_mix"]), tuple(lc["shading"]))
    import torch
    device = "cuda:0" if cfg["run"].get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    post = LibraryPosterior(A, prior, cfg["noise_sigma"], cfg, rng, device)
    N = proj.N_basis
    xs, cs = post.calib
    v_R = _unit(A.T @ rng.standard_normal(A.shape[0]))
    v_N = _null_pred_direction(N, post.Sigma, post.cov_mean)
    u = post.u
    sd_R = float(np.std(xs @ v_R))
    sd_r = float(np.sqrt(u @ post.E_cov @ u))
    med = post.cond_median(u, cs)
    mix_score = xs @ v_R / sd_R + (xs @ u - med) / sd_r
    tasks = {
        "T_range": TaskDef("T_range", "linear", v_R, float(np.median(xs @ v_R))),
        "T_null_pred": TaskDef("T_null_pred", "linear", v_N, float(np.median(xs @ v_N))),
        "T_resid": TaskDef("T_resid", "resid", u, 0.0),
        "T_mixed": TaskDef("T_mixed", "mixed", v_R, float(np.median(mix_score)), u=u, sd_R=sd_R, sd_r=sd_r),
    }
    info = {
        "prior": "library",
        "n_library": int(len(prior.R)),
        "rank_A": int(A.shape[1] - N.shape[1]),
        "r2_T_range": post.r2_of_direction(v_R),
        "r2_T_null_pred": post.r2_of_direction(v_N),
        "r2_T_resid_score_given_c": 0.0,
        "norm_A_vN": float(np.linalg.norm(A @ v_N)),
        "sigma_post_top_eigs": post.E_cov_eigs[:6].tolist(),  # reference E_c[Cov(x|c)]
        "sigma_top_eigs": np.linalg.eigvalsh(post.Sigma)[::-1][:6].tolist(),
        "mix_range_weight_sd": sd_R,
        "mix_resid_weight_sd": sd_r,
        "reference": f"MLP trend + kNN residual correction (k={lc['ref_knn']}) over {lc['n_calib']:,} calibration pixels",
    }
    return World(cfg, A, proj, prior, post, tasks, info)


def _gaussian_world(cfg, A, proj, rng) -> World:
    prior = build_prior(cfg, rng)
    Sigma = prior.Sigma
    post = GaussianPosterior(A, prior.mu, Sigma, cfg["noise_sigma"])
    N = proj.N_basis

    # T_range: v_R in row space of A; v_R^T x_R = v_R^T x.
    v_R = _unit(A.T @ rng.standard_normal(A.shape[0]))
    # T_null_pred: v_N in null(A) maximizing R^2(v^T x | c) (generalized eigenproblem on null basis).
    M = post.gain @ post.S_c @ post.gain.T  # = Cov(E[x|c])
    v_N = _null_pred_direction(N, Sigma, M)
    # T_resid: u = top eigenvector of Sigma_post.
    pe, pv = np.linalg.eigh(post.Sigma_post)
    u = _unit(pv[:, -1])
    sd_R = np.sqrt(v_R @ Sigma @ v_R)
    sd_r = np.sqrt(u @ post.Sigma_post @ u)

    tasks = {
        "T_range": TaskDef("T_range", "linear", v_R, float(v_R @ prior.mu)),
        "T_null_pred": TaskDef("T_null_pred", "linear", v_N, float(v_N @ prior.mu)),
        "T_resid": TaskDef("T_resid", "resid", u, 0.0),
        "T_mixed": TaskDef("T_mixed", "mixed", v_R, float(v_R @ prior.mu / sd_R), u=u, sd_R=sd_R, sd_r=sd_r),
    }
    info = {
        "prior": "gaussian",
        "rank_A": int(A.shape[1] - N.shape[1]),
        "r2_T_range": post.r2_of_direction(v_R),
        "r2_T_null_pred": post.r2_of_direction(v_N),
        "r2_T_resid_score_given_c": 0.0,  # residual is independent of c by construction
        "norm_A_vN": float(np.linalg.norm(A @ v_N)),
        "sigma_post_top_eigs": pe[::-1][:6].tolist(),
        "sigma_top_eigs": np.linalg.eigvalsh(Sigma)[::-1][:6].tolist(),
        "mix_range_weight_sd": float(sd_R),
        "mix_resid_weight_sd": float(sd_r),
    }
    return World(cfg, A, proj, prior, post, tasks, info)


def _mixture_world(cfg, A, proj, rng) -> World:
    prior = build_mixture_prior(cfg, A, proj, rng)
    post = MixturePosterior(A, prior.pis, [c.mu for c in prior.comps], [c.Sigma for c in prior.comps], cfg["noise_sigma"])
    N = proj.N_basis
    Sigma = post.Sigma  # mixture (marginal) covariance

    # Monte-Carlo sample from the data seed for directions and thresholds.
    xs = prior.sample(cfg["spectra"]["mixture"]["n_calib"], rng)
    cs = xs @ A.T + post.sigma * rng.standard_normal((len(xs), A.shape[0]))
    ms = post.mean(cs)
    E_cov = post.cond_cov(cs[:20000]).mean(0)  # E_c[Cov(x|c)]

    v_R = _unit(A.T @ rng.standard_normal(A.shape[0]))
    M = np.cov(ms.T)  # Cov(E[x|c]) (MC)
    v_N = _null_pred_direction(N, Sigma, M)
    pe, pv = np.linalg.eigh(E_cov)
    u = _unit(pv[:, -1])
    sd_R = float(np.std(xs @ v_R))
    sd_r = float(np.sqrt(u @ E_cov @ u))
    med = post.cond_median(u, cs)
    mix_score = xs @ v_R / sd_R + (xs @ u - med) / sd_r

    tasks = {
        "T_range": TaskDef("T_range", "linear", v_R, float(np.median(xs @ v_R))),
        "T_null_pred": TaskDef("T_null_pred", "linear", v_N, float(np.median(xs @ v_N))),
        "T_resid": TaskDef("T_resid", "resid", u, 0.0),
        "T_mixed": TaskDef("T_mixed", "mixed", v_R, float(np.median(mix_score)), u=u, sd_R=sd_R, sd_r=sd_r),
    }
    r2 = lambda v: float(1 - v @ E_cov @ v / (v @ Sigma @ v))
    w = post.weights(cs[:20000])
    info = {
        "prior": "mog",
        "rank_A": int(A.shape[1] - N.shape[1]),
        "r2_T_range": r2(v_R),
        "r2_T_null_pred": r2(v_N),
        "r2_T_resid_score_given_c": 0.0,
        "norm_A_vN": float(np.linalg.norm(A @ v_N)),
        "sigma_post_top_eigs": pe[::-1][:6].tolist(),  # eigenvalues of E_c[Cov(x|c)]
        "sigma_top_eigs": np.linalg.eigvalsh(Sigma)[::-1][:6].tolist(),
        "mix_range_weight_sd": sd_R,
        "mix_resid_weight_sd": sd_r,
        "component_posterior_entropy_mean": float(-(w * np.log(w + 1e-300)).sum(1).mean()),
        "component_posterior_max_mean": float(w.max(1).mean()),
    }
    return World(cfg, A, proj, prior, post, tasks, info)


@dataclass
class Split:
    x: np.ndarray
    c: np.ndarray
    y: dict  # task -> labels


def make_split(world: World, n: int, rng: np.random.Generator) -> Split:
    x = world.prior.sample(n, rng)
    c = world.observe(x, rng)
    return Split(x, c, world.labels(x, c))


def make_metameric_pairs(world: World, n_pairs: int, rng: np.random.Generator, task: str = "T_resid") -> Split:
    """Pairs (x, x') with identical c and different `task` labels (rejection sampling).

    Rows 2i and 2i+1 form a pair. x' is drawn from the prior conditioned on A x' = A x,
    and the observation noise is shared, so c' == c exactly.
    """
    xs, xps, cs = [], [], []
    got = 0
    t = world.tasks[task]
    while got < n_pairs:
        x = world.prior.sample(4 * n_pairs, rng)
        c = world.observe(x, rng)
        xp = world.post.sample_metamer(x, rng)
        med = world.post.cond_median(world.u, c)
        keep = t.label(x, med) != t.label(xp, med)
        xs.append(x[keep]); xps.append(xp[keep]); cs.append(c[keep])
        got += int(keep.sum())
    x = np.concatenate(xs)[:n_pairs]; xp = np.concatenate(xps)[:n_pairs]; c = np.concatenate(cs)[:n_pairs]
    X = np.empty((2 * n_pairs, x.shape[1])); X[0::2] = x; X[1::2] = xp
    C = np.repeat(c, 2, axis=0)
    return Split(X, C, world.labels(X, C))


@dataclass
class Data:
    world: World
    unlabeled: Split
    labeled: Split
    test: Split
    metameric: Split


def build_data(cfg: dict) -> Data:
    world = build_world(cfg)
    rng = np.random.default_rng([cfg["data_seed"], 1])
    s = cfg["sizes"]
    data = Data(
        world=world,
        unlabeled=make_split(world, s["n_unlabeled"], rng),
        labeled=make_split(world, s["n_labeled_pool"], rng),
        test=make_split(world, s["n_test"], rng),
        metameric=make_metameric_pairs(world, s["n_metameric_pairs"], rng),
    )
    if hasattr(world.post, "finalise"):  # cache reference predictions for the splits, drop the calibration bank
        world.post.finalise([data.unlabeled.c, data.labeled.c, data.test.c, data.metameric.c])
    return data


def diagnostics(data: Data) -> dict:
    """Label-free difficulty diagnostics (do not involve any compared method).

    - linear_r2_of_posterior_mean: share of Var(E[x|c]) explained by a linear map of c
      (1.0 for the Gaussian prior; lower = more non-linear).
    - ceiling_*: test balanced accuracy of a large-data (20k) logistic regression on x and on the
      oracle features [c, x - E[x|c]]: how far a *linear* head on raw x can go in principle.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    w, L, T = data.world, data.labeled, data.test
    m = w.post.mean(L.c)
    lin = LinearRegression().fit(L.c, m)
    r2 = 1 - ((m - lin.predict(L.c)) ** 2).sum() / ((m - m.mean(0)) ** 2).sum()
    out = {"linear_r2_of_posterior_mean": float(r2)}
    feats = {"x": (L.x, T.x),
             "oracle": (np.c_[L.c, w.post.residual(L.x, L.c)], np.c_[T.c, w.post.residual(T.x, T.c)])}
    for task in ["T_resid", "T_mixed"]:
        for name, (ftr, fte) in feats.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(ftr, L.y[task])
            out[f"ceiling_{task}_{name}"] = float(balanced_accuracy_score(T.y[task], clf.predict(fte)))
    return out


def sample_k_per_class(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = [rng.choice(np.flatnonzero(y == cls), size=k, replace=False) for cls in (0, 1)]
    return np.concatenate(idx)
