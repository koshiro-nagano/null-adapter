"""Unit tests (1)-(5) for the non-linear (mixture of Gaussians) prior."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data import TASKS, make_metameric_pairs, make_split


def test_mog_null_projection(mog_world):
    x = mog_world.prior.sample(1000, np.random.default_rng(0))
    assert np.abs(x @ mog_world.proj.P_N.T @ mog_world.A.T).max() < 1e-10


def test_mog_metameric_pairs(mog_world):
    mp = make_metameric_pairs(mog_world, 300, np.random.default_rng(1))
    np.testing.assert_array_equal(mp.c[0::2], mp.c[1::2])
    assert np.abs((mp.x[1::2] - mp.x[0::2]) @ mog_world.A.T).max() < 1e-9
    assert np.all(mp.y["T_resid"][0::2] != mp.y["T_resid"][1::2])


def test_mog_posterior_mean_monte_carlo(mog_world):
    """E[x|c] closed form vs MC: (a) residual orthogonal to non-linear functions of c,
    (b) local-linear nearest-neighbour estimate at fixed c."""
    w = mog_world
    sp = make_split(w, 300000, np.random.default_rng(2))
    R = w.post.residual(sp.x, sp.c)
    cs = (sp.c - sp.c.mean(0)) / sp.c.std(0)
    feats = np.c_[np.ones(len(cs)), cs, cs**2, np.sin(2 * cs), cs[:, :1] * cs[:, 1:2]]
    proj = feats.T @ R / len(R)                       # E[f(c) r] ~ 0 for any f
    se = np.sqrt((feats**2).T @ (R**2) / len(R)) / np.sqrt(len(R))
    assert np.all(np.abs(proj) < 5 * se + 1e-12), np.abs(proj / se).max()
    # a linear regression on c is clearly worse than the closed form (non-linearity is real)
    Cc = np.c_[np.ones(len(sp.c)), sp.c]
    coef, *_ = np.linalg.lstsq(Cc, sp.x, rcond=None)
    assert ((sp.x - Cc @ coef) ** 2).sum() > 1.02 * (R**2).sum()
    for i in range(3):
        c0 = sp.c[i]
        nn = np.argsort(np.linalg.norm(sp.c - c0, axis=1))[:3000]
        Cl = np.c_[np.ones(len(nn)), sp.c[nn] - c0]
        local, *_ = np.linalg.lstsq(Cl, sp.x[nn], rcond=None)
        resid_sd = np.sqrt(np.diag(w.post.cond_cov(c0[None])[0]).max())
        assert np.abs(local[0] - w.post.mean(c0[None])[0]).max() < 6 * resid_sd / np.sqrt(len(nn)) + 0.02


def test_mog_cond_median(mog_world):
    w = mog_world
    rng = np.random.default_rng(3)
    x0 = w.prior.sample(1, rng)
    c0 = w.observe(x0, rng)
    med = w.post.cond_median(w.u, c0)[0]
    # sample x | c0 exactly: component by responsibility, then Gaussian posterior
    wk = w.post.weights(c0)[0]
    ks = rng.choice(len(wk), size=200000, p=wk)
    vals = np.concatenate([rng.multivariate_normal(g.mean(c0)[0], g.Sigma_post, size=(ks == j).sum()) @ w.u
                           for j, g in enumerate(w.post.comps)])
    assert abs((vals > med).mean() - 0.5) < 0.005


def test_mog_resid_label_independent_of_c(mog_world):
    rng = np.random.default_rng(5)
    tr, te = make_split(mog_world, 20000, rng), make_split(mog_world, 20000, rng)
    for clf in [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                make_pipeline(StandardScaler(), PolynomialFeatures(3), StandardScaler(), LogisticRegression(max_iter=5000)),
                make_pipeline(StandardScaler(), KNeighborsClassifier(50))]:
        clf.fit(tr.c, tr.y["T_resid"])
        assert abs(balanced_accuracy_score(te.y["T_resid"], clf.predict(te.c)) - 0.5) < 0.02


def test_mog_task_balance(mog_world):
    sp = make_split(mog_world, 50000, np.random.default_rng(6))
    for t in TASKS:
        assert abs(sp.y[t].mean() - 0.5) < 0.02, (t, sp.y[t].mean())
