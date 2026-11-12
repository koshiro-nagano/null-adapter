import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

from src.data import make_metameric_pairs, make_split, TASKS


def test_A_shape_and_rank(world):
    assert world.A.shape == (3, 31)
    assert world.info["rank_A"] == 3
    assert np.all(world.A >= -1e-12)


# (1) ||A P_N x|| ~ 0
def test_null_projection_annihilated(world):
    rng = np.random.default_rng(0)
    x = world.prior.sample(1000, rng)
    xN = x @ world.proj.P_N.T
    assert np.abs(xN @ world.A.T).max() < 1e-10 * np.abs(x @ world.A.T).max()
    # and x_R alone reproduces A x
    xR = x @ world.proj.P_R.T
    np.testing.assert_allclose(xR @ world.A.T, x @ world.A.T, atol=1e-12)


# (2) P_R + P_N = I, P_R P_N = 0, both idempotent & symmetric
def test_projector_algebra(world):
    P_R, P_N = world.proj.P_R, world.proj.P_N
    n = P_R.shape[0]
    np.testing.assert_allclose(P_R + P_N, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(P_R @ P_N, 0, atol=1e-10)
    np.testing.assert_allclose(P_R @ P_R, P_R, atol=1e-10)
    np.testing.assert_allclose(P_R, P_R.T, atol=1e-10)
    assert abs(np.trace(P_R) - 3) < 1e-8


# (3) metameric pairs share c, differ in T_resid label, x' - x in null(A)
def test_metameric_pairs(world):
    rng = np.random.default_rng(1)
    mp = make_metameric_pairs(world, 300, rng)
    np.testing.assert_array_equal(mp.c[0::2], mp.c[1::2])
    d = mp.x[1::2] - mp.x[0::2]
    assert np.abs(d @ world.A.T).max() < 1e-9
    assert np.all(mp.y["T_resid"][0::2] != mp.y["T_resid"][1::2])
    assert abs(mp.y["T_resid"].mean() - 0.5) < 1e-12


def test_metamer_sampler_is_conditional_prior(world):
    """x' | (A x' = A x) has the noiseless-conditional covariance."""
    rng = np.random.default_rng(2)
    x0 = np.repeat(world.prior.sample(1, rng), 50000, axis=0)
    xp = world.post.sample_metamer(x0, rng)
    S, A = world.post.Sigma, world.A
    G0 = S @ A.T @ np.linalg.inv(A @ S @ A.T)
    C_true = S - G0 @ A @ S
    C_mc = np.cov(xp.T)
    assert np.abs(C_mc - C_true).max() < 0.05 * np.abs(C_true).max()


# (4) closed-form E[x|c] equals Monte-Carlo (OLS of x on c == linear MMSE == E[x|c] for Gaussians)
def test_posterior_mean_matches_monte_carlo(noisy_world):
    w = noisy_world
    rng = np.random.default_rng(3)
    sp = make_split(w, 400000, rng)
    Cc = np.c_[np.ones(len(sp.c)), sp.c]
    coef, *_ = np.linalg.lstsq(Cc, sp.x, rcond=None)
    m_mc = Cc @ coef
    m_cf = w.post.mean(sp.c)
    rel = np.linalg.norm(m_mc - m_cf) / np.linalg.norm(m_cf - w.prior.mu)
    assert rel < 0.01, rel
    # residual covariance equals Sigma_post
    R = sp.x - m_cf
    assert np.abs(np.cov(R.T) - w.post.Sigma_post).max() < 0.02 * np.abs(w.post.Sigma_post).max()
    # local MC at fixed c0: local-linear fit of x on c among nearest neighbours, evaluated at c0
    for i in range(3):
        c0 = sp.c[i]
        nn = np.argsort(np.linalg.norm(sp.c - c0, axis=1))[:4000]
        Cl = np.c_[np.ones(len(nn)), sp.c[nn] - c0]
        local, *_ = np.linalg.lstsq(Cl, sp.x[nn], rcond=None)
        err = np.abs(local[0] - w.post.mean(c0[None])[0]).max()
        assert err < 6 * np.sqrt(np.diag(w.post.Sigma_post).max() / len(nn)), err


def test_posterior_mean_also_on_default_noise(world):
    rng = np.random.default_rng(4)
    sp = make_split(world, 200000, rng)
    Cc = np.c_[np.ones(len(sp.c)), sp.c]
    coef, *_ = np.linalg.lstsq(Cc, sp.x, rcond=None)
    m_cf = world.post.mean(sp.c)
    rel = np.linalg.norm(Cc @ coef - m_cf) / np.linalg.norm(m_cf - world.prior.mu)
    assert rel < 0.01, rel


# (5) T_resid labels are independent of c: classifiers on c are at chance
def test_resid_label_independent_of_c(world):
    rng = np.random.default_rng(5)
    tr = make_split(world, 20000, rng)
    te = make_split(world, 20000, rng)
    clfs = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "poly3": make_pipeline(StandardScaler(), PolynomialFeatures(3), StandardScaler(),
                               LogisticRegression(max_iter=5000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(50)),
    }
    for name, clf in clfs.items():
        clf.fit(tr.c, tr.y["T_resid"])
        acc = balanced_accuracy_score(te.y["T_resid"], clf.predict(te.c))
        assert abs(acc - 0.5) < 0.02, (name, acc)
    # contrast: T_range is solvable from c
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr.c, tr.y["T_range"])
    assert balanced_accuracy_score(te.y["T_range"], clf.predict(te.c)) > 0.95


def test_task_balance_and_directions(world):
    rng = np.random.default_rng(6)
    sp = make_split(world, 50000, rng)
    for t in TASKS:
        assert abs(sp.y[t].mean() - 0.5) < 0.02, (t, sp.y[t].mean())
    assert world.info["norm_A_vN"] < 1e-10
    assert world.info["r2_T_range"] > 0.99
    assert 0.0 < world.info["r2_T_null_pred"] < 1.0
    u = world.tasks["T_resid"].a
    assert world.post.r2_of_direction(u) < world.info["r2_T_null_pred"]
