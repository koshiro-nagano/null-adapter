"""Unit tests (1), (3), (5) and reference-posterior checks for the real-spectra world (small calibration)."""
import copy

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data import TASKS, build_world, make_metameric_pairs, make_split


@pytest.fixture(scope="module", params=["real_nikon.yaml", "real_sony.yaml"])
def lib_world(request):
    cfg = load_config(f"configs/{request.param}")
    cfg = copy.deepcopy(cfg)
    cfg["spectra"]["library"]["n_calib"] = 300000
    cfg["spectra"]["library"]["n_estimate"] = 50000
    cfg["spectra"]["library"]["ref_net"].update(epochs=8)
    return build_world(cfg)


def test_library_data_and_operator(lib_world):
    w = lib_world
    assert w.prior.R.shape == (1995, 31)
    assert w.prior.R.min() >= 0 and w.prior.R.max() <= 1.0
    assert w.A.shape == (3, 31) and np.all(w.A >= -1e-9)
    np.testing.assert_allclose(w.A.sum(1), 1, atol=1e-9)
    x = w.prior.sample(1000, np.random.default_rng(0))
    assert x.min() >= 0
    assert np.abs(x @ w.proj.P_N.T @ w.A.T).max() < 1e-10


def test_library_metamers(lib_world):
    mp = make_metameric_pairs(lib_world, 200, np.random.default_rng(1))
    np.testing.assert_array_equal(mp.c[0::2], mp.c[1::2])
    assert np.abs((mp.x[1::2] - mp.x[0::2]) @ lib_world.A.T).max() < 1e-9
    assert np.all(mp.y["T_resid"][0::2] != mp.y["T_resid"][1::2])


def test_library_reference_mean(lib_world):
    """Reference E[x|c]: residual nearly orthogonal to non-linear functions of c, and better than linear."""
    w = lib_world
    sp = make_split(w, 100000, np.random.default_rng(2))
    R = w.post.residual(sp.x, sp.c)
    cs = (sp.c - sp.c.mean(0)) / sp.c.std(0)
    feats = np.c_[cs, cs**2, np.sin(2 * cs)]
    corr = np.abs(np.corrcoef(np.c_[feats, R].T)[: feats.shape[1], feats.shape[1]:])
    assert corr.max() < 0.05, corr.max()
    Cc = np.c_[np.ones(len(sp.c)), sp.c]
    coef, *_ = np.linalg.lstsq(Cc, sp.x, rcond=None)
    assert ((sp.x - Cc @ coef) ** 2).sum() > (R**2).sum()


def test_library_resid_label_nearly_independent_of_c(lib_world):
    rng = np.random.default_rng(5)
    tr, te = make_split(lib_world, 20000, rng), make_split(lib_world, 20000, rng)
    assert abs(te.y["T_resid"].mean() - 0.5) < 0.03
    for clf in [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                make_pipeline(StandardScaler(), KNeighborsClassifier(50))]:
        clf.fit(tr.c, tr.y["T_resid"])
        assert abs(balanced_accuracy_score(te.y["T_resid"], clf.predict(te.c)) - 0.5) < 0.03


def test_library_task_balance(lib_world):
    sp = make_split(lib_world, 50000, np.random.default_rng(6))
    for t in TASKS:
        assert abs(sp.y[t].mean() - 0.5) < 0.03, (t, sp.y[t].mean())
