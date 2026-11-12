"""GPULogReg reaches the sklearn pipeline's optimum on a small synthetic problem."""
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase2.src.gpu_head import GPULogReg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 20)) * rng.uniform(0.1, 10, 20) + rng.normal(size=20)
    X[:, 3] = 2.0  # constant column: StandardScaler leaves it unscaled
    y = (X[:, :5] @ rng.normal(size=(5, 4)) + rng.normal(size=(600, 4))).argmax(1) + 7  # labels 7..10
    s = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, tol=1e-8)).fit(X, y)
    g = GPULogReg(C=1.0, max_iter=5000, tol=1e-8).fit(X, y)
    np.testing.assert_allclose(g.W_.T.cpu().numpy(), s[-1].coef_, atol=1e-5)
    np.testing.assert_allclose(g.b_.cpu().numpy(), s[-1].intercept_, atol=1e-5)
    assert (g.predict(X) == s.predict(X)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_col_weight_scales_standardised_columns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)); y = rng.integers(0, 3, 300)
    w = np.array([1, 1, 1, 0.5, 0.5, 2.0])
    a = GPULogReg(tol=1e-10, col_weight=w).fit(X, y)
    Z = (X - X.mean(0)) / X.std(0) * w
    za = a._z(X).cpu().numpy()
    np.testing.assert_allclose(za, Z, atol=1e-10)
