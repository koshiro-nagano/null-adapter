import numpy as np
import torch

from src.data import make_split
from src.encoders import RNDReconstructor, Standardizer, apply


def test_rnd_reconstruction_is_range_consistent(world):
    """RND-SCI property: A x_hat = A A^+ c for any generator output (data consistency)."""
    rng = np.random.default_rng(0)
    sp = make_split(world, 2000, rng)
    sc, sx = Standardizer.fit(sp.c), Standardizer.fit(sp.x)
    torch.manual_seed(0)
    rnd = RNDReconstructor(world.proj.A_pinv, world.proj.P_N, sc, sx, 32, 2)  # untrained
    xh = sx.inv(apply(rnd, sc(sp.c)))
    np.testing.assert_allclose(xh @ world.A.T, sp.c, atol=1e-4)
