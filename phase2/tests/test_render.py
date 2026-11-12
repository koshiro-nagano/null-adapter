"""Operator A and RGB rendering (no LIB-HSI data needed except the last test)."""
import os

import numpy as np
import pytest

from phase2.src import libhsi, render


@pytest.fixture(scope="module")
def A():
    return render.build_operator()


def test_operator_shape_and_normalisation(A):
    assert A.shape == (3, libhsi.N_BANDS)
    assert np.all(A >= 0)
    np.testing.assert_allclose(A @ np.ones(libhsi.N_BANDS), 1.0, atol=1e-12)


def test_nir_is_invisible(A):
    assert np.all(A[:, libhsi.WAVELENGTHS > 780] == 0)
    assert (libhsi.WAVELENGTHS > 780).mean() > 0.35


def test_channel_order_peaks(A):
    peak = libhsi.WAVELENGTHS[A.argmax(1)]
    assert peak[0] > peak[1] > peak[2]  # R, G, B
    assert 570 < peak[0] < 660 and 500 < peak[1] < 570 and 420 < peak[2] < 490


def test_projectors(A):
    P_R, P_N = render.projectors(A)
    np.testing.assert_allclose(A @ P_N, 0, atol=1e-10)
    np.testing.assert_allclose(P_N @ P_N, P_N, atol=1e-10)
    assert np.linalg.matrix_rank(P_R) == 3


def test_display_encoding():
    np.testing.assert_array_equal(render.to_display(np.array([[-1, 0, 1, 2]] * 3).T),
                                  np.array([[0, 0, 255, 255]] * 3).T)


ROOT = os.environ.get("LIBHSI_ROOT", "phase2/data/partial")


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "train", "reflectance_cubes")), reason="no LIB-HSI data")
def test_rendering_resembles_shipped_rgb(A):
    """Same channel order as the shipped PNG: each rendered channel correlates best with its own PNG channel."""
    for sid in libhsi.list_ids(ROOT, "train")[:3]:
        img = render.to_display(libhsi.load_cube(ROOT, "train", sid) @ A.T).reshape(-1, 3).astype(float)
        png = libhsi.load_rgb(ROOT, "train", sid).reshape(-1, 3).astype(float)
        C = np.corrcoef(img.T, png.T)[:3, 3:]
        assert np.all(np.diag(C) > 0.5)
