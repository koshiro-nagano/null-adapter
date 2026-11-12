"""Sanity checks for the inferred LIB-HSI layout. Skipped when no data is present.

LIBHSI_ROOT (default phase2/data/partial) must contain label_info.json and train/{reflectance_cubes,rgb,labels}.
"""
import os

import numpy as np
import pytest

from phase2.src import libhsi

ROOT = os.environ.get("LIBHSI_ROOT", "phase2/data/partial")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "train", "reflectance_cubes")), reason="no LIB-HSI data")


SPLITS = [s for s in ("train", "validation", "test") if os.path.isdir(os.path.join(ROOT, s, "reflectance_cubes"))]


@pytest.fixture(scope="module", params=SPLITS)
def scenes(request):
    split = request.param
    ids = libhsi.list_ids(ROOT, split)[:6]
    cls = libhsi.load_classes(ROOT)
    return [(libhsi.load_cube(ROOT, split, s), libhsi.load_rgb(ROOT, split, s), libhsi.load_labels(ROOT, split, s, cls)) for s in ids], cls


def test_alignment_with_rgb(scenes):
    """A green-ish band must correlate strongly with the shipped RGB in the returned orientation only."""
    b = int(np.argmin(abs(libhsi.WAVELENGTHS - 540)))
    for cube, rgb, _ in scenes[0]:
        g = rgb[..., 1].astype(float).ravel()
        r_ok = np.corrcoef(cube[..., b].ravel(), g)[0, 1]
        r_rot = np.corrcoef(np.rot90(cube[..., b]).ravel(), g)[0, 1]
        assert r_ok > 0.5 and r_ok > abs(r_rot) + 0.2


def test_labels_known(scenes):
    for *_, lab in scenes[0]:
        assert (lab >= 0).mean() > 0.99


def test_spectral_landmarks(scenes):
    """Vegetation: red trough < green < NIR plateau, red edge in 680-760 nm; scene mean dips near 940 nm (H2O)."""
    lam = libhsi.WAVELENGTHS
    at = lambda s, w: s[int(np.argmin(abs(lam - w)))]
    veg_id = [c["name"] for c in scenes[1]].index("Vegetation Plant")
    veg = [cube[lab == veg_id].mean(0) for cube, _, lab in scenes[0] if (lab == veg_id).sum() > 1000]
    if veg:
        v = np.mean(veg, 0)
        assert at(v, 670) < at(v, 550) < at(v, 800)
        win = (lam > 650) & (lam < 800)
        assert 680 < lam[win][np.argmax(np.diff(v)[win[:-1]])] < 760
    m = np.mean([c.reshape(-1, libhsi.N_BANDS).mean(0) for c, *_ in scenes[0]], 0)
    win = (lam > 880) & (lam < 1000)
    assert 920 < lam[win][np.argmin(m[win] - np.convolve(m, np.ones(15) / 15, "same")[win])] < 960
