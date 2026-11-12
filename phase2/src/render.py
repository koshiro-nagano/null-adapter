"""Physical operator A for LIB-HSI and the RGB rendering fed to the frozen RGB model (PREREGISTRATION.md).

A (3 x 204) = Nikon D5100 (NPL) SSF x D65 x d-lambda at the Specim IQ band centres, rows normalised to sum to 1,
so c = A x is linear camera RGB on the reflectance scale (white reflector -> (1, 1, 1)).
"""
from __future__ import annotations

import numpy as np

from .libhsi import WAVELENGTHS

CAMERA = "Nikon 5100 (NPL)"
ILLUMINANT = "D65"
GAMMA = 2.2


def _interp(sd_wl, sd_val, lam):
    return np.interp(lam, sd_wl, sd_val, left=0.0, right=0.0)


def build_operator(lam: np.ndarray = WAVELENGTHS) -> np.ndarray:
    import colour

    ssf = colour.characterisation.MSDS_CAMERA_SENSITIVITIES[CAMERA]
    ill = colour.SDS_ILLUMINANTS[ILLUMINANT]
    S = np.stack([_interp(ssf.wavelengths, ssf.values[:, k], lam) for k in range(3)], 1)  # (n, 3), 0 outside 380-780
    E = _interp(ill.wavelengths, ill.values, lam)
    dlam = np.gradient(lam)  # band centres are not evenly spaced
    A = (S * (E * dlam)[:, None]).T
    return A / A.sum(axis=1, keepdims=True)


def projectors(A: np.ndarray):
    """P_R = A^+ A (range of A^T) and P_N = I - P_R."""
    P_R = np.linalg.pinv(A) @ A
    return P_R, np.eye(A.shape[1]) - P_R


def to_display(c: np.ndarray) -> np.ndarray:
    """Linear c (..., 3) -> 8-bit image, clip(c, 0, 1) ** (1 / 2.2)."""
    return np.round(255 * np.clip(c, 0, 1) ** (1 / GAMMA)).astype(np.uint8)
