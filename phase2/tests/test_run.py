"""Pixel-level feature sampling equals PyTorch bilinear upsampling of the patch grid."""
import numpy as np
import torch

from phase2.src.run import sample_patch_features, scores, confusion


def test_bilinear_matches_interpolate():
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(2, 32, 32, 5)).astype(np.float16)
    up = torch.nn.functional.interpolate(torch.tensor(grid.astype(np.float32)).permute(0, 3, 1, 2), size=(512, 512),
                                         mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    img = rng.integers(0, 2, 5000)
    pix = np.r_[rng.integers(0, 512 * 512, 4990), [0, 511, 511 * 512, 512 * 512 - 1, 7, 8, 9, 15, 16, 17]]
    np.testing.assert_allclose(sample_patch_features(grid, img, pix), up[img, pix // 512, pix % 512], atol=1e-5)


def test_scores_ignore_unscored_rows():
    y = np.array([0, 0, 1, 1, 2]); p = np.array([0, 1, 1, 1, 2])
    bacc, miou = scores(confusion(y, p), np.array([0, 1, 2]))
    assert np.isclose(bacc, (0.5 + 1 + 1) / 3)
    assert np.isclose(miou, (1 / 2 + 2 / 3 + 1) / 3)
