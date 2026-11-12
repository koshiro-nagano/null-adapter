import numpy as np

from phase4.src.labels import NS, labelled


def test_nested_whole_images():
    a, b = labelled(0), labelled(1)
    g = np.load("phase4/cache/train_grid.npz")
    per_img = np.bincount(g["img"])
    for N in NS:
        sel = a["l_rank"] < N
        imgs = np.unique(a["l_img"][sel])
        assert len(imgs) == N
        assert sel.sum() == per_img[imgs].sum()  # every grid pixel of those images
    for N, M in zip(NS[:-1], NS[1:]):
        assert set(np.unique(a["l_img"][a["l_rank"] < N])) <= set(np.unique(a["l_img"][a["l_rank"] < M]))
    assert not np.array_equal(np.unique(a["l_img"][a["l_rank"] < 5]), np.unique(b["l_img"][b["l_rank"] < 5]))


def test_grid_matches_cube_layout():
    g = np.load("phase4/cache/train_grid.npz")
    assert ((g["pix"] // 512) % 16 == 8).all() and ((g["pix"] % 512) % 16 == 8).all()
    assert g["x"].shape[1] == 204 and (g["y"] >= 0).all()
