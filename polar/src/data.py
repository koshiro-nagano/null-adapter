"""ZJU-RGBP and RGBP-Glass (SimCMF's `zju-rgbp` / `pgsnet` folders) as native-resolution uint8 stacks.

Caches (polar/scripts/prepare.py): polar/cache/<ds>/<split>_raw.npy (n, C, H, W) uint8, <split>_mask.npy (n, H, W)
instance ids, <split>_names.txt.
  zju:    C = 12 = intensities at polariser angles 0, 45, 90, 135 deg (RGB each, as stored), H x W = 1024 x 1224
  pgsnet: C = 9  = RGB, AoLP, DoLP (uint8 as stored), H x W = 512 x 612

Features are computed on the GPU at native resolution and then resized to 1024 x 1024 (the order SimCMF uses):
  c      intensity image on a 0..255 scale (ZJU: S0 / 2 = mean over the 4 angles; RGBP-Glass: the RGB file)
  p      normalised linear Stokes s1/S0, s2/S0 per colour = DoLP * (cos 2 AoLP, sin 2 AoLP), 6 channels
  simcmf SimCMF's 9-channel input [I, AoLP, DoLP] with SimCMF's own normalisation (dataset.py, unchanged)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

CACHE = Path("polar/cache")
SIZE, OUT = 1024, 256
SPLITS = {"zju": ("train", "val"), "pgsnet": ("train", "test")}
CNT_THRESHOLD = {"zju": 10000, "pgsnet": 100}   # SimCMF: minimum region size (pixels at 1024 x 1024)
SAM_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
# SimCMF dataset.py, ZJU_RGBP / PGSNet_RGBP
SIMCMF_STATS = {
    "zju": ([141.1251, 141.37535, 138.99559, -0.01452719, -0.014152956, -0.016532706, 0.15173349, 0.1489902, 0.17018148],
            [139.9553, 140.5757, 141.13524, 0.38503787, 0.38248295, 0.3834839, 0.13344945, 0.13081141, 0.14260896]),
    "pgsnet": ([0.20545432, 0.21862635, 0.15604725, 0.47199312, 0.4637443, 0.48603472, 0.102636375, 0.10431647, 0.11925355],
               [0.19210099, 0.20711783, 0.18405455, 0.28202084, 0.28334478, 0.25937688, 0.11682395, 0.118849464, 0.12313089]),
}


def resize(x: torch.Tensor, size: int = SIZE) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


@torch.no_grad()
def features(ds: str, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """raw: (B, C, H, W) uint8 on the GPU -> c (B,3,1024,1024) in 0..255, p (B,6,..), simcmf (B,9,..) normalised."""
    x = raw.float()
    if ds == "zju":
        B, _, H, W = x.shape
        I1, I2, I3, I4 = x.view(B, 4, 3, H, W).unbind(1)
        # SimCMF ZJU_RGBP.raw2rgbp, verbatim
        I = 0.5 * (I1 + I2 + I3 + I4) + 1e-4
        Q, U = I1 - I3, I2 - I4
        Qs = torch.where(Q == 0, torch.full_like(Q, 1e-4), Q)
        I = torch.where(I == 0, torch.full_like(I, 1e-4), I)
        dolp = (torch.sqrt(Qs ** 2 + U ** 2) / I).clamp(max=1)
        aolp = 0.5 * torch.atan(U / Qs)
        sim = torch.cat([I, aolp, dolp], 1)
        c = I / 2
        p = torch.cat([Q / I, U / I], 1)
    elif ds == "pgsnet":
        sim = x / 255  # ToTensor on uint8
        c = x[:, :3]
        theta, rho = x[:, 3:6] / 255 * math.pi, x[:, 6:9] / 255
        p = torch.cat([rho * torch.cos(2 * theta), rho * torch.sin(2 * theta)], 1)
    else:
        raise KeyError(ds)
    mu, sd = (torch.tensor(v, device=x.device).view(1, -1, 1, 1) for v in SIMCMF_STATS[ds])
    return {"c": resize(c), "p": resize(p), "simcmf": (resize(sim) - mu) / sd}


def sam_norm(c: torch.Tensor) -> torch.Tensor:
    return (c - SAM_MEAN.to(c.device)) / SAM_STD.to(c.device)


def load_split(ds: str, split: str):
    d = CACHE / ds
    raw = np.load(d / f"{split}_raw.npy", mmap_mode="r")
    mask = np.load(d / f"{split}_mask.npy", mmap_mode="r")
    names = (d / f"{split}_names.txt").read_text().split()
    assert len(raw) == len(mask) == len(names)
    return raw, mask, names


def mask1024(mask: np.ndarray) -> np.ndarray:
    """SimCMF: PIL NEAREST resize of the instance mask to 1024 x 1024."""
    return np.array(Image.fromarray(np.ascontiguousarray(mask)).resize((SIZE, SIZE), Image.NEAREST))


def regions(m: np.ndarray, thr: int) -> tuple[np.ndarray, np.ndarray]:
    ids, cnt = np.unique(m, return_counts=True)
    keep = (ids > 0) & (cnt >= thr)
    return ids[keep], cnt[keep]


def click(m: np.ndarray, rid: int, rng: np.random.Generator | None) -> np.ndarray:
    """SimCMF utils.random_click: (x, y) of a random pixel of region rid, or the middle one (rng None)."""
    ind = np.argwhere(m == rid)
    i = len(ind) // 2 if rng is None else rng.integers(len(ind))
    return ind[i, ::-1].astype(np.float32).copy()


class TrainSet(Dataset):
    """One random region (>= threshold) of a labelled image and one random click inside it per item."""

    def __init__(self, ds: str, idx: np.ndarray, seed: int):
        self.ds, self.idx, self.seed = ds, np.asarray(idx), seed
        self.raw, self.mask, _ = load_split(ds, "train")
        self.thr = CNT_THRESHOLD[ds]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        k, draw = key  # from DrawSampler: the draw number makes every sample's randomness reproducible
        i = int(self.idx[k])
        rng = np.random.default_rng([self.seed, i, draw])
        m = mask1024(self.mask[i])
        ids, _ = regions(m, self.thr)
        rid = int(rng.choice(ids))
        pt = click(m, rid, rng)
        target = (m[::SIZE // OUT, ::SIZE // OUT] == rid).astype(np.float32)[None]
        return torch.from_numpy(np.array(self.raw[i])), torch.from_numpy(pt)[None], torch.from_numpy(target)


class DrawSampler(torch.utils.data.Sampler):
    """Endless (k, draw) keys: epochs of random permutations of range(n), seeded."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed

    def __iter__(self):
        g, draw = np.random.default_rng([self.seed, 7]), 0
        while True:
            for k in g.permutation(self.n):
                yield int(k), draw
                draw += 1


def eligible(ds: str, split: str) -> np.ndarray:
    """Indices of images with at least one region above the threshold (SimCMF would crash on the others)."""
    _, mask, _ = load_split(ds, split)
    return np.array([i for i in range(len(mask)) if len(regions(mask1024(mask[i]), CNT_THRESHOLD[ds])[0])])


class EvalSet(Dataset):
    """All regions >= threshold of an image, each with its middle click (SimCMF validation_sam)."""

    def __init__(self, ds: str, split: str):
        self.ds = ds
        self.raw, self.mask, self.names = load_split(ds, split)
        self.thr = CNT_THRESHOLD[ds]

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        m = mask1024(self.mask[i])
        ids, cnt = regions(m, self.thr)
        pts = np.stack([click(m, r, None) for r in ids]) if len(ids) else np.zeros((0, 2), np.float32)
        tg = np.stack([(m[::SIZE // OUT, ::SIZE // OUT] == r) for r in ids]).astype(np.float32) if len(ids) \
            else np.zeros((0, OUT, OUT), np.float32)
        return torch.from_numpy(np.array(self.raw[i])), torch.from_numpy(pts), torch.from_numpy(tg), \
            torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(cnt.astype(np.int64)), i
