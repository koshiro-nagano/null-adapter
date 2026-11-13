"""Pack ZJU-RGBP / RGBP-Glass (as distributed with SimCMF) into uint8 caches, see polar/src/data.py.

  python -I -m polar.scripts.prepare zju    polar/data/raw/zju/zju-rgbp
  python -I -m polar.scripts.prepare pgsnet polar/data/raw/pgsnet/pgsnet
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from polar.src.data import CACHE, SPLITS

PGSNET_BROKEN = ("20210808013065", "20210811002205")  # skipped by SimCMF (PGSNet_RGBP)


def items(ds: str, root: Path, split: str):
    """(name, [channel image paths], instance mask path), sorted by name."""
    if ds == "zju":
        for m in sorted((root / split / "label_sam").glob("*.png")):
            yield m.stem, [root / split / a / f"{m.stem}_{a}°.png" for a in ("0", "45", "90", "135")], m
    else:
        for f in sorted((root / split / "image").glob("*_rgb.tiff")):
            stem = f.name[: -len("_rgb.tiff")]
            if any(b in stem for b in PGSNET_BROKEN):
                continue
            yield stem, [root / split / k / f"{stem}_{k if k != 'image' else 'rgb'}.tiff" for k in ("image", "aolp", "dolp")], \
                root / split / "instance_mask" / f"{stem}_instance_mask.png"


def load(it):
    name, chans, mpath = it
    raw = np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in chans], -1).transpose(2, 0, 1)
    m = np.asarray(Image.open(mpath))
    assert m.ndim == 2 and m.shape == raw.shape[1:], (name, m.shape, raw.shape)
    return name, raw, m


def main(ds: str, root: str):
    out = CACHE / ds
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS[ds]:
        its = list(items(ds, Path(root), split))
        name0, r0, m0 = load(its[0])
        raw = np.lib.format.open_memmap(out / f"{split}_raw.npy", "w+", np.uint8, (len(its), *r0.shape))
        mask = np.lib.format.open_memmap(out / f"{split}_mask.npy", "w+", np.uint8, (len(its), *m0.shape))
        with ThreadPoolExecutor(16) as ex:
            names = []
            for i, (name, r, m) in enumerate(ex.map(load, its)):
                raw[i], mask[i] = r, m
                names.append(name)
        raw.flush(); mask.flush()
        (out / f"{split}_names.txt").write_text("\n".join(names) + "\n")
        print(ds, split, raw.shape, mask.shape, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
