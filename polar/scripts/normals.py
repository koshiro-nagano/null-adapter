"""Monocular surface normals of the intensity image c with Metric3D v2 (ViT-Small), cached for null_fres.

  PYTHONPATH=../ext/shim_mmcv:../ext/pydeps_normals:../ext/Metric3D python -m polar.scripts.normals zju   (then pgsnet)
  (../ext/shim_mmcv: a two-function mmcv.utils stand-in; mmcv itself is not needed for inference)
  -> polar/cache/<ds>/<split>_normal.npy  (n, 3, 512, 512) float16, unit camera-frame normals (x right, y down, z forward)

Metric3D: https://github.com/YvanYin/Metric3D (commit eb5b6fa), weights metric_depth_vit_small_800k.pth (HuggingFace
JUGGHM/Metric3D). Input as in its hubconf example: keep the aspect ratio, fit into 616 x 1064, pad with the ImageNet
mean, normalise; the normal head's output is unpadded and resized to the 1024 x 1024 square the features use, then
average-pooled to 512 like m(c).
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch.nn import functional as F

from polar.src.data import CACHE, SAM_MEAN, SAM_STD, SPLITS, features, load_split
from polar.src.inputs import LOW

IN_H, IN_W = 616, 1064


@torch.no_grad()
def main(ds: str):
    import hubconf  # Metric3D's torch.hub entry (on PYTHONPATH)
    dev = torch.device("cuda")
    model = hubconf.metric3d_vit_small(pretrain=True).to(dev).eval()
    mean, std = SAM_MEAN.to(dev), SAM_STD.to(dev)  # the same ImageNet statistics Metric3D uses
    for split in SPLITS[ds]:
        raw, _, _ = load_split(ds, split)
        out = np.lib.format.open_memmap(CACHE / ds / f"{split}_normal.npy", "w+", np.float16, (len(raw), 3, LOW, LOW))
        for i in range(len(raw)):
            x = torch.from_numpy(np.asarray(raw[i:i + 1])).to(dev)
            H, W = x.shape[-2:]
            # c at native resolution (features() resizes to 1024^2; undo that by taking c before the resize)
            c = features(ds, x)["c"]                                                   # 1,3,1024,1024
            c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
            s = min(IN_H / H, IN_W / W)
            h, w = round(H * s), round(W * s)
            img = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
            ph, pw = IN_H - h, IN_W - w
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            img = F.pad((img - mean) / std, pad, value=0.0)                            # pad = mean after normalising
            _, _, o = model.inference({"input": img})
            n = o["prediction_normal"][:, :3]
            n = n[..., pad[2]:IN_H - pad[3], pad[0]:IN_W - pad[1]]
            n = F.interpolate(n, size=(1024, 1024), mode="bilinear", align_corners=False)
            n = F.avg_pool2d(n, 1024 // LOW)
            n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-6)
            out[i] = n[0].cpu().numpy().astype(np.float16)
            if i % 200 == 0:
                print(ds, split, i, len(raw), flush=True)
        out.flush()
        print(ds, split, "done", out.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
