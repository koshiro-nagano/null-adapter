"""Frozen DINOv3 patch features for every LIB-HSI image (PREREGISTRATION.md, "Frozen RGB foundation model").

Usage (from null-adapter/):  python -I phase2/scripts/features.py <root> <cache> <model: S|B|L> <source: render|png>
  source=render: <cache>/render/<split>/<id>.png (our A x rendering); source=png: the RGB PNG shipped with LIB-HSI.
Writes <cache>/feat/<model>_<source>/<split>.npy, float16 (n_img, 32, 32, D): last-layer patch tokens at 512 x 512.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {"S": "facebook/dinov3-vits16-pretrain-lvd1689m", "B": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "L": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
SIZE = 512
BATCH = 8


@torch.no_grad()
def main(root, cache, model_key, source):
    from transformers import AutoImageProcessor, AutoModel

    root, cache = Path(root), Path(cache)
    ids = json.load(open(cache / "ids.json"))
    name = MODELS[model_key]
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to("cuda").eval()
    n_skip = 1 + model.config.num_register_tokens
    g = SIZE // model.config.patch_size
    out_dir = cache / "feat" / f"{model_key}_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, sids in ids.items():
        paths = [(cache / "render" / split / f"{s}.png") if source == "render" else (root / split / "rgb" / f"{s}.png")
                 for s in sids]
        feats = []
        for i in range(0, len(paths), BATCH):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
            assert all(im.size == (SIZE, SIZE) for im in imgs)
            inp = proc(images=imgs, size={"height": SIZE, "width": SIZE}, return_tensors="pt").to("cuda")
            assert inp["pixel_values"].shape[-2:] == (SIZE, SIZE)
            h = model(**inp).last_hidden_state[:, n_skip:]
            assert h.shape[1] == g * g
            feats.append(h.reshape(len(imgs), g, g, -1).half().cpu().numpy())
        np.save(out_dir / f"{split}.npy", np.concatenate(feats))
        print(model_key, source, split, np.concatenate(feats).shape, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
