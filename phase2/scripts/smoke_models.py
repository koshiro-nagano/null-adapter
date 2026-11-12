"""Smoke test: DINOv3 (ViT-S/B/L) and SAM 3 load and run on GPU via transformers.

Usage (from null-adapter/):  VARIANT=phase2 ./run_docker.sh python phase2/scripts/smoke_models.py
Test images are scikit-learn's bundled samples (china.jpg, flower.jpg) - no download needed.
Writes phase2/results/smoke/{smoke.json, dinov3_pca.png, sam3_masks.png}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_sample_image

OUT = Path("phase2/results/smoke")
DINO = ["facebook/dinov3-vits16-pretrain-lvd1689m", "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "facebook/dinov3-vitl16-pretrain-lvd1689m"]
SAM = "facebook/sam3"
PROMPTS = {"flower": ["flower", "bee"], "china": ["building", "tree"]}


def dinov3(name, images, device):
    from transformers import AutoImageProcessor, AutoModel
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    load = time.time() - t0
    inp = proc(images=list(images.values()), return_tensors="pt").to(device)
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        out = model(**inp)
        torch.cuda.synchronize(); fwd = time.time() - t0
    h = out.last_hidden_state                      # (B, 1 + n_register + n_patch, D)
    n_reg = model.config.num_register_tokens
    patch = h[:, 1 + n_reg:]
    H, W = inp["pixel_values"].shape[-2:]
    ps = model.config.patch_size
    grid = (H // ps, W // ps)
    assert patch.shape[1] == grid[0] * grid[1], (patch.shape, grid)
    info = dict(model=name, params_M=sum(p.numel() for p in model.parameters()) / 1e6, input=list(inp["pixel_values"].shape),
                hidden=list(h.shape), n_register=n_reg, patch_grid=list(grid), pooler=list(out.pooler_output.shape),
                finite=bool(torch.isfinite(h).all()), load_s=round(load, 1), forward_s=round(fwd, 3),
                gpu_mem_GB=round(torch.cuda.max_memory_allocated(device) / 1e9, 2))
    pca = []
    for b in range(len(images)):  # 3-component PCA of patch features -> RGB (standard DINO sanity picture)
        f = patch[b].float().cpu().numpy(); f = f - f.mean(0)
        _, _, vt = np.linalg.svd(f, full_matrices=False)
        p = f @ vt[:3].T; p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-8)
        pca.append(p.reshape(*grid, 3))
    del model; torch.cuda.empty_cache()
    return info, pca


def sam3(images, device):
    from transformers import Sam3Model, Sam3Processor
    t0 = time.time()
    proc = Sam3Processor.from_pretrained(SAM)
    model = Sam3Model.from_pretrained(SAM).to(device).eval()
    load = time.time() - t0
    info, masks = dict(model=SAM, params_M=sum(p.numel() for p in model.parameters()) / 1e6, load_s=round(load, 1), runs=[]), {}
    for key, img in images.items():
        for text in PROMPTS[key]:
            inp = proc(images=img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.time()
                out = model(**inp)
                torch.cuda.synchronize(); fwd = time.time() - t0
            r = proc.post_process_instance_segmentation(out, threshold=0.5, mask_threshold=0.5,
                                                        target_sizes=inp["original_sizes"].tolist())[0]
            m = r["masks"].cpu().numpy().astype(bool)
            info["runs"].append(dict(image=key, text=text, n_instances=int(len(m)), scores=[round(float(s), 3) for s in r["scores"]],
                                     mask_shape=list(m.shape[1:]) if len(m) else None,
                                     area_frac=[round(float(x.mean()), 4) for x in m], forward_s=round(fwd, 3)))
            masks[(key, text)] = m
    info["gpu_mem_GB"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    del model; torch.cuda.empty_cache()
    return info, masks


def main():
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    images = {k: Image.fromarray(load_sample_image(f"{k}.jpg")) for k in ["china", "flower"]}
    res = dict(env=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                        transformers=__import__("transformers").__version__), dinov3=[], sam3=None)
    pcas = {}
    for name in DINO:
        torch.cuda.reset_peak_memory_stats(device)
        info, pca = dinov3(name, images, device); res["dinov3"].append(info); pcas[name] = pca
        print(json.dumps(info))
    torch.cuda.reset_peak_memory_stats(device)
    res["sam3"], masks = sam3(images, device)
    print(json.dumps(res["sam3"]))
    (OUT / "smoke.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(2, 1 + len(DINO), figsize=(4 * (1 + len(DINO)), 6))
    for i, (k, img) in enumerate(images.items()):
        ax[i, 0].imshow(img); ax[i, 0].set_title(k)
        for j, name in enumerate(DINO):
            ax[i, j + 1].imshow(pcas[name][i]); ax[i, j + 1].set_title(name.split("/")[1].split("-")[1] + " patch PCA")
    for a in ax.flat: a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "dinov3_pca.png", dpi=80); plt.close(fig)

    fig, ax = plt.subplots(1, len(masks), figsize=(4 * len(masks), 3.4))
    for a, ((k, text), m) in zip(ax, masks.items()):
        a.imshow(images[k])
        if len(m):
            ov = np.zeros((*m.shape[1:], 4)); cols = plt.cm.tab10(np.arange(len(m)) % 10)
            for c, mm in zip(cols, m): ov[mm] = [*c[:3], 0.55]
            a.imshow(ov)
        a.set_title(f'{k}: "{text}" ({len(m)})'); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "sam3_masks.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
