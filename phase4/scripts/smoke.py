"""Smoke test before the freeze: cost of the largest heads, on train labels only (no val / test metric).

Usage (from null-adapter/):  python -m phase4.scripts.smoke --device cuda:0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from phase2.src.gpu_head import GPULogReg
from phase2.src.run import sample_patch_features
from phase4.src.labels import NS, labelled
from phase4.src.run_linear import HEAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    dev = ap.parse_args().device
    torch.cuda.set_device(dev)
    lab = labelled(0)
    f = sample_patch_features(np.load("phase2/cache/feat/B_render/train.npy", mmap_mode="r"), lab["l_img"], lab["l_pix"])
    X = np.c_[f, lab["l_x"]].astype(np.float64)  # the 972-d shape of M2 / M5_*
    for N in NS:
        sel = lab["l_rank"] < N
        torch.cuda.reset_peak_memory_stats(dev)
        t = time.time()
        h = GPULogReg(**HEAD, device=dev).fit(X[sel], lab["l_y"][sel])
        print(f"N={N:3d} n={sel.sum():6d} classes={len(h.classes_):2d} iter {h.n_iter_} {time.time() - t:.0f}s "
              f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
