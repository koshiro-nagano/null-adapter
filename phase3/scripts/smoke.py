"""Pipeline smoke test (no metric is computed): component training time, GPU memory, seconds per head step.

Usage (from null-adapter/):  python -m phase3.scripts.smoke --device cuda:0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from phase3.src import run as R
from phase3.src.head import FusionHead, read_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    dev = a.device
    R.E.set_device(dev)
    t0 = time.time()
    tr = R.Split("train", dev)
    comp = R.Components(tr, 0, dev)
    print(f"components {time.time() - t0:.0f}s, final losses", {k: round(v[-1], 4) for k, v in comp.losses.items()}, flush=True)
    for u, (mu, sd) in comp.u_stats.items():
        print(f"u={u}: mean|mu| {mu.abs().mean():.4f} mean sd {sd.mean():.4f} finite {bool(torch.isfinite(mu).all() and torch.isfinite(sd).all())}")
    pool = np.load(R.CACHE2 / "pools" / "seed0.npz")
    sel = pool["l_rank"] < 1000
    li, lp = torch.from_numpy(pool["l_img"][sel]).to(dev), torch.from_numpy(pool["l_pix"][sel]).to(dev)
    ly = torch.from_numpy(pool["l_y"][sel].astype(np.int64)).to(dev)
    imgs = torch.unique(li)
    for method in R.METHODS:
        torch.manual_seed(0)
        head = FusionHead(None if method == "S1" else 768, None if method == "S0" else 204).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
        losses = []
        for step in range(a.steps):
            if step == 5:
                torch.cuda.synchronize(dev); t1 = time.time()
            bs = torch.sort(imgs[torch.randperm(len(imgs))[:16].to(dev)]).values
            m = torch.isin(li, bs)
            f32, u = comp.inputs(method, tr, bs)
            loss = F.cross_entropy(read_at(head(f32, u), torch.searchsorted(bs, li[m]), lp[m]), ly[m])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        torch.cuda.synchronize(dev)
        dt = (time.time() - t1) / (a.steps - 5)
        print(f"{method:8s} {dt * 1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB, "
              f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, finite {all(np.isfinite(losses))}", flush=True)


if __name__ == "__main__":
    main()
