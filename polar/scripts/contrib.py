"""How much the polarisation branch actually adds to SAM's patch tokens: per trained fuse_p / null_p model,
||sum_i w_i E_i(x_i)|| / ||E(c)|| over the patch tokens of the first N evaluation images (median over tokens and images).

  python -m polar.scripts.contrib [--n 16]   -> polar/results/contrib.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from polar.src.data import EvalSet
from polar.src.inputs import Inputs
from polar.src.train import EVAL_SPLIT, build

ROOT = Path("polar/results")
SETTINGS = [("zju", "all", 344), ("zju", "20", 20), ("pgsnet", "all", 3205), ("pgsnet", "100", 100)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    for ds, n, nl in SETTINGS:
        ev = EvalSet(ds, EVAL_SPLIT[ds])
        raws = [ev.raw[i] for i in range(a.n)]
        for m in ("fuse_p", "null_p"):
            for s in (0, 1, 2):
                d = ROOT / ds / f"{m}_n{n}_s{s}"
                if not (d / "trainable.pt").exists():
                    continue
                inputs = Inputs(ds, m, s, dev, lambda *_: None)
                net, _ = build(m, ds)
                net.load_state_dict(torch.load(d / "trainable.pt", map_location="cpu"), strict=False)
                net = net.to(dev).eval()
                pj, pe = net.image_encoder.projector, net.image_encoder.patch_embed.proj
                ratios = []
                for r in raws:
                    x = inputs(torch.from_numpy(np.array(r))[None].to(dev))
                    base = pe(x[:, :3])
                    u = x[:, 3:].unsqueeze(2)
                    nov = torch.stack([pj.group_patch_embedding[i](u[:, i]) for i in range(pj.n_extra)], -1)
                    add = pj.fc(nov.permute(0, 2, 3, 1, 4))[..., 0].permute(0, 3, 1, 2)
                    ratios.append((add.norm(dim=1) / base.norm(dim=1)).flatten().cpu())
                v = torch.cat(ratios)
                out[f"{ds}-{nl}-{m}-{s}"] = {"median": float(v.median()), "p90": float(v.quantile(0.9))}
                print(ds, nl, m, s, out[f"{ds}-{nl}-{m}-{s}"])
                del net; torch.cuda.empty_cache()
    (ROOT / "contrib.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
