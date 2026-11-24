"""Neural m(c) (null_p) vs non-neural alternatives (null_orth, null_orthf, null_phys), with fuse_p / rgb as references.
Mean mIoU over the seeds present, differences with a two-level (seed x instance) bootstrap 95 % CI, the label-free
R^2 of each m(c) on the evaluation split, plus the same comparisons restricted to RGB-hard instances (LOSO, IoU < 0.5).

  python -m polar.scripts.nonneural   -> polar/results/nonneural.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polar.scripts.strata import SETTINGS, boot, load_iou

ROOT = Path("polar/results")
METHODS = ("rgb", "fuse_p", "null_p", "null_orth", "null_orthf", "null_phys", "null_fres")
NEW = ("null_orth", "null_orthf", "null_phys", "null_fres")


def main():
    rng = np.random.default_rng(0)
    out = {}
    for ds, n in SETTINGS:
        tag = f"{ds}-{n}"
        have = {m: [s for s in range(10) if (ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").exists()] for m in METHODS}
        res = {"seeds": have, "miou": {}, "r2": {}, "pairs": {}, "hard": {}}
        for m, ss in have.items():
            if ss:
                rs = [json.loads((ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").read_text()) for s in ss]
                res["miou"][m] = [100 * r["mIoU"] for r in rs]
                r2 = [r.get("inputs_info", {}).get("predictor", {}).get("r2_eval") for r in rs]
                if r2[0] is not None:
                    res["r2"][m] = [float(np.mean(v)) for v in r2]
        for a in NEW:
            for b in ("null_p", "fuse_p", "rgb"):
                ss = [s for s in have[a] if s in have[b]]
                if not ss:
                    continue
                A = {(m, s): load_iou(ds, n, m, s) for m in (a, b) for s in ss}
                keys = sorted(A[(b, ss[0])])
                d = [np.array([A[(a, s)][k] - A[(b, s)][k] for k in keys]) for s in ss]
                res["pairs"][f"{a}-{b}"] = boot(d, rng) + [len(ss)]
                # RGB-hard instances, leave-one-seed-out hardness from the rgb runs
                rgb = {s: load_iou(ds, n, "rgb", s) for s in have["rgb"]}
                dh = []
                for i, s in enumerate(ss):
                    h = np.array([np.mean([rgb[t][k] for t in rgb if t != s]) for k in keys])
                    dh.append(d[i][h < .5])
                res["hard"][f"{a}-{b}"] = boot(dh, rng) + [int(np.mean([len(x) for x in dh]))]
        out[tag] = res
        print(f"\n# {tag}")
        for m, v in res["miou"].items():
            r2 = f"  R2 {np.mean(res['r2'][m]):.3f}" if m in res["r2"] else ""
            print(f"  {m:11s} {np.mean(v):6.2f}  seeds {np.round(v, 2).tolist()}{r2}")
        for k, v in res["pairs"].items():
            h = res["hard"][k]
            print(f"  {k:22s} {v[0]:+5.2f} [{v[1]:+5.2f},{v[2]:+5.2f}] ({v[3]} seeds) | RGB-hard {h[0]:+5.1f} [{h[1]:+5.1f},{h[2]:+5.1f}]")
    (ROOT / "nonneural.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
