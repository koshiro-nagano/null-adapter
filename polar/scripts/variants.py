"""null_orth improvements (null_orth2: + token-level orthogonalisation; null_orth_g: + RGB-token gate and modality
dropout; null_orth2_g: both) vs null_orth and RGB only, on the seeds shared by each pair. Two-level bootstrap CI,
overall and on RGB-hard instances (LOSO, IoU < 0.5). Also the mean learned gate is not stored, so only accuracy here.

  python -m polar.scripts.variants   -> polar/results/variants.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polar.scripts.strata import boot, load_iou

ROOT = Path("polar/results")
SETS = [("zju", "20"), ("pgsnet", "100")]
VARS = ("null_orth2", "null_orth_g", "null_orth2_g")
REFS = ("null_orth", "rgb")


def seeds(ds, n, m):
    return [s for s in range(10) if (ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").exists()]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for ds, n in SETS:
        tag = f"{ds}-{n}"
        res = {"miou": {}, "pairs": {}}
        for m in REFS + VARS:
            ss = seeds(ds, n, m)
            res["miou"][m] = {s: 100 * json.loads((ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").read_text())["mIoU"] for s in ss}
        rgb_all = {s: load_iou(ds, n, "rgb", s) for s in res["miou"]["rgb"]}
        print(f"# {tag}")
        for m, v in res["miou"].items():
            v3 = [v[s] for s in (0, 1, 2) if s in v]
            print(f"  {m:13s} seeds0-2 mean {np.mean(v3):6.2f}  {np.round(v3, 2).tolist()}")
        for a in VARS:
            for b in REFS:
                ss = [s for s in (0, 1, 2) if s in res["miou"][a] and s in res["miou"][b]]
                A = {(m, s): load_iou(ds, n, m, s) for m in (a, b) for s in ss}
                keys = sorted(A[(b, ss[0])])
                d = [np.array([A[(a, s)][k] - A[(b, s)][k] for k in keys]) for s in ss]
                dh = []
                for i, s in enumerate(ss):
                    h = np.array([np.mean([rgb_all[t][k] for t in rgb_all if t != s]) for k in keys])
                    dh.append(d[i][h < .5])
                v, h = boot(d, rng), boot(dh, rng)
                res["pairs"][f"{a}-{b}"] = {"all": v, "hard": h, "per_seed": [100 * x.mean() for x in d]}
                print(f"  {a}-{b:9s} {v[0]:+5.2f} [{v[1]:+5.2f},{v[2]:+5.2f}] per seed {np.round(res['pairs'][f'{a}-{b}']['per_seed'], 2).tolist()}"
                      f" | RGB-hard {h[0]:+5.1f} [{h[1]:+5.1f},{h[2]:+5.1f}]")
        res["miou"] = {m: {str(s): x for s, x in v.items()} for m, v in res["miou"].items()}
        out[tag] = res
    (ROOT / "variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
