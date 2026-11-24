"""Label-budget curve: null_orth vs RGB only (and the other methods where present) at every budget.
Two-level (seed x instance) bootstrap CI of null_orth - rgb, overall and on RGB-hard instances (LOSO, IoU < 0.5).

  python -m polar.scripts.fewshot   -> polar/results/fewshot.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polar.scripts.strata import boot, load_iou

ROOT = Path("polar/results")
BUDGETS = {"zju": ("5", "10", "20", "all"), "pgsnet": ("10", "30", "100", "all")}
METHODS = ("rgb", "fuse_p", "null_p", "null_orth", "null_orthf")


def seeds(ds, n, m):
    return [s for s in range(10) if (ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").exists()]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for ds, ns in BUDGETS.items():
        for n in ns:
            tag = f"{ds}-{n}"
            res = {"miou": {}, "seeds": {}}
            for m in METHODS:
                ss = seeds(ds, n, m)
                if ss:
                    res["seeds"][m] = ss
                    res["miou"][m] = [100 * json.loads((ROOT / ds / f"{m}_n{n}_s{s}" / "result.json").read_text())["mIoU"] for s in ss]
            ss = [s for s in res["seeds"].get("null_orth", []) if s in res["seeds"].get("rgb", [])]
            if ss:
                A = {(m, s): load_iou(ds, n, m, s) for m in ("null_orth", "rgb") for s in ss}
                keys = sorted(A[("rgb", ss[0])])
                d = [np.array([A[("null_orth", s)][k] - A[("rgb", s)][k] for k in keys]) for s in ss]
                rgb = {s: load_iou(ds, n, "rgb", s) for s in res["seeds"]["rgb"]}
                dh = []
                for i, s in enumerate(ss):
                    h = np.array([np.mean([rgb[t][k] for t in rgb if t != s]) for k in keys])
                    dh.append(d[i][h < .5])
                res["diff"] = boot(d, rng) + [len(ss)]
                res["hard"] = boot(dh, rng) + [int(np.mean([len(x) for x in dh]))]
                res["per_seed"] = [100 * x.mean() for x in d]
            out[tag] = res
            print(f"# {tag}  " + "  ".join(f"{m} {np.mean(v):.2f}({len(v)})" for m, v in res["miou"].items()))
            if "diff" in res:
                v, h = res["diff"], res["hard"]
                print(f"   null_orth-rgb {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}] ({v[3]} seeds; per seed {np.round(res['per_seed'], 2).tolist()})"
                      f" | RGB-hard {h[0]:+.1f} [{h[1]:+.1f},{h[2]:+.1f}] (n~{h[3]})")
    (ROOT / "fewshot.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
