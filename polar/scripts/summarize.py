"""Tables from polar/results/<ds>/<method>_n<labels>_s<seed>/{result.json, instances.csv}.

  python -m polar.scripts.summarize [--root polar/results] [--out polar/results/summary.json]

Per (dataset, label budget, method): final-checkpoint instance mIoU (SimCMF's metric) as mean +- sd over seeds,
the SimCMF-style peak over the logged evaluations, and paired differences against the other methods: per instance,
IoU averaged over the seeds both methods have, then a bootstrap over instances (95% CI).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["simcmf", "rgb", "fuse_p", "null_p"]
PAIRS = [("null_p", "simcmf"), ("null_p", "fuse_p"), ("null_p", "rgb"), ("fuse_p", "rgb"), ("simcmf", "rgb")]


def load(root: Path):
    runs = defaultdict(dict)  # (ds, n) -> method -> seed -> (result, iou array keyed by (image, region))
    for f in sorted(root.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        with open(f.parent / "instances.csv") as fh:
            rows = list(csv.DictReader(fh))
        iou = {(int(x["image"]), int(x["region"])): float(x["iou"]) for x in rows}
        runs[(r["ds"], r["n_label"])].setdefault(r["method"], {})[r["seed"]] = (r, iou)
    return runs


def paired(a: dict, b: dict, n_boot=10000, rng=np.random.default_rng(0)):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    keys = sorted(set.intersection(*[set(a[s][1]) & set(b[s][1]) for s in seeds]))
    d = np.array([np.mean([a[s][1][k] - b[s][1][k] for s in seeds]) for k in keys])
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    per_seed = [float(np.mean([a[s][1][k] for k in keys]) - np.mean([b[s][1][k] for k in keys])) for s in seeds]
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_seed": per_seed, "seeds": seeds, "n_inst": len(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="polar/results")
    ap.add_argument("--out", default="polar/results/summary.json")
    a = ap.parse_args()
    runs = load(Path(a.root))
    summary = []
    for (ds, n) in sorted(runs, key=lambda k: (k[0], -k[1])):
        g = runs[(ds, n)]
        print(f"\n== {ds}, {n} labelled images ==")
        print(f"{'method':8s} {'seeds':>5s} {'mIoU (final)':>16s} {'peak':>14s} {'trainable':>10s}")
        rows = {}
        for m in [m for m in ORDER if m in g]:
            fin = np.array([v[0]["mIoU"] for v in g[m].values()]) * 100
            pk = np.array([v[0]["peak_mIoU"] for v in g[m].values()]) * 100
            tr = next(iter(g[m].values()))[0]["trainable"]
            rows[m] = {"seeds": sorted(g[m]), "mIoU": fin.tolist(), "peak": pk.tolist(), "trainable": tr}
            sd = lambda x: f"{x.std(ddof=1):.2f}" if len(x) > 1 else "  - "
            print(f"{m:8s} {len(fin):5d} {fin.mean():9.2f} +- {sd(fin)} {pk.mean():7.2f} +- {sd(pk)} {tr / 1e6:9.2f}M")
        diffs = {}
        for x, y in PAIRS:
            if x in g and y in g:
                p = paired(g[x], g[y])
                if p:
                    diffs[f"{x}-{y}"] = p
                    print(f"  {x:7s} - {y:7s}: {100 * p['diff']:+6.2f}  CI95 [{100 * p['ci95'][0]:+.2f}, {100 * p['ci95'][1]:+.2f}]"
                          f"  per seed {[round(100 * v, 2) for v in p['per_seed']]}  ({p['n_inst']} instances)")
        info = {m: next(iter(g[m].values()))[0].get("inputs_info", {}) for m in g}
        summary.append({"ds": ds, "n_label": n, "methods": rows, "paired": diffs, "inputs_info": info})
    Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
