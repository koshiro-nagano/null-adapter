"""Phase 4, part B (PREREGISTRATION.md): the Phase 3 methods (shared spatial head, src of phase3) with
whole-image labels. Components, head, optimiser and batches are the Phase 3 ones; N replaces K everywhere
(head seed 1000 * seed + N, batches of 16 of the N labelled images).

Usage (from null-adapter/):
  python -m phase4.src.run_spatial --mode select --seeds 0 2 --device cuda:0 --out phase4/results/spatial_select
  python -m phase4.src.run_spatial --mode select --choose --out phase4/results/spatial_select
  python -m phase4.src.run_spatial --mode sanity --seeds 0 --device cuda:0 --out phase4/results/spatial_sanity
  python -m phase4.src.run_spatial --mode main --seeds 0 2 4 6 8 --device cuda:0 --out phase4/results/spatial
Writes <out>/seed<s>/{runs.csv, confusion.npz, losses.json, curves.json}; select writes <out>/steps.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase2.src.run import E, MIN_TEST_PIXELS, N_CLASSES, scores
from phase3.src import run as R3
from .labels import NS, labelled

STEPS_FILE = "phase4/results/spatial_select/steps.json"


def run_seed(s, mode, out, dev, steps_file):
    t0 = time.time()
    E.set_device(dev)
    split = "test" if mode == "main" else "validation"
    out = Path(out) / f"seed{s}"
    out.mkdir(parents=True, exist_ok=True)
    tr, ev = R3.Split("train", dev), R3.Split(split, dev)
    comp = R3.Components(tr, s, dev)
    print(f"seed {s}: components {time.time() - t0:.0f}s", flush=True)
    pool = labelled(s)
    grid = np.load(R3.CACHE2 / f"{split}.npz")
    n = np.bincount(grid["y"][grid["y"] >= 0], minlength=N_CLASSES)
    scored = np.flatnonzero(n >= MIN_TEST_PIXELS)
    keep = np.isin(grid["y"], scored)
    pts = (grid["img"][keep], grid["pix"][keep], grid["y"][keep].astype(np.int64))

    if mode == "select":
        methods = ["S0"]
    else:
        chosen = {int(k): v for k, v in json.load(open(steps_file))["steps"].items()}
        methods = R3.METHODS
    rows, cms, curves = [], {}, {}
    for method in methods:
        for N in NS:
            cps = R3.STEP_GRID if mode == "select" else [chosen[N]]
            res, curve = R3.train_eval(comp, tr, ev, method, N, s, pool, pts, cps, dev)
            curves[f"{method}/N{N}"] = curve
            for step, (cm, g, dt) in res.items():
                bacc, miou = scores(cm, scored)
                rows.append(dict(seed=s, N=N, method=method, steps=step, bacc=bacc, miou=miou, gamma=g, train_s=round(dt, 1)))
                cms[f"{method}/N{N}/S{step}"] = cm
                print(f"seed {s} {method:8s} N={N:3d} steps {step:4d} bacc {bacc:.3f} miou {miou:.3f} |γ| {g:.4f} ({dt:.0f}s)", flush=True)
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "confusion.npz", scored=scored, **cms)
    json.dump(comp.losses, open(out / "losses.json", "w"))
    json.dump(curves, open(out / "curves.json", "w"))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)


def choose_steps(out):
    """The S_N rule: per N, the step count with the highest mean val bacc of S0 over the select seeds
    (ties: fewer steps)."""
    rows = [r for p in sorted(Path(out).glob("seed*/runs.csv")) for r in csv.DictReader(open(p))]
    seeds = sorted({int(r["seed"]) for r in rows})
    mean = {}
    for r in rows:
        mean.setdefault((int(r["N"]), int(r["steps"])), []).append(float(r["bacc"]))
    steps = {N: max(R3.STEP_GRID, key=lambda S: (round(np.mean(mean[(N, S)]), 6), -S)) for N in NS}
    table = {f"{N}/{S}": float(np.mean(v)) for (N, S), v in mean.items()}
    json.dump(dict(seeds=seeds, steps=steps, mean_bacc=table), open(Path(out) / "steps.json", "w"), indent=1)
    print("steps", steps, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["select", "sanity", "main"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps-file", default=STEPS_FILE)
    ap.add_argument("--choose", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.choose:
        return choose_steps(a.out)
    for s in a.seeds:
        run_seed(s, a.mode, a.out, a.device, a.steps_file)


if __name__ == "__main__":
    main()
