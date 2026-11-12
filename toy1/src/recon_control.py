"""M6b + reconstruction loss (H7a-H7c) from results dirs that contain the M6b_rec* variants.

Usage (from toy1/):  python -m src.recon_control results/main_rec results/mog_rec [results/real_nikon ...]
Writes results/recon/recon_control.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .analysis import paired
from .run import summarize

VARIANTS = ["M6", "M6b_direct", "M6b_ema", "M6b_vic", "M6b_rec01", "M6b_rec1", "M6b_rec10"]


def one(res: Path) -> dict:
    df = pd.read_csv(res / "runs.csv")
    dyn = pd.read_json(res / "dynamics.jsonl", lines=True)
    last = dyn.loc[dyn.groupby(["variant", "seed"]).step.idxmax()]
    s = summarize(df)
    acc = lambda m, K, t="T_resid": float(s[(s.task == t) & (s.method == m) & (s.K == K)].bal_acc_mean.iloc[0])
    Ks = sorted(df.K.unique())
    out = dict(
        dynamics={v: dict(unpred_end=float((1 - last[last.variant == v].pred_r2).mean()),
                          r_norm_end=float(last[last.variant == v].r_norm.mean()),
                          g_var_end=float(last[last.variant == v].g_var_total.mean()))
                  for v in VARIANTS if v in set(last.variant)},
        acc={v: {int(K): acc(v, K) for K in Ks} for v in VARIANTS},
        acc_mixed={v: {int(K): acc(v, K, "T_mixed") for K in Ks} for v in VARIANTS},
        H7b=paired(df, "T_resid", "M6b_rec1", "M6b_direct", 20),
        H7c=paired(df, "T_resid", "M6b_rec1", "M6", 20),
        rec_vs_M6={v: [paired(df, "T_resid", v, "M6", int(K)) for K in Ks] for v in ["M6b_rec01", "M6b_rec1", "M6b_rec10"]},
    )
    out["verdicts"] = dict(
        H7a="supported" if out["dynamics"]["M6b_rec1"]["unpred_end"] >= 0.05 else "not supported",
        H7b="supported" if out["H7b"]["lo"] > 0 else "not supported",
        H7c="supported" if out["H7c"]["hi"] >= 0 else "not supported",
    )
    return out


def main(dirs):
    res = {Path(d).name: one(Path(d)) for d in dirs}
    Path("results/recon").mkdir(parents=True, exist_ok=True)
    Path("results/recon/recon_control.json").write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(f"== {name}: {r['verdicts']}")
        for v, d in r["dynamics"].items():
            print(f"   {v:11s} 1-R2 {d['unpred_end']:.2e}  ||r|| {d['r_norm_end']:.3f}  T_resid K=5/20/500: "
                  f"{r['acc'][v][5]:.3f} {r['acc'][v][20]:.3f} {r['acc'][v][500]:.3f}")
        for k in ["H7b", "H7c"]:
            x = r[k]
            print(f"   {k}: {x['a']} - {x['b']} = {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
