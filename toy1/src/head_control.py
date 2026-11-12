"""Head-capacity control: compare linear-head runs with MLP-head runs (H6a-H6d).

Usage (from toy1/):  python -m src.head_control
Reads results/{main,mog}{,_mlp}/runs.csv and hypotheses.json; writes results/head_mlp/head_control.json.
Seeds share encoders and labeled subsets across heads, so gaps are paired seed by seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import paired
from .run import summarize

RES = Path("results")
RUNS = {("gauss", "linear"): "main", ("gauss", "mlp"): "main_mlp", ("mog", "linear"): "mog", ("mog", "mlp"): "mog_mlp"}
METHODS = ["M0", "M1", "M2", "M3", "M4", "M5_ridge", "M5_mlp", "M6", "M6b_direct", "B_RNDres", "B_HSIA"]


def ci(v):
    v = np.asarray(v)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h))


def gap(df, a, b, K, task="T_resid"):
    d = df[(df.task == task) & (df.K == K)].pivot(index="seed", columns="method", values="bal_acc")
    return (d[a] - d[b]).sort_index()


def main():
    dfs = {k: pd.read_csv(RES / v / "runs.csv") for k, v in RUNS.items()}
    hyps = {k: json.loads((RES / v / "hypotheses.json").read_text()) for k, v in RUNS.items()}
    Ks = sorted(dfs[("mog", "mlp")].K.unique())
    out = {}

    out["H6a"] = [paired(dfs[("mog", "mlp")], "T_resid", a, "M1", K) for K in [10, 20] for a in ["M4", "M5_mlp", "M6"]]
    out["H6b"] = []
    for K in Ks:
        g_lin, g_mlp = gap(dfs[("mog", "linear")], "M4", "M1", K), gap(dfs[("mog", "mlp")], "M4", "M1", K)
        out["H6b"].append(dict(K=int(K), gap_linear=ci(g_lin), gap_mlp=ci(g_mlp), diff=ci(g_mlp.values - g_lin.values)))
    out["H6c"] = {f"{p}_{h}": dict(chance=hyps[(p, h)]["verdicts"]["H2_M0_chance"],
                                   acc=[r["acc"] for r in hyps[(p, h)]["H2a"] if r["method"] == "M0"],
                                   meta=[r["meta"] for r in hyps[(p, h)]["H2a"] if r["method"] == "M0"])
                  for (p, h) in RUNS}
    out["H6d"] = [paired(dfs[("gauss", "mlp")], "T_resid", "M4", "M1", K) for K in [10, 20]]

    v = {}
    v["H6a"] = "supported" if all(r["lo"] > 0 for r in out["H6a"]) else (
        "partially supported" if any(r["lo"] > 0 for r in out["H6a"]) else "not supported")
    b500 = next(r for r in out["H6b"] if r["K"] == 500)
    v["H6b"] = "supported" if b500["diff"]["hi"] < 0 else "not supported"
    v["H6c"] = "supported" if all(x["chance"] for (p, h), x in zip(RUNS, out["H6c"].values()) if h == "mlp") else "not supported"
    v["H6d"] = "supported" if all(r["lo"] <= 0 <= r["hi"] for r in out["H6d"]) else "not supported"
    out["verdicts"] = v

    # gap curves (method - M1 on T_resid) and accuracy curves for all four configurations
    curves, gaps = {}, {}
    for (p, h), df in dfs.items():
        s = summarize(df)
        for m in METHODS:
            d = s[(s.task == "T_resid") & (s.method == m)].sort_values("K")
            curves[f"{p}:{h}:{m}"] = [dict(K=int(k), y=float(y), ci=float(c)) for k, y, c in zip(d.K, d.bal_acc_mean, d.bal_acc_ci95)]
        for m in ["M4", "M5_mlp", "M6", "M2"]:
            gaps[f"{p}:{h}:{m}"] = [dict(K=int(K), **ci(gap(df, m, "M1", K))) for K in Ks]
    out["curves"], out["gaps"] = curves, gaps
    out["preregistered_verdicts"] = {f"{p}_{h}": hyps[(p, h)]["verdicts"] for (p, h) in RUNS}
    def at(s, t, m, K):
        return float(s[(s.task == t) & (s.method == m) & (s.K == K)].bal_acc_mean.iloc[0])

    out["T_all"] = {}
    for (p, h), df in dfs.items():
        s = summarize(df)
        out["T_all"][f"{p}_{h}"] = {t: {m: {int(K): at(s, t, m, K) for K in [5, 20, 100, 500]} for m in METHODS}
                                    for t in ["T_range", "T_null_pred", "T_resid", "T_mixed"]}
    (RES / "head_mlp" / "head_control.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v, indent=1))
    for r in out["H6a"]:
        print("H6a K=%d %s-M1 %+.3f [%+.3f,%+.3f]" % (r["K"], r["a"], r["mean"], r["lo"], r["hi"]))
    for r in out["H6b"]:
        print("H6b K=%3d gap lin %+.3f  mlp %+.3f  diff %+.3f [%+.3f,%+.3f]" % (
            r["K"], r["gap_linear"]["mean"], r["gap_mlp"]["mean"], r["diff"]["mean"], r["diff"]["lo"], r["diff"]["hi"]))
    for r in out["H6d"]:
        print("H6d K=%d M4-M1 %+.3f [%+.3f,%+.3f]" % (r["K"], r["mean"], r["lo"], r["hi"]))


if __name__ == "__main__":
    main()
