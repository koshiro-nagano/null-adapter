"""Hypothesis checks H1-H4 from runs.csv / dynamics.jsonl (paired over seeds).

Decision rule (fixed by us because the pre-registration does not define "clearly better"):
  A is "clearly better" than B at K  <=>  the 95% t-CI of the per-seed paired difference
  acc(A) - acc(B) lies entirely above 0. "Chance level" <=> |mean - 0.5| < 2 SE of a fair coin on the
  test set (2 * 0.5 / sqrt(n_test) = 0.01) AND metameric-pair accuracy == 0.5 exactly.
  (All seeds share one test set, so the across-seed CI omits test-set sampling error and is not
  used for the chance test.)

Usage (from toy1/):  python -m src.analysis results/main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

RESID = ["M4", "M5_ridge", "M5_mlp", "M6"]
FULL = ["M1", "M2", "M3"]
SMALL_K = [2, 5, 10, 20]


def paired(df, task, a, b, K, col="bal_acc"):
    d = df[(df.task == task) & (df.K == K)].pivot(index="seed", columns="method", values=col)
    diff = (d[a] - d[b]).dropna().values
    n = len(diff)
    h = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_rel(d[a], d[b]).pvalue if diff.std() > 0 else 0.0
    return dict(a=a, b=b, task=task, K=int(K), metric=col, mean=float(diff.mean()),
                lo=float(diff.mean() - h), hi=float(diff.mean() + h), p=float(p), n=n)


def cell(df, task, m, K, col="bal_acc"):
    v = df[(df.task == task) & (df.method == m) & (df.K == K)][col].values
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), ci=float(h))


def analyse(res: Path) -> dict:
    df = pd.read_csv(res / "runs.csv")
    n_test = yaml.safe_load((res / "config_used.yaml").read_text())["sizes"]["n_test"]
    tol = 2 * 0.5 / np.sqrt(n_test)
    Ks = sorted(df.K.unique())
    pre = ["M0", "M1", "M2", "M3", "M4", "M5_ridge", "M5_mlp", "M6", "M6b_direct", "M6b_ema", "M6b_vic"]
    out = {}

    # H1: T_range, M0 best among pre-registered methods at small K
    h1 = []
    for K in SMALL_K:
        means = {m: cell(df, "T_range", m, K)["mean"] for m in pre}
        best = max(means, key=means.get)
        comps = [paired(df, "T_range", "M0", m, K) for m in pre if m != "M0"]
        h1.append(dict(K=K, best=best, M0=means["M0"], best_acc=means[best],
                       n_sig_better=sum(c["lo"] > 0 for c in comps),
                       n_sig_worse=sum(c["hi"] < 0 for c in comps),
                       comps=comps))
    out["H1"] = h1

    # H2a: M0 at chance on T_resid for all phi scales and all K (test + metameric)
    h2a = []
    for m in ["M0_S", "M0", "M0_L"]:
        for K in Ks:
            c = cell(df, "T_resid", m, K)
            mm = cell(df, "T_resid", m, K, "meta_acc")
            h2a.append(dict(method=m, K=int(K), acc=c["mean"], ci=c["ci"], meta=mm["mean"],
                            chance=bool(abs(c["mean"] - 0.5) < tol and abs(mm["mean"] - 0.5) < 1e-9)))
    out["H2a"] = h2a
    # H2b: residual methods vs full-channel / algebraic-null methods at K<=20
    h2b = []
    for K in SMALL_K:
        for a in RESID:
            for b in FULL:
                h2b.append(paired(df, "T_resid", a, b, K))
    out["H2b"] = h2b

    # H2-NL (MoG pre-registration): the advantage persists at K = 10, 20 vs M1
    out["H2NL"] = [paired(df, "T_resid", a, "M1", K) for K in [10, 20] for a in ["M4", "M5_mlp", "M6"]]
    # H5: misspecified linear estimate of E[x|c] is worse than the MLP estimate
    out["H5"] = [paired(df, "T_resid", "M5_mlp", "M5_ridge", K) for K in Ks if K >= 20]

    # H3: T_null_pred is (partly) solvable by M0
    out["H3"] = [dict(K=int(K), **cell(df, "T_null_pred", "M0", K),
                      vs_M1=paired(df, "T_null_pred", "M0", "M1", K)) for K in Ks]

    # H4: M6b residual shrinks and T_resid accuracy drops vs M6
    dyn = pd.read_json(res / "dynamics.jsonl", lines=True)
    last = dyn.loc[dyn.groupby(["variant", "seed"]).step.idxmax()]
    first = dyn[dyn.step == 0]
    dsum = []
    for v in ["M6", "M6b_direct", "M6b_ema", "M6b_vic"]:
        l, f = last[last.variant == v], first[first.variant == v]
        dsum.append(dict(variant=v, r_norm_start=float(f.r_norm.mean()), r_norm_end=float(l.r_norm.mean()),
                         g_var_end=float(l.g_var_total.mean()), g_var_min_end=float(l.g_var_min.mean()),
                         unpred_end=float((1 - l.pred_r2).mean())))
    out["H4_dynamics"] = dsum
    out["H4_acc"] = [paired(df, "T_resid", v, "M6", K) for v in ["M6b_direct", "M6b_ema", "M6b_vic"] for K in Ks]
    return out


def verdicts(a: dict) -> dict:
    v = {}
    h1 = a["H1"]
    m0_never_worse = all(r["n_sig_worse"] == 0 for r in h1)
    m0_best = sum(r["best"] == "M0" for r in h1)
    v["H1"] = ("supported" if m0_never_worse and m0_best >= 3 else
               "inconclusive" if m0_never_worse else "not supported")
    chance = all(r["chance"] for r in a["H2a"])
    by_k = {}
    for r in a["H2b"]:
        by_k.setdefault(r["K"], []).append(r["lo"] > 0)
    v["H2_K_all_sig"] = {k: all(x) for k, x in by_k.items()}
    v["H2_M0_chance"] = chance
    allk = all(v["H2_K_all_sig"].values())
    somek = any(v["H2_K_all_sig"].values())
    v["H2"] = "supported" if chance and allk else ("partially supported" if chance and somek else "not supported")
    # same rule restricted to the non-linear estimators named in the spec (M4, M5_mlp, M6)
    ex = {}
    for r in a["H2b"]:
        if r["a"] != "M5_ridge":
            ex.setdefault(r["K"], []).append(r["lo"] > 0)
    v["H2_K_all_sig_excl_ridge"] = {k: all(x) for k, x in ex.items()}
    v["H2_excl_ridge"] = "supported" if chance and all(v["H2_K_all_sig_excl_ridge"].values()) else (
        "partially supported" if chance and any(v["H2_K_all_sig_excl_ridge"].values()) else "not supported")
    v["H2NL"] = "supported" if all(r["lo"] > 0 for r in a["H2NL"]) else "not supported"
    v["H5"] = "supported" if all(r["lo"] > 0 for r in a["H5"]) else (
        "partially supported" if any(r["lo"] > 0 for r in a["H5"]) else "not supported")
    h3 = a["H3"]
    v["H3"] = "supported" if h3[0]["mean"] > 0.7 and h3[-1]["mean"] > 0.95 else "not supported"
    dyn = {d["variant"]: d for d in a["H4_dynamics"]}
    shrink = all(dyn[f"M6b_{x}"]["r_norm_end"] < dyn["M6"]["r_norm_end"] for x in ["direct", "ema"])
    drop = all(r["hi"] < 0 for r in a["H4_acc"] if r["K"] >= 5)
    v["H4"] = "supported" if shrink and drop else "not supported"
    return v


def main(res_dir: str):
    res = Path(res_dir)
    a = analyse(res)
    a["verdicts"] = verdicts(a)
    (res / "hypotheses.json").write_text(json.dumps(a, indent=1))
    print(json.dumps(a["verdicts"], indent=1))
    for r in a["H1"]:
        print("H1 K=%d best=%s(%.3f) M0=%.3f  M0 sig better than %d, sig worse than %d"
              % (r["K"], r["best"], r["best_acc"], r["M0"], r["n_sig_better"], r["n_sig_worse"]))
    for r in a["H2b"]:
        print("H2b K=%2d %-8s - %s: %+.3f [%+.3f, %+.3f]" % (r["K"], r["a"], r["b"], r["mean"], r["lo"], r["hi"]))
    for d in a["H4_dynamics"]:
        print("H4", d)


if __name__ == "__main__":
    main(sys.argv[1])
