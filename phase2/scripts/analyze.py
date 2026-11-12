"""Phase 2 analysis exactly as pre-registered (PREREGISTRATION.md "Metrics", "Hypotheses", "Exploratory").

Usage (from null-adapter/):  python -m phase2.scripts.analyze phase2/results/main [label_info.json]
Reads <results>/seed*/{runs.csv, confusion.npz, diagnostics.json}; writes <results>/analysis.json and
<results>/analysis.md. Paired by seed, 95 % t-CI of the mean difference; "clearly better" = CI excludes 0 and
the mean difference is positive. K set for a hypothesis = {5, 20, 100} unless the hypothesis names its K.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

KS = [5, 20, 100, 1000]
DEFAULT_KS = [5, 20, 100]
# (id, A, B, Ks, kind): kind "better" = A clearly better than B at every K; "close" = |mean(A - B)| < 0.02 at every K
HYP = [("P1", "M1", "M0c", DEFAULT_KS, "better"),
       ("P2", "M5_mlp", "M0", DEFAULT_KS, "better"),
       ("P3", "M5_mlp", "M2", [5, 20], "better"),
       ("P4", "M5_mlp", "M1", DEFAULT_KS, "better"),
       ("P5", "M5_mlp", "M5_ridge", [100, 1000], "better"),
       ("P6", "M5_mlp", "B_RNDres", DEFAULT_KS, "close")]
EXPL = [("E1", "M0", "M0_png"), ("E3", "M6", "M5_mlp"), ("E3", "B_HSIA", "M2"), ("E3", "M0", "M0_S"),
        ("E3", "M0", "M0_L"), ("E4", "M5_ctx", "M5_mlp")]


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    m = float(d.mean())
    if len(d) < 2:
        return dict(mean=m, lo=float("nan"), hi=float("nan"), n=len(d))
    h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return dict(mean=m, lo=m - h, hi=m + h, n=len(d))


def group_scores(cm, scored, groups):
    """bacc / mIoU after merging classes into their class_group (true rows: scored classes only)."""
    G = groups.max() + 1
    M = np.zeros((G, G))
    for t in scored:
        np.add.at(M[groups[t]], groups, cm[t])
    present = np.flatnonzero(M.sum(1) > 0)
    tp = np.diag(M)[present]
    return float((tp / M[present].sum(1)).mean()), float((tp / (M[present].sum(1) + M[:, present].sum(0) - tp)).mean())


def main(results, label_info="phase2/data/full/LIB-HSI/label_info.json"):
    results = Path(results)
    groups = np.array([c["class_group"] for c in json.load(open(label_info))["items"]])
    seeds = sorted((d for d in results.glob("seed*") if (d / "runs.csv").exists()), key=lambda d: int(d.name[4:]))
    score = defaultdict(dict)  # (metric, method, K) -> {seed: value}
    diags = []
    for d in seeds:
        s = int(d.name[4:])
        for r in csv.DictReader(open(d / "runs.csv")):
            for key in ("bacc", "miou"):
                score[(key, r["method"], int(r["K"]))][s] = float(r[key])
        cz = np.load(d / "confusion.npz")
        for name in cz.files:
            if name == "scored":
                continue
            m, K = name.split("/K")
            gb, gi = group_scores(cz[name], cz["scored"], groups)
            score[("group_bacc", m, int(K))][s] = gb
            score[("group_miou", m, int(K))][s] = gi
        diags.append(json.load(open(d / "diagnostics.json")))
    seed_ids = [int(d.name[4:]) for d in seeds]
    methods = list(dict.fromkeys(m for (_, m, _) in score))

    def vals(metric, m, K):
        return [score[(metric, m, K)][s] for s in seed_ids]

    table = {metric: {m: {K: dict(mean=float(np.mean(vals(metric, m, K))), sd=float(np.std(vals(metric, m, K), ddof=1))
                                  if len(seed_ids) > 1 else float("nan")) for K in KS} for m in methods}
             for metric in ("bacc", "miou", "group_bacc", "group_miou")}
    hyps = []
    for hid, a, b, ks, kind in HYP:
        per_k = {K: paired(vals("bacc", a, K), vals("bacc", b, K)) for K in KS}
        if kind == "better":
            ok = [per_k[K]["lo"] > 0 for K in ks]
        else:
            ok = [abs(per_k[K]["mean"]) < 0.02 for K in ks]
        hyps.append(dict(id=hid, A=a, B=b, Ks=ks, kind=kind, supported=bool(all(ok)), per_K=per_k,
                         secondary_miou={K: paired(vals("miou", a, K), vals("miou", b, K)) for K in KS}))
    expl = [dict(id=e, A=a, B=b, per_K={K: paired(vals("bacc", a, K), vals("bacc", b, K)) for K in KS})
            for e, a, b in EXPL]
    diag_mean = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
    out = dict(seeds=seed_ids, table=table, hypotheses=hyps, exploratory=expl, diagnostics_mean=diag_mean)
    json.dump(out, open(results / "analysis.json", "w"), indent=1)

    L = [f"# Phase 2 results ({len(seed_ids)} seeds: {seed_ids})", "",
         "## Balanced accuracy, test, 39 scored classes (mean ± sd over seeds)", "",
         "| method | " + " | ".join(f"K={K}" for K in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{table['bacc'][m][K]['mean']:.3f} ± {table['bacc'][m][K]['sd']:.3f}"
                                         for K in KS) + " |")
    L += ["", "## Pre-registered hypotheses (paired 95 % t-CI of bacc(A) - bacc(B))", "",
          "| id | A vs B | Ks | " + " | ".join(f"K={K}" for K in KS) + " | verdict |", "|---|---|---|" + "---|" * len(KS) + "---|"]
    for h in hyps:
        cells = [f"{h['per_K'][K]['mean']:+.4f} [{h['per_K'][K]['lo']:+.4f}, {h['per_K'][K]['hi']:+.4f}]" for K in KS]
        L.append(f"| {h['id']} | {h['A']} vs {h['B']} | {','.join(map(str, h['Ks']))} | " + " | ".join(cells)
                 + f" | {'supported' if h['supported'] else 'not supported'} |")
    L += ["", "## Exploratory (no decision)", "", "| id | A vs B | " + " | ".join(f"K={K}" for K in KS) + " |",
          "|---|---|" + "---|" * len(KS)]
    for e in expl:
        L.append(f"| {e['id']} | {e['A']} vs {e['B']} | " + " | ".join(
            f"{e['per_K'][K]['mean']:+.4f} [{e['per_K'][K]['lo']:+.4f}, {e['per_K'][K]['hi']:+.4f}]" for K in KS) + " |")
    L += ["", "## Diagnostics (mean over seeds)", ""] + [f"- {k}: {v:.4f}" for k, v in diag_mean.items()]
    (results / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:3])
