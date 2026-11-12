"""Experiment loop: (task, method, K, seed) -> balanced accuracy + metameric-pair accuracy.

Usage (from toy1/):
  python -m src.run --out results/smoke --seeds 1 --K 2 20 500
  python -m src.run --out results/main
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .baselines import fit_predict_hsia
from .config import load_config
from .data import TASKS, build_data, diagnostics, sample_k_per_class
from .encoders import set_device, train_encoders
from .methods import LINEAR_METHODS, FeatureFactory, make_head

DATA = None
CFG = None
METAMERIC_TASK = "T_resid"


def _init_worker(cfg, data_path):
    global DATA, CFG
    import pickle
    CFG = cfg
    with open(data_path, "rb") as f:  # data built once in the parent (identical for every worker)
        DATA = pickle.load(f)


def _pick_device(seed: int) -> str:
    want = CFG["run"].get("device", "cpu")
    if want == "cuda" and torch.cuda.is_available():
        return f"cuda:{seed % torch.cuda.device_count()}"
    return "cpu"


def run_seed(args):
    seed, Ks = args
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # L-BFGS iteration cap (fixed a priori)
    torch.set_num_threads(CFG["run"]["torch_threads"])
    device = str(set_device(_pick_device(seed)))
    t0 = time.time()
    enc = train_encoders(DATA, CFG, seed)
    t_enc = time.time() - t0
    ff = FeatureFactory(DATA, enc, CFG)
    L, T, Mp = DATA.labeled, DATA.test, DATA.metameric
    methods = LINEAR_METHODS + [f"M6b_{v}" for v in enc.m6b if f"M6b_{v}" not in LINEAR_METHODS]
    FL, FT, FM = (ff.all(S.c, S.x, methods) for S in (L, T, Mp))
    rows = []
    for ti, task in enumerate(TASKS):
        for K in Ks:
            rng = np.random.default_rng([seed, ti, K])
            idx = sample_k_per_class(L.y[task], K, rng)  # same subset for every method (paired)
            ytr = L.y[task][idx]
            preds = {}
            for m in FL:
                head = make_head(CFG, seed).fit(FL[m][idx], ytr)
                preds[m] = (head.predict(FT[m]), head.predict(FM[m]))
            ph = fit_predict_hsia(enc.phi[CFG["encoders"]["phi"]["main_scale"]], enc.sx, L.x[idx], ytr,
                                  {"test": T.x, "meta": Mp.x}, CFG, seed * 1000 + ti * 10 + Ks.index(K))
            preds["B_HSIA"] = (ph["test"], ph["meta"])
            for m, (pt, pm) in preds.items():
                rows.append(dict(task=task, method=m, K=K, seed=seed,
                                 bal_acc=balanced_accuracy_score(T.y[task], pt),
                                 meta_acc=accuracy_score(Mp.y[task], pm) if task == METAMERIC_TASK else np.nan))
    dyn = pd.DataFrame(enc.dynamics).assign(seed=seed)
    losses = {k: v[-1] for k, v in enc.losses.items()}
    timing = dict(seed=seed, device=device, encoders_s=t_enc, total_s=time.time() - t0)
    return rows, dyn, losses, timing


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def ci(s):
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        return stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))

    g = df.groupby(["task", "method", "K"])
    out = g.agg(bal_acc_mean=("bal_acc", "mean"), meta_acc_mean=("meta_acc", "mean"), n=("seed", "count"))
    out["bal_acc_ci95"] = g["bal_acc"].apply(ci)
    out["meta_acc_ci95"] = g["meta_acc"].apply(ci)
    return out.reset_index()


def main():
    global DATA, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--K", type=int, nargs="*")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    CFG = load_config(a.config)
    seeds = a.seeds if a.seeds is not None else CFG["seeds"]
    Ks = a.K or CFG["K"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(CFG, sort_keys=False))
    (out / "run_args.json").write_text(json.dumps(dict(seeds=seeds, K=Ks), indent=1))

    t0 = time.time()
    DATA = build_data(CFG)
    info = dict(DATA.world.info, **diagnostics(DATA), label_balance={t: float(DATA.test.y[t].mean()) for t in TASKS},
                data_build_s=time.time() - t0)
    (out / "world_info.json").write_text(json.dumps(info, indent=1))
    print("world:", json.dumps(info), flush=True)

    workers = min(a.workers or CFG["run"]["n_workers"], len(seeds))
    jobs = [(s, Ks) for s in seeds]
    if workers > 1:
        # spawn: CUDA cannot be used in forked children; each worker rebuilds the same data
        import pickle
        data_path = out / "data_cache.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(DATA, f)
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(CFG, str(data_path))) as pool:
            results = pool.map(run_seed, jobs, chunksize=1)
    else:
        torch.set_num_threads(CFG["run"]["torch_threads"])
        results = [run_seed(j) for j in jobs]

    df = pd.DataFrame([r for res in results for r in res[0]])
    df.to_csv(out / "runs.csv", index=False)
    pd.concat([res[1] for res in results]).to_json(out / "dynamics.jsonl", orient="records", lines=True)
    pd.DataFrame([dict(seed=s, **res[2]) for s, res in zip(seeds, results)]).to_csv(out / "encoder_losses.csv", index=False)
    pd.DataFrame([res[3] for res in results]).to_csv(out / "timing.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "summary.csv", index=False)
    if (out / "data_cache.pkl").exists():
        (out / "data_cache.pkl").unlink()
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    piv = summ.pivot_table(index=["task", "method"], columns="K", values="bal_acc_mean")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(piv.round(3))


if __name__ == "__main__":
    main()
