"""Choose MoG prior parameters from label-free / large-data diagnostics only.

No compared method (M0-M6b, baselines) is run here. For each candidate we report how
non-linear E[x|c] is and how far a large-data *linear* head on raw x falls short of the
oracle residual features on T_resid. Usage (from toy1/): python -m scripts.prior_design
"""
import copy
import json

import numpy as np

from src.config import load_config
from src.data import Data, build_world, diagnostics, make_split

base = load_config("configs/mog.yaml")
import sys
if len(sys.argv) > 1 and sys.argv[1] == "round2":
    # round 2: identifiable components (large range offsets) + strongly different covariances
    cands = [dict(n_components=k, range_offset=r, null_offset=n, lambda_scale_range=l)
             for k in (4, 6) for r in (2.0, 3.0) for n in (1.5, 3.0) for l in ([0.2, 5.0],)]
else:
    cands = [dict(range_offset=r, null_offset=n, lambda_scale_range=l)
             for r in (0.0, 0.5, 1.0) for n in (0.5, 1.5, 3.0) for l in ([0.5, 2.0],)]
    cands.append(dict(range_offset=0.5, null_offset=1.5, lambda_scale_range=[0.2, 5.0]))
rows = []
for c in cands:
    cfg = copy.deepcopy(base)
    cfg["spectra"]["mixture"].update(c)
    cfg["spectra"]["mixture"]["n_calib"] = 50000
    w = build_world(cfg)
    rng = np.random.default_rng(123)
    L, T = make_split(w, 20000, rng), make_split(w, 10000, rng)
    d = diagnostics(Data(w, None, L, T, None))
    rows.append(dict(c, **d, post_max=w.info["component_posterior_max_mean"],
                     r2_null=w.info["r2_T_null_pred"], bal_resid=float(T.y["T_resid"].mean()),
                     bal_mixed=float(T.y["T_mixed"].mean())))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
