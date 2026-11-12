"""Gate before the test run (PREREGISTRATION.md "Run order and sanity"): both val sanity runs have every
method x N row, finite scores and losses, and Part B training losses decrease. Prints the reasons and exits 1 on
failure. Scores are not compared between methods.

Usage (from null-adapter/):  python -m phase4.scripts.check_sanity phase4/results/linear_sanity phase4/results/spatial_sanity
"""
import csv
import json
import math
import sys
from pathlib import Path

from phase2.src.run import METHODS as LINEAR
from phase3.src.run import METHODS as SPATIAL
from phase4.src.labels import NS
from phase4.src.run_linear import SANITY_NS

lin, spa = Path(sys.argv[1]) / "seed0", Path(sys.argv[2]) / "seed0"
bad = []
for d, methods, ns in ((lin, LINEAR + ["B_HSIA"], SANITY_NS), (spa, SPATIAL, NS)):
    rows = list(csv.DictReader(open(d / "runs.csv")))
    have = {(r["method"], int(r["N"])) for r in rows}
    bad += [f"{d}: missing {m} N={n}" for m in methods for n in ns if (m, n) not in have]
    bad += [f"{d}: non-finite score {r['method']} N={r['N']}" for r in rows
            if not all(math.isfinite(float(r[c])) for c in ("bacc", "miou"))]
    comp = json.load(open(d / "losses.json"))
    bad += [f"{d}: component loss not finite or not decreasing {k}" for k, v in comp.items()
            if not all(math.isfinite(x) for x in v) or v[-1] >= v[0]]
curves = json.load(open(spa / "curves.json"))
bad += [f"non-finite loss {k}" for k, c in curves.items() if not all(math.isfinite(v) for _, v in c)]
bad += [f"loss did not decrease {k}" for k, c in curves.items() if c and c[-1][1] >= c[0][1]]
print("\n".join(bad) or "sanity OK")
sys.exit(1 if bad else 0)
