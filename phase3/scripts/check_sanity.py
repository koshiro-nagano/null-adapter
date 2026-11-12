"""Gate before the test run: the val sanity run has every method x K row, finite scores, and finite losses.
Prints the reasons and exits 1 on failure. Scores are not compared between methods."""
import csv
import json
import math
import sys
from pathlib import Path

from phase3.src.run import KS, METHODS

out = Path(sys.argv[1])
bad = []
rows = list(csv.DictReader(open(out / "seed0" / "runs.csv")))
have = {(r["method"], int(r["K"])) for r in rows}
bad += [f"missing {m} K={k}" for m in METHODS for k in KS if (m, k) not in have]
bad += [f"non-finite score {r['method']} K={r['K']}" for r in rows if not all(math.isfinite(float(r[c])) for c in ("bacc", "miou"))]
curves = json.load(open(out / "seed0" / "curves.json"))
bad += [f"non-finite loss {k}" for k, c in curves.items() if not all(math.isfinite(v) for _, v in c)]
bad += [f"loss did not decrease {k}" for k, c in curves.items() if c and c[-1][1] >= c[0][1]]
comp = json.load(open(out / "seed0" / "losses.json"))
bad += [f"component loss did not decrease {k}" for k, v in comp.items() if v[-1] >= v[0]]
print("\n".join(bad) or "sanity OK")
sys.exit(1 if bad else 0)
