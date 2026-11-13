#!/usr/bin/env bash
# Polarisation comparison (SimCMF vs RGB-only vs simple fusion vs Null-Adapter conditional residual), in priority
# order on one GPU; finished runs (result.json) are skipped, so the script can be restarted.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
METHODS="simcmf rgb fuse_p null_p"
run() {  # ds n_label seed iters out
  for m in $METHODS; do
    d="$5/${m}_n$2_s$3"
    [ -f "$d/result.json" ] && continue
    echo "[$(date '+%F %T')] $1 $m n=$2 seed=$3"
    $PY -m polar.src.train --ds "$1" --method "$m" --n_label "$2" --seed "$3" ${4:+--iters $4} --out "$5" \
      > /dev/null 2>"$5/${m}_n$2_s$3.err" || echo "  FAILED (see $5/${m}_n$2_s$3.err)"
  done
}
mkdir -p polar/results/zju polar/results/pgsnet
for s in 0 1 2; do run zju all $s "" polar/results/zju; done          # P1  SimCMF schedule (51 epochs)
for s in 0 1 2; do run zju 20 $s "" polar/results/zju; done           # P2  20 labelled images, same iterations
run pgsnet all 0 8020 polar/results/pgsnet                            # P3  10 epochs' worth
run pgsnet 100 0 8020 polar/results/pgsnet                            # P4
echo "[$(date '+%F %T')] queue done"
