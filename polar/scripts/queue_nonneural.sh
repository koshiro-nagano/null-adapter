#!/usr/bin/env bash
# Non-neural m(c) comparison: pixel-level per-patch orthogonalisation (null_orth), token-level orthogonalisation
# against the RGB token (null_orthf), then the closed-form physics operator (null_phys). Same schedules as queue.sh;
# finished runs (result.json) are skipped, so the script can be restarted.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
run() {  # methods ds n_label seed iters out
  for m in $1; do
    d="$6/${m}_n$3_s$4"
    [ -f "$d/result.json" ] && continue
    echo "[$(date '+%F %T')] $2 $m n=$3 seed=$4"
    $PY -m polar.src.train --ds "$2" --method "$m" --n_label "$3" --seed "$4" ${5:+--iters $5} --out "$6" \
      > /dev/null 2>"$6/${m}_n$3_s$4.err" || echo "  FAILED (see $6/${m}_n$3_s$4.err)"
  done
}
for M in "null_orth null_orthf" "null_phys"; do
  for s in 0 1 2; do run "$M" zju 20 $s "" polar/results/zju; done
  for s in 0 1 2; do run "$M" zju all $s "" polar/results/zju; done
  for s in 0 1 2; do
    run "$M" pgsnet 100 $s 8020 polar/results/pgsnet
    run "$M" pgsnet all $s 8020 polar/results/pgsnet
  done
done
echo "[$(date '+%F %T')] nonneural queue done"
