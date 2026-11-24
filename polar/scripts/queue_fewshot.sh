#!/usr/bin/env bash
# Fewer labels for the most promising operator: null_orth vs RGB only at ZJU 10 / 5 and RGBP-Glass 30 / 10 labelled
# images (nested within a seed, same iteration schedules as queue.sh). Finished runs (result.json) are skipped.
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
M="rgb null_orth"
for n in 10 5; do for s in 0 1 2; do run "$M" zju $n $s "" polar/results/zju; done; done
for n in 30 10; do for s in 0 1 2; do run "$M" pgsnet $n $s 8020 polar/results/pgsnet; done; done
echo "[$(date '+%F %T')] fewshot queue done"
