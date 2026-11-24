#!/usr/bin/env bash
# Seeds 3, 4 for the two promising non-neural configurations and their RGB-only reference (paired comparison):
#   ZJU 20 labels: null_orth vs rgb;   RGBP-Glass all labels: null_orthf vs rgb. Finished runs are skipped.
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
for s in 3 4; do run "null_orth rgb" zju 20 $s "" polar/results/zju; done
for s in 3 4; do run "null_orthf rgb" pgsnet all $s 8020 polar/results/pgsnet; done
echo "[$(date '+%F %T')] more-seeds queue done"
