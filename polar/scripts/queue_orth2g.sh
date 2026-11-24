#!/usr/bin/env bash
# Checks of null_orth2_g (pixel + token orthogonalisation, RGB-token gate, modality dropout 0.3):
#  (A) stability: seeds 3, 4 at ZJU 20 / RGBP-Glass 100 (plus rgb and null_orth where those seeds are missing)
#  (B) extreme few-shot: ZJU 5, RGBP-Glass 10 (seeds 0-2)
#  (C) full labels: ZJU all, RGBP-Glass all (seeds 0-2)
# Finished runs (result.json) are skipped, so the script can be restarted.
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
for s in 3 4; do                                                                   # (A)
  run "null_orth2_g" zju 20 $s "" polar/results/zju
  run "null_orth2_g rgb null_orth" pgsnet 100 $s 8020 polar/results/pgsnet
done
for s in 0 1 2; do                                                                 # (B)
  run "null_orth2_g" zju 5 $s "" polar/results/zju
  run "null_orth2_g" pgsnet 10 $s 8020 polar/results/pgsnet
done
for s in 0 1 2; do                                                                 # (C)
  run "null_orth2_g" zju all $s "" polar/results/zju
  run "null_orth2_g" pgsnet all $s 8020 polar/results/pgsnet
done
echo "[$(date '+%F %T')] orth2g queue done"
