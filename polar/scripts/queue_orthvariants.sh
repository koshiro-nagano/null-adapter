#!/usr/bin/env bash
# null_orth improvements, after the few-shot queue: (1) pixel- + token-level orthogonalisation (null_orth2),
# (2) RGB-token gate + modality dropout 0.3 (null_orth_g), and both (null_orth2_g); ZJU 20 / RGBP-Glass 100 labels,
# seeds interleaved. Waits until queue_fewshot.sh has exited. Finished runs (result.json) are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
while pgrep -f queue_fewshot.sh > /dev/null; do sleep 60; done
run() {  # methods ds n_label seed iters out
  for m in $1; do
    d="$6/${m}_n$3_s$4"
    [ -f "$d/result.json" ] && continue
    echo "[$(date '+%F %T')] $2 $m n=$3 seed=$4"
    $PY -m polar.src.train --ds "$2" --method "$m" --n_label "$3" --seed "$4" ${5:+--iters $5} --out "$6" \
      > /dev/null 2>"$6/${m}_n$3_s$4.err" || echo "  FAILED (see $6/${m}_n$3_s$4.err)"
  done
}
M="null_orth2 null_orth_g null_orth2_g"
for s in 0 1 2; do
  run "$M" zju 20 $s "" polar/results/zju
  run "$M" pgsnet 100 $s 8020 polar/results/pgsnet
done
echo "[$(date '+%F %T')] orth variants queue done"
