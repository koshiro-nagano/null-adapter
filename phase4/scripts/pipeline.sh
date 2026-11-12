#!/usr/bin/env bash
# Phase 4 run order (PREREGISTRATION.md): S_N rule on val (Part B, S0 only) -> val sanity (A and B) -> test.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=phase4/results
dk() { gpu=$1; shift; docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp --gpus "device=$gpu" --ipc=host \
  -v "$PWD":/work -w /work --cpus=8 null-adapter:phase2 "$@"; }
echo "[$(date '+%F %T')] select"
dk 0 python -m phase4.src.run_spatial --mode select --seeds 0 2 --device cuda:0 --out $R/spatial_select > $R/spatial_select_gpu0.log 2>&1 &
dk 1 python -m phase4.src.run_spatial --mode select --seeds 1 --device cuda:0 --out $R/spatial_select > $R/spatial_select_gpu1.log 2>&1 &
wait
dk 0 python -m phase4.src.run_spatial --mode select --choose --out $R/spatial_select
echo "[$(date '+%F %T')] sanity"
dk 0 python -m phase4.src.run_spatial --mode sanity --seeds 0 --device cuda:0 --out $R/spatial_sanity > $R/spatial_sanity.log 2>&1 &
dk 1 python -m phase4.src.run_linear --mode sanity --seeds 0 --device cuda:0 --out $R/linear_sanity > $R/linear_sanity.log 2>&1 &
wait
dk 0 python -m phase4.scripts.check_sanity $R/linear_sanity $R/spatial_sanity
echo "[$(date '+%F %T')] main"
dk 0 python -m phase4.src.run_linear --mode main --seeds 0 2 4 6 8 --device cuda:0 --out $R/linear > $R/linear_gpu0.log 2>&1 &
dk 1 python -m phase4.src.run_linear --mode main --seeds 1 3 5 7 9 --device cuda:0 --out $R/linear > $R/linear_gpu1.log 2>&1 &
dk 0 python -m phase4.src.run_spatial --mode main --seeds 0 2 4 6 8 --device cuda:0 --out $R/spatial > $R/spatial_gpu0.log 2>&1 &
dk 1 python -m phase4.src.run_spatial --mode main --seeds 1 3 5 7 9 --device cuda:0 --out $R/spatial > $R/spatial_gpu1.log 2>&1 &
wait
echo "[$(date '+%F %T')] done"
