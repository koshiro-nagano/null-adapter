#!/usr/bin/env bash
# Deviation D1 (phase4/results/DEVIATIONS.md): Part A test run after Part B, alone on each GPU, because running
# both parts on one GPU ran out of memory before any Part A fit.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=phase4/results
dk() { gpu=$1; shift; docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp --gpus "device=$gpu" --ipc=host \
  -v "$PWD":/work -w /work --cpus=8 null-adapter:phase2 "$@"; }
while pgrep -f phase4/scripts/pipeline.sh >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] linear main"
dk 0 python -m phase4.src.run_linear --mode main --seeds 0 2 4 6 8 --device cuda:0 --out $R/linear > $R/linear_gpu0.log 2>&1 &
dk 1 python -m phase4.src.run_linear --mode main --seeds 1 3 5 7 9 --device cuda:0 --out $R/linear > $R/linear_gpu1.log 2>&1 &
wait
echo "[$(date '+%F %T')] done"
