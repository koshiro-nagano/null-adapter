#!/usr/bin/env bash
# Phase 3 run order (PREREGISTRATION.md): S_K rule on val (S0 only) -> val sanity -> test, seeds split over 2 GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=phase3/results
dk() { gpu=$1; shift; docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp --gpus "device=$gpu" --ipc=host \
  -v "$PWD":/work -w /work --cpus=8 null-adapter:phase2 "$@"; }
echo "[$(date '+%F %T')] select"
dk 0 python -m phase3.src.run --mode select --seeds 0 2 --device cuda:0 --out $R/select > $R/select_gpu0.log 2>&1 &
dk 1 python -m phase3.src.run --mode select --seeds 1 --device cuda:0 --out $R/select > $R/select_gpu1.log 2>&1 &
wait
dk 0 python -m phase3.src.run --mode select --choose --out $R/select
echo "[$(date '+%F %T')] sanity"
dk 0 python -m phase3.src.run --mode sanity --seeds 0 --device cuda:0 --out $R/val_sanity > $R/val_sanity.log 2>&1
dk 0 python -m phase3.scripts.check_sanity $R/val_sanity
echo "[$(date '+%F %T')] main"
dk 0 python -m phase3.src.run --mode main --seeds 0 2 4 6 8 --device cuda:0 --out $R/main > $R/main_gpu0.log 2>&1 &
dk 1 python -m phase3.src.run --mode main --seeds 1 3 5 7 9 --device cuda:0 --out $R/main > $R/main_gpu1.log 2>&1 &
wait
echo "[$(date '+%F %T')] done"
