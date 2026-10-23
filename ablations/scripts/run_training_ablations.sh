#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the existing FineDance directory containing manifest.json}"
: "${LODGE_ROOT:?Set LODGE_ROOT to the existing LODGE-main source directory}"
ARGS=(--data-root "$DATA_ROOT" --lodge-root "$LODGE_ROOT"
      --output "${OUTPUT_ROOT:-/root/autodl-fs/TempoGraph_ablation34_25k_results/seed_${TRAIN_SEED:-1234}}"
      --training-seed "${TRAIN_SEED:-1234}" --group "${GROUP:-all}")
if [[ "${RESUME:-0}" == "1" ]]; then ARGS+=(--resume); fi
python scripts/run_training_ablations.py "${ARGS[@]}" "$@"
