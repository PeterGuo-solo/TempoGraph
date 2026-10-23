#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${CONFIG:-configs/base.json}"
MAX_STEPS="${MAX_STEPS:-}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-backup}"
RUN_LODGE="${RUN_LODGE:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

mkdir -p artifacts/checkpoints artifacts/generated artifacts/reports

if [[ -z "$RESUME_CHECKPOINT" && "$ALLOW_EXISTING" != "1" ]]; then
  if [[ -f artifacts/checkpoints/tempograph.pt || -f artifacts/checkpoints/last.pt ]]; then
    echo "Existing training checkpoints found. Set RESUME_CHECKPOINT to an exact checkpoint, or ALLOW_EXISTING=1 for an intentional fresh overwrite." >&2
    exit 2
  fi
fi

python scripts/check_assets.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee artifacts/reports/gpu_inventory.txt

TRAIN_ARGS=(--config "$CONFIG" --stage all --device cuda)
if [[ -n "$MAX_STEPS" ]]; then
  TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
PYTHONUNBUFFERED=1 python scripts/train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee artifacts/reports/full_train.log

python scripts/infer.py \
  --checkpoint artifacts/checkpoints/tempograph.pt \
  --split test --device cuda \
  --output artifacts/generated/tempograph
python scripts/evaluate.py \
  --predictions artifacts/generated/tempograph \
  --split test --output artifacts/reports/tempograph_metrics.json

if [[ "$RUN_LODGE" == "1" ]]; then
  bash scripts/run_lodge_autodl.sh
else
  python scripts/evaluate_baselines.py \
    --predictions artifacts/generated/lodge \
    --output artifacts/reports/lodge_metrics.json || true
fi

python scripts/build_comparison.py

mkdir -p "$BACKUP_DIR"
cp -a artifacts/checkpoints "$BACKUP_DIR/"
cp -a artifacts/reports "$BACKUP_DIR/"
cp -a configs "$BACKUP_DIR/"
tar -czf "$BACKUP_DIR/TempoGraph_full_results.tar.gz" \
  artifacts/checkpoints artifacts/reports artifacts/generated configs
echo "Training, inference, evaluation, and backup completed: $BACKUP_DIR"
