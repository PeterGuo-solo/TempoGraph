#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the V7.3 checkpoints/tempograph.pt}"
SEED="${SEED:-1234}"
ROOT="${ROOT:-artifacts/v74_12k/seed_${SEED}}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-v74-backup}"
CONFIG="${CONFIG:-configs/v74_12k_train.json}"
RESUME="${RESUME:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

test -f "$BASE_CHECKPOINT"
if [[ -e "$ROOT" && "$RESUME" != "1" ]]; then
  echo "Run directory already exists: $ROOT" >&2
  echo "Set a new ROOT, or use RESUME=1 only for this exact V7.4 run." >&2
  exit 2
fi
mkdir -p "$ROOT/checkpoints" "$ROOT/reports" "$ROOT/val_candidates" "$BACKUP_DIR"

PYTHONPATH=src python -m pytest \
  tests/test_v7_objectives.py tests/test_motion_prior.py tests/test_v73_objectives.py \
  tests/test_v74_short_tune.py -q
python scripts/prepare_finedance.py --seed 1234
python scripts/cache_lodge_beats.py
python scripts/check_assets.py

TRAIN_ARGS=(
  --config "$CONFIG" --stage joint_finetune --max-steps 12000
  --device cuda --seed "$SEED"
  --output-dir "$ROOT/checkpoints" --reports-dir "$ROOT/reports"
)
if [[ "$RESUME" == "1" && -f "$ROOT/checkpoints/last.pt" ]]; then
  TRAIN_ARGS+=(--resume-auto)
else
  TRAIN_ARGS+=(--init-checkpoint "$BASE_CHECKPOINT")
fi
PYTHONUNBUFFERED=1 python scripts/train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$ROOT/reports/train_12k.log"

evaluate_candidate() {
  local name="$1"
  local profile="$2"
  local prior_weight="$3"
  local run="$ROOT/val_candidates/$name"
  mkdir -p "$run/generated/tempograph" "$run/generated/tempograph_npy139" "$run/reports"
  python scripts/infer.py --checkpoint "$ROOT/checkpoints/tempograph.pt" \
    --split val --device cuda --seed "$SEED" --max-frames 1024 \
    --motion-prior-weight "$prior_weight" --generation-profile "$profile" \
    --output "$run/generated/tempograph"
  python scripts/export_finedance139.py --input "$run/generated/tempograph" \
    --output "$run/generated/tempograph_npy139" --frames 1024
  python scripts/evaluate_lodge_protocol.py --predictions "$run/generated/tempograph_npy139" \
    --output "$run/reports/lodge_protocol_metrics.json"
  python scripts/check_sota_targets.py --metrics "$run/reports/lodge_protocol_metrics.json" \
    --output "$run/reports/sota_target_check.json" || true
}

# High prior weight is intentional: the earlier V6.2 validation showed that
# 0.95 strongly reduced FID_g.  Profiles and weights are fixed before reading
# V7.4 validation metrics; the untouched test split is not opened here.
evaluate_candidate quality  v74_quality  0.95
evaluate_candidate balanced v74_balanced 0.94
evaluate_candidate rhythmic v74_rhythmic 0.92

SELECTION="$ROOT/reports/validation_selection.json"
if ! python scripts/select_v71_candidate.py \
  --candidates-root "$ROOT/val_candidates" --output "$SELECTION"; then
  tar -czf "$BACKUP_DIR/TempoGraph_v74_12k_VALIDATION_FAILED_seed_${SEED}.tar.gz" \
    "$ROOT" "$CONFIG"
  echo "Validation did not beat all six targets; untouched test was not run." >&2
  echo "Diagnostic result: $BACKUP_DIR/TempoGraph_v74_12k_VALIDATION_FAILED_seed_${SEED}.tar.gz" >&2
  exit 4
fi

# Intentionally stop after validation.  Run the one-shot test only after the
# validation gate passes and after freezing the selected profile.
tar -czf "$BACKUP_DIR/TempoGraph_v74_12k_VALIDATION_PASSED_seed_${SEED}.tar.gz" \
  "$ROOT" "$CONFIG"
echo "Validation passed: $BACKUP_DIR/TempoGraph_v74_12k_VALIDATION_PASSED_seed_${SEED}.tar.gz"
