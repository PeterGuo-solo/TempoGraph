#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the frozen V7.3 checkpoints/tempograph.pt}"
SEED="${SEED:-1234}"
ROOT="${ROOT:-artifacts/v75_constrained/seed_${SEED}}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-v75-backup}"
CONFIG="${CONFIG:-configs/v75_constrained_train.json}"
RESUME="${RESUME:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

test -f "$BASE_CHECKPOINT"
if [[ -e "$ROOT" && "$RESUME" != "1" ]]; then
  echo "Run directory already exists: $ROOT" >&2
  echo "Use a new ROOT, or RESUME=1 only for this exact run." >&2
  exit 2
fi
mkdir -p "$ROOT/checkpoints" "$ROOT/reports" "$ROOT/val_candidates" "$BACKUP_DIR"

PYTHONPATH=src python -m pytest \
  tests/test_v7_objectives.py tests/test_motion_prior.py tests/test_v73_objectives.py \
  tests/test_v74_short_tune.py tests/test_v75_constrained.py -q
python scripts/prepare_finedance.py --seed 1234
python scripts/cache_lodge_beats.py
python scripts/check_assets.py

TRAIN_ARGS=(
  --config "$CONFIG" --stage joint_finetune --max-steps 15000
  --device cuda --seed "$SEED"
  --output-dir "$ROOT/checkpoints" --reports-dir "$ROOT/reports"
)
if [[ "$RESUME" == "1" && -f "$ROOT/checkpoints/last.pt" ]]; then
  TRAIN_ARGS+=(--resume-auto)
else
  TRAIN_ARGS+=(--init-checkpoint "$BASE_CHECKPOINT")
fi
PYTHONUNBUFFERED=1 python scripts/train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$ROOT/reports/train_15k.log"

evaluate_candidate() {
  local profile="$1"
  local prior_weight="$2"
  local sampling_seed="$3"
  local run="$ROOT/val_candidates/$profile/seed_${sampling_seed}"
  mkdir -p "$run/generated/tempograph" "$run/generated/tempograph_npy139" "$run/reports"
  python scripts/infer.py --checkpoint "$ROOT/checkpoints/tempograph.pt" \
    --split val --device cuda --seed "$sampling_seed" --max-frames 1024 \
    --motion-prior-weight "$prior_weight" --generation-profile "$profile" \
    --output "$run/generated/tempograph"
  python scripts/export_finedance139.py --input "$run/generated/tempograph" \
    --output "$run/generated/tempograph_npy139" --frames 1024
  python scripts/evaluate_lodge_protocol.py --predictions "$run/generated/tempograph_npy139" \
    --output "$run/reports/lodge_protocol_metrics.json"
  python scripts/check_sota_targets.py --metrics "$run/reports/lodge_protocol_metrics.json" \
    --output "$run/reports/sota_target_check.json" || true
}

# Fast two-stage validation: screen all three predeclared profiles on seed
# 1234, then spend the remaining validation budget only on the selected
# profile.  The final gate still requires three independent sampling seeds.
evaluate_candidate v75_quality  0.97 1234
evaluate_candidate v75_balanced 0.95 1234
evaluate_candidate v75_rhythmic 0.93 1234

SCREEN_SELECTION="$ROOT/reports/screen_selection.json"
SELECTED_PROFILE="$(python scripts/select_v75_screen.py \
  --candidates-root "$ROOT/val_candidates" --sampling-seed 1234 \
  --output "$SCREEN_SELECTION")"
case "$SELECTED_PROFILE" in
  v75_quality)  SELECTED_PRIOR=0.97 ;;
  v75_balanced) SELECTED_PRIOR=0.95 ;;
  v75_rhythmic) SELECTED_PRIOR=0.93 ;;
  *) echo "Unexpected selected profile: $SELECTED_PROFILE" >&2; exit 3 ;;
esac
evaluate_candidate "$SELECTED_PROFILE" "$SELECTED_PRIOR" 2025
evaluate_candidate "$SELECTED_PROFILE" "$SELECTED_PRIOR" 3407

SELECTION="$ROOT/reports/validation_selection.json"
if ! python scripts/select_v75_group.py \
  --candidates-root "$ROOT/val_candidates" --required-seeds 3 --output "$SELECTION"; then
  tar -czf "$BACKUP_DIR/TempoGraph_v75_VALIDATION_FAILED_seed_${SEED}.tar.gz" \
    "$ROOT" "$CONFIG"
  echo "No profile passed all six targets on all three sampling seeds." >&2
  echo "Untouched test split was not opened." >&2
  exit 4
fi

# Stop after validation.  The selected profile and prior weight must be frozen
# before a separately authorized one-shot test run.
tar -czf "$BACKUP_DIR/TempoGraph_v75_VALIDATION_PASSED_seed_${SEED}.tar.gz" \
  "$ROOT" "$CONFIG"
echo "Validation passed on all seeds: $BACKUP_DIR/TempoGraph_v75_VALIDATION_PASSED_seed_${SEED}.tar.gz"
