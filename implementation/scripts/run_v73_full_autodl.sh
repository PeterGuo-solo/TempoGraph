#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to a compatible completed V5/V6/V7 tempograph.pt}"
SEED="${SEED:-1234}"
TRAIN_HOURS="${TRAIN_HOURS:-6.75}"
ROOT="${ROOT:-artifacts/v73_full/seed_${SEED}}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-v73-backup}"
CONFIG="${CONFIG:-configs/v73_full_train.json}"
RESUME="${RESUME:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

test -f "$BASE_CHECKPOINT"
if [[ -e "$ROOT" && "$RESUME" != "1" ]]; then
  echo "Run directory already exists: $ROOT" >&2
  echo "Use RESUME=1 only when continuing this exact V7.3 run." >&2
  exit 2
fi
mkdir -p "$ROOT/checkpoints" "$ROOT/reports" "$ROOT/val_candidates" "$BACKUP_DIR"

PYTHONPATH=src python -m pytest \
  tests/test_v7_objectives.py tests/test_motion_prior.py tests/test_v73_objectives.py -q
python scripts/prepare_finedance.py --seed 1234
python scripts/cache_lodge_beats.py
python scripts/check_assets.py

TRAIN_ARGS=(
  --config "$CONFIG" --stage joint_finetune --max-steps 90000
  --max-hours "$TRAIN_HOURS" --device cuda --seed "$SEED"
  --output-dir "$ROOT/checkpoints" --reports-dir "$ROOT/reports"
)
if [[ "$RESUME" == "1" && -f "$ROOT/checkpoints/last.pt" ]]; then
  TRAIN_ARGS+=(--resume-auto)
else
  TRAIN_ARGS+=(--init-checkpoint "$BASE_CHECKPOINT")
fi
PYTHONUNBUFFERED=1 python scripts/train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$ROOT/reports/full_train.log"

evaluate_candidate() {
  local split="$1"
  local name="$2"
  local profile="$3"
  local run="$ROOT/${split}_candidates/$name"
  if [[ "$split" == "test" ]]; then run="$ROOT/test"; fi
  mkdir -p "$run/generated/tempograph" "$run/generated/tempograph_npy139" "$run/reports"
  python scripts/infer.py --checkpoint "$ROOT/checkpoints/tempograph.pt" \
    --split "$split" --device cuda --seed "$SEED" --max-frames 1024 \
    --motion-prior-weight 0.90 --generation-profile "$profile" \
    --output "$run/generated/tempograph"
  python scripts/export_finedance139.py --input "$run/generated/tempograph" \
    --output "$run/generated/tempograph_npy139" --frames 1024
  python scripts/evaluate_lodge_protocol.py --predictions "$run/generated/tempograph_npy139" \
    --output "$run/reports/lodge_protocol_metrics.json"
  python scripts/check_sota_targets.py --metrics "$run/reports/lodge_protocol_metrics.json" \
    --output "$run/reports/sota_target_check.json" || true
}

# Three predeclared validation profiles expose the fidelity/rhythm trade-off.
# They never inspect test labels and all use the same single trained checkpoint.
evaluate_candidate val fidelity v73_fidelity
evaluate_candidate val balanced v73_balanced
evaluate_candidate val rhythmic v73_rhythmic

SELECTION="$ROOT/reports/validation_selection.json"
if ! python scripts/select_v71_candidate.py \
  --candidates-root "$ROOT/val_candidates" --output "$SELECTION"; then
  tar -czf "$BACKUP_DIR/TempoGraph_v73_VALIDATION_FAILED_seed_${SEED}.tar.gz" \
    "$ROOT" "$CONFIG"
  echo "Validation did not beat all six targets; untouched test was not run." >&2
  echo "Diagnostic result: $BACKUP_DIR/TempoGraph_v73_VALIDATION_FAILED_seed_${SEED}.tar.gz" >&2
  exit 4
fi

SELECTED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$SELECTION")"
case "$SELECTED" in
  fidelity) PROFILE=v73_fidelity ;;
  balanced) PROFILE=v73_balanced ;;
  rhythmic) PROFILE=v73_rhythmic ;;
  *) echo "Unknown selected candidate: $SELECTED" >&2; exit 5 ;;
esac

# One untouched-test evaluation after validation-only selection.
evaluate_candidate test "$SELECTED" "$PROFILE"
python scripts/evaluate.py --predictions "$ROOT/test/generated/tempograph" --split test \
  --output "$ROOT/test/reports/tempograph_metrics.json"
python scripts/quality_gate.py --profile strict \
  --metrics "$ROOT/test/reports/tempograph_metrics.json" \
  --lodge-metrics "$ROOT/test/reports/lodge_protocol_metrics.json" \
  --output "$ROOT/test/reports/quality_gate_strict.json" || true
tar -czf "$BACKUP_DIR/TempoGraph_v73_full_seed_${SEED}.tar.gz" "$ROOT" "$CONFIG"
echo "V7.3 result: $BACKUP_DIR/TempoGraph_v73_full_seed_${SEED}.tar.gz"
