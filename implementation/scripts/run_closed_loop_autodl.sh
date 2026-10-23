#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

V77_CHECKPOINT="${V77_CHECKPOINT:?Set V77_CHECKPOINT to the completed V7.7 15000-step tempograph.pt}"
ROOT="${ROOT:-artifacts/v78_final_phase_locked}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-v78-FINAL-backup}"
mkdir -p "$ROOT/val_candidates" "$ROOT/reports" "$BACKUP_DIR"
test -f "$V77_CHECKPOINT"

PYTHONPATH=src python -m pytest tests/test_v77_official_bas.py tests/test_closed_loop.py -q
python scripts/prepare_finedance.py --seed 1234
python scripts/cache_lodge_beats.py
python scripts/check_assets.py

evaluate_candidate() {
  local profile="$1" prior_weight="$2" sampling_seed="$3"
  local run="$ROOT/val_candidates/$profile/seed_${sampling_seed}"
  mkdir -p "$run/generated/tempograph" "$run/generated/tempograph_npy139" "$run/reports"
  python scripts/infer.py --checkpoint "$V77_CHECKPOINT" \
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

# Every profile is evaluated on every predeclared seed.  Incomplete groups can
# never win, fixing the V7.7 selection/reporting defect.
for seed in 1234 2025 3407; do
  evaluate_candidate v78_protected 0.95 "$seed"
  evaluate_candidate v78_balanced  0.93 "$seed"
  evaluate_candidate v78_rhythmic  0.90 "$seed"
done

SELECTION="$ROOT/reports/validation_selection.json"
if ! python scripts/select_complete_validation.py \
  --candidates-root "$ROOT/val_candidates" --output "$SELECTION"; then
  tar -czf "$BACKUP_DIR/TempoGraph_v78_FINAL_VALIDATION_FAILED.tar.gz" "$ROOT"
  echo "No complete three-seed profile passed all six validation gates." >&2
  exit 4
fi
tar -czf "$BACKUP_DIR/TempoGraph_v78_FINAL_VALIDATION_PASSED.tar.gz" "$ROOT"
echo "All six gates passed on every seed. Test split remains untouched."
