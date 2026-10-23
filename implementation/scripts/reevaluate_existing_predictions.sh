#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
ROOT="${ROOT:-artifacts/v78_final_phase_locked}"
BACKUP_DIR="${BACKUP_DIR:-/root/autodl-fs/TempoGraph-v78-FINAL-R2-backup}"
mkdir -p "$ROOT/reports" "$BACKUP_DIR"

for profile in v78_protected v78_balanced v78_rhythmic; do
  for seed in 1234 2025 3407; do
    run="$ROOT/val_candidates/$profile/seed_${seed}"
    predictions="$run/generated/tempograph_npy139"
    test -d "$predictions"
    python scripts/evaluate_lodge_protocol.py --predictions "$predictions" \
      --output "$run/reports/lodge_protocol_metrics.json"
    python scripts/check_sota_targets.py \
      --metrics "$run/reports/lodge_protocol_metrics.json" \
      --output "$run/reports/sota_target_check.json" || true
  done
done

SELECTION="$ROOT/reports/validation_selection.json"
if ! python scripts/select_complete_validation.py \
  --candidates-root "$ROOT/val_candidates" --output "$SELECTION"; then
  tar -czf "$BACKUP_DIR/TempoGraph_v78_FINAL_R2_VALIDATION_FAILED.tar.gz" "$ROOT"
  echo "Corrected duration-matched evaluation still did not pass all six gates." >&2
  exit 4
fi
tar -czf "$BACKUP_DIR/TempoGraph_v78_FINAL_R2_VALIDATION_PASSED.tar.gz" "$ROOT"
echo "Corrected duration-matched evaluation passed all six gates on every seed."
