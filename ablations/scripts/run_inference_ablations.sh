#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CHECKPOINT:?Set CHECKPOINT to the original matched final checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the FineDance directory containing manifest.json}"
: "${LODGE_ROOT:?Set LODGE_ROOT to the official LODGE source directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
test -f "$CHECKPOINT"
test -f "$DATA_ROOT/manifest.json"
test ! -e "$OUTPUT_ROOT"
for group in full no_relation_message linear_boundary; do
  run="$OUTPUT_ROOT/$group"
  python scripts/infer.py --checkpoint "$CHECKPOINT" --manifest "$DATA_ROOT/manifest.json" \
    --split val --device cuda --seed "${SAMPLING_SEED:-3407}" --max-frames 1024 \
    --motion-prior-weight 0.90 --generation-profile v78_rhythmic --ablation "$group" --output "$run/npz"
  python scripts/export_finedance139.py --input "$run/npz" --output "$run/npy139" --frames 1024
  python scripts/evaluate_lodge_protocol.py --predictions "$run/npy139" --data-root "$DATA_ROOT" \
    --lodge-root "$LODGE_ROOT" --output "$run/metrics.json"
done
