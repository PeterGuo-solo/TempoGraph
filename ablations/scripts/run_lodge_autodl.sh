#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_DIR="$PROJECT_DIR"
while [[ ! -d "$REPOSITORY_DIR/reference_code/baselines/LODGE-main" ]]; do
  [[ "$REPOSITORY_DIR" != "/" ]] || { echo "LODGE source not found" >&2; exit 2; }
  REPOSITORY_DIR="$(dirname "$REPOSITORY_DIR")"
done
ENV_NAME="${LODGE_ENV_NAME:-lodge-tempograph}"
cd "$PROJECT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required for the official pinned LODGE environment." >&2
  exit 2
fi

if ! conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  SANITIZED_ENV="$(mktemp --suffix=.yaml)"
  trap 'rm -f "$SANITIZED_ENV"' EXIT
  grep -v '^prefix:' "$REPOSITORY_DIR/reference_code/baselines/LODGE-main/lodge.yaml" > "$SANITIZED_ENV"
  conda env create -n "$ENV_NAME" -f "$SANITIZED_ENV"
fi

conda run -n "$ENV_NAME" python scripts/evaluate_baselines.py \
  --baseline lodge \
  --checkpoint artifacts/baselines/lodge \
  --predictions artifacts/generated/lodge \
  --manifest data/finedance/manifest.json \
  --split test --run-lodge \
  --output artifacts/reports/lodge_metrics.json

python scripts/build_comparison.py

echo "LODGE inference and unified evaluation completed."
