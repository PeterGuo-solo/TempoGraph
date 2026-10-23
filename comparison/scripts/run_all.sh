#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/logs
python scripts/prepare.py
python scripts/preflight.py
for method in tempograph lodge megadance; do python scripts/run_method.py "$method"; done
python scripts/evaluate.py
python scripts/render_comparison.py
# Download this archive after completion; inputs and provenance accompany outputs.
tar -czf TempoGraph_comparison_results.tar.gz work environment_actual.txt config
