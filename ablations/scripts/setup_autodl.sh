#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python --version
nvidia-smi
python -m pip install --upgrade pip
python -m pip install -e ".[train,audio,body,dev]"
mkdir -p artifacts/checkpoints artifacts/generated artifacts/reports
python scripts/prepare_finedance.py
PYTHONPATH=src python -m pytest -q
python scripts/check_assets.py

python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; do not start paid training")
PY

echo "AutoDL setup and asset verification completed."
