from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.training import inspect_training_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TempoGraph data, models, and runtime")
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--allow-cpu", action="store_true", help="do not require CUDA")
    parser.add_argument("--no-fail", action="store_true", help="always exit successfully")
    args = parser.parse_args()
    report = inspect_training_assets(args.root, require_cuda=not args.allow_cpu)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_training"] or args.no_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
