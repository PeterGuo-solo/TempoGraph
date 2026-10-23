from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.data.finedance import build_finedance_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the aligned FineDance@Genre manifest")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data" / "finedance")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    summary = build_finedance_manifest(
        args.data_root, args.manifest, args.validation_fraction, args.seed
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
