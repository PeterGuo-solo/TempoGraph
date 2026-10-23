from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.lodge_protocol import evaluate_lodge_protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TempoGraph outputs with released LODGE/FineDance metric definitions"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data" / "finedance")
    parser.add_argument(
        "--lodge-root", type=Path,
        default=next(p for p in PROJECT.parents if (p / "reference_code" / "baselines" / "LODGE-main").is_dir()) / "reference_code" / "baselines" / "LODGE-main",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="diagnostic only: omit BAS when librosa/audio is unavailable",
    )
    args = parser.parse_args()
    report = evaluate_lodge_protocol(
        args.predictions, args.data_root, args.lodge_root,
        compute_audio=not args.skip_audio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "ids"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
