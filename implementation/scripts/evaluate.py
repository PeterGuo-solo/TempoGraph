from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.benchmarking import evaluate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated motion on the FineDance manifest")
    parser.add_argument("--predictions", type=Path, default=PROJECT / "artifacts" / "generated" / "tempograph")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "finedance" / "manifest.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts" / "reports" / "tempograph_metrics.json")
    parser.add_argument(
        "--format", default="canonical_npz",
        choices=["canonical_npz", "npy139", "npy151", "npy319"],
    )
    args = parser.parse_args()
    report = evaluate_predictions(args.predictions, args.manifest, args.split, args.format)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sequences"}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
