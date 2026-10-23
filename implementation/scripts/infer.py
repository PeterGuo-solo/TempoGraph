from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tempograph.generation import generate_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate motion from FineDance music with a TempoGraph checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "artifacts" / "checkpoints" / "tempograph.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "finedance" / "manifest.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts" / "generated" / "tempograph")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--motion-prior-weight", type=float,
        help="V6 training-split motion-prior weight; use 0 to disable",
    )
    parser.add_argument(
        "--generation-profile", default="checkpoint",
        choices=[
            "checkpoint", "v62_sota", "v63_balanced", "v63_beat", "v7_sota",
            "v72_safe", "v72_balanced", "v72_diverse",
            "v73_fidelity", "v73_balanced", "v73_rhythmic",
            "v74_quality", "v74_balanced", "v74_rhythmic",
            "v75_quality", "v75_balanced", "v75_rhythmic",
            "v76_protected", "v76_balanced", "v76_rhythmic",
            "v77_protected", "v77_balanced", "v77_rhythmic",
            "v78_protected", "v78_balanced", "v78_rhythmic",
        ],
        help="validated deterministic generation corrections",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    report = generate_split(
        args.checkpoint, args.manifest, args.split, args.output, args.device,
        args.max_sequences, args.max_frames, args.seed, args.motion_prior_weight,
        args.generation_profile,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
