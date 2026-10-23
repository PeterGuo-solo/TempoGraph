"""Export canonical TempoGraph NPZ predictions as FineDance-compatible NPY139."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.data.adapters import load_motion_npz


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TempoGraph canonical NPZ files to [T,139] NPY files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "artifacts" / "generated" / "tempograph",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "artifacts" / "generated" / "tempograph_npy139",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1024,
        help="crop every sequence to this many frames; use 0 to preserve full length",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(args.input.glob("*.npz")):
        features = load_motion_npz(source).to_features().astype(np.float32, copy=False)
        if args.frames > 0:
            if len(features) < args.frames:
                continue
            features = features[: args.frames]
        destination = args.output / f"{source.stem}.npy"
        np.save(destination, features, allow_pickle=False)
        written.append({"id": source.stem, "frames": len(features), "path": destination.name})
    report = {
        "source": str(args.input.resolve()),
        "destination": str(args.output.resolve()),
        "format": "FineDance canonical 139D",
        "requested_frames": args.frames,
        "sample_count": len(written),
        "items": written,
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
