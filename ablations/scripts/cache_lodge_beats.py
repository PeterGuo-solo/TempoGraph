from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.data.dataset import read_manifest
from tempograph.lodge_protocol import _music_beats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache the exact WAV/librosa beats used by the LODGE BAS metric"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT / "data" / "finedance" / "manifest.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = manifest.parent / "lodge_beats"
    output.mkdir(parents=True, exist_ok=True)
    written = reused = 0
    counts: dict[str, int] = {}
    for record in read_manifest(manifest):
        destination = output / f"{record.name}.npy"
        if destination.is_file() and not args.force:
            beats = np.asarray(np.load(destination, allow_pickle=False), dtype=np.int64)
            reused += 1
        else:
            if not record.audio_path:
                raise ValueError(f"record {record.name} has no audio_path")
            beats = _music_beats(manifest.parent / record.audio_path)
            np.save(destination, beats.astype(np.int64), allow_pickle=False)
            written += 1
        if len(beats) < 2:
            raise ValueError(f"fewer than two official beats for {record.name}")
        counts[record.name] = int(len(beats))
    report = {
        "detector": "LODGE librosa WAV beat tracker",
        "manifest": str(manifest),
        "written": written,
        "reused": reused,
        "sequences": len(counts),
        "minimum_beats": min(counts.values()),
        "maximum_beats": max(counts.values()),
    }
    (output / "cache_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
