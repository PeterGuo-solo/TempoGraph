from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.data.adapters import (
    convert_joint_count,
    motion_from_axis_angle,
    motion_from_edge151,
    motion_from_finedance319,
    motion_from_lodge139,
    save_motion_npz,
)
from tempograph.data.dataset import DatasetRecord, write_manifest


def convert(array: np.ndarray, source_format: str, fps: float):
    if source_format == "lodge139":
        return motion_from_lodge139(array, fps)
    if source_format == "edge151":
        return motion_from_edge151(array, fps)
    if source_format == "finedance319":
        return motion_from_finedance319(array, fps)
    if source_format == "axis22":
        return motion_from_axis_angle(array, 22, fps)
    if source_format == "axis52":
        return motion_from_axis_angle(array, 52, fps)
    raise ValueError(source_format)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert reference datasets to TempoGraph NPZ")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--format",
        required=True,
        choices=["lodge139", "edge151", "finedance319", "axis22", "axis52"],
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--target-joints", type=int, default=22, choices=[22, 52])
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-dataset", default="unknown")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(args.input.rglob("*.npy")):
        motion = convert_joint_count(convert(np.load(source), args.format, args.fps), args.target_joints)
        target = args.output / f"{source.stem}.npz"
        save_motion_npz(target, motion)
        records.append(
            DatasetRecord(
                name=source.stem,
                split=args.split,
                motion_path=target.name,
                source_dataset=args.source_dataset,
            )
        )
    write_manifest(args.output / "manifest.json", records)
    print(f"Converted {len(records)} sequences to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
