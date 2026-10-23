from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.data.adapters import load_motion_npz
from tempograph.rendering import render_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_motion(load_motion_npz(args.motion), args.output)
