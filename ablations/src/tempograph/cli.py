"""Small dependency-light command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.adapters import save_motion_npz
from .evaluation import boundary_continuity_metrics, evaluate_sequence
from .pipeline import TempoGraphPipeline, graph_to_dict, plan_to_dict
from .synthetic import synthetic_motion, synthetic_music
from .training import inspect_training_assets


def run_smoke(output: Path | None = None) -> dict[str, object]:
    motion = synthetic_motion()
    music = synthetic_music()
    pipeline = TempoGraphPipeline()
    result = pipeline.fill_gap(motion, music, 75, 105)
    left = motion.slice(60, 75)
    right = motion.slice(105, 120)
    report: dict[str, object] = {
        "graph": {"nodes": len(result.graph.nodes), "edges": len(result.graph.edges)},
        "plan": plan_to_dict(result.plan),
        "boundary": boundary_continuity_metrics(left, result.transition, right),
        "sequence": evaluate_sequence(result.motion, music),
        "output_frames": result.motion.frames,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        save_motion_npz(output / "smoke_motion.npz", result.motion)
        (output / "smoke_graph.json").write_text(
            json.dumps(graph_to_dict(result.graph), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output / "smoke_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tempograph")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the synthetic end-to-end pipeline")
    smoke.add_argument("--output", type=Path)
    assets = subparsers.add_parser("assets", help="inspect real-training assets")
    assets.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(run_smoke(args.output), indent=2, ensure_ascii=False))
    elif args.command == "assets":
        print(json.dumps(inspect_training_assets(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

