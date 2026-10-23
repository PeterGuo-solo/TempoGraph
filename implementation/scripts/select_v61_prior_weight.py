from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def objective(metrics: dict[str, float]) -> float:
    """Validation-only multi-objective distance from the supplied LODGE targets."""
    div_k_ratio = metrics["div_k"] / max(metrics["div_k_gt"], 1e-8)
    div_g_ratio = metrics["div_g"] / max(metrics["div_g_gt"], 1e-8)
    return float(
        metrics["fid_k"] / 45.56
        + metrics["fid_g"] / 28.17
        + 2.0 * metrics["foot_skating_ratio"] / 0.0276
        + abs(math.log(max(div_k_ratio, 1e-8)))
        + abs(math.log(max(div_g_ratio, 1e-8)))
        - 2.0 * metrics["bas"] / 0.2397
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select V6.1 prior weight on validation data only")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-weight", action="store_true")
    args = parser.parse_args()
    candidates = []
    for generation_index in sorted(args.root.glob("weight_*/generated/tempograph/generation_index.json")):
        generation = json.loads(generation_index.read_text(encoding="utf-8"))
        if generation.get("split") != "val":
            raise ValueError(f"weight selection received non-validation output: {generation_index}")
        lodge_path = generation_index.parents[2] / "reports" / "lodge_protocol_metrics.json"
        lodge = json.loads(lodge_path.read_text(encoding="utf-8"))
        metrics = lodge["aggregate"]
        weight = float(generation["items"][0]["motion_prior"]["blend_weight"])
        candidates.append({"weight": weight, "objective": objective(metrics), "metrics": metrics})
    if not candidates:
        raise ValueError(f"no completed validation candidates found under {args.root}")
    chosen = min(candidates, key=lambda item: (item["objective"], item["weight"]))
    result = {
        "schema_version": 1,
        "selection_split": "val",
        "test_metrics_used": False,
        "chosen_weight": chosen["weight"],
        "objective": "normalized FID + 2x foot skating + diversity distance - 2x BAS",
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.print_weight:
        print(chosen["weight"])
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
