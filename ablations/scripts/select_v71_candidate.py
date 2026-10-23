from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a generation setting on validation metrics only"
    )
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = json.loads(
        (PROJECT / "configs" / "sota_targets.json").read_text(encoding="utf-8")
    )["lodge_protocol"]["targets"]
    candidates = []
    for path in sorted(args.candidates_root.glob("*/reports/lodge_protocol_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        violations = []
        checks = {}
        for name, target in targets.items():
            measured = float(metrics[name])
            threshold = float(target["value"])
            if target["direction"] == "lower":
                violation = max(0.0, measured / threshold - 1.0)
                passed = measured < threshold
            else:
                violation = max(0.0, threshold / max(measured, 1e-12) - 1.0)
                passed = measured > threshold
            violations.append(violation)
            checks[name] = passed
        candidates.append(
            {
                "name": path.parents[1].name,
                "metrics": metrics,
                "checks": checks,
                "all_targets_beaten": all(checks.values()),
                "normalized_target_violation": sum(violations),
            }
        )

    if not candidates:
        raise FileNotFoundError(f"No validation metrics under {args.candidates_root}")
    eligible = [item for item in candidates if item["all_targets_beaten"]]
    pool = eligible or candidates
    selected = min(pool, key=lambda item: item["normalized_target_violation"])
    result = {
        "selection_split": "val",
        "selected": selected["name"],
        "eligible_for_one_shot_test": bool(eligible),
        "candidates": candidates,
        "rule": "Test is run once only when a validation candidate beats all six declared LODGE-protocol targets.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(selected["name"])
    return 0 if eligible else 4


if __name__ == "__main__":
    raise SystemExit(main())
