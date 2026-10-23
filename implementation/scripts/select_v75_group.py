from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
METRICS = ("fid_k", "fid_g", "foot_skating_ratio", "div_k", "div_g", "bas")


def _check(metrics: dict[str, float], targets: dict[str, dict]) -> tuple[dict[str, bool], float]:
    checks: dict[str, bool] = {}
    violation = 0.0
    for name in METRICS:
        measured = float(metrics[name])
        threshold = float(targets[name]["value"])
        if targets[name]["direction"] == "lower":
            checks[name] = measured < threshold
            violation += max(0.0, measured / threshold - 1.0)
        else:
            checks[name] = measured > threshold
            violation += max(0.0, threshold / max(measured, 1e-12) - 1.0)
    return checks, violation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select one V7.5 profile only when every declared sampling seed passes all gates"
    )
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    args = parser.parse_args()

    targets = json.loads(
        (PROJECT / "configs" / "sota_targets.json").read_text(encoding="utf-8")
    )["lodge_protocol"]["targets"]
    groups = []
    for profile_dir in sorted(path for path in args.candidates_root.iterdir() if path.is_dir()):
        runs = []
        for report in sorted(profile_dir.glob("seed_*/reports/lodge_protocol_metrics.json")):
            metrics = json.loads(report.read_text(encoding="utf-8"))["aggregate"]
            checks, violation = _check(metrics, targets)
            runs.append(
                {
                    "sampling_seed": int(report.parents[1].name[len("seed_"):]),
                    "metrics": {name: float(metrics[name]) for name in METRICS},
                    "checks": checks,
                    "all_targets_beaten": all(checks.values()),
                    "normalized_target_violation": violation,
                }
            )
        if not runs:
            continue
        aggregate = {
            name: {
                "mean": statistics.fmean(run["metrics"][name] for run in runs),
                "sample_std": statistics.stdev(run["metrics"][name] for run in runs)
                if len(runs) > 1 else None,
            }
            for name in METRICS
        }
        complete = len(runs) == args.required_seeds
        eligible = complete and all(run["all_targets_beaten"] for run in runs)
        groups.append(
            {
                "profile": profile_dir.name,
                "complete_seed_group": complete,
                "eligible_for_one_shot_test": eligible,
                "runs": runs,
                "aggregate": aggregate,
                "total_normalized_violation": sum(
                    run["normalized_target_violation"] for run in runs
                ),
            }
        )

    if not groups:
        raise FileNotFoundError(f"No V7.5 validation metrics under {args.candidates_root}")
    eligible = [group for group in groups if group["eligible_for_one_shot_test"]]
    if eligible:
        # FID_g is the primary rescue objective; BAS breaks near-ties.  The
        # other four metrics are hard gates and cannot be traded away.
        selected = min(
            eligible,
            key=lambda group: (
                group["aggregate"]["fid_g"]["mean"],
                -group["aggregate"]["bas"]["mean"],
            ),
        )
    else:
        selected = min(groups, key=lambda group: group["total_normalized_violation"])
    result = {
        "selection_split": "val",
        "required_sampling_seeds": args.required_seeds,
        "selected_profile": selected["profile"],
        "eligible_for_one_shot_test": bool(eligible),
        "groups": groups,
        "rule": (
            "A profile is test-eligible only when every predeclared sampling seed "
            "beats all six LODGE-protocol targets. FID_g then ranks eligible profiles; "
            "BAS breaks ties. Test ground truth is never consulted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(selected["profile"])
    return 0 if eligible else 4


if __name__ == "__main__":
    raise SystemExit(main())
