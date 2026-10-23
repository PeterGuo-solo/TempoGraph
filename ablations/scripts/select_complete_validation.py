from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
METRICS = ("fid_k", "fid_g", "foot_skating_ratio", "div_k", "div_g", "bas")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank only complete three-seed V7.8 profiles")
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, nargs="+", default=[1234, 2025, 3407])
    args = parser.parse_args()
    targets = json.loads((PROJECT / "configs" / "sota_targets.json").read_text())["lodge_protocol"]["targets"]
    groups = []
    for profile in sorted(path for path in args.candidates_root.iterdir() if path.is_dir()):
        runs = []
        for seed in args.required_seeds:
            path = profile / f"seed_{seed}" / "reports" / "lodge_protocol_metrics.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text())["aggregate"]
            checks = {}
            violation = 0.0
            for name in METRICS:
                measured, threshold = float(metrics[name]), float(targets[name]["value"])
                if targets[name]["direction"] == "lower":
                    checks[name] = measured < threshold
                    violation += max(0.0, measured / threshold - 1.0)
                else:
                    checks[name] = measured > threshold
                    violation += max(0.0, threshold / max(measured, 1e-12) - 1.0)
            runs.append({"seed": seed, "metrics": metrics, "checks": checks,
                         "all_targets_beaten": all(checks.values()), "violation": violation})
        complete = len(runs) == len(args.required_seeds)
        eligible = complete and all(run["all_targets_beaten"] for run in runs)
        groups.append({
            "profile": profile.name, "complete": complete, "eligible": eligible,
            "runs": runs,
            "mean": {name: statistics.fmean(float(run["metrics"][name]) for run in runs)
                     for name in METRICS} if complete else None,
            "total_violation": sum(run["violation"] for run in runs) if complete else None,
        })
    complete_groups = [group for group in groups if group["complete"]]
    if not complete_groups:
        raise RuntimeError("No profile has all required seeds")
    eligible = [group for group in complete_groups if group["eligible"]]
    selected = min(
        eligible or complete_groups,
        key=lambda group: (
            0.0 if group["eligible"] else group["total_violation"],
            group["mean"]["fid_g"], -group["mean"]["bas"],
        ),
    )
    result = {
        "selection_split": "val", "required_seeds": args.required_seeds,
        "selected_profile": selected["profile"], "all_six_pass": bool(eligible),
        "groups": groups,
        "rule": "Only complete three-seed groups are ranked; all six metrics must pass on every seed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(selected["profile"])
    return 0 if eligible else 4


if __name__ == "__main__":
    raise SystemExit(main())
