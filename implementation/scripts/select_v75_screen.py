from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PROTECTED = ("fid_k", "foot_skating_ratio", "div_k", "div_g")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast-screen V7.5 profiles on one predeclared validation sampling seed"
    )
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = json.loads(
        (PROJECT / "configs" / "sota_targets.json").read_text(encoding="utf-8")
    )["lodge_protocol"]["targets"]
    candidates = []
    for profile in sorted(path for path in args.candidates_root.iterdir() if path.is_dir()):
        report = (
            profile / f"seed_{args.sampling_seed}" / "reports"
            / "lodge_protocol_metrics.json"
        )
        if not report.is_file():
            continue
        metrics = json.loads(report.read_text(encoding="utf-8"))["aggregate"]
        checks = {}
        violations = {}
        for name, target in targets.items():
            measured, threshold = float(metrics[name]), float(target["value"])
            if target["direction"] == "lower":
                checks[name] = measured < threshold
                violations[name] = max(0.0, measured / threshold - 1.0)
            else:
                checks[name] = measured > threshold
                violations[name] = max(0.0, threshold / max(measured, 1e-12) - 1.0)
        candidates.append(
            {
                "profile": profile.name,
                "metrics": metrics,
                "checks": checks,
                "protected_metrics_pass": all(checks[name] for name in PROTECTED),
                "target_violation": sum(violations.values()),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No seed {args.sampling_seed} metrics under {args.candidates_root}")

    protected = [item for item in candidates if item["protected_metrics_pass"]]
    pool = protected or candidates
    # Preserve the four already-won metrics first.  Within that feasible set,
    # minimize remaining normalized six-metric violation; use FID_g and BAS as
    # deterministic tie breakers for the rescue objectives.
    selected = min(
        pool,
        key=lambda item: (
            item["target_violation"],
            float(item["metrics"]["fid_g"]),
            -float(item["metrics"]["bas"]),
        ),
    )
    result = {
        "selection_split": "val",
        "screen_sampling_seed": args.sampling_seed,
        "selected_profile": selected["profile"],
        "protected_pool_available": bool(protected),
        "candidates": candidates,
        "rule": (
            "Screen on one predeclared seed. Preserve FID_k, foot skating, Div_k and "
            "Div_g first; then minimize normalized six-target violation. The selected "
            "profile must still pass the independent three-seed final gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(selected["profile"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
