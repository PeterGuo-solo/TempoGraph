from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STRICT_LIMITS = {
    "contact_balanced_accuracy": (0.55, 1.0),
    "contact_macro_f1": (0.55, 1.0),
    "contact_positive_rate": (0.40, 0.995),
    "diversity_ratio": (0.50, 1.50),
    "energy_ratio": (0.50, 1.50),
    "root_velocity_ratio": (0.50, 1.50),
}

# Exploratory continuation limits requested for V5.1.  They decide whether a
# second seed may run; they must not be presented as paper-quality thresholds.
EXPLORATORY_LIMITS = {
    "contact_balanced_accuracy": (0.35, 1.0),
    "contact_macro_f1": (0.35, 1.0),
    "contact_positive_rate": (0.25, 0.999),
    "diversity_ratio": (0.25, 3.00),
    "energy_ratio": (0.25, 3.00),
    "root_velocity_ratio": (0.25, 3.00),
}


def evaluate_gate(
    report: dict[str, Any],
    minimum_samples: int = 19,
    lodge_report: dict[str, Any] | None = None,
    profile: str = "strict",
) -> dict[str, Any]:
    if profile not in {"strict", "exploratory"}:
        raise ValueError(f"unknown quality-gate profile: {profile}")
    limits = STRICT_LIMITS if profile == "strict" else EXPLORATORY_LIMITS
    aggregate = report.get("aggregate", {})
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "value": value, "requirement": requirement}
        )

    add("status", report.get("status") == "complete", report.get("status"), "complete")
    sample_count = int(report.get("sample_count", 0))
    add("sample_count", sample_count >= minimum_samples, sample_count, f">= {minimum_samples}")
    for name, (lower, upper) in limits.items():
        value = aggregate.get(name)
        passed = value is not None and lower <= float(value) <= upper
        add(name, passed, value, f"{lower} <= value <= {upper}")

    predicted_rate = aggregate.get("contact_positive_rate")
    reference_rate = aggregate.get("reference_contact_positive_rate")
    gap = None
    if predicted_rate is not None and reference_rate is not None:
        gap = abs(float(predicted_rate) - float(reference_rate))
    rate_gap_limit = 0.10 if profile == "strict" else 0.20
    add(
        "contact_rate_gap", gap is not None and gap <= rate_gap_limit,
        gap, f"<= {rate_gap_limit}",
    )

    if lodge_report is not None:
        lodge = lodge_report.get("aggregate", {})
        add(
            "lodge_status", lodge_report.get("status") == "complete",
            lodge_report.get("status"), "complete",
        )
        foot_skating = lodge.get("foot_skating_ratio")
        foot_upper = 0.40 if profile == "strict" else 0.90
        add(
            "lodge_foot_skating_ratio",
            foot_skating is not None and 0.0 <= float(foot_skating) <= foot_upper,
            foot_skating, f"0.0 <= value <= {foot_upper}",
        )
        diversity_lower, diversity_upper = ((0.50, 1.50) if profile == "strict" else (0.25, 8.00))
        for name in ("div_k", "div_g"):
            value, reference = lodge.get(name), lodge.get(f"{name}_gt")
            ratio = None
            if value is not None and reference is not None and abs(float(reference)) > 1e-12:
                ratio = float(value) / float(reference)
            add(
                f"lodge_{name}_ratio",
                ratio is not None and diversity_lower <= ratio <= diversity_upper,
                ratio, f"{diversity_lower} <= generated/GT <= {diversity_upper}",
            )
        bas = lodge.get("bas")
        bas_lower = 0.15 if profile == "strict" else 0.08
        add("lodge_bas", bas is not None and float(bas) >= bas_lower, bas, f">= {bas_lower}")

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 2,
        "passed": passed,
        "profile": profile,
        "purpose": (
            "fail-fast protection against the previously observed all-contact, "
            "abnormal-energy, foot-skating and diversity failures; "
            "not a claim of paper superiority; exploratory limits only control "
            "whether another seed may run"
        ),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject a collapsed TempoGraph result")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lodge-metrics", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=19)
    parser.add_argument("--profile", choices=["strict", "exploratory"], default="strict")
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    lodge_report = (
        json.loads(args.lodge_metrics.read_text(encoding="utf-8"))
        if args.lodge_metrics else None
    )
    result = evaluate_gate(report, args.minimum_samples, lodge_report, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
