from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _passes(value: float | None, rule: dict) -> bool:
    if value is None or not math.isfinite(float(value)):
        return False
    value = float(value)
    if rule["direction"] == "lower":
        return value <= float(rule.get("target", rule.get("limit")))
    if rule["direction"] == "higher":
        return value >= float(rule.get("target", rule.get("limit")))
    return float(rule["minimum"]) <= value <= float(rule["maximum"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.metrics.read_text(encoding="utf-8"))["aggregate"]
    policy = json.loads((PROJECT / "configs" / "v63_metric_policy.json").read_text(encoding="utf-8"))

    def render(group: str) -> list[dict]:
        result = []
        for name, rule in policy[group].items():
            value = aggregate.get(name)
            result.append({"metric": name, "measured": value, "passed": _passes(value, rule), **rule})
        return result

    primary, secondary = render("primary_sota_claims"), render("secondary_health_checks")
    result = {
        "schema_version": 1,
        "split": args.split,
        "primary_sota_claims": primary,
        "secondary_health_checks": secondary,
        "primary_targets_met": all(item["passed"] for item in primary),
        "secondary_metrics_healthy": all(item["passed"] for item in secondary),
        "paper_claim_ready": args.split == "test" and all(item["passed"] for item in primary),
        "unavailable_automatic_metrics": policy["unavailable_automatic_metrics"],
        "claim_rule": policy["claim_rule"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["primary_targets_met"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
