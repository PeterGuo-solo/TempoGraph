from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare measured LODGE metrics with supplied SOTA targets")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measured = json.loads(args.metrics.read_text(encoding="utf-8")).get("aggregate", {})
    target_file = PROJECT / "configs" / "sota_targets.json"
    targets = json.loads(target_file.read_text(encoding="utf-8"))["lodge_protocol"]["targets"]
    checks = []
    for name, target in targets.items():
        value = measured.get(name)
        direction = target["direction"]
        passed = value is not None and (
            float(value) < float(target["value"])
            if direction == "lower"
            else float(value) > float(target["value"])
        )
        checks.append({
            "metric": name,
            "measured": value,
            "target": target["value"],
            "direction": direction,
            "reference_method": target["method"],
            "beats_target": bool(passed),
        })
    result = {
        "protocol": "LODGE FineDance",
        "all_targets_beaten": all(item["beats_target"] for item in checks),
        "checks": checks,
        "claim_rule": "A SOTA claim requires the matching official protocol; exploratory quality gates are irrelevant.",
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["all_targets_beaten"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
