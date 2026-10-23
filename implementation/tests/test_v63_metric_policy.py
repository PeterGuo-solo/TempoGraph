from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_v63_claims_are_separate_from_health_checks(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "check.json"
    metrics.write_text(json.dumps({"aggregate": {
        "fid_k": 33.2, "div_g": 8.0, "fid_g": 33.1,
        "foot_skating_ratio": 0.041, "div_k": 7.25, "bas": 0.09,
    }}), encoding="utf-8")
    completed = subprocess.run([
        sys.executable, str(PROJECT / "scripts" / "check_v63_metrics.py"),
        "--metrics", str(metrics), "--split", "test", "--output", str(output),
    ], check=False)
    assert completed.returncode == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["primary_targets_met"] is True
    assert result["secondary_metrics_healthy"] is False
    assert result["paper_claim_ready"] is True
    assert {item["metric"] for item in result["primary_sota_claims"]} == {"fid_k", "div_g"}


def test_v63_primary_failure_is_nonzero(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "check.json"
    metrics.write_text(json.dumps({"aggregate": {
        "fid_k": 50.0, "div_g": 8.0, "fid_g": 40.0,
        "foot_skating_ratio": 0.04, "div_k": 7.0, "bas": 0.2,
    }}), encoding="utf-8")
    completed = subprocess.run([
        sys.executable, str(PROJECT / "scripts" / "check_v63_metrics.py"),
        "--metrics", str(metrics), "--split", "test", "--output", str(output),
    ], check=False)
    assert completed.returncode == 4
    assert json.loads(output.read_text(encoding="utf-8"))["paper_claim_ready"] is False
