import importlib.util
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_seed_aggregation_keeps_individual_runs_and_sample_std(tmp_path: Path):
    module = _load_script("aggregate_seed_results")
    for seed, fid in ((1234, 10.0), (2025, 14.0), (3407, 12.0)):
        root = tmp_path / f"seed_{seed}"
        _write(root / "reports" / "run_manifest.json", {"seed": seed})
        _write(
            root / "generated" / "tempograph" / "generation_index.json",
            {"seed": seed},
        )
        _write(
            root / "reports" / "tempograph_metrics.json",
            {"sample_count": 19, "aggregate": {"diversity_ratio": 1.0}},
        )
        _write(
            root / "reports" / "lodge_protocol_metrics.json",
            {"aggregate": {"fid_k": fid}},
        )
    result = module.aggregate_seed_root(tmp_path)
    assert result["seeds"] == [1234, 2025, 3407]
    assert result["aggregate"]["lodge_fid_k"]["mean"] == pytest.approx(12.0)
    assert result["aggregate"]["lodge_fid_k"]["sample_std"] == pytest.approx(2.0)


def test_quality_gate_rejects_first_run_failure_pattern():
    module = _load_script("quality_gate")
    report = {
        "status": "complete",
        "sample_count": 19,
        "aggregate": {
            "contact_balanced_accuracy": 0.5,
            "contact_macro_f1": 0.45,
            "contact_positive_rate": 1.0,
            "reference_contact_positive_rate": 0.88,
            "diversity_ratio": 0.046,
            "energy_ratio": 0.22,
            "root_velocity_ratio": 0.10,
        },
    }
    result = module.evaluate_gate(report)
    assert not result["passed"]
    assert {item["name"] for item in result["checks"] if not item["passed"]} >= {
        "contact_balanced_accuracy", "contact_positive_rate", "diversity_ratio",
        "energy_ratio", "root_velocity_ratio", "contact_rate_gap",
    }


def test_quality_gate_rejects_bad_official_lodge_motion_metrics():
    module = _load_script("quality_gate")
    local = {
        "status": "complete", "sample_count": 19,
        "aggregate": {
            "contact_balanced_accuracy": 0.7, "contact_macro_f1": 0.7,
            "contact_positive_rate": 0.84, "reference_contact_positive_rate": 0.84,
            "diversity_ratio": 1.0, "energy_ratio": 1.0, "root_velocity_ratio": 1.0,
        },
    }
    lodge = {
        "status": "complete",
        "aggregate": {
            "foot_skating_ratio": 0.88, "div_k": 39.0, "div_k_gt": 9.73,
            "div_g": 153.0, "div_g_gt": 7.44, "bas": 0.10,
        },
    }
    result = module.evaluate_gate(local, lodge_report=lodge)
    failed = {item["name"] for item in result["checks"] if not item["passed"]}
    assert {
        "lodge_foot_skating_ratio", "lodge_div_k_ratio",
        "lodge_div_g_ratio", "lodge_bas",
    }.issubset(failed)


def test_exploratory_gate_accepts_v5_observed_ranges_but_strict_does_not():
    module = _load_script("quality_gate")
    local = {
        "status": "complete", "sample_count": 19,
        "aggregate": {
            "contact_balanced_accuracy": 0.5062, "contact_macro_f1": 0.4854,
            "contact_positive_rate": 0.8676, "reference_contact_positive_rate": 0.8385,
            "diversity_ratio": 1.5106, "energy_ratio": 1.9524,
            "root_velocity_ratio": 2.1509,
        },
    }
    lodge = {
        "status": "complete",
        "aggregate": {
            "foot_skating_ratio": 0.7566, "div_k": 12.28, "div_k_gt": 9.73,
            "div_g": 49.84, "div_g_gt": 7.44, "bas": 0.1083,
        },
    }
    assert not module.evaluate_gate(local, lodge_report=lodge, profile="strict")["passed"]
    assert module.evaluate_gate(local, lodge_report=lodge, profile="exploratory")["passed"]
