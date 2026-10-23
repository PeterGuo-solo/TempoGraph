from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_v74_is_exactly_twelve_thousand_steps() -> None:
    config = json.loads((PROJECT / "configs" / "v74_12k_train.json").read_text())
    training = config["training"]
    assert training["stage_steps"]["joint_finetune"] == 12_000
    assert training["stage_hyperparameters"]["joint_finetune"]["learning_rate"] <= 4e-6


def test_v74_jointly_targets_geometry_and_beats() -> None:
    config = json.loads((PROJECT / "configs" / "v74_12k_train.json").read_text())
    weights = config["training"]["loss_weights"]
    assert weights["geometric_distribution"] >= 8.0
    assert weights["beat_alignment"] >= 10.0
    assert weights["kinetic_coverage"] > 0.0
    assert weights["fk_position"] > weights["fk_velocity"] > weights["fk_acceleration"]


def test_v74_freezes_nonlocal_models() -> None:
    config = json.loads((PROJECT / "configs" / "v74_12k_train.json").read_text())
    stage = config["training"]["stage_hyperparameters"]["joint_finetune"]
    assert stage["trainable_models"] == ["local_generator"]


def test_v74_runner_never_opens_test_split() -> None:
    runner = (PROJECT / "scripts" / "run_v74_12k_autodl.sh").read_text()
    assert "--max-steps 12000" in runner
    assert "--split test" not in runner
    assert "--split val" in runner
