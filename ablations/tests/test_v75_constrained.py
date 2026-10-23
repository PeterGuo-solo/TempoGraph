from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tempograph.generation import _align_motion_beats
from tempograph.losses import manual_geometry_distribution_loss, soft_manual_descriptor


PROJECT = Path(__file__).resolve().parents[1]


def test_soft_manual_descriptor_is_32d_finite_and_differentiable() -> None:
    joints = torch.randn(3, 24, 22, 3, requires_grad=True)
    descriptor = soft_manual_descriptor(joints)
    assert descriptor.shape == (3, 32)
    assert torch.isfinite(descriptor).all()
    descriptor.mean().backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()


def test_manual_distribution_prefers_identical_geometry() -> None:
    reference = torch.randn(4, 24, 22, 3)
    same = manual_geometry_distribution_loss(reference, reference)
    shifted = reference.clone()
    shifted[:, :, 20:, 1] += 0.8
    different = manual_geometry_distribution_loss(shifted, reference)
    assert float(same) < 1e-7
    assert different > same


def test_sparse_alignment_reports_protective_controls() -> None:
    frames = 64
    features = np.zeros((frames, 139), dtype=np.float32)
    features[:, 7:] = np.tile(np.asarray([1, 0, 0, 0, 1, 0], np.float32), 22)
    features[:, 4] = np.arange(frames, dtype=np.float32)
    music = np.zeros((frames, 35), dtype=np.float32)
    beats = np.asarray([4, 12, 20, 28, 36, 44, 52, 60])
    aligned, diagnostics = _align_motion_beats(
        features,
        music,
        np.zeros((24, 3), dtype=np.float32),
        beat_frames=beats,
        anchor_stride=4,
        punctuation_stride=4,
        warp_strength=0.0,
        punctuation_strength=0.1,
    )
    assert aligned.shape == features.shape
    assert diagnostics["anchor_stride"] == 4
    assert diagnostics["punctuation_stride"] == 4
    assert diagnostics["warp_strength"] == 0.0


def test_v75_config_freezes_nonlocal_models_and_targets_official_geometry() -> None:
    config = json.loads((PROJECT / "configs" / "v75_constrained_train.json").read_text())
    training = config["training"]
    stage = training["stage_hyperparameters"]["joint_finetune"]
    weights = training["loss_weights"]
    assert training["stage_steps"]["joint_finetune"] == 15_000
    assert training["batch_size"] == 4
    assert stage["trainable_models"] == ["local_generator"]
    assert weights["manual_geometry_distribution"] > weights["geometric_distribution"]
    assert weights["beat_alignment"] > 0.0
    assert weights["diversity_preservation"] > 0.0


def test_group_selector_requires_every_seed_to_pass(tmp_path: Path) -> None:
    profile = tmp_path / "v75_quality"
    passing = {
        "fid_k": 30.0, "fid_g": 20.0, "foot_skating_ratio": 0.02,
        "div_k": 8.4, "div_g": 7.8, "bas": 0.25,
    }
    for seed in (1234, 2025, 3407):
        report = profile / f"seed_{seed}" / "reports" / "lodge_protocol_metrics.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"aggregate": passing}), encoding="utf-8")
    output = tmp_path / "selection.json"
    completed = subprocess.run(
        [
            sys.executable, str(PROJECT / "scripts" / "select_v75_group.py"),
            "--candidates-root", str(tmp_path), "--output", str(output),
        ],
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(output.read_text())["eligible_for_one_shot_test"] is True


def test_fast_screen_preserves_won_metrics_before_rescue_targets(tmp_path: Path) -> None:
    rows = {
        "v75_quality": {
            "fid_k": 30.0, "fid_g": 24.0, "foot_skating_ratio": 0.02,
            "div_k": 8.3, "div_g": 7.8, "bas": 0.21,
        },
        # Better BAS/FID_g, but it loses protected Div_k and must be rejected.
        "v75_rhythmic": {
            "fid_k": 30.0, "fid_g": 18.0, "foot_skating_ratio": 0.02,
            "div_k": 7.0, "div_g": 7.8, "bas": 0.27,
        },
    }
    for profile, metrics in rows.items():
        report = tmp_path / profile / "seed_1234" / "reports" / "lodge_protocol_metrics.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"aggregate": metrics}), encoding="utf-8")
    output = tmp_path / "screen.json"
    completed = subprocess.run(
        [
            sys.executable, str(PROJECT / "scripts" / "select_v75_screen.py"),
            "--candidates-root", str(tmp_path), "--output", str(output),
        ],
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(output.read_text())["selected_profile"] == "v75_quality"


def test_v75_runner_never_reads_test_split() -> None:
    runner = (PROJECT / "scripts" / "run_v75_constrained_autodl.sh").read_text()
    assert "--split val" in runner
    assert "--split test" not in runner
    assert "--max-steps 15000" in runner
    assert "select_v75_screen.py" in runner
    assert '"$SELECTED_PROFILE" "$SELECTED_PRIOR" 2025' in runner
    assert '"$SELECTED_PROFILE" "$SELECTED_PRIOR" 3407' in runner
