from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tempograph.generation import _align_key_motion_anchors
from tempograph.losses import key_motion_anchor_loss


PROJECT = Path(__file__).resolve().parents[1]


def _root_motion(frames: int, minima: tuple[int, ...]) -> torch.Tensor:
    time = torch.arange(frames - 1, dtype=torch.float32)
    speed = torch.full_like(time, 1.0)
    for center in minima:
        speed -= 0.82 * torch.exp(-0.5 * ((time - center) / 4.0).square())
    root = torch.nn.functional.pad(torch.cumsum(speed, dim=0), (1, 0))
    joints = torch.zeros((1, frames, 22, 3), dtype=torch.float32)
    joints[0, :, :, 0] = root[:, None]
    return joints


def _canonical_features(frames: int, minima: tuple[int, ...]) -> np.ndarray:
    features = np.zeros((frames, 139), dtype=np.float32)
    features[:, 7:] = np.tile(
        np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), (frames, 22)
    )
    time = np.arange(frames - 1, dtype=np.float32)
    speed = np.ones_like(time)
    for center in minima:
        speed -= 0.82 * np.exp(-0.5 * ((time - center) / 4.0) ** 2)
    features[1:, 4] = np.cumsum(speed)
    return features


def test_key_motion_loss_prefers_salient_minima_on_music_beats() -> None:
    frames = 128
    joints = _root_motion(frames, (20, 52, 84, 116)).requires_grad_()
    aligned = torch.zeros((1, frames), dtype=torch.float32)
    shifted = torch.zeros_like(aligned)
    aligned[0, [20, 52, 84, 116]] = 1.0
    shifted[0, [32, 64, 96, 124]] = 1.0
    aligned_loss = key_motion_anchor_loss(joints, aligned, downbeat_stride=1)
    shifted_loss = key_motion_anchor_loss(joints, shifted, downbeat_stride=1)
    assert aligned_loss < shifted_loss
    aligned_loss.backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()


def test_local_soft_anchor_warp_is_sparse_finite_and_monotonic() -> None:
    frames = 112
    features = _canonical_features(frames, (18, 50, 82))
    beats = np.asarray([22, 54, 86], dtype=np.int64)
    aligned, diagnostics = _align_key_motion_anchors(
        features,
        np.zeros((frames, 35), dtype=np.float32),
        np.zeros((24, 3), dtype=np.float32),
        beat_frames=beats,
        strength=0.55,
        maximum_shift=8,
        downbeat_stride=1,
        minimum_anchor_spacing=12,
        local_radius=10,
    )
    assert aligned.shape == features.shape
    assert np.isfinite(aligned).all()
    assert diagnostics["used"] is True
    assert 1 <= diagnostics["anchor_count"] <= len(beats)
    assert diagnostics["applied_strength"] <= diagnostics["strength"]
    assert all(abs(source - target) <= 8 for source, target in diagnostics["pairs"])
    # Contacts remain discrete/nearest-sampled, while rotations are projected
    # back onto the valid 6D representation after local interpolation.
    assert set(np.unique(aligned[:, :4])).issubset(set(np.unique(features[:, :4])))
    rotations = aligned[:, 7:].reshape(frames, 22, 6)
    first = rotations[..., :3]
    second = rotations[..., 3:]
    assert np.allclose(np.linalg.norm(first, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(second, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.sum(first * second, axis=-1), 0.0, atol=1e-5)


def test_v76_config_prioritizes_sparse_anchor_and_protects_quality() -> None:
    config = json.loads((PROJECT / "configs" / "v76_rhythm_anchor_train.json").read_text())
    training = config["training"]
    weights = training["loss_weights"]
    stage = training["stage_hyperparameters"]["joint_finetune"]
    assert training["stage_steps"]["joint_finetune"] == 15_000
    assert stage["trainable_models"] == ["local_generator"]
    assert weights["key_motion_anchor"] > weights["beat_alignment"] > 0.0
    assert weights["manual_geometry_distribution"] > 0.0
    assert weights["foot_skating"] > 0.0
    assert weights["diversity_preservation"] > 0.0
    assert weights["kinetic_coverage"] > 0.0


def test_v76_runner_uses_validation_only_and_three_seed_hard_gate() -> None:
    runner = (PROJECT / "scripts" / "run_v76_rhythm_anchor_autodl.sh").read_text()
    assert "--max-steps 15000" in runner
    assert "--split val" in runner
    assert "--split test" not in runner
    assert "v76_protected" in runner
    assert "v76_balanced" in runner
    assert "v76_rhythmic" in runner
    assert '"$SELECTED_PROFILE" "$SELECTED_PRIOR" 2025' in runner
    assert '"$SELECTED_PROFILE" "$SELECTED_PRIOR" 3407' in runner
    assert "select_v75_group.py" in runner
