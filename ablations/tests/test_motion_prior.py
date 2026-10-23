from pathlib import Path

import numpy as np
import torch

from tempograph.data.dataset import DatasetRecord, write_manifest
from tempograph.motion_prior import (
    TrainingMotionPrior,
    beat_synchronous_resample,
    blend_with_motion_prior,
    music_descriptor,
)
from tempograph.generation import _contact_aware_foot_lock, _torch_forward_kinematics
from tempograph.lodge_protocol import forward_kinematics_139


def _identity_motion(frames: int, offset: float = 0.0) -> np.ndarray:
    motion = np.zeros((frames, 139), dtype=np.float32)
    motion[:, :4] = np.asarray([1, 0, 1, 0], dtype=np.float32)
    motion[:, 4] = np.arange(frames, dtype=np.float32) + offset
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:] = np.tile(identity, 22)
    return motion


def test_motion_descriptor_has_fixed_shape() -> None:
    descriptor = music_descriptor(np.zeros((12, 35), dtype=np.float32))
    assert descriptor.shape == (109,)
    assert np.isfinite(descriptor).all()


def test_prior_bank_uses_training_split_only(tmp_path: Path) -> None:
    music_train = np.zeros((12, 35), dtype=np.float32)
    music_test = np.ones((12, 35), dtype=np.float32)
    np.save(tmp_path / "train_motion.npy", _identity_motion(12))
    np.save(tmp_path / "test_motion.npy", _identity_motion(12, 100.0))
    np.save(tmp_path / "train_music.npy", music_train)
    np.save(tmp_path / "test_music.npy", music_test)
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            DatasetRecord("train-id", "train", "train_motion.npy", "train_music.npy", motion_format="finedance319"),
            DatasetRecord("test-id", "test", "test_motion.npy", "test_music.npy", motion_format="finedance319"),
        ],
    )
    bank = TrainingMotionPrior.from_manifest(manifest, window_frames=8, stride=4)
    assert {candidate.record_id for candidate in bank.candidates} == {"train-id"}
    motion, metadata = bank.retrieve(music_test[:8], 8, 1, np.random.default_rng(0))
    assert motion.shape == (8, 139)
    assert metadata["source_split"] == "train"
    assert metadata["source_id"] == "train-id"


def test_prior_blend_preserves_shape_and_rotation_manifold() -> None:
    generated = _identity_motion(10)
    generated[:, :4] = 0.0  # logits
    prior = _identity_motion(10)
    result = blend_with_motion_prior(generated, prior, 0.85)
    assert result.shape == generated.shape
    assert np.isfinite(result).all()
    rotations = result[:, 7:].reshape(10, 22, 6)
    np.testing.assert_allclose(np.linalg.norm(rotations[..., :3], axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(rotations[..., 3:], axis=-1), 1.0, atol=1e-5)


def test_beat_synchronous_resample_maps_source_beats_to_target_beats() -> None:
    motion = _identity_motion(20)
    motion[:, 4] = np.arange(20, dtype=np.float32)
    warped, used = beat_synchronous_resample(
        motion,
        np.asarray([4, 8, 12, 16], dtype=np.float32),
        np.asarray([2, 7, 13, 18], dtype=np.float32),
        20,
    )
    assert used
    np.testing.assert_allclose(warped[[2, 7, 13, 18], 4], [4, 8, 12, 16], atol=1e-4)


def test_contact_foot_lock_reduces_planted_root_motion() -> None:
    features = _identity_motion(12)
    features[:, :4] = 1.0
    features[:, 4] = np.linspace(0.0, 0.22, 12, dtype=np.float32)
    rest = np.zeros((24, 3), dtype=np.float32)
    locked, diagnostics = _contact_aware_foot_lock(
        features, rest, threshold=0.5, height_tolerance=0.1,
        strength=1.0, maximum_step=0.1,
    )
    assert diagnostics["corrected_frames"] > 0
    assert np.ptp(locked[:, 4]) < np.ptp(features[:, 4])


def test_differentiable_fk_matches_metric_fk() -> None:
    features = _identity_motion(8)
    features[:, 4] = np.linspace(0.0, 0.12, len(features), dtype=np.float32)
    rest = np.zeros((24, 3), dtype=np.float32)
    rest[0] = (0.0, 1.0, 0.0)
    rest[1], rest[2] = (0.2, 0.9, 0.0), (-0.2, 0.9, 0.0)
    rest[4], rest[5] = (0.2, 0.5, 0.0), (-0.2, 0.5, 0.0)
    rest[7], rest[8] = (0.2, 0.05, 0.0), (-0.2, 0.05, 0.0)
    rest[10], rest[11] = (0.2, 0.0, 0.2), (-0.2, 0.0, 0.2)
    expected = forward_kinematics_139(features, rest)
    actual = _torch_forward_kinematics(
        torch.from_numpy(features), torch.from_numpy(rest)
    ).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
