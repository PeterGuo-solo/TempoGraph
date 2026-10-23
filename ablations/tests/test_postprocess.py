from __future__ import annotations

import numpy as np

from tempograph.postprocess import (
    GenerationCalibration,
    calibrate_contact_thresholds,
    correct_long_form_motion,
)


def _identity_features(frames: int = 240) -> np.ndarray:
    features = np.zeros((frames, 139), dtype=np.float32)
    rotations = features[:, 7:].reshape(frames, 22, 6)
    rotations[..., 0] = 1.0
    rotations[..., 4] = 1.0
    return features


def test_long_form_correction_removes_vertical_drift_and_preserves_so3() -> None:
    features = _identity_features()
    features[:, 5] = np.linspace(0.0, -3.0, len(features))
    rng = np.random.default_rng(7)
    features[:, 7:] += rng.normal(0, 0.05, features[:, 7:].shape)
    calibration = GenerationCalibration(1.0, 30.0, (0.5,) * 4)
    corrected = correct_long_form_motion(features, calibration)
    assert np.ptp(corrected[:, 5]) < 0.5
    rotations = corrected[:, 7:].reshape(len(corrected), 22, 6)
    assert np.allclose(np.linalg.norm(rotations[..., :3], axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(rotations[..., 3:], axis=-1), 1.0, atol=1e-5)
    assert np.allclose((rotations[..., :3] * rotations[..., 3:]).sum(-1), 0.0, atol=1e-5)


def test_contact_calibration_uses_signal_instead_of_majority_threshold() -> None:
    targets = np.tile(np.array([[0], [0], [1], [1]], dtype=np.float32), (50, 4))
    probabilities = np.where(targets > 0, 0.85, 0.60).astype(np.float32)
    thresholds = calibrate_contact_thresholds(probabilities, targets)
    assert all(0.60 < threshold <= 0.851 for threshold in thresholds)
