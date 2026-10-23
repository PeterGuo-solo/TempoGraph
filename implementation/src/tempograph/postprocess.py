"""Validation-calibrated corrections for long-form generated motion.

The local model is trained on finite windows.  Small vertical-velocity biases and
high-frequency rotation noise therefore accumulate during long-form generation.
This module applies deterministic corrections whose parameters are selected on
the validation split only; test labels are never used for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix
from .lodge_protocol import _foot_skating_ratio, forward_kinematics_139


@dataclass(frozen=True)
class GenerationCalibration:
    rotation_sigma: float
    vertical_trend_sigma: float
    contact_thresholds: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _contact_scores(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = int(np.logical_and(prediction, target).sum())
    true_negative = int(np.logical_and(~prediction, ~target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    positive_f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    negative_f1 = 2 * true_negative / max(2 * true_negative + false_positive + false_negative, 1)
    return 0.5 * (sensitivity + specificity), 0.5 * (positive_f1 + negative_f1), float(prediction.mean())


def calibrate_contact_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    maximum_rate_gap: float = 0.10,
) -> tuple[float, float, float, float]:
    """Select per-channel thresholds on validation data with a rate constraint."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32) >= 0.5
    if probabilities.shape != targets.shape or probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("contact calibration expects matching [N, 4] arrays")
    grid = np.linspace(0.05, 0.95, 181, dtype=np.float32)
    selected: list[float] = []
    for channel in range(4):
        target = targets[:, channel]
        target_rate = float(target.mean())
        candidates: list[tuple[float, float, float]] = []
        for threshold in grid:
            balanced_accuracy, macro_f1, predicted_rate = _contact_scores(
                probabilities[:, channel] >= threshold, target
            )
            rate_gap = abs(predicted_rate - target_rate)
            # Prefer a feasible rate, then maximize both balanced metrics.  The
            # rate term breaks ties without allowing a majority-class solution.
            feasible = 1.0 if rate_gap <= maximum_rate_gap else 0.0
            score = feasible * 10.0 + balanced_accuracy + macro_f1 - 0.25 * rate_gap
            candidates.append((score, -rate_gap, float(threshold)))
        selected.append(max(candidates)[2])
    return tuple(selected)  # type: ignore[return-value]


def correct_long_form_motion(
    features: np.ndarray,
    calibration: GenerationCalibration,
) -> np.ndarray:
    """Remove long-form drift, suppress rotation jitter, and calibrate contacts."""
    result = np.asarray(features, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 139:
        raise ValueError("features must have shape [T, 139]")
    if len(result) < 2:
        return result

    # Only the vertical channel is high-pass corrected.  Horizontal root
    # translation is intentional locomotion and must not be detrended.
    vertical = result[:, 5]
    trend = gaussian_filter1d(
        vertical,
        sigma=max(calibration.vertical_trend_sigma, 1e-3),
        mode="nearest",
    )
    vertical = vertical - trend
    result[:, 5] = vertical - vertical[0]

    sigma = max(float(calibration.rotation_sigma), 0.0)
    if sigma > 0:
        result[:, 4:7] = gaussian_filter1d(result[:, 4:7], sigma=sigma, axis=0, mode="nearest")
        result[:, 4:7] -= result[:1, 4:7]
        rotations = result[:, 7:].reshape(len(result), 22, 6)
        rotations = gaussian_filter1d(rotations, sigma=sigma, axis=0, mode="nearest")
        result[:, 7:] = matrix_to_rotation_6d(
            rotation_6d_to_matrix(rotations)
        ).reshape(len(result), -1)

    thresholds = np.asarray(calibration.contact_thresholds, dtype=np.float32)
    result[:, :4] = (result[:, :4] >= thresholds[None]).astype(np.float32)
    return result


def select_rotation_sigma(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    rest_joints: np.ndarray,
    vertical_trend_sigma: float = 30.0,
) -> float:
    """Choose the least aggressive smoothing satisfying validation physics."""
    candidates = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
    best: tuple[float, float] | None = None
    for sigma in candidates:
        calibration = GenerationCalibration(sigma, vertical_trend_sigma, (0.5,) * 4)
        energy_ratios, skating = [], []
        for prediction, target in zip(predictions, targets):
            corrected = correct_long_form_motion(prediction, calibration)
            predicted_energy = np.linalg.norm(np.diff(corrected[:, 7:], axis=0), axis=1).mean()
            target_energy = np.linalg.norm(np.diff(target[:, 7:], axis=0), axis=1).mean()
            energy_ratios.append(float(predicted_energy / max(target_energy, 1e-8)))
            joints = forward_kinematics_139(corrected, rest_joints)
            skating.append(_foot_skating_ratio(joints))
        energy_ratio = float(np.mean(energy_ratios))
        skating_ratio = float(np.mean(skating))
        violation = max(0.5 - energy_ratio, 0.0) + max(energy_ratio - 1.5, 0.0) + max(skating_ratio - 0.40, 0.0)
        objective = violation * 100.0 + abs(np.log(max(energy_ratio, 1e-8))) + 0.25 * skating_ratio
        candidate = (objective, sigma)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return float(best[1])
