"""LODGE/FineDance paper-protocol metrics for canonical 139D predictions.

This evaluates TempoGraph predictions only.  Published baseline rows remain
cited paper values and are never presented as locally reproduced runs.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.rotations import rotation_6d_to_matrix


SMPLX_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
     16, 17, 18, 19, 15, 15],
    dtype=np.int64,
)


def forward_kinematics_139(features: np.ndarray, rest_joints: np.ndarray) -> np.ndarray:
    """Match LODGE's rigid-joint FK without importing its PyTorch3D renderer."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 139:
        raise ValueError("features must have shape [T, 139]")
    if rest_joints.shape[0] < 24 or rest_joints.shape[1] != 3:
        raise ValueError("rest_joints must contain at least 24 xyz joints")
    frames = len(features)
    rotations = rotation_6d_to_matrix(features[:, 7:].reshape(frames, 22, 6))
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 2, 3, 3))
    rotations = np.concatenate([rotations, identity], axis=1)
    rest = np.asarray(rest_joints[:24], dtype=np.float32)
    global_rotations = np.empty_like(rotations)
    positions = np.empty((frames, 24, 3), dtype=np.float32)
    global_rotations[:, 0] = rotations[:, 0]
    positions[:, 0] = rest[0] + features[:, 4:7]
    for joint in range(1, 24):
        parent = int(SMPLX_PARENTS[joint])
        global_rotations[:, joint] = global_rotations[:, parent] @ rotations[:, joint]
        offset = rest[joint] - rest[parent]
        positions[:, joint] = (
            positions[:, parent]
            + np.einsum("tij,j->ti", global_rotations[:, parent], offset)
        )
    return positions


def _load_feature_extractors(lodge_root: Path):
    metric_root = lodge_root / "metric"
    if not (metric_root / "features" / "kinetic.py").is_file():
        raise FileNotFoundError(f"LODGE metric sources not found under {metric_root}")
    sys.path.insert(0, str(metric_root))
    try:
        kinetic = importlib.import_module("features.kinetic")
        manual = importlib.import_module("features.manual_new")
    finally:
        sys.path.pop(0)
    return kinetic.extract_kinetic_features, manual.extract_manual_features


def _relative_joints(joints: np.ndarray) -> np.ndarray:
    flat = joints[:, :22].reshape(len(joints), 66)
    flat = flat - np.tile(flat[:1, :3], (1, 22))
    relative = flat.reshape(len(joints), 22, 3)
    relative[:, 1:] -= relative[:, :1]
    return relative


def _frechet(first: np.ndarray, second: np.ndarray) -> float:
    from scipy import linalg

    mean_first, mean_second = first.mean(0), second.mean(0)
    covariance_first = np.cov(first, rowvar=False)
    covariance_second = np.cov(second, rowvar=False)
    # Calling sqrtm without ``disp`` works on both older SciPy releases and
    # releases where that deprecated keyword (and the tuple return) was removed.
    covariance_mean = linalg.sqrtm(covariance_first.dot(covariance_second))
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_first.shape[0]) * 1e-5
        covariance_mean = linalg.sqrtm(
            (covariance_first + offset).dot(covariance_second + offset)
        )
    covariance_mean = covariance_mean.real
    difference = mean_first - mean_second
    return float(
        difference.dot(difference)
        + np.trace(covariance_first)
        + np.trace(covariance_second)
        - 2 * np.trace(covariance_mean)
    )


def _diversity(features: np.ndarray) -> float:
    distances = np.linalg.norm(features[:, None] - features[None], axis=-1)
    return float(distances[np.triu_indices(len(features), 1)].mean())


def _motion_beats(joints: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelextrema

    velocity = np.mean(np.linalg.norm(joints[1:] - joints[:-1], axis=2), axis=1)
    return argrelextrema(gaussian_filter(velocity, 5), np.less)[0]


def _music_beats(audio_path: Path) -> np.ndarray:
    import librosa

    sample_rate = 30 * 512
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo = float(np.asarray(librosa.feature.tempo(y=audio, sr=sample_rate)).reshape(-1)[0])
    _, beats = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=512,
        start_bpm=tempo,
        tightness=100,
    )
    return np.asarray(beats, dtype=np.int64)


def _beat_alignment(music_beats: np.ndarray, motion_beats: np.ndarray) -> float:
    if not len(music_beats) or not len(motion_beats):
        return 0.0
    distances = np.min((motion_beats[None] - music_beats[:, None]) ** 2, axis=1)
    return float(np.exp(-distances / 18.0).mean())


def _foot_skating_ratio(joints: np.ndarray) -> float:
    joints = joints.copy()
    left_toe, right_toe = joints[0, 10, 1], joints[0, 11, 1]
    height = (left_toe + right_toe) / 2 if abs(left_toe - right_toe) < 0.02 else min(left_toe, right_toe)
    joints[:, :, 1] -= height
    velocity = np.zeros_like(joints[:, (7, 8, 10, 11)])
    velocity[:-1] = joints[1:, (7, 8, 10, 11)] - joints[:-1, (7, 8, 10, 11)]
    left_contact = (joints[:, 7, 1] <= 0.08) & (joints[:, 10, 1] <= 0.05)
    right_contact = (joints[:, 8, 1] <= 0.08) & (joints[:, 11, 1] <= 0.05)

    def side_ratio(indices: tuple[int, int], contact: np.ndarray) -> float:
        tangent = np.abs(velocity[:, indices][:, :, (0, 2)].mean(axis=-1))
        slide = np.any((tangent > 0.01) & contact[:, None], axis=-1)
        return float(slide.sum() / max(int(contact.sum()), 1))

    return 0.5 * (side_ratio((0, 2), left_contact) + side_ratio((1, 3), right_contact))


def evaluate_lodge_protocol(
    predictions: str | Path,
    data_root: str | Path,
    lodge_root: str | Path,
    compute_audio: bool = True,
) -> dict[str, Any]:
    predictions = Path(predictions).resolve()
    data_root = Path(data_root).resolve()
    lodge_root = Path(lodge_root).resolve()
    rest_joints = np.load(data_root / "smplx_neu_J_1.npy", allow_pickle=False)
    extract_kinetic, extract_manual = _load_feature_extractors(lodge_root)
    kinetic, geometric, beat_scores, skating = [], [], [], []
    ids = []
    for path in sorted(predictions.glob("*.npy")):
        features = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)[:1024, :139]
        if len(features) < 1024:
            continue
        joints = forward_kinematics_139(features, rest_joints)
        relative = _relative_joints(joints)
        kinetic.append(extract_kinetic(relative))
        geometric.append(extract_manual(relative))
        centered = joints - joints[:1, :1]
        if compute_audio:
            # Match music evidence to the evaluated motion duration.  The WAV
            # files can be longer than the fixed 1024-frame prediction; beats
            # after the last motion-velocity sample have no possible match and
            # must not be included in the BAS denominator.
            music_beats = _music_beats(data_root / "music_wav" / f"{path.stem}.wav")
            music_beats = music_beats[
                (music_beats >= 0) & (music_beats < max(0, len(centered) - 1))
            ]
            beat_scores.append(
                _beat_alignment(
                    music_beats,
                    _motion_beats(centered),
                )
            )
        skating.append(_foot_skating_ratio(joints))
        ids.append(path.stem)
    if len(ids) < 2:
        raise ValueError("at least two 1024-frame prediction files are required")

    predicted_k = np.stack(kinetic)
    predicted_g = np.stack(geometric)
    gt_k = np.stack([
        np.load(path, allow_pickle=False)
        for path in sorted((data_root / "mofea319" / "kinetic_features").glob("*.npy"))
    ])
    gt_g = np.stack([
        np.load(path, allow_pickle=False)
        for path in sorted((data_root / "mofea319" / "manual_features_new").glob("*.npy"))
    ])
    if gt_k.shape[1] == 72:
        gt_k = gt_k[:, :66]
    if predicted_k.shape[1] == 72:
        predicted_k = predicted_k[:, :66]
    mean_k, std_k = gt_k.mean(0), gt_k.std(0)
    mean_g, std_g = gt_g.mean(0), gt_g.std(0)
    gt_k = (gt_k - mean_k) / (std_k + 1e-10)
    predicted_k = (predicted_k - mean_k) / (std_k + 1e-10)
    gt_g = (gt_g - mean_g) / (std_g + 1e-10)
    predicted_g = (predicted_g - mean_g) / (std_g + 1e-10)
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol": "LODGE released FineDance metric definitions; TempoGraph predictions only",
        "sample_count": len(ids),
        "ids": ids,
        "aggregate": {
            "fid_k": _frechet(predicted_k, gt_k),
            "fid_g": _frechet(predicted_g, gt_g),
            "div_k": _diversity(predicted_k),
            "div_g": _diversity(predicted_g),
            "div_k_gt": _diversity(gt_k),
            "div_g_gt": _diversity(gt_g),
            "bas": float(np.mean(beat_scores)) if beat_scores else None,
            "foot_skating_ratio": float(np.mean(skating)),
        },
        "provenance": {
            "predictions": str(predictions),
            "data_root": str(data_root),
            "feature_source": str(lodge_root / "metric" / "features"),
            "note": "No baseline model was run; compare only with cited paper rows using this protocol.",
        },
    }
