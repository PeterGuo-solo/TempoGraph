"""Deterministic synthetic data used only for implementation smoke tests."""

from __future__ import annotations

import numpy as np

from .geometry.rotations import matrix_to_rotation_6d, so3_exp
from .types import MotionSequence, MusicFeatures, MusicSegment


def synthetic_motion(frames: int = 180, joints: int = 22, fps: float = 30.0) -> MotionSequence:
    time = np.arange(frames, dtype=np.float32) / fps
    phases = np.arange(joints, dtype=np.float32)[None] * 0.13
    angles = 0.18 * np.sin(time[:, None] * 2.0 + phases)
    axes = np.zeros((frames, joints, 3), dtype=np.float32)
    axes[..., 1] = angles
    rotations = matrix_to_rotation_6d(so3_exp(axes))
    roots = np.stack(
        [0.25 * time, 0.02 * np.sin(time * 2), 0.15 * np.sin(time * 0.7)], axis=-1
    ).astype(np.float32)
    contacts = np.zeros((frames, 4), dtype=np.float32)
    half_beat = max(1, int(round(fps * 0.25)))
    support = (np.arange(frames) // half_beat) % 2
    contacts[np.ix_(support == 0, [0, 2])] = 1
    contacts[np.ix_(support == 1, [1, 3])] = 1

    offsets = np.zeros((joints, 3), dtype=np.float32)
    offsets[:, 1] = np.linspace(0.0, 1.6, joints)
    offsets[:, 0] = 0.08 * np.sin(np.arange(joints))
    local = offsets[None] + 0.03 * np.sin(time[:, None, None] * 3 + np.arange(joints)[None, :, None])
    return MotionSequence(rotations, roots, contacts, fps, local.astype(np.float32), {"synthetic": True})


def synthetic_music(frames: int = 180, fps: float = 30.0, feature_dim: int = 35) -> MusicFeatures:
    time = np.arange(frames, dtype=np.float32) / fps
    frequencies = np.linspace(0.2, 4.0, feature_dim, dtype=np.float32)
    features = np.sin(time[:, None] * frequencies[None] * 2 * np.pi).astype(np.float32)
    duration = frames / fps
    beats = np.arange(0.0, duration, 0.5, dtype=np.float32)
    downbeats = beats[::4]
    positions = (np.arange(beats.size) % 4 + 1).astype(np.int64)
    midpoint = max(0.1, duration / 2)
    segments = [MusicSegment(0.0, midpoint, "verse"), MusicSegment(midpoint, duration, "chorus")]
    return MusicFeatures(features, fps, beats, downbeats, positions, segments, {"synthetic": True})
