"""Adapters from reference-code formats into the canonical TempoGraph schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from ..types import MotionSequence


def motion_from_lodge139(features: np.ndarray, fps: float = 30.0) -> MotionSequence:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 139:
        raise ValueError("LODGE motion must have shape [T, 139]")
    return MotionSequence.from_features(features, fps=fps)


def motion_from_finedance319(features: np.ndarray, fps: float = 30.0) -> MotionSequence:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 319:
        raise ValueError("FineDance SMPL-H motion must have shape [T, 319]")
    return MotionSequence.from_features(features, fps=fps)


def motion_from_edge151(features: np.ndarray, fps: float = 30.0) -> MotionSequence:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 151:
        raise ValueError("EDGE/AtomicDance motion must have shape [T, 151]")
    return MotionSequence.from_features(features, fps=fps)


def motion_from_axis_angle(
    array: np.ndarray,
    joint_count: int,
    fps: float = 30.0,
    contacts: np.ndarray | None = None,
) -> MotionSequence:
    """Convert `[root_xyz, J*axis_angle]` arrays to canonical rotation-6D."""
    array = np.asarray(array, dtype=np.float32)
    expected = 3 + 3 * joint_count
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(f"axis-angle motion must have shape [T, {expected}]")
    rotations = array[:, 3:].reshape(array.shape[0], joint_count, 3)
    rotations_6d = matrix_to_rotation_6d(axis_angle_to_matrix(rotations))
    if contacts is None:
        contacts = np.zeros((array.shape[0], 4), dtype=np.float32)
    return MotionSequence(rotations_6d, array[:, :3], contacts, fps=fps)


def canonicalize_root_frame(motion: MotionSequence) -> MotionSequence:
    """Translate the first root position to the origin without modifying rotations."""
    root = motion.root_positions - motion.root_positions[0]
    local = motion.local_joint_positions
    return MotionSequence(
        motion.rotations_6d.copy(),
        root,
        motion.contacts.copy(),
        motion.fps,
        None if local is None else local.copy(),
        {**motion.metadata, "root_translation_origin": motion.root_positions[0].tolist()},
    )


def save_motion_npz(path: str | Path, motion: MotionSequence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rotations_6d": motion.rotations_6d,
        "root_positions": motion.root_positions,
        "contacts": motion.contacts,
        "fps": np.asarray(motion.fps, dtype=np.float32),
    }
    if motion.local_joint_positions is not None:
        payload["local_joint_positions"] = motion.local_joint_positions
    np.savez_compressed(path, **payload)


def load_motion_npz(path: str | Path) -> MotionSequence:
    with np.load(Path(path), allow_pickle=False) as archive:
        local = archive["local_joint_positions"] if "local_joint_positions" in archive else None
        return MotionSequence(
            archive["rotations_6d"],
            archive["root_positions"],
            archive["contacts"],
            float(archive["fps"]),
            local,
        )


def estimate_contacts(
    local_joint_positions: np.ndarray,
    fps: float,
    foot_indices: tuple[int, int, int, int] = (7, 8, 10, 11),
    velocity_threshold: float = 0.15,
    height_threshold: float = 0.10,
) -> np.ndarray:
    """Estimate four foot-contact probabilities from joint positions."""
    positions = np.asarray(local_joint_positions, dtype=np.float32)
    feet = positions[:, foot_indices]
    velocity = np.zeros(feet.shape[:-1], dtype=np.float32)
    velocity[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
    height = feet[..., 1] - np.min(feet[..., 1], axis=0, keepdims=True)
    velocity_score = np.clip(1 - velocity / velocity_threshold, 0, 1)
    height_score = np.clip(1 - height / height_threshold, 0, 1)
    return (velocity_score * height_score).astype(np.float32)


def convert_joint_count(motion: MotionSequence, target_joints: int) -> MotionSequence:
    """Deterministically keep or pad joints; intended for adapters, not retargeting."""
    if target_joints <= 0:
        raise ValueError("target_joints must be positive")
    if target_joints == motion.joint_count:
        return motion
    if target_joints < motion.joint_count:
        rotations = motion.rotations_6d[:, :target_joints]
        local = None if motion.local_joint_positions is None else motion.local_joint_positions[:, :target_joints]
    else:
        identity_6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        pad = np.broadcast_to(
            identity_6d,
            (motion.frames, target_joints - motion.joint_count, 6),
        )
        rotations = np.concatenate([motion.rotations_6d, pad], axis=1)
        local = None
        if motion.local_joint_positions is not None:
            local_pad = np.zeros((motion.frames, target_joints - motion.joint_count, 3), np.float32)
            local = np.concatenate([motion.local_joint_positions, local_pad], axis=1)
    return MotionSequence(rotations, motion.root_positions, motion.contacts, motion.fps, local, motion.metadata)

