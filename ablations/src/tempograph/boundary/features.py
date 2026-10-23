"""Boundary diagnostics and motion-energy proxies from the theory."""

from __future__ import annotations

import numpy as np

from ..geometry.rotations import geodesic_distance, rotation_6d_to_matrix, so3_log
from ..types import MotionSequence


def finite_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    velocity = np.zeros_like(values)
    velocity[1:] = np.diff(values, axis=0) * fps
    return velocity


def articulated_energy(
    local_joint_positions: np.ndarray,
    fps: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(local_joint_positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("local_joint_positions must have shape [T, J, 3]")
    velocity = finite_velocity(positions, fps)
    if weights is None:
        weights = np.ones(positions.shape[1], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (positions.shape[1],):
        raise ValueError("joint weights must have shape [J]")
    squared_speed = np.sum(velocity**2, axis=-1)
    return (squared_speed * weights[None]).sum(axis=-1) / max(float(weights.sum()), 1e-8)


def root_angular_velocity(root_orientations: np.ndarray, fps: float) -> np.ndarray:
    orientations = np.asarray(root_orientations, dtype=np.float32)
    relative = np.swapaxes(orientations[:-1], -1, -2) @ orientations[1:]
    result = np.zeros((orientations.shape[0], 3), dtype=np.float32)
    result[1:] = so3_log(relative) * fps
    return result


def root_energy(
    root_positions: np.ndarray,
    root_orientations: np.ndarray,
    fps: float,
    angular_weight: float = 0.25,
) -> np.ndarray:
    linear = finite_velocity(root_positions, fps)
    angular = root_angular_velocity(root_orientations, fps)
    return np.sum(linear**2, axis=-1) + angular_weight * np.sum(angular**2, axis=-1)


def proxy_articulated_energy(motion: MotionSequence) -> np.ndarray:
    if motion.local_joint_positions is not None:
        return articulated_energy(motion.local_joint_positions, motion.fps)
    rotation_velocity = finite_velocity(motion.rotations_6d, motion.fps)
    return np.mean(np.sum(rotation_velocity**2, axis=-1), axis=-1)


def boundary_conflict(
    left: MotionSequence,
    right: MotionSequence,
    angular_weight: float = 0.25,
) -> np.ndarray:
    """Return `[pose, velocity, root, contact, energy]` boundary discrepancy."""
    if left.joint_count != right.joint_count or abs(left.fps - right.fps) > 1e-6:
        raise ValueError("boundary contexts must share skeleton and frame rate")
    left_matrices = rotation_6d_to_matrix(left.rotations_6d[-1])
    right_matrices = rotation_6d_to_matrix(right.rotations_6d[0])
    pose = float(np.mean(geodesic_distance(left_matrices, right_matrices)))

    left_velocity = (left.to_features()[-1, 7:] - left.to_features()[-2, 7:]) * left.fps
    right_velocity = (right.to_features()[1, 7:] - right.to_features()[0, 7:]) * right.fps
    velocity = float(np.mean(np.abs(left_velocity - right_velocity)))

    left_root_v = (left.root_positions[-1] - left.root_positions[-2]) * left.fps
    right_root_v = (right.root_positions[1] - right.root_positions[0]) * right.fps
    left_root_r = rotation_6d_to_matrix(left.rotations_6d[-2:, 0])
    right_root_r = rotation_6d_to_matrix(right.rotations_6d[:2, 0])
    left_omega = so3_log(left_root_r[0].T @ left_root_r[1]) * left.fps
    right_omega = so3_log(right_root_r[0].T @ right_root_r[1]) * right.fps
    root = float(np.linalg.norm(left_root_v - right_root_v) + angular_weight * np.linalg.norm(left_omega - right_omega))

    contact = float(np.sum(np.abs(left.contacts[-1] - right.contacts[0])))
    left_energy = float(np.mean(proxy_articulated_energy(left)[-min(5, left.frames) :]))
    right_energy = float(np.mean(proxy_articulated_energy(right)[: min(5, right.frames)]))
    energy = abs(left_energy - right_energy)
    return np.asarray([pose, velocity, root, contact, energy], dtype=np.float32)


def gated_rigid_penalty(
    root_energy_values: np.ndarray,
    articulated_energy_values: np.ndarray,
    reference_energy: np.ndarray,
    cold_threshold: float,
    hot_threshold: float,
    root_threshold: float,
    articulation_threshold: float,
) -> float:
    reference = np.asarray(reference_energy, dtype=np.float32)
    alpha = np.clip((reference - cold_threshold) / max(hot_threshold - cold_threshold, 1e-8), 0, 1)
    gate = alpha * alpha * (3 - 2 * alpha)
    root_excess = np.maximum(np.asarray(root_energy_values) - root_threshold, 0)
    articulation_deficit = np.maximum(articulation_threshold - np.asarray(articulated_energy_values), 0)
    return float(np.mean(gate * root_excess * articulation_deficit))

