"""NumPy SO(3) utilities using FineDance/PyTorch3D rotation-6D packing."""

from __future__ import annotations

import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert the first two matrix rows, matching PyTorch3D/FineDance."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-1] != 6:
        raise ValueError("rotation must end with dimension 6")
    first = _normalize(rotation[..., :3])
    second_raw = rotation[..., 3:]
    second = _normalize(second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-2).astype(np.float32)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end with shape [3, 3]")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def _skew(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector)
    result = np.zeros(vector.shape[:-1] + (3, 3), dtype=vector.dtype)
    x, y, z = np.moveaxis(vector, -1, 0)
    result[..., 0, 1] = -z
    result[..., 0, 2] = y
    result[..., 1, 0] = z
    result[..., 1, 2] = -x
    result[..., 2, 0] = -y
    result[..., 2, 1] = x
    return result


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    return so3_exp(axis_angle)


def so3_exp(tangent: np.ndarray) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=np.float64)
    if tangent.shape[-1] != 3:
        raise ValueError("SO(3) tangent must end with dimension 3")
    theta = np.linalg.norm(tangent, axis=-1, keepdims=True)
    theta2 = theta * theta
    safe_theta = np.maximum(theta, 1e-16)
    a = np.where(
        theta < 1e-5,
        1 - theta2 / 6 + theta2 * theta2 / 120,
        np.sin(theta) / safe_theta,
    )
    b = np.where(
        theta < 1e-5,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - np.cos(theta)) / np.maximum(theta2, 1e-16),
    )
    omega = _skew(tangent)
    identity = np.broadcast_to(np.eye(3), omega.shape)
    return (identity + a[..., None] * omega + b[..., None] * (omega @ omega)).astype(np.float32)


def so3_log(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end with [3, 3]")
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cosine)
    vee = np.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    scale = np.where(theta < 1e-5, 0.5 + theta**2 / 12, theta / np.maximum(2 * np.sin(theta), 1e-8))
    return (scale[..., None] * vee).astype(np.float32)


def geodesic_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(so3_log(np.swapaxes(left, -1, -2) @ right), axis=-1)
