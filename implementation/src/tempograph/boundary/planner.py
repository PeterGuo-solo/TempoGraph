"""Analytic, testable baseline for the five-part Boundary Fusion plan."""

from __future__ import annotations

import numpy as np

from ..geometry.hermite import cubic_hermite
from ..geometry.rotations import rotation_6d_to_matrix, so3_exp, so3_log
from ..types import BoundaryPlan, MotionSequence
from .features import boundary_conflict, proxy_articulated_energy


def _smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3 - 2 * values)


class AnalyticBoundaryPlanner:
    """Produces valid supervision targets before a learned planner is trained."""

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 60,
        beat_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.beat_multipliers = beat_multipliers

    def choose_length(self, conflict: np.ndarray, beat_interval: float, fps: float) -> int:
        normalized = float(np.tanh(np.mean(np.log1p(np.maximum(conflict, 0)))))
        index = min(len(self.beat_multipliers) - 1, int(normalized * len(self.beat_multipliers)))
        frames = int(round(fps * beat_interval * self.beat_multipliers[index]))
        return int(np.clip(frames, self.min_length, self.max_length))

    def plan(
        self,
        left: MotionSequence,
        right: MotionSequence,
        beat_interval: float,
        edge_type: str = "transition",
    ) -> BoundaryPlan:
        conflict = boundary_conflict(left, right)
        length = self.choose_length(conflict, beat_interval, left.fps)
        phase = np.linspace(0, 1, length, dtype=np.float32)
        ease = _smoothstep(phase)[:, None]

        left_velocity = (left.root_positions[-1] - left.root_positions[-2]) * left.fps
        right_velocity = (right.root_positions[1] - right.root_positions[0]) * right.fps
        duration = length / left.fps
        roots = cubic_hermite(
            left.root_positions[-1],
            right.root_positions[0],
            left_velocity,
            right_velocity,
            duration,
            phase,
        )

        left_orientation = rotation_6d_to_matrix(left.rotations_6d[-1, 0])
        right_orientation = rotation_6d_to_matrix(right.rotations_6d[0, 0])
        delta = so3_log(left_orientation.T @ right_orientation)
        orientations = np.stack([left_orientation @ so3_exp(value * delta) for value in phase])

        contacts = (1 - ease) * left.contacts[-1] + ease * right.contacts[0]
        left_support = int(np.argmax(left.contacts[-1, :2]))
        right_support = int(np.argmax(right.contacts[0, :2]))
        if left_support != right_support and edge_type != "contrast":
            middle = np.abs(phase - 0.5) < 0.15
            contacts[middle, 0] = np.maximum(contacts[middle, 0], 0.75)
            contacts[middle, 1] = np.maximum(contacts[middle, 1], 0.75)

        left_energy = float(np.mean(proxy_articulated_energy(left)[-min(left.frames, 5) :]))
        right_energy = float(np.mean(proxy_articulated_energy(right)[: min(right.frames, 5)]))
        energy = ((1 - _smoothstep(phase)) * left_energy + _smoothstep(phase) * right_energy).astype(np.float32)
        return BoundaryPlan(length, contacts, roots, orientations, energy, conflict)

