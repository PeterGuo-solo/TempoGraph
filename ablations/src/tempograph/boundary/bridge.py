"""Boundary bridges: analytic baseline and shared interpolation helpers."""

from __future__ import annotations

import numpy as np

from ..geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix, so3_exp, so3_log
from ..types import BoundaryPlan, MotionSequence


class AnalyticBoundaryBridge:
    """A deterministic endpoint-respecting baseline used for tests and ablations."""

    def generate(
        self,
        left: MotionSequence,
        right: MotionSequence,
        plan: BoundaryPlan,
    ) -> MotionSequence:
        if left.joint_count != right.joint_count:
            raise ValueError("left and right boundaries need the same skeleton")
        phases = np.linspace(0, 1, plan.length, dtype=np.float32)
        rotations = np.empty((plan.length, left.joint_count, 6), dtype=np.float32)
        left_matrix = rotation_6d_to_matrix(left.rotations_6d[-1])
        right_matrix = rotation_6d_to_matrix(right.rotations_6d[0])
        delta = so3_log(np.swapaxes(left_matrix, -1, -2) @ right_matrix)
        for frame, phase in enumerate(phases):
            rotations[frame] = matrix_to_rotation_6d(left_matrix @ so3_exp(phase * delta))
        rotations[:, 0] = matrix_to_rotation_6d(plan.root_orientations)
        local = None
        if left.local_joint_positions is not None and right.local_joint_positions is not None:
            ease = (phases * phases * (3 - 2 * phases))[:, None, None]
            local = (
                (1 - ease) * left.local_joint_positions[-1][None]
                + ease * right.local_joint_positions[0][None]
            ).astype(np.float32)
        return MotionSequence(
            rotations,
            plan.root_positions,
            plan.contact_probabilities,
            fps=left.fps,
            local_joint_positions=local,
            metadata={"generator": "analytic_boundary_bridge"},
        )


def concatenate_with_transition(
    left_core: MotionSequence,
    transition: MotionSequence,
    right_core: MotionSequence,
) -> MotionSequence:
    if len({left_core.joint_count, transition.joint_count, right_core.joint_count}) != 1:
        raise ValueError("all motion pieces must share a skeleton")
    local = None
    if all(piece.local_joint_positions is not None for piece in (left_core, transition, right_core)):
        local = np.concatenate(
            [left_core.local_joint_positions, transition.local_joint_positions, right_core.local_joint_positions]
        )
    return MotionSequence(
        np.concatenate([left_core.rotations_6d, transition.rotations_6d, right_core.rotations_6d]),
        np.concatenate([left_core.root_positions, transition.root_positions, right_core.root_positions]),
        np.concatenate([left_core.contacts, transition.contacts, right_core.contacts]),
        fps=left_core.fps,
        local_joint_positions=local,
    )
