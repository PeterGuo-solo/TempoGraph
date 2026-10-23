"""Small inference-only contact correction that preserves generated motion."""

from __future__ import annotations

import numpy as np

from ..types import MotionSequence


def contact_aware_root_correction(
    transition: MotionSequence,
    foot_indices: tuple[int, int, int, int] = (7, 8, 10, 11),
    threshold: float = 0.5,
    strength: float = 0.5,
    iterations: int = 8,
) -> MotionSequence:
    """Reduce contact-foot drift by small root translations.

    Requires local joint positions. Rotations are intentionally untouched so the correction cannot
    redesign the generated transition.
    """
    if transition.local_joint_positions is None:
        return transition
    roots = transition.root_positions.copy()
    local_feet = transition.local_joint_positions[:, foot_indices]
    original = roots.copy()
    for _ in range(iterations):
        world_feet = local_feet + roots[:, None, :]
        correction = np.zeros_like(roots)
        counts = np.zeros((transition.frames, 1), dtype=np.float32)
        for frame in range(1, transition.frames):
            active = transition.contacts[frame] >= threshold
            if np.any(active):
                drift = world_feet[frame, active] - world_feet[frame - 1, active]
                correction[frame] -= drift.mean(axis=0) * strength
                counts[frame] += 1
        roots += correction / np.maximum(counts, 1)
        endpoint_weight = np.linspace(0, 1, transition.frames, dtype=np.float32)[:, None]
        endpoint_delta = (1 - endpoint_weight) * (original[0] - roots[0]) + endpoint_weight * (
            original[-1] - roots[-1]
        )
        roots += endpoint_delta
    return MotionSequence(
        transition.rotations_6d.copy(),
        roots,
        transition.contacts.copy(),
        transition.fps,
        transition.local_joint_positions.copy(),
        {**transition.metadata, "contact_corrected": True},
    )
