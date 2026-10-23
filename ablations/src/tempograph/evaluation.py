"""TempoGraph motion, music, and boundary evaluation metrics."""

from __future__ import annotations

import numpy as np

from .boundary.features import finite_velocity, proxy_articulated_energy
from .types import MotionSequence, MusicFeatures


def boundary_continuity_metrics(
    left: MotionSequence,
    transition: MotionSequence,
    right: MotionSequence,
) -> dict[str, float]:
    if not (left.joint_count == transition.joint_count == right.joint_count):
        raise ValueError("motion pieces must share a skeleton")
    joined = np.concatenate([left.to_features(), transition.to_features(), right.to_features()], axis=0)
    velocity = finite_velocity(joined, left.fps)
    acceleration = finite_velocity(velocity, left.fps)
    seams = [left.frames, left.frames + transition.frames]
    position_jumps = []
    velocity_jumps = []
    acceleration_jumps = []
    for seam in seams:
        position_jumps.append(float(np.mean(np.abs(joined[seam] - joined[seam - 1]))))
        velocity_jumps.append(float(np.mean(np.abs(velocity[seam] - velocity[seam - 1]))))
        acceleration_jumps.append(float(np.mean(np.abs(acceleration[seam] - acceleration[seam - 1]))))
    return {
        "position_jump": float(np.mean(position_jumps)),
        "velocity_jump": float(np.mean(velocity_jumps)),
        "acceleration_jump": float(np.mean(acceleration_jumps)),
        "contact_jump": float(
            0.5
            * (
                np.mean(np.abs(left.contacts[-1] - transition.contacts[0]))
                + np.mean(np.abs(transition.contacts[-1] - right.contacts[0]))
            )
        ),
    }


def foot_skating_metric(
    motion: MotionSequence,
    foot_indices: tuple[int, int, int, int] = (7, 8, 10, 11),
    threshold: float = 0.5,
) -> float:
    if motion.local_joint_positions is None:
        return float("nan")
    feet = motion.local_joint_positions[:, foot_indices] + motion.root_positions[:, None]
    velocity = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * motion.fps
    active = motion.contacts[1:] >= threshold
    return float(np.mean(velocity[active])) if np.any(active) else 0.0


def energy_envelope_error(motion: MotionSequence, reference: np.ndarray) -> float:
    actual = proxy_articulated_energy(motion)
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape != actual.shape:
        reference = np.interp(
            np.linspace(0, 1, actual.size), np.linspace(0, 1, reference.size), reference
        )
    return float(np.mean(np.abs(np.log(actual + 1e-6) - np.log(reference + 1e-6))))


def beat_alignment_score(motion: MotionSequence, music: MusicFeatures, tolerance: float = 0.1) -> float:
    if music.beats.size == 0:
        return float("nan")
    energy = proxy_articulated_energy(motion)
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:])) + 1
    if minima.size == 0:
        return 0.0
    motion_events = minima / motion.fps
    distances = np.min(np.abs(music.beats[:, None] - motion_events[None]), axis=1)
    return float(np.mean(np.exp(-(distances**2) / (2 * tolerance**2))))


def evaluate_sequence(motion: MotionSequence, music: MusicFeatures) -> dict[str, float | None]:
    foot_skating = foot_skating_metric(motion)
    return {
        "foot_skating": foot_skating if np.isfinite(foot_skating) else None,
        "beat_alignment": beat_alignment_score(motion, music),
        "root_path_length": float(np.sum(np.linalg.norm(np.diff(motion.root_positions, axis=0), axis=-1))),
        "mean_articulated_energy": float(np.mean(proxy_articulated_energy(motion))),
    }
