"""Cubic Hermite reference trajectories for boundary planning."""

from __future__ import annotations

import numpy as np


def cubic_hermite(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    duration: float,
    phases: np.ndarray,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float64).reshape(-1, 1)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start_velocity = np.asarray(start_velocity, dtype=np.float64)
    end_velocity = np.asarray(end_velocity, dtype=np.float64)
    s = phases
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return (
        h00 * start + h10 * duration * start_velocity + h01 * end + h11 * duration * end_velocity
    ).astype(np.float32)


def cubic_hermite_velocity(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    duration: float,
    phases: np.ndarray,
) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration must be positive")
    phases = np.asarray(phases, dtype=np.float64).reshape(-1, 1)
    s = phases
    derivative = (
        (6 * s**2 - 6 * s) * np.asarray(start)
        + (3 * s**2 - 4 * s + 1) * duration * np.asarray(start_velocity)
        + (-6 * s**2 + 6 * s) * np.asarray(end)
        + (3 * s**2 - 2 * s) * duration * np.asarray(end_velocity)
    )
    return (derivative / duration).astype(np.float32)
