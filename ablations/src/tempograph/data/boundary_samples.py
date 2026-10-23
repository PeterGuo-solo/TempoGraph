"""Leakage-safe training samples for Boundary Motion Bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MotionSequence


@dataclass
class BoundarySample:
    left_context: MotionSequence
    transition: MotionSequence
    right_context: MotionSequence
    transition_start: int
    transition_end: int

    @property
    def masked_features(self) -> np.ndarray:
        left = self.left_context.to_features()
        right = self.right_context.to_features()
        mask = np.full((self.transition.frames, left.shape[1]), np.nan, dtype=np.float32)
        return np.concatenate([left, mask, right], axis=0)

    def assert_no_leakage(self) -> None:
        middle = self.masked_features[self.left_context.frames : -self.right_context.frames]
        if not np.isnan(middle).all():
            raise AssertionError("ground-truth transition leaked into conditioning")


def extract_boundary_sample(
    motion: MotionSequence,
    transition_start: int,
    transition_length: int,
    context_frames: int,
) -> BoundarySample:
    transition_end = transition_start + transition_length
    if context_frames < 2 or transition_length < 2:
        raise ValueError("context and transition lengths must be at least two")
    if transition_start - context_frames < 0 or transition_end + context_frames > motion.frames:
        raise ValueError("boundary sample exceeds sequence")
    sample = BoundarySample(
        motion.slice(transition_start - context_frames, transition_start),
        motion.slice(transition_start, transition_end),
        motion.slice(transition_end, transition_end + context_frames),
        transition_start,
        transition_end,
    )
    sample.assert_no_leakage()
    return sample


def enumerate_boundary_samples(
    motion: MotionSequence,
    context_frames: int,
    transition_lengths: tuple[int, ...] = (8, 15, 30),
    stride: int = 15,
) -> list[BoundarySample]:
    samples: list[BoundarySample] = []
    for length in transition_lengths:
        first = context_frames
        last = motion.frames - context_frames - length
        for start in range(first, max(first, last + 1), stride):
            samples.append(extract_boundary_sample(motion, start, length, context_frames))
    return samples

