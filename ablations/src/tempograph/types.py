"""Canonical typed containers shared by every TempoGraph stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray
EDGE_TYPES = {"repeat", "variation", "contrast", "transition"}


def _as_float32(value: Array) -> Array:
    return np.asarray(value, dtype=np.float32)


@dataclass
class MotionSequence:
    rotations_6d: Array
    root_positions: Array
    contacts: Array
    fps: float = 30.0
    local_joint_positions: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rotations_6d = _as_float32(self.rotations_6d)
        self.root_positions = _as_float32(self.root_positions)
        self.contacts = _as_float32(self.contacts)
        if self.local_joint_positions is not None:
            self.local_joint_positions = _as_float32(self.local_joint_positions)
        self.validate()

    @property
    def frames(self) -> int:
        return int(self.rotations_6d.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.rotations_6d.shape[1])

    @property
    def feature_dim(self) -> int:
        return 4 + 3 + 6 * self.joint_count

    def validate(self) -> None:
        if self.rotations_6d.ndim != 3 or self.rotations_6d.shape[-1] != 6:
            raise ValueError("rotations_6d must have shape [T, J, 6]")
        if self.root_positions.shape != (self.frames, 3):
            raise ValueError("root_positions must have shape [T, 3]")
        if self.contacts.shape != (self.frames, 4):
            raise ValueError("contacts must have shape [T, 4]")
        if self.local_joint_positions is not None and self.local_joint_positions.shape != (
            self.frames,
            self.joint_count,
            3,
        ):
            raise ValueError("local_joint_positions must have shape [T, J, 3]")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not np.isfinite(self.rotations_6d).all() or not np.isfinite(self.root_positions).all():
            raise ValueError("motion contains non-finite values")
        if np.any((self.contacts < 0) | (self.contacts > 1)):
            raise ValueError("contacts must be probabilities in [0, 1]")

    def to_features(self) -> Array:
        """Pack to `[T, 4 + 3 + J*6]`, matching LODGE/FineDance ordering."""
        return np.concatenate(
            [self.contacts, self.root_positions, self.rotations_6d.reshape(self.frames, -1)],
            axis=-1,
        ).astype(np.float32, copy=False)

    @classmethod
    def from_features(cls, features: Array, fps: float = 30.0) -> "MotionSequence":
        features = _as_float32(features)
        if features.ndim != 2 or features.shape[1] < 13:
            raise ValueError("features must have shape [T, 7 + 6*J]")
        remainder = features.shape[1] - 7
        if remainder % 6:
            raise ValueError("feature dimension must equal 4 + 3 + 6*J")
        joints = remainder // 6
        return cls(
            rotations_6d=features[:, 7:].reshape(features.shape[0], joints, 6),
            root_positions=features[:, 4:7],
            contacts=features[:, :4],
            fps=fps,
        )

    def slice(self, start: int, end: int) -> "MotionSequence":
        if not 0 <= start < end <= self.frames:
            raise ValueError("invalid motion slice")
        local = None if self.local_joint_positions is None else self.local_joint_positions[start:end]
        return MotionSequence(
            self.rotations_6d[start:end],
            self.root_positions[start:end],
            self.contacts[start:end],
            self.fps,
            local,
            dict(self.metadata),
        )


@dataclass(frozen=True)
class MusicSegment:
    start: float
    end: float
    label: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("music segment must satisfy 0 <= start < end")


@dataclass
class MusicFeatures:
    frame_features: Array
    fps: float
    beats: Array = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    downbeats: Array = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    beat_positions: Array = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    segments: list[MusicSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_features = _as_float32(self.frame_features)
        self.beats = _as_float32(self.beats).reshape(-1)
        self.downbeats = _as_float32(self.downbeats).reshape(-1)
        self.beat_positions = np.asarray(self.beat_positions, dtype=np.int64).reshape(-1)
        if self.frame_features.ndim != 2:
            raise ValueError("frame_features must have shape [T, F]")
        if self.fps <= 0:
            raise ValueError("music fps must be positive")
        if self.beat_positions.size not in (0, self.beats.size):
            raise ValueError("beat_positions must be empty or align with beats")

    @property
    def duration_seconds(self) -> float:
        return self.frame_features.shape[0] / self.fps


@dataclass
class ChoreographyNode:
    index: int
    start_frame: int
    end_frame: int
    role: str
    semantic: str
    keyframes: list[int]
    confidence: float
    embedding: Array | None = None
    boundary_summary: dict[str, Array] = field(default_factory=dict)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ChoreographyEdge:
    source: int
    target: int
    edge_type: str
    confidence: float

    def __post_init__(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"unsupported choreography edge type: {self.edge_type}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("edge confidence must lie in [0, 1]")


@dataclass
class ChoreographyGraph:
    nodes: list[ChoreographyNode]
    edges: list[ChoreographyEdge]
    total_frames: int

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("choreography graph must contain nodes")
        ordered = sorted(self.nodes, key=lambda node: node.start_frame)
        if ordered[0].start_frame != 0 or ordered[-1].end_frame != self.total_frames:
            raise ValueError("nodes must cover the complete sequence")
        for left, right in zip(ordered, ordered[1:]):
            if left.end_frame != right.start_frame:
                raise ValueError("nodes must be contiguous and non-overlapping")
        valid_ids = {node.index for node in self.nodes}
        if any(edge.source not in valid_ids or edge.target not in valid_ids for edge in self.edges):
            raise ValueError("edge references an unknown node")


@dataclass
class BoundaryPlan:
    length: int
    contact_probabilities: Array
    root_positions: Array
    root_orientations: Array
    energy: Array
    conflict: Array

    def __post_init__(self) -> None:
        self.contact_probabilities = _as_float32(self.contact_probabilities)
        self.root_positions = _as_float32(self.root_positions)
        self.root_orientations = _as_float32(self.root_orientations)
        self.energy = _as_float32(self.energy).reshape(-1)
        self.conflict = _as_float32(self.conflict).reshape(-1)
        if self.length < 2:
            raise ValueError("boundary plan needs at least two frames")
        if self.contact_probabilities.shape != (self.length, 4):
            raise ValueError("contact plan must have shape [L, 4]")
        if self.root_positions.shape != (self.length, 3):
            raise ValueError("root path must have shape [L, 3]")
        if self.root_orientations.shape != (self.length, 3, 3):
            raise ValueError("root orientations must have shape [L, 3, 3]")
        if self.energy.shape != (self.length,):
            raise ValueError("energy plan must have shape [L]")
        if self.conflict.shape != (5,):
            raise ValueError("conflict vector must have five components")

