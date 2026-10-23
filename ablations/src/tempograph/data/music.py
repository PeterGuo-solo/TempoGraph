"""Music feature alignment and All-In-One structure import."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import MusicFeatures, MusicSegment


def resample_frame_features(
    features: np.ndarray,
    source_fps: float,
    target_fps: float,
    target_frames: int | None = None,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("features must be [T, F] and frame rates must be positive")
    duration = features.shape[0] / source_fps
    if target_frames is None:
        target_frames = max(1, int(round(duration * target_fps)))
    source_times = np.arange(features.shape[0], dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.clip(target_times, source_times[0], source_times[-1])
    columns = [np.interp(target_times, source_times, features[:, index]) for index in range(features.shape[1])]
    return np.stack(columns, axis=-1).astype(np.float32)


def load_allin1_result(
    json_path: str | Path,
    frame_features: np.ndarray,
    feature_fps: float,
) -> MusicFeatures:
    with Path(json_path).open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    segments = [
        MusicSegment(float(item["start"]), float(item["end"]), str(item["label"]))
        for item in result.get("segments", [])
        if float(item["end"]) > float(item["start"])
    ]
    beats = np.asarray(result.get("beats", []), dtype=np.float32)
    positions = np.asarray(result.get("beat_positions", []), dtype=np.int64)
    return MusicFeatures(
        frame_features,
        fps=feature_fps,
        beats=beats,
        downbeats=np.asarray(result.get("downbeats", []), dtype=np.float32),
        beat_positions=positions,
        segments=segments,
        metadata={"bpm": result.get("bpm"), "source": str(json_path)},
    )


def section_role_at(time_seconds: float, segments: list[MusicSegment]) -> str:
    for segment in segments:
        if segment.start <= time_seconds < segment.end:
            mapping = {
                "start": "introduction",
                "intro": "introduction",
                "verse": "development",
                "chorus": "climax",
                "solo": "climax",
                "bridge": "transition",
                "break": "transition",
                "outro": "closure",
                "end": "closure",
            }
            return mapping.get(segment.label.lower(), segment.label.lower())
    return "development"


def beat_interval_near(time_seconds: float, beats: np.ndarray, default: float = 0.5) -> float:
    beats = np.asarray(beats, dtype=np.float32)
    if beats.size < 2:
        return default
    index = int(np.clip(np.searchsorted(beats, time_seconds), 1, beats.size - 1))
    return float(max(1e-3, beats[index] - beats[index - 1]))

