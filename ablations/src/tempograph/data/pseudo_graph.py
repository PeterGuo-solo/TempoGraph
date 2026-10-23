"""Confidence-weighted pseudo choreography graph construction."""

from __future__ import annotations

import numpy as np

from ..types import ChoreographyEdge, ChoreographyGraph, ChoreographyNode, MotionSequence, MusicFeatures
from .music import section_role_at


def _smooth(values: np.ndarray, radius: int = 3) -> np.ndarray:
    if radius <= 0:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    return np.convolve(values, kernel, mode="same")


def motion_saliency(motion: MotionSequence, music: MusicFeatures | None = None) -> np.ndarray:
    packed = motion.to_features()[:, 4:]
    velocity = np.zeros(motion.frames, dtype=np.float32)
    velocity[1:] = np.linalg.norm(np.diff(packed, axis=0), axis=-1)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = np.abs(np.diff(velocity))
    saliency = 0.65 * _smooth(acceleration) + 0.35 * _smooth(velocity)
    if music is not None and music.beats.size:
        beat_frames = np.clip(np.rint(music.beats * motion.fps).astype(int), 0, motion.frames - 1)
        impulse = np.zeros(motion.frames, dtype=np.float32)
        impulse[beat_frames] = 1.0
        saliency += 0.25 * _smooth(impulse, radius=2)
    maximum = float(np.max(saliency))
    return saliency / maximum if maximum > 0 else saliency


def _segment_boundaries(
    saliency: np.ndarray,
    min_frames: int,
    max_frames: int,
) -> list[int]:
    if min_frames < 2 or max_frames < min_frames:
        raise ValueError("invalid segment limits")
    peaks = [
        index
        for index in range(1, len(saliency) - 1)
        if saliency[index] >= saliency[index - 1] and saliency[index] > saliency[index + 1]
    ]
    candidates = sorted(peaks, key=lambda index: float(saliency[index]), reverse=True)
    boundaries = [0, len(saliency)]
    for candidate in candidates:
        left = max(boundary for boundary in boundaries if boundary < candidate)
        right = min(boundary for boundary in boundaries if boundary > candidate)
        if candidate - left >= min_frames and right - candidate >= min_frames:
            boundaries.append(candidate)
            boundaries.sort()
    changed = True
    while changed:
        changed = False
        result = [boundaries[0]]
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left > max_frames:
                result.append(left + max_frames)
                changed = True
            result.append(right)
        boundaries = sorted(set(result))
    return boundaries


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _classify_similarity(similarity: float, repeat: float, variation: float, contrast: float) -> tuple[str, float]:
    if similarity >= repeat:
        return "repeat", min(1.0, (similarity - repeat) / max(1e-6, 1 - repeat) * 0.5 + 0.5)
    if similarity >= variation:
        midpoint = (repeat + variation) / 2
        confidence = 0.5 + min(0.5, abs(similarity - midpoint) / max(1e-6, repeat - variation))
        return "variation", confidence
    if similarity <= contrast:
        return "contrast", min(1.0, (contrast - similarity) / max(1e-6, 1 + contrast) * 0.5 + 0.5)
    return "transition", 0.5 + min(0.45, abs(similarity - 0.5) * 0.4)


def build_pseudo_choreography_graph(
    motion: MotionSequence,
    music: MusicFeatures | None = None,
    min_segment_frames: int = 30,
    max_segment_frames: int = 240,
    repeat_threshold: float = 0.90,
    variation_threshold: float = 0.70,
    contrast_threshold: float = 0.20,
) -> ChoreographyGraph:
    saliency = motion_saliency(motion, music)
    boundaries = _segment_boundaries(saliency, min_segment_frames, max_segment_frames)
    packed = motion.to_features()
    nodes: list[ChoreographyNode] = []
    embeddings: list[np.ndarray] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        embedding = packed[start:end].mean(axis=0)
        embeddings.append(embedding)
        keyframe = start + int(np.argmax(saliency[start:end]))
        role = section_role_at((start + end) / (2 * motion.fps), music.segments) if music else "development"
        confidence = float(np.clip(0.5 + saliency[keyframe] * 0.5, 0, 1))
        nodes.append(
            ChoreographyNode(
                index=index,
                start_frame=start,
                end_frame=end,
                role=role,
                semantic=f"motion_cluster_{index}",
                keyframes=[keyframe],
                confidence=confidence,
                embedding=embedding,
                boundary_summary={"start": packed[start], "end": packed[end - 1]},
            )
        )
    edges: list[ChoreographyEdge] = []
    for index in range(len(nodes) - 1):
        similarity = _cosine(embeddings[index], embeddings[index + 1])
        edge_type, confidence = _classify_similarity(
            similarity, repeat_threshold, variation_threshold, contrast_threshold
        )
        edges.append(ChoreographyEdge(index, index + 1, edge_type, confidence))
    for source in range(len(nodes)):
        for target in range(source + 2, len(nodes)):
            similarity = _cosine(embeddings[source], embeddings[target])
            if similarity >= variation_threshold:
                edge_type, confidence = _classify_similarity(
                    similarity, repeat_threshold, variation_threshold, contrast_threshold
                )
                edges.append(ChoreographyEdge(source, target, edge_type, confidence))
    graph = ChoreographyGraph(nodes, edges, motion.frames)
    graph.validate()
    return graph
