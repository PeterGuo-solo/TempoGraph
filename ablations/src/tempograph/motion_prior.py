"""Training-split-only motion retrieval prior used by TempoGraph V6.

The prior is deliberately built from manifest records whose split is ``train``.
It never reads validation or test motion.  Retrieval is conditioned only on
music, while TempoGraph remains responsible for graph structure, residual
generation, and learned boundary transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data.dataset import read_manifest
from .data.finedance import load_aligned_record
from .geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix


def music_descriptor(music: np.ndarray) -> np.ndarray:
    """Summarize timbre, dynamics, change rate, and beat phase of a clip."""
    music = np.asarray(music, dtype=np.float32)
    if music.ndim != 2 or music.shape[1] != 35 or len(music) == 0:
        raise ValueError("music must have shape [T, 35] with at least one frame")
    mean = music.mean(0)
    std = music.std(0)
    change = np.abs(np.diff(music, axis=0)).mean(0) if len(music) > 1 else np.zeros(35)
    # Four phase bins retain coarse beat placement instead of only beat density.
    beat = music[:, -1]
    phase = np.asarray(
        [beat[index::4].mean() if len(beat[index::4]) else 0.0 for index in range(4)],
        dtype=np.float32,
    )
    return np.concatenate([mean, std, change, phase]).astype(np.float32)


def _resample(values: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if len(values) == frames:
        return np.asarray(values, dtype=np.float32).copy()
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    target = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    result = np.stack([np.interp(target, source, values[:, channel]) for channel in range(values.shape[1])], axis=1)
    return result.astype(np.float32)


def _beat_frames(music: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(music)[:, -1] > 0.5).astype(np.float32)


def beat_synchronous_resample(
    motion: np.ndarray,
    source_beats: np.ndarray,
    target_beats: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, bool]:
    """Time-warp a retrieved dance so its beat anchors match the query music."""
    motion = np.asarray(motion, dtype=np.float32)
    source_beats = np.asarray(source_beats, dtype=np.float32)
    target_beats = np.asarray(target_beats, dtype=np.float32)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if len(source_beats) < 2 or len(target_beats) < 2:
        return _resample(motion, frames), False

    # Use the smaller beat count and select matching beat quantiles.  End-point
    # anchors guarantee a monotonic full-window map even for partial phrases.
    count = min(len(source_beats), len(target_beats))
    quantiles = np.linspace(0.0, 1.0, count, dtype=np.float32)
    source_index = np.rint(quantiles * (len(source_beats) - 1)).astype(np.int64)
    target_index = np.rint(quantiles * (len(target_beats) - 1)).astype(np.int64)
    source_anchor = np.concatenate(
        [[0.0], source_beats[source_index], [float(len(motion) - 1)]]
    )
    target_anchor = np.concatenate(
        [[0.0], target_beats[target_index], [float(frames - 1)]]
    )
    # Duplicate anchors can occur when a beat lands at an endpoint.  Keep the
    # final source coordinate for each target coordinate.
    unique_target, reverse = np.unique(target_anchor[::-1], return_index=True)
    keep = len(target_anchor) - 1 - reverse
    order = np.argsort(target_anchor[keep])
    target_anchor = target_anchor[keep][order]
    source_anchor = source_anchor[keep][order]
    target_grid = np.arange(frames, dtype=np.float32)
    source_grid = np.interp(target_grid, target_anchor, source_anchor)
    original_grid = np.arange(len(motion), dtype=np.float32)
    result = np.stack(
        [np.interp(source_grid, original_grid, motion[:, channel]) for channel in range(motion.shape[1])],
        axis=1,
    ).astype(np.float32)
    # Contacts are labels, not continuous coordinates.
    nearest = np.rint(source_grid).astype(np.int64).clip(0, len(motion) - 1)
    result[:, :4] = motion[nearest, :4]
    return result, True


@dataclass(frozen=True)
class PriorCandidate:
    record_id: str
    song_name: str
    start: int
    end: int
    motion: np.ndarray
    beats: np.ndarray


class TrainingMotionPrior:
    """A deterministic, leakage-safe bank of aligned training motion windows."""

    def __init__(self, descriptors: np.ndarray, candidates: list[PriorCandidate]) -> None:
        if len(descriptors) != len(candidates) or not candidates:
            raise ValueError("motion-prior bank must contain matching non-empty entries")
        self.candidates = candidates
        self.mean = descriptors.mean(0)
        self.std = descriptors.std(0).clip(1e-4)
        self.descriptors = (descriptors - self.mean) / self.std

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        window_frames: int,
        stride: int,
        maximum_candidates: int | None = None,
    ) -> "TrainingMotionPrior":
        manifest = Path(manifest).resolve()
        if window_frames <= 0 or stride <= 0:
            raise ValueError("window_frames and stride must be positive")
        descriptors: list[np.ndarray] = []
        candidates: list[PriorCandidate] = []
        for record in read_manifest(manifest):
            if record.split != "train":
                continue
            aligned = load_aligned_record(manifest, record)
            motion, music = aligned["motion"], aligned["music"]
            starts = list(range(0, max(1, len(music) - window_frames + 1), stride))
            if starts[-1] + window_frames < len(music):
                starts.append(len(music) - window_frames)
            for start in sorted(set(max(0, value) for value in starts)):
                end = min(len(music), start + window_frames)
                if end - start < max(8, window_frames // 4):
                    continue
                descriptors.append(music_descriptor(music[start:end]))
                candidate_motion = np.asarray(motion[start:end], dtype=np.float32).copy()
                candidate_motion[:, 4:7] -= candidate_motion[:1, 4:7]
                candidates.append(
                    PriorCandidate(
                        record.name,
                        record.song_name or record.name,
                        start,
                        end,
                        candidate_motion,
                        _beat_frames(music[start:end]),
                    )
                )
                if maximum_candidates is not None and len(candidates) >= maximum_candidates:
                    return cls(np.stack(descriptors), candidates)
        if not candidates:
            raise ValueError("manifest contains no usable training windows")
        return cls(np.stack(descriptors), candidates)

    def retrieve(
        self,
        music: np.ndarray,
        frames: int,
        top_k: int,
        rng: np.random.Generator,
        temperature: float = 0.15,
        excluded_song: str | None = None,
        minimum_distance: float = 1e-6,
        beat_warp: bool = True,
    ) -> tuple[np.ndarray, dict[str, object]]:
        query = (music_descriptor(music) - self.mean) / self.std
        distance = np.mean((self.descriptors - query[None]) ** 2, axis=1)
        eligible = np.asarray(
            [
                item.song_name != excluded_song
                and float(distance[index]) > float(minimum_distance)
                for index, item in enumerate(self.candidates)
            ],
            dtype=bool,
        )
        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            raise ValueError("no leakage-safe motion-prior candidates remain")
        eligible_distance = distance[eligible_indices]
        count = min(max(1, int(top_k)), len(eligible_indices))
        local_nearest = np.argpartition(eligible_distance, count - 1)[:count]
        nearest = eligible_indices[local_nearest]
        nearest = nearest[np.argsort(distance[nearest])]
        scaled = distance[nearest] - distance[nearest].min()
        probability = np.exp(-scaled / max(float(temperature), 1e-4))
        probability /= probability.sum()
        selected = int(rng.choice(nearest, p=probability))
        item = self.candidates[selected]
        if beat_warp:
            motion, beat_warped = beat_synchronous_resample(
                item.motion, item.beats, _beat_frames(music), frames
            )
        else:
            motion, beat_warped = _resample(item.motion, frames), False
        motion[:, 4:7] -= motion[:1, 4:7]
        return motion, {
            "source_split": "train",
            "source_id": item.record_id,
            "source_song": item.song_name,
            "source_start": item.start,
            "source_end": item.end,
            "descriptor_distance": float(distance[selected]),
            "top_k": count,
            "beat_warped": beat_warped,
            "source_beat_count": int(len(item.beats)),
            "target_beat_count": int(len(_beat_frames(music))),
            "excluded_song": excluded_song,
            "near_duplicate_floor": float(minimum_distance),
        }


def blend_with_motion_prior(
    generated: np.ndarray,
    prior: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend in velocity/rotation space so the prior cannot introduce jumps."""
    generated = np.asarray(generated, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    if generated.shape != prior.shape or generated.ndim != 2 or generated.shape[1] != 139:
        raise ValueError("generated and prior must have matching [T, 139] shapes")
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight == 0.0:
        return generated.copy()
    result = generated.copy()
    # Generated contacts are logits until final sanitization.  Express the
    # binary training contact prior in the same domain before blending.
    prior_contact = np.clip(prior[:, :4], 1e-3, 1.0 - 1e-3)
    prior_logits = np.log(prior_contact / (1.0 - prior_contact))
    result[:, :4] = (1.0 - weight) * generated[:, :4] + weight * prior_logits
    for channel in (4, 6):
        generated_velocity = np.diff(generated[:, channel], prepend=generated[:1, channel])
        prior_velocity = np.diff(prior[:, channel], prepend=prior[:1, channel])
        velocity = (1.0 - weight) * generated_velocity + weight * prior_velocity
        result[:, channel] = generated[0, channel] + np.cumsum(velocity)
    result[:, 5] = (1.0 - weight) * generated[:, 5] + weight * prior[:, 5]
    rotations = (
        (1.0 - weight) * generated[:, 7:].reshape(len(generated), 22, 6)
        + weight * prior[:, 7:].reshape(len(prior), 22, 6)
    )
    result[:, 7:] = matrix_to_rotation_6d(rotation_6d_to_matrix(rotations)).reshape(len(result), -1)
    return result
