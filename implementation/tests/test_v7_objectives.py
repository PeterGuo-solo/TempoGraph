from __future__ import annotations

import numpy as np
import torch

from tempograph.generation import _align_motion_beats
from tempograph.losses import (
    beat_alignment_loss,
    contact_transition_loss,
    diversity_preservation_loss,
    foot_skating_loss,
    geometric_distribution_loss,
)


def test_beat_alignment_loss_prefers_energy_minima_near_beats() -> None:
    frames = 24
    beats = torch.zeros(1, frames)
    beats[:, (6, 16)] = 1.0
    aligned_speed = torch.ones(frames)
    aligned_speed[[6, 16]] = 0.01
    shifted_speed = torch.ones(frames)
    shifted_speed[[11, 21]] = 0.01

    def joints(speed: torch.Tensor) -> torch.Tensor:
        position = torch.cumsum(speed, dim=0)
        result = torch.zeros(1, frames, 2, 3)
        result[0, :, :, 0] = position[:, None]
        return result

    assert beat_alignment_loss(joints(aligned_speed), beats, 2) < beat_alignment_loss(
        joints(shifted_speed), beats, 2
    )


def test_geometric_distribution_is_zero_for_reference_and_detects_scale() -> None:
    reference = torch.randn(2, 20, 22, 3)
    same = geometric_distribution_loss(reference, reference)
    scaled = geometric_distribution_loss(reference * 2.0, reference)
    assert float(same) < 1e-7
    assert scaled > same


def test_geometric_contact_penalizes_skating_even_with_zero_contact_labels() -> None:
    feet = torch.zeros(1, 12, 4, 3)
    feet[0, :, :, 0] = torch.arange(12)[:, None] * 0.03
    contacts = torch.zeros(1, 12, 4)
    assert foot_skating_loss(feet, contacts) > 0


def test_contact_transition_loss_detects_wrong_switch_timing() -> None:
    target = torch.zeros(1, 12, 4)
    target[:, 4:8] = 1.0
    aligned = torch.logit(target.clamp(0.01, 0.99))
    shifted = torch.roll(aligned, 2, dims=1)
    assert contact_transition_loss(aligned, target) < contact_transition_loss(shifted, target)


def test_diversity_preservation_detects_batch_collapse() -> None:
    reference = torch.randn(4, 20, 22, 3)
    collapsed = reference[:1].expand_as(reference).clone()
    assert diversity_preservation_loss(reference, reference) < diversity_preservation_loss(
        collapsed, reference
    )


def test_inference_beat_warp_is_finite_and_uses_music_only() -> None:
    frames = 32
    features = np.zeros((frames, 139), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    features[:, 7:] = np.tile(identity, 22)
    speed = np.ones(frames, dtype=np.float32)
    speed[[8, 18, 27]] = 0.01
    features[:, 4] = np.cumsum(speed)
    music = np.zeros((frames, 35), dtype=np.float32)
    music[[6, 16, 25], -1] = 1.0
    warped, diagnostics = _align_motion_beats(
        features, music, np.zeros((24, 3), dtype=np.float32), maximum_shift=4
    )
    assert warped.shape == features.shape
    assert np.isfinite(warped).all()
    assert diagnostics["enabled"] is True


def test_wav_beat_override_and_punctuation_create_energy_minima() -> None:
    frames = 64
    features = np.zeros((frames, 139), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    features[:, 7:] = np.tile(identity, 22)
    frame_axis = np.arange(frames, dtype=np.float32)
    wav_beats = np.asarray([12, 32, 52])
    speed_profile = np.ones(frames, dtype=np.float32)
    for beat in wav_beats:
        speed_profile -= 0.8 * np.exp(-0.5 * ((frame_axis - beat) / 4.0) ** 2)
    features[:, 4] = np.cumsum(speed_profile)
    music = np.zeros((frames, 35), dtype=np.float32)
    # Deliberately wrong proxy beats: the explicit WAV beats must win.
    music[[5, 25, 45], -1] = 1.0
    warped, diagnostics = _align_motion_beats(
        features,
        music,
        np.zeros((24, 3), dtype=np.float32),
        maximum_shift=64,
        beat_frames=wav_beats,
        punctuation_strength=1.0,
    )
    speed = np.abs(np.diff(warped[:, 4], prepend=warped[:1, 4]))
    assert diagnostics["beat_source"] == "wav"
    assert diagnostics["punctuation_strength"] == 1.0
    for beat in wav_beats:
        assert speed[beat] <= speed[max(0, beat - 5) : min(frames, beat + 6)].mean()
