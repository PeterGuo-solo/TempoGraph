from __future__ import annotations

import numpy as np
import torch

from tempograph.generation import _align_motion_beats
from tempograph.engine import build_models, stage_loss
from tempograph.losses import (
    kinetic_distribution_coverage_loss,
    lodge_kinetic_descriptor,
)


def test_lodge_kinetic_descriptor_has_official_shape_and_gradients() -> None:
    joints = torch.randn(4, 32, 22, 3, requires_grad=True)
    descriptor = lodge_kinetic_descriptor(joints)
    assert descriptor.shape == (4, 66)
    descriptor.mean().backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()


def test_kinetic_coverage_prefers_reference_over_collapsed_batch() -> None:
    reference = torch.randn(4, 32, 22, 3)
    collapsed = reference[:1].expand_as(reference).clone()
    matched = kinetic_distribution_coverage_loss(
        reference, reference, diversity_gain=1.0
    )
    failed = kinetic_distribution_coverage_loss(
        collapsed, reference, diversity_gain=1.0
    )
    assert matched < failed


def test_sparse_accents_preserve_shape_and_use_exact_wav_beats() -> None:
    frames = 96
    features = np.zeros((frames, 139), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    features[:, 7:] = np.tile(identity, 22)
    features[:, 4] = np.arange(frames, dtype=np.float32) * 0.01
    music = np.zeros((frames, 35), dtype=np.float32)
    music[[3, 23, 43], -1] = 1.0
    wav_beats = np.asarray([12, 30, 48, 66, 84])
    result, diagnostics = _align_motion_beats(
        features,
        music,
        np.zeros((24, 3), dtype=np.float32),
        beat_frames=wav_beats,
        punctuation_strength=0.4,
        accent_every=4,
        accent_strength=0.8,
    )
    assert result.shape == features.shape
    assert np.isfinite(result).all()
    assert diagnostics["beat_source"] == "wav"
    assert diagnostics["music_beat_count"] == len(wav_beats)
    assert diagnostics["accent_every"] == 4
    assert diagnostics["accent_strength"] == 0.8


def test_v73_joint_objective_is_finite_with_official_beat_batch() -> None:
    config = {
        "seed": 7,
        "motion": {"feature_dim": 139, "contact_dim": 4},
        "music": {"feature_dim": 35},
        "boundary": {},
        "model": {
            "latent_dim": 8,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
        "training": {
            "graph_nodes": 4,
            "training_rollout_steps": 2,
            "boundary_training_rollout_steps": 2,
            "kinetic_diversity_gain": 1.03,
            "loss_weights": {
                "beat_alignment": 6.0,
                "kinetic_coverage": 2.0,
                "graph_edge_coverage": 0.5,
                "geometric_distribution": 4.0,
            },
        },
    }
    models = build_models(config, torch.device("cpu"))
    rest = torch.zeros(24, 3)
    rest[:, 1] = torch.linspace(0.0, -1.0, 24)
    models["local_generator"].configure_skeleton(rest)
    motion = torch.randn(4, 24, 139)
    motion[:, :, :4] = torch.randint(0, 2, (4, 24, 4)).float()
    music = torch.randn(4, 24, 35)
    beat_evidence = torch.zeros(4, 24)
    beat_evidence[:, (4, 10, 16, 22)] = 1.0
    total, metrics = stage_loss(
        "joint_finetune",
        models,
        {
            "motion": motion,
            "music": music,
            "beat_evidence": beat_evidence,
            "style_fine": ["a", "b", "c", "d"],
        },
        config,
    )
    assert torch.isfinite(total)
    assert torch.isfinite(metrics["local_kinetic_coverage"])
    assert torch.isfinite(metrics["graph_edge_coverage"])
    total.backward()
