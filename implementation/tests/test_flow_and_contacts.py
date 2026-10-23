import numpy as np
import pytest


torch = pytest.importorskip("torch")

from tempograph.benchmarking import _contact_metrics
from tempograph.losses import balanced_contact_loss, soft_contact_f1_loss
from tempograph.models.local_generator import (
    LocalMotionGenerator,
    expand_graph_condition,
    rectified_flow_training_pair,
    sample_local_flow,
)
from tempograph.engine import STAGES, build_models, stage_loss
from tempograph.models.motion_autoencoder import MotionAutoencoder


def test_graph_nodes_expand_to_requested_frames():
    nodes = torch.arange(1 * 3 * 2, dtype=torch.float32).reshape(1, 3, 2)
    durations = torch.tensor([[1.0, 2.0, 1.0]])
    expanded = expand_graph_condition(nodes, 8, durations)
    assert expanded.shape == (1, 8, 2)
    assert torch.equal(expanded[:, :2], nodes[:, :1].expand(-1, 2, -1))
    assert torch.equal(expanded[:, 2:6], nodes[:, 1:2].expand(-1, 4, -1))
    assert torch.equal(expanded[:, 6:], nodes[:, 2:3].expand(-1, 2, -1))


def test_rectified_flow_pair_reaches_data_at_one():
    data = torch.randn(2, 5, 7)
    noise = torch.randn_like(data)
    time = torch.ones(2)
    interpolation, velocity, returned_time = rectified_flow_training_pair(data, noise, time)
    torch.testing.assert_close(interpolation, data)
    torch.testing.assert_close(velocity, data - noise)
    torch.testing.assert_close(returned_time, time)


def test_local_flow_model_shapes_and_sampler():
    model = LocalMotionGenerator(
        motion_dim=19,
        music_dim=5,
        graph_dim=6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    noise = torch.randn(2, 12, 15)
    music = torch.randn(2, 12, 5)
    graph = torch.randn(2, 3, 6)
    durations = torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]])
    continuous, contact_logits = sample_local_flow(
        model, noise, music, graph, durations, steps=2
    )
    assert continuous.shape == (2, 12, 15)
    assert contact_logits.shape == (2, 12, 4)
    assert torch.isfinite(continuous).all()
    assert torch.isfinite(contact_logits).all()


def test_local_flow_codec_round_trip_uses_planar_velocity_and_vertical_height():
    model = LocalMotionGenerator(
        motion_dim=19, music_dim=5, graph_dim=6, hidden_dim=16,
        num_layers=1, num_heads=4, dropout=0.0,
    )
    raw = torch.randn(2, 12, 15)
    raw[:, :, :3] = torch.cumsum(raw[:, :, :3] * 0.05, dim=1) + 10.0
    relative = model.to_relative_representation(raw)
    model.configure_statistics(
        relative.mean((0, 1)), relative.std((0, 1)).clamp_min(1e-3),
        torch.tensor([0.8, 0.7, 0.6, 0.5]),
    )
    decoded = model.decode_motion(model.encode_motion(raw))
    torch.testing.assert_close(decoded[:, :, 3:], raw[:, :, 3:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[:, :, 1], raw[:, :, 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decoded[:, 1:, [0, 2]] - decoded[:, :-1, [0, 2]],
        raw[:, 1:, [0, 2]] - raw[:, :-1, [0, 2]], atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(decoded[:, 0, [0, 2]], torch.zeros_like(decoded[:, 0, [0, 2]]))


def test_graph_attributes_and_dense_relations_change_conditioning():
    from tempograph.models.graph_planner import GraphPlanner
    model = GraphPlanner(
        music_dim=5, hidden_dim=16, num_layers=1, num_heads=4,
        semantic_classes=8, graph_dim=6, dropout=0.0,
    ).eval()
    output = model(torch.randn(2, 7, 5))
    assert output["dense_edge_logits"].shape == (2, 7, 7, 4)
    assert output["graph_condition"].shape == (2, 7, 6)
    loss = output["graph_condition"].square().mean()
    loss.backward()
    assert model.role_head.weight.grad is not None
    assert model.keyframe_head.weight.grad is not None
    assert model.boundary_head.weight.grad is not None
    assert model.edge_head.weight.grad is not None


def test_balanced_contact_loss_penalizes_all_positive_solution():
    target = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    all_positive = torch.full_like(target, 8.0)
    correct = torch.tensor([[[8.0], [8.0], [8.0], [-8.0]]])
    assert balanced_contact_loss(correct, target) < balanced_contact_loss(all_positive, target)


def test_fixed_prior_and_soft_f1_penalize_all_positive_contacts():
    target = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    prior = torch.tensor([0.75])
    all_positive = torch.full_like(target, 8.0)
    correct = torch.tensor([[[8.0], [8.0], [8.0], [-8.0]]])
    assert balanced_contact_loss(correct, target, prior) < balanced_contact_loss(all_positive, target, prior)
    assert soft_contact_f1_loss(correct, target) < soft_contact_f1_loss(all_positive, target)
    # With the real FineDance-like 88% positive prior, positive-class F1 alone
    # would make all-positive look deceptively good.  Macro-F1 must retain a
    # substantial penalty through the negative class.
    imbalanced_target = torch.cat([torch.ones(88), torch.zeros(12)]).reshape(1, 100, 1)
    collapsed = torch.full_like(imbalanced_target, 12.0)
    assert float(soft_contact_f1_loss(collapsed, imbalanced_target)) > 0.45


def test_contact_diagnostics_expose_all_positive_collapse():
    prediction = np.ones((10, 139), dtype=np.float32)
    reference = np.zeros((10, 139), dtype=np.float32)
    reference[:8, :4] = 1.0
    metrics = _contact_metrics(prediction, reference)
    assert metrics["contact_accuracy"] == pytest.approx(0.8)
    assert metrics["contact_balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["contact_positive_rate"] == pytest.approx(1.0)


def test_all_training_stages_remain_finite_after_audit_changes():
    config = {
        "seed": 3,
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
        "training": {"graph_nodes": 4, "loss_weights": {}},
    }
    models = build_models(config, torch.device("cpu"))
    rest = torch.zeros(24, 3)
    rest[:, 1] = torch.linspace(0.0, -1.0, 24)
    models["local_generator"].configure_skeleton(rest)
    motion = torch.randn(4, 24, 139)
    motion[:, :, :4] = torch.randint(0, 2, (4, 24, 4)).float()
    music = torch.randn(4, 24, 35)
    music[:, :, -1] = 0
    music[:, ::6, -1] = 1
    batch = {
        "motion": motion,
        "music": music,
        "style_fine": ["a", "b", "c", "d"],
    }
    for stage in STAGES:
        total, metrics = stage_loss(stage, models, batch, config)
        assert torch.isfinite(total), stage
        assert all(torch.isfinite(value) for value in metrics.values()), stage
        total.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for model in models.values() for parameter in model.parameters()
        ), stage
        for model in models.values():
            model.zero_grad(set_to_none=True)


def test_sampled_rotations_are_projected_to_orthonormal_rows():
    model = LocalMotionGenerator(
        motion_dim=139, music_dim=5, graph_dim=6, hidden_dim=16,
        num_layers=1, num_heads=4, dropout=0.0,
    ).eval()
    continuous, _ = sample_local_flow(
        model,
        torch.randn(1, 8, 135),
        torch.randn(1, 8, 5),
        torch.randn(1, 2, 6),
        steps=2,
    )
    rotations = continuous[..., 3:].reshape(1, 8, 22, 6)
    first, second = rotations[..., :3], rotations[..., 3:]
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=-1), torch.ones(1, 8, 22))
    torch.testing.assert_close(torch.linalg.vector_norm(second, dim=-1), torch.ones(1, 8, 22))
    torch.testing.assert_close((first * second).sum(-1), torch.zeros(1, 8, 22), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("frames", [14, 15, 16, 29, 30])
def test_motion_autoencoder_reconstructs_exact_requested_length(frames):
    model = MotionAutoencoder(feature_dim=19, latent_dim=8, hidden_dim=16)
    motion = torch.randn(2, frames, 19)
    reconstruction, _ = model(motion)
    assert reconstruction.shape == motion.shape
    assert torch.isfinite(reconstruction).all()


def test_boundary_bridge_handles_full_window_fifteen_frame_target():
    config = {
        "seed": 3,
        "motion": {"feature_dim": 139, "contact_dim": 4},
        "music": {"feature_dim": 35},
        "boundary": {},
        "model": {"latent_dim": 8, "hidden_dim": 16, "num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "training": {"graph_nodes": 4, "boundary_training_rollout_steps": 2, "loss_weights": {}},
    }
    models = build_models(config, torch.device("cpu"))
    models["local_generator"].configure_skeleton(torch.zeros(24, 3))
    motion = torch.randn(2, 240, 139)
    motion[:, :, :4] = torch.randint(0, 2, (2, 240, 4)).float()
    music = torch.randn(2, 240, 35)
    batch = {"motion": motion, "music": music, "style_fine": ["a", "b"]}
    total, metrics = stage_loss("boundary_bridge", models, batch, config)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in metrics.values())
    total.backward()
