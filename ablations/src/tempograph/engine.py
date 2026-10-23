"""Real FineDance multi-stage training and checkpoint orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.torch_data import FineDanceWindowDataset
from .relation_supervision import relation_targets
from .losses import (
    balanced_contact_loss,
    boundary_reconstruction_losses,
    beat_alignment_loss,
    contact_transition_loss,
    diversity_preservation_loss,
    energy_envelope_loss,
    finite_difference,
    foot_skating_loss,
    forward_kinematics_continuous,
    fk_reconstruction_losses,
    geometric_distribution_loss,
    kinetic_distribution_coverage_loss,
    key_motion_anchor_loss,
    manual_geometry_distribution_loss,
    motion_energy,
    normalized_joint_energy_loss,
    rotation_6d_regularization,
    soft_contact_f1_loss,
)
from .models.boundary_bridge import (
    BoundaryPlanNetwork,
    BoundaryVelocityField,
    integrate_boundary_flow,
    rectified_flow_training_pair as boundary_flow_training_pair,
)
from .models.graph_planner import GraphPlanner, duration_coverage_loss
from .models.local_generator import (
    LocalMotionGenerator,
    integrate_local_flow,
    rectified_flow_training_pair,
    sample_local_flow,
)
from .models.motion_autoencoder import MotionAutoencoder


STAGES = (
    "motion_autoencoder",
    "graph_and_local",
    "boundary_planner",
    "boundary_bridge",
    "joint_finetune",
)
CHECKPOINT_NAMES = {
    "motion_autoencoder": "motion_autoencoder.pt",
    "graph_and_local": "graph_local.pt",
    "boundary_planner": "boundary_planner.pt",
    "boundary_bridge": "boundary_bridge.pt",
    "joint_finetune": "tempograph.pt",
}
EDGE_TYPES = ("repeat", "variation", "contrast", "transition")


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible seed runs are more important here than a small throughput
    # gain.  warn_only avoids aborting on a third-party CUDA kernel that has no
    # deterministic implementation while still recording the intended policy.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def build_models(config: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    model = config["model"]
    motion_dim = config["motion"]["feature_dim"]
    music_dim = config["music"]["feature_dim"]
    common = {
        "hidden_dim": int(model["hidden_dim"]),
    }
    models: dict[str, torch.nn.Module] = {
        "motion_autoencoder": MotionAutoencoder(
            motion_dim, int(model["latent_dim"]), **common
        ),
        "graph_planner": GraphPlanner(
            music_dim=music_dim,
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            graph_dim=int(model["latent_dim"]),
            dropout=float(model["dropout"]),
        ),
        "local_generator": LocalMotionGenerator(
            motion_dim=motion_dim,
            music_dim=music_dim,
            graph_dim=int(model["latent_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            dropout=float(model["dropout"]),
        ),
        "boundary_planner": BoundaryPlanNetwork(motion_dim, music_dim, int(model["hidden_dim"])),
        "boundary_bridge": BoundaryVelocityField(
            latent_dim=int(model["latent_dim"]),
            condition_dim=(
                2 * int(model["latent_dim"])
                + music_dim
                + 3  # transition-length distribution
                + 8  # endpoint contacts
                + 6  # root endpoint residuals
                + 4  # energy statistics
                + 4  # typed edge
            ),
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
        ),
    }
    return {name: module.to(device) for name, module in models.items()}


def _segment_mean(values: torch.Tensor, segments: int) -> torch.Tensor:
    boundaries = torch.linspace(0, values.shape[1], segments + 1, device=values.device).round().long()
    return torch.stack(
        [values[:, boundaries[i] : boundaries[i + 1]].mean(dim=1) for i in range(segments)],
        dim=1,
    )


def _segment_sum(values: torch.Tensor, segments: int) -> torch.Tensor:
    boundaries = torch.linspace(0, values.shape[1], segments + 1, device=values.device).round().long()
    return torch.stack(
        [values[:, boundaries[i] : boundaries[i + 1]].sum(dim=1) for i in range(segments)],
        dim=1,
    )


def _style_ids(styles: Iterable[str], classes: int = 128) -> torch.Tensor:
    values = [int.from_bytes(hashlib.sha256(style.encode("utf-8")).digest()[:4], "big") % classes for style in styles]
    return torch.tensor(values, dtype=torch.long)


def graph_targets(
    motion: torch.Tensor,
    music: torch.Tensor,
    styles: list[str],
    nodes: int,
    beat_evidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    music_nodes = _segment_mean(music, nodes)
    pose_nodes = _segment_mean(motion[:, :, 7:], nodes)
    pose_std = torch.stack([
        part.std(dim=1, unbiased=False)
        for part in torch.tensor_split(motion[:, :, 7:], nodes, dim=1)
    ], dim=1)
    contact_nodes = _segment_mean(motion[:, :, :4], nodes)
    velocity = torch.linalg.vector_norm(motion[:, 1:, 7:] - motion[:, :-1, 7:], dim=-1)
    energy = _segment_mean(velocity.unsqueeze(-1), nodes).squeeze(-1)
    ranks = torch.argsort(torch.argsort(energy, dim=1), dim=1)
    roles = torch.clamp((ranks * 4) // max(nodes, 1), max=3).long()
    semantic = _style_ids(styles).to(motion.device)[:, None].expand(-1, nodes)
    # FineDance music features store beat evidence in the final channel.  Use
    # segment beat counts instead of the former constant two-beat target.
    beat_source = music[:, :, -1] if beat_evidence is None else beat_evidence
    durations = _segment_sum(beat_source[:, :, None].clamp_min(0), nodes).squeeze(-1)
    fallback = motion.new_full((motion.shape[0], nodes), motion.shape[1] / nodes / 15.0)
    no_beats = durations.sum(dim=-1, keepdim=True) < 0.5
    durations = torch.where(no_beats, fallback, durations.clamp_min(0.25))
    descriptor = torch.cat([pose_nodes, pose_std, contact_nodes, energy[..., None]], dim=-1)
    descriptor = descriptor - descriptor.mean(dim=1, keepdim=True)
    normalized = torch.nn.functional.normalize(descriptor, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    contact_gap = torch.cdist(contact_nodes, contact_nodes, p=1) / contact_nodes.shape[-1]
    energy_scale = energy.mean(dim=1, keepdim=True).clamp_min(1e-4)
    energy_gap = torch.abs(energy[:, :, None] - energy[:, None, :]) / energy_scale[:, :, None]
    repeat = (similarity >= 0.90) & (contact_gap <= 0.15) & (energy_gap <= 0.35)
    variation = (similarity >= 0.45) & ~repeat
    contrast = similarity <= -0.20
    dense_edge_types = torch.full_like(similarity, 3, dtype=torch.long)
    dense_edge_types = torch.where(contrast, 2, dense_edge_types)
    dense_edge_types = torch.where(variation, 1, dense_edge_types)
    dense_edge_types = torch.where(repeat, 0, dense_edge_types)
    diagonal = torch.eye(nodes, device=motion.device, dtype=torch.bool)[None]
    dense_confidence = torch.maximum(
        torch.abs(similarity - 0.45), torch.abs(similarity + 0.20)
    ).clamp(0.10, 1.0).masked_fill(diagonal, 0.0)
    indices = torch.arange(nodes - 1, device=motion.device)
    edge_types = dense_edge_types[:, indices, indices + 1]
    confidence = dense_confidence[:, indices, indices + 1]
    return {
        "music_nodes": music_nodes,
        "roles": roles,
        "semantics": semantic,
        "durations": durations,
        "total_beats": durations.sum(dim=-1),
        "edge_types": edge_types,
        "edge_confidence": confidence,
        "dense_edge_types": dense_edge_types,
        "dense_edge_confidence": dense_confidence,
    }


def boundary_targets(
    motion: torch.Tensor,
    music: torch.Tensor,
    graph_edge_types: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    frames = motion.shape[1]
    length = min(15, max(8, frames // 6))
    start = max(2, frames // 2 - length // 2)
    end = start + length
    context = min(15, start, frames - end)
    left = motion[:, start - context : start]
    transition = motion[:, start:end]
    right = motion[:, end : end + context]
    left_summary = left.mean(dim=1)
    right_summary = right.mean(dim=1)
    music_summary = music[:, start:end].mean(dim=1)
    difference = torch.abs(right_summary - left_summary)
    conflict = torch.stack(
        [chunk.mean(dim=-1) for chunk in torch.tensor_split(difference, 5, dim=-1)], dim=-1
    )
    if graph_edge_types is not None and graph_edge_types.shape[1] > 0:
        center_edge = graph_edge_types[:, graph_edge_types.shape[1] // 2]
        edge_one_hot = torch.nn.functional.one_hot(center_edge, num_classes=4).to(motion.dtype)
    else:
        edge_one_hot = motion.new_zeros((motion.shape[0], 4))
        edge_one_hot[:, 3] = 1
    contacts = torch.cat([transition[:, 0, :4], transition[:, -1, :4]], dim=-1)
    root = torch.cat([transition[:, 0, 4:7], transition[:, -1, 4:7]], dim=-1)
    velocity = torch.linalg.vector_norm(transition[:, 1:, 7:] - transition[:, :-1, 7:], dim=-1)
    energy = torch.stack(
        [velocity.mean(1), velocity.std(1), velocity.amin(1), velocity.amax(1)], dim=-1
    )
    conflict_score = difference.mean(dim=-1)
    relative_conflict = conflict_score / conflict_score.mean().clamp_min(1e-6)
    length_class = torch.ones_like(conflict_score, dtype=torch.long)
    length_class = torch.where(relative_conflict < 0.75, 0, length_class)
    length_class = torch.where(relative_conflict > 1.25, 2, length_class)
    return {
        "left": left,
        "right": right,
        "transition": transition,
        "left_summary": left_summary,
        "right_summary": right_summary,
        "music_summary": music_summary,
        "conflict": conflict,
        "edge_one_hot": edge_one_hot,
        "contact_summary": contacts,
        "root_residual": root,
        "energy_parameters": energy,
        "length_class": length_class,
    }


def _loss_weight(config: dict[str, Any], name: str, default: float = 1.0) -> float:
    return float(config.get("training", {}).get("loss_weights", {}).get(name, default))


def _stage_hyperparameter(
    config: dict[str, Any], stage: str, name: str, default: Any
) -> Any:
    training = config["training"]
    return training.get("stage_hyperparameters", {}).get(stage, {}).get(
        name, training.get(name, default)
    )


def _reconstruction_total(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    contact_dim = int(config["motion"].get("contact_dim", 4))
    losses = boundary_reconstruction_losses(
        prediction[:, :, contact_dim:], target[:, :, contact_dim:]
    )
    contact = balanced_contact_loss(
        prediction[:, :, :contact_dim], target[:, :, :contact_dim]
    )
    predicted_energy = motion_energy(prediction[:, :, 7:])
    reference_energy = motion_energy(target[:, :, 7:])
    energy = energy_envelope_loss(
        predicted_energy.clamp_min(1e-6), reference_energy.clamp_min(1e-6)
    )
    root_velocity = torch.nn.functional.l1_loss(
        finite_difference(prediction[:, :, 4:7]),
        finite_difference(target[:, :, 4:7]),
    )
    total = (
        _loss_weight(config, "pose") * losses["pose"]
        + _loss_weight(config, "velocity", 0.5) * losses["velocity"]
        + _loss_weight(config, "acceleration", 0.25) * losses["acceleration"]
        + _loss_weight(config, "contact", 2.0) * contact
        + _loss_weight(config, "energy", 0.5) * energy
        + _loss_weight(config, "root_velocity", 1.0) * root_velocity
    )
    return total, {
        **losses,
        "contact": contact,
        "energy": energy,
        "root_velocity": root_velocity,
    }


def _distribution_moment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match motion-dynamics moments without using validation/test statistics.

    FID collapse in V5 was accompanied by extreme generated velocity variance.
    Matching per-channel first and second moments of root/pose velocities inside
    each training batch constrains that failure while retaining stochasticity.
    """
    predicted_velocity = finite_difference(prediction[:, :, 4:])
    target_velocity = finite_difference(target[:, :, 4:])
    predicted_flat = predicted_velocity.flatten(0, 1)
    target_flat = target_velocity.flatten(0, 1)
    target_mean = target_flat.mean(0)
    predicted_mean = predicted_flat.mean(0)
    target_std = target_flat.std(0, unbiased=False).clamp_min(1e-4)
    predicted_std = predicted_flat.std(0, unbiased=False).clamp_min(1e-4)
    mean = torch.nn.functional.smooth_l1_loss(
        predicted_mean / target_std.detach(), target_mean / target_std.detach()
    )
    spread = torch.nn.functional.smooth_l1_loss(
        torch.log(predicted_std), torch.log(target_std)
    )
    return mean + spread


def _local_flow_total(
    model: LocalMotionGenerator,
    motion: torch.Tensor,
    music: torch.Tensor,
    graph_output: dict[str, torch.Tensor],
    config: dict[str, Any],
    beat_evidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    contact_dim = int(config["motion"].get("contact_dim", 4))
    raw_data = motion[:, :, contact_dim:]
    data = model.encode_motion(raw_data)
    source_noise = torch.randn_like(data)
    interpolation, target_velocity, time = rectified_flow_training_pair(data, noise=source_noise)
    output = model(
        interpolation,
        music,
        graph_output["graph_condition"],
        time,
        graph_output["durations"],
    )
    flow = torch.nn.functional.mse_loss(output["velocity"], target_velocity)
    view = (data.shape[0],) + (1,) * (data.ndim - 1)
    estimated_data = interpolation + (1.0 - time.view(view)) * output["velocity"]
    # Train the contact head on the estimated generated endpoint as well as
    # clean and corrupted-clean inputs.  The former code saw only clean motion
    # here but inference saw sampled motion, causing severe covariate shift and
    # near-100% contact predictions.
    clean_time = torch.ones(data.shape[0], device=data.device, dtype=data.dtype)
    clean_contact_logits = model(
        data,
        music,
        graph_output["graph_condition"],
        clean_time,
        graph_output["durations"],
    )["contact_logits"]
    endpoint_contact_logits = model(
        estimated_data.detach(), music, graph_output["graph_condition"], clean_time,
        graph_output["durations"],
    )["contact_logits"]
    contact_noise = float(config["training"].get("contact_input_noise", 0.15))
    corrupted_contact_logits = model(
        data + contact_noise * torch.randn_like(data), music,
        graph_output["graph_condition"], clean_time, graph_output["durations"],
    )["contact_logits"]
    target_contacts = motion[:, :, :contact_dim]
    endpoint_contact = balanced_contact_loss(
        endpoint_contact_logits, target_contacts, model.contact_prior
    )
    clean_contact = balanced_contact_loss(
        clean_contact_logits, target_contacts, model.contact_prior
    )
    corrupted_contact = balanced_contact_loss(
        corrupted_contact_logits, target_contacts, model.contact_prior
    )
    contact_f1 = soft_contact_f1_loss(endpoint_contact_logits, target_contacts)
    estimated_raw = model.decode_motion(estimated_data, project_rotations=True)
    estimated_motion = torch.cat([endpoint_contact_logits, estimated_raw], dim=-1)
    reconstruction, losses = _reconstruction_total(estimated_motion, motion, config)
    # Replace the batch-prior contact term inside reconstruction with the fixed
    # training-set-prior term above.
    reconstruction = reconstruction - _loss_weight(config, "contact", 2.0) * losses["contact"]
    reconstruction = reconstruction + _loss_weight(config, "contact", 2.0) * endpoint_contact
    total = (
        _loss_weight(config, "local_flow", 1.0) * flow
        + _loss_weight(config, "local_reconstruction", 1.0) * reconstruction
        + _loss_weight(config, "contact_clean", 0.25) * clean_contact
        + _loss_weight(config, "contact_corrupt", 0.5) * corrupted_contact
        + _loss_weight(config, "contact_f1", 0.5) * contact_f1
    )
    # V3 constrained a one-step algebraic endpoint, while inference integrated
    # a 20-step ODE.  Train on a short differentiable rollout from the exact
    # same source noise so contact timing and physical losses reach the sampled
    # distribution instead of only clean/teacher-forced states.
    rollout_steps = int(config["training"].get("training_rollout_steps", 2))
    rollout_raw, rollout_logits, rollout_endpoint = integrate_local_flow(
        model,
        source_noise,
        music,
        graph_output["graph_condition"],
        graph_output["durations"],
        steps=rollout_steps,
        method="euler",
    )
    rollout_motion = torch.cat([rollout_logits, rollout_raw], dim=-1)
    rollout_reconstruction, rollout_losses = _reconstruction_total(
        rollout_motion, motion, config
    )
    rollout_contact = balanced_contact_loss(
        rollout_logits, target_contacts, model.contact_prior
    )
    rollout_contact_f1 = soft_contact_f1_loss(rollout_logits, target_contacts)
    rollout_contact_transition = contact_transition_loss(rollout_logits, target_contacts)
    rollout_reconstruction = (
        rollout_reconstruction
        - _loss_weight(config, "contact", 2.0) * rollout_losses["contact"]
        + _loss_weight(config, "contact", 2.0) * rollout_contact
    )
    unprojected = rollout_endpoint * model.feature_std + model.feature_mean
    rotation_regularizer = rotation_6d_regularization(
        unprojected[..., 3:].reshape(*unprojected.shape[:-1], -1, 6)
    )
    if not bool(model.skeleton_ready):
        raise RuntimeError("neutral skeleton was not configured before local-flow training")
    predicted_joints = forward_kinematics_continuous(rollout_raw, model.rest_joints)
    reference_joints = forward_kinematics_continuous(raw_data, model.rest_joints)
    fk_losses = fk_reconstruction_losses(predicted_joints, reference_joints)
    foot_indices = (7, 8, 10, 11)
    predicted_feet = predicted_joints[:, :, foot_indices]
    reference_feet = reference_joints[:, :, foot_indices]
    skating = foot_skating_loss(predicted_feet, target_contacts)
    foot_position = torch.nn.functional.smooth_l1_loss(
        predicted_feet, reference_feet, beta=0.01
    )
    beat_evidence = music[:, :, -1] if beat_evidence is None else beat_evidence
    beat_energy = normalized_joint_energy_loss(
        predicted_joints[:, :, :22], reference_joints[:, :, :22], beat_evidence
    )
    beat_alignment = beat_alignment_loss(
        predicted_joints[:, :, :22], beat_evidence,
        tolerance_frames=int(config["training"].get("beat_tolerance_frames", 3)),
    )
    key_motion_anchor = key_motion_anchor_loss(
        predicted_joints[:, :, :22],
        beat_evidence,
        downbeat_stride=int(config["training"].get("key_motion_downbeat_stride", 4)),
    )
    geometric_distribution = geometric_distribution_loss(
        predicted_joints[:, :, :22], reference_joints[:, :, :22]
    )
    manual_geometry_distribution = manual_geometry_distribution_loss(
        predicted_joints[:, :, :22],
        reference_joints[:, :, :22],
        temperature=float(config["training"].get("manual_feature_temperature", 0.05)),
    )
    diversity_preservation = diversity_preservation_loss(
        predicted_joints[:, :, :22], reference_joints[:, :, :22]
    )
    kinetic_coverage = kinetic_distribution_coverage_loss(
        predicted_joints[:, :, :22],
        reference_joints[:, :, :22],
        diversity_gain=float(config["training"].get("kinetic_diversity_gain", 1.03)),
    )
    distribution_moments = _distribution_moment_loss(rollout_motion, motion)
    total = (
        total
        + _loss_weight(config, "rollout_reconstruction", 1.0) * rollout_reconstruction
        + _loss_weight(config, "rollout_contact_f1", 1.0) * rollout_contact_f1
        + _loss_weight(config, "contact_transition", 0.0) * rollout_contact_transition
        + _loss_weight(config, "rotation_regularization", 0.1) * rotation_regularizer
        + _loss_weight(config, "foot_skating", 0.5) * skating
        + _loss_weight(config, "foot_position", 0.5) * foot_position
        + _loss_weight(config, "fk_position", 0.0) * fk_losses["position"]
        + _loss_weight(config, "fk_velocity", 0.0) * fk_losses["velocity"]
        + _loss_weight(config, "fk_acceleration", 0.0) * fk_losses["acceleration"]
        + _loss_weight(config, "beat_energy", 0.5) * beat_energy
        + _loss_weight(config, "beat_alignment", 0.0) * beat_alignment
        + _loss_weight(config, "key_motion_anchor", 0.0) * key_motion_anchor
        + _loss_weight(config, "distribution_moments", 1.0) * distribution_moments
        + _loss_weight(config, "geometric_distribution", 0.0) * geometric_distribution
        + _loss_weight(config, "manual_geometry_distribution", 0.0)
        * manual_geometry_distribution
        + _loss_weight(config, "diversity_preservation", 0.0) * diversity_preservation
        + _loss_weight(config, "kinetic_coverage", 0.0) * kinetic_coverage
    )
    return total, {
        "flow": flow, **losses, "contact": endpoint_contact,
        "contact_clean": clean_contact, "contact_corrupt": corrupted_contact,
        "contact_f1": contact_f1,
        "rollout_pose": rollout_losses["pose"],
        "rollout_velocity": rollout_losses["velocity"],
        "rollout_acceleration": rollout_losses["acceleration"],
        "rollout_energy": rollout_losses["energy"],
        "rollout_root_velocity": rollout_losses["root_velocity"],
        "rollout_contact": rollout_contact,
        "rollout_contact_f1": rollout_contact_f1,
        "contact_transition": rollout_contact_transition,
        "rotation_regularization": rotation_regularizer,
        "foot_skating": skating,
        "foot_position": foot_position,
        "fk_position": fk_losses["position"],
        "fk_velocity": fk_losses["velocity"],
        "fk_acceleration": fk_losses["acceleration"],
        "beat_energy": beat_energy,
        "beat_alignment": beat_alignment,
        "key_motion_anchor": key_motion_anchor,
        "distribution_moments": distribution_moments,
        "geometric_distribution": geometric_distribution,
        "manual_geometry_distribution": manual_geometry_distribution,
        "diversity_preservation": diversity_preservation,
        "kinetic_coverage": kinetic_coverage,
    }


def stage_loss(
    stage: str,
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    motion = batch["motion"]
    music = batch["music"]
    beat_evidence = batch.get("beat_evidence", music[:, :, -1])
    nodes = int(config["training"].get("graph_nodes", 8))
    metrics: dict[str, torch.Tensor] = {}

    if stage == "motion_autoencoder":
        reconstruction, _ = models["motion_autoencoder"](motion)
        total, losses = _reconstruction_total(reconstruction, motion, config)
        return total, {**losses, "total": total}

    graph_batch = graph_targets(
        motion, music, list(batch["style_fine"]), nodes, beat_evidence
    )
    graph_output = models["graph_planner"](graph_batch["music_nodes"])
    role = torch.nn.functional.cross_entropy(graph_output["role_logits"].flatten(0, 1), graph_batch["roles"].flatten())
    semantic = torch.nn.functional.cross_entropy(graph_output["semantic_logits"].flatten(0, 1), graph_batch["semantics"].flatten())
    duration = duration_coverage_loss(graph_output["durations"], graph_batch["durations"], graph_batch["total_beats"])
    relation_labels, relation_confidence, shuffled_fraction = relation_targets(
        graph_batch, batch, config, training=torch.is_grad_enabled()
    )
    metrics["relation_shuffle_changed_fraction"] = shuffled_fraction
    target_flat = relation_labels.reshape(-1)
    dense_weight = relation_confidence.reshape(-1)
    active = dense_weight > 0
    counts = torch.bincount(target_flat[active], minlength=4).to(dense_weight)
    class_weight = (counts.sum() / counts.clamp_min(1.0)).sqrt().clamp(0.5, 3.0)
    class_weight = class_weight / class_weight.mean().clamp_min(1e-6)
    dense_edge = torch.nn.functional.cross_entropy(
        graph_output["dense_edge_logits"].reshape(-1, 4),
        target_flat, weight=class_weight, reduction="none",
    )
    edge = (dense_edge * dense_weight).sum() / dense_weight.sum().clamp_min(1.0)
    target_histogram = torch.nn.functional.one_hot(target_flat, 4).to(dense_weight)
    target_histogram = (target_histogram * dense_weight[:, None]).sum(0)
    target_histogram = target_histogram / target_histogram.sum().clamp_min(1.0)
    predicted_histogram = (
        torch.softmax(graph_output["dense_edge_logits"].reshape(-1, 4), -1)
        * dense_weight[:, None]
    ).sum(0)
    predicted_histogram = predicted_histogram / predicted_histogram.sum().clamp_min(1.0)
    edge_coverage = torch.nn.functional.kl_div(
        predicted_histogram.clamp_min(1e-6).log(),
        target_histogram.detach().clamp_min(1e-6),
        reduction="sum",
    )
    graph_total = (
        _loss_weight(config, "graph_role") * role
        + _loss_weight(config, "graph_semantic") * semantic
        + _loss_weight(config, "graph_duration") * duration
        + _loss_weight(config, "graph_edge") * edge
        + _loss_weight(config, "graph_edge_coverage", 0.0) * edge_coverage
    )
    metrics.update(
        graph_role=role, graph_semantic=semantic, graph_duration=duration,
        graph_edge=edge, graph_edge_coverage=edge_coverage,
    )

    if stage == "graph_and_local":
        local_total, local_losses = _local_flow_total(
            models["local_generator"], motion, music, graph_output, config,
            beat_evidence,
        )
        total = (
            _loss_weight(config, "graph") * graph_total
            + _loss_weight(config, "local") * local_total
        )
        return total, {**metrics, **{f"local_{k}": v for k, v in local_losses.items()}, "total": total}

    boundary = boundary_targets(motion, music, graph_batch["edge_types"])
    plan = models["boundary_planner"](
        boundary["left_summary"], boundary["right_summary"], boundary["music_summary"],
        boundary["conflict"], boundary["edge_one_hot"],
    )
    plan_length = torch.nn.functional.cross_entropy(plan["length_logits"], boundary["length_class"])
    plan_contact = balanced_contact_loss(
        plan["contact_summary"], boundary["contact_summary"]
    )
    plan_root = torch.nn.functional.smooth_l1_loss(plan["root_residual"], boundary["root_residual"])
    plan_energy = torch.nn.functional.smooth_l1_loss(plan["energy_parameters"], boundary["energy_parameters"])
    plan_total = (
        _loss_weight(config, "plan_length") * plan_length
        + _loss_weight(config, "plan_contact") * plan_contact
        + _loss_weight(config, "plan_root") * plan_root
        + _loss_weight(config, "plan_energy") * plan_energy
    )
    metrics.update(plan_length=plan_length, plan_contact=plan_contact, plan_root=plan_root, plan_energy=plan_energy)
    if stage == "boundary_planner":
        return plan_total, {**metrics, "total": plan_total}

    autoencoder = models["motion_autoencoder"]
    with torch.set_grad_enabled(stage == "joint_finetune"):
        transition_latent = autoencoder.encode(boundary["transition"])
        left_latent = autoencoder.encode(boundary["left"]).mean(dim=1)
        right_latent = autoencoder.encode(boundary["right"]).mean(dim=1)
    plan_condition = torch.cat(
        [
            torch.softmax(plan["length_logits"], dim=-1),
            torch.sigmoid(plan["contact_summary"]),
            plan["root_residual"],
            plan["energy_parameters"],
            boundary["music_summary"],
            boundary["edge_one_hot"],
        ],
        dim=-1,
    )
    if stage == "boundary_bridge":
        plan_condition = plan_condition.detach()
    condition = torch.cat([left_latent, right_latent, plan_condition], dim=-1)
    boundary_noise = torch.randn_like(transition_latent)
    boundary_interpolation, boundary_target, boundary_time = boundary_flow_training_pair(
        transition_latent, noise=boundary_noise
    )
    boundary_velocity = models["boundary_bridge"](
        boundary_interpolation, boundary_time, condition
    )
    flow = torch.nn.functional.mse_loss(boundary_velocity, boundary_target)
    metrics["flow"] = flow
    boundary_rollout_latent = integrate_boundary_flow(
        models["boundary_bridge"],
        boundary_noise,
        condition,
        steps=int(config["training"].get("boundary_training_rollout_steps", 2)),
        method="euler",
    )
    boundary_rollout = autoencoder.decode(
        boundary_rollout_latent, boundary["transition"].shape[1]
    )
    boundary_continuous = boundary_rollout[:, :, 4:]
    projected_pose = models["local_generator"].project_rotation_6d(
        boundary_continuous[:, :, 3:].reshape(*boundary_continuous.shape[:-1], -1, 6)
    ).flatten(-2)
    boundary_rollout = torch.cat(
        [boundary_rollout[:, :, :7], projected_pose], dim=-1
    )
    boundary_physical, boundary_losses = _reconstruction_total(
        boundary_rollout, boundary["transition"], config
    )
    metrics.update({f"boundary_rollout_{key}": value for key, value in boundary_losses.items()})
    if stage == "boundary_bridge":
        total = (
            _loss_weight(config, "flow") * flow
            + _loss_weight(config, "boundary_rollout", 1.0) * boundary_physical
        )
        return total, {**metrics, "total": total}

    local_total, local_losses = _local_flow_total(
        models["local_generator"], motion, music, graph_output, config,
        beat_evidence,
    )
    total = (
        _loss_weight(config, "graph") * graph_total
        + _loss_weight(config, "local") * local_total
        + _loss_weight(config, "plan") * plan_total
        + _loss_weight(config, "flow") * flow
        + _loss_weight(config, "boundary_rollout", 1.0) * boundary_physical
    )
    return total, {**metrics, **{f"local_{k}": v for k, v in local_losses.items()}, "total": total}


def _stage_model_names(stage: str) -> tuple[str, ...]:
    return {
        "motion_autoencoder": ("motion_autoencoder",),
        "graph_and_local": ("graph_planner", "local_generator"),
        "boundary_planner": ("boundary_planner",),
        "boundary_bridge": ("boundary_bridge",),
        "joint_finetune": tuple(build for build in ("motion_autoencoder", "graph_planner", "local_generator", "boundary_planner", "boundary_bridge")),
    }[stage]


def save_checkpoint(
    path: Path,
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    stage: str,
    step: int,
    validation_loss: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    best_validation_loss: float | None = None,
    best_step: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "schema_version": 7,
            "stage": stage,
            "step": step,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "best_step": best_step,
            "config": config,
            "models": {name: model.state_dict() for name, model in models.items()},
            "optimizer": optimizer.state_dict(),
            "amp_scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") not in {5, 6, 7}:
        raise ValueError(
            f"checkpoint {path} is not a V5/V5.1/V6 compatible checkpoint; "
            "start a new run instead of resuming an incompatible checkpoint"
        )
    for name, state in payload["models"].items():
        if name in models:
            models[name].load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and payload.get("rng_state"):
        state = payload["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        # ``torch.load(..., map_location=cuda)`` also moves the saved CPU RNG
        # byte tensor to CUDA.  torch.set_rng_state explicitly requires a CPU
        # ByteTensor, so move RNG payloads back to CPU before restoration.
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    return payload


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def _local_rollout_penalty(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generation-state validation used when selecting local checkpoints."""
    motion = batch["motion"]
    music = batch["music"]
    contact_dim = int(config["motion"].get("contact_dim", 4))
    nodes = int(config["training"].get("graph_nodes", 8))
    targets = graph_targets(
        motion, music, list(batch["style_fine"]), nodes,
        batch.get("beat_evidence", music[:, :, -1]),
    )
    graph = models["graph_planner"](targets["music_nodes"])
    model = models["local_generator"]
    noise = torch.randn(
        motion.shape[0], motion.shape[1], model.continuous_dim,
        device=motion.device, dtype=motion.dtype,
    )
    continuous, contact_logits = sample_local_flow(
        model, noise, music, graph["graph_condition"], graph["durations"],
        steps=int(config["training"].get("validation_flow_steps", 8)),
    )
    prediction = torch.cat([torch.sigmoid(contact_logits), continuous], dim=-1)
    target_contacts = motion[:, :, :contact_dim]
    predicted_contacts = (prediction[:, :, :contact_dim] >= 0.5).to(motion.dtype)
    positives = target_contacts.sum().clamp_min(1.0)
    negatives = (1.0 - target_contacts).sum().clamp_min(1.0)
    sensitivity = (predicted_contacts * target_contacts).sum() / positives
    specificity = ((1.0 - predicted_contacts) * (1.0 - target_contacts)).sum() / negatives
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    true_positive = (predicted_contacts * target_contacts).sum()
    true_negative = ((1.0 - predicted_contacts) * (1.0 - target_contacts)).sum()
    false_positive = (predicted_contacts * (1.0 - target_contacts)).sum()
    false_negative = ((1.0 - predicted_contacts) * target_contacts).sum()
    shared_errors = false_positive + false_negative
    positive_f1 = 2.0 * true_positive / (2.0 * true_positive + shared_errors).clamp_min(1.0)
    negative_f1 = 2.0 * true_negative / (2.0 * true_negative + shared_errors).clamp_min(1.0)
    macro_f1 = 0.5 * (positive_f1 + negative_f1)
    predicted_energy = motion_energy(prediction[:, :, 7:]).mean().clamp_min(1e-6)
    target_energy = motion_energy(motion[:, :, 7:]).mean().clamp_min(1e-6)
    energy_ratio = predicted_energy / target_energy
    predicted_root = finite_difference(prediction[:, :, 4:7]).abs().mean().clamp_min(1e-6)
    target_root = finite_difference(motion[:, :, 4:7]).abs().mean().clamp_min(1e-6)
    root_ratio = predicted_root / target_root
    rate_gap = torch.abs(predicted_contacts.mean() - target_contacts.mean())
    predicted_joints = forward_kinematics_continuous(continuous, model.rest_joints)
    target_joints = forward_kinematics_continuous(
        motion[:, :, contact_dim:], model.rest_joints
    )
    foot_indices = (7, 8, 10, 11)
    predicted_skating = foot_skating_loss(
        predicted_joints[:, :, foot_indices], target_contacts
    )
    target_skating = foot_skating_loss(
        target_joints[:, :, foot_indices], target_contacts
    ).clamp_min(1e-4)
    skating_ratio = predicted_skating / target_skating
    beat_energy = normalized_joint_energy_loss(
        predicted_joints[:, :, :22], target_joints[:, :, :22],
        batch.get("beat_evidence", music[:, :, -1]),
    )
    ratio_penalty = torch.abs(torch.log(energy_ratio)) + torch.abs(torch.log(root_ratio))
    contact_penalty = (
        10.0 * torch.relu(0.55 - balanced_accuracy)
        + 10.0 * torch.relu(0.55 - macro_f1)
        + 5.0 * torch.relu(rate_gap - 0.10)
    )
    physical_penalty = torch.abs(torch.log(skating_ratio.clamp_min(1e-4))) + beat_energy
    penalty = ratio_penalty + contact_penalty + physical_penalty
    return penalty, {
        "rollout_contact_balanced_accuracy": float(balanced_accuracy.cpu()),
        "rollout_contact_macro_f1": float(macro_f1.cpu()),
        "rollout_contact_rate_gap": float(rate_gap.cpu()),
        "rollout_energy_ratio": float(energy_ratio.cpu()),
        "rollout_root_velocity_ratio": float(root_ratio.cpu()),
        "rollout_foot_skating_ratio": float(skating_ratio.cpu()),
        "rollout_beat_energy_error": float(beat_energy.cpu()),
        "rollout_penalty": float(penalty.cpu()),
    }


@torch.no_grad()
def validate(
    stage: str,
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    for model in models.values():
        model.eval()
    losses = []
    rollout_penalties: list[float] = []
    rollout_diagnostics: list[dict[str, float]] = []
    limit = int(config["training"].get("validation_batches", 0))
    available_batches = len(loader) if limit <= 0 else min(len(loader), limit)
    requested_rollout_batches = max(
        1, int(config["training"].get("validation_rollout_batches", 4))
    )
    rollout_indices = set(
        np.linspace(
            0, max(available_batches - 1, 0),
            min(requested_rollout_batches, max(available_batches, 1)),
            dtype=int,
        ).tolist()
    )
    validation_seed = int(config.get("seed", 0)) + 100_003 + STAGES.index(stage)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Flow validation contains random times and noise.  Forking and resetting
    # RNG makes every checkpoint see identical validation draws, so "best"
    # means a real loss improvement rather than a lucky noise sample.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(validation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(validation_seed)
        for index, batch in enumerate(loader):
            if limit > 0 and index >= limit:
                break
            moved = _move_batch(batch, device)
            loss, _ = stage_loss(stage, models, moved, config)
            losses.append(float(loss.detach().cpu()))
            if index in rollout_indices and stage in {"graph_and_local", "joint_finetune"}:
                penalty, diagnostics = _local_rollout_penalty(models, moved, config)
                rollout_penalties.append(float(penalty.cpu()))
                rollout_diagnostics.append(diagnostics)
    base = float(np.mean(losses)) if losses else math.inf
    weight = float(config["training"].get("validation_rollout_weight", 1.0))
    rollout_penalty = float(np.mean(rollout_penalties)) if rollout_penalties else 0.0
    if rollout_diagnostics:
        names = rollout_diagnostics[0].keys()
        aggregate = {
            name: float(np.mean([item[name] for item in rollout_diagnostics]))
            for name in names
        }
        aggregate["batches"] = len(rollout_diagnostics)
        print(json.dumps({"stage": stage, "validation_rollout": aggregate}), flush=True)
    return base + weight * rollout_penalty


def train_stage(
    stage: str,
    models: dict[str, torch.nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    max_steps: int,
    resume: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    stage_names = _stage_model_names(stage)
    requested_names = _stage_hyperparameter(config, stage, "trainable_models", None)
    if requested_names is None:
        names = stage_names
    else:
        names = tuple(str(name) for name in requested_names)
        invalid = sorted(set(names) - set(stage_names))
        if invalid or not names:
            raise ValueError(
                f"invalid trainable_models for {stage}: {invalid or list(names)}"
            )
    # Frozen stages still participate in conditioning/decoding.  Explicitly
    # disable their parameter gradients while retaining gradients with respect
    # to their inputs; otherwise physical bridge losses accumulate stale grads
    # in the autoencoder during boundary-only training.
    for model_name, model in models.items():
        trainable = model_name in names
        for parameter in model.parameters():
            parameter.requires_grad_(trainable)
    parameters = [parameter for name in names for parameter in models[name].parameters()]
    learning_rate = float(_stage_hyperparameter(config, stage, "learning_rate", 2e-4))
    weight_decay = float(_stage_hyperparameter(config, stage, "weight_decay", 1e-4))
    betas_value = _stage_hyperparameter(config, stage, "adam_betas", [0.9, 0.99])
    betas = (float(betas_value[0]), float(betas_value[1]))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    scheduler_name = str(_stage_hyperparameter(config, stage, "scheduler", "constant"))
    warmup_updates = int(_stage_hyperparameter(config, stage, "warmup_updates", 0))
    min_lr_ratio = float(_stage_hyperparameter(config, stage, "min_lr_ratio", 1.0))
    total_updates = max(1, math.ceil(max_steps / accumulation))

    def lr_multiplier(update_index: int) -> float:
        if warmup_updates > 0 and update_index < warmup_updates:
            return max(1e-8, (update_index + 1) / warmup_updates)
        if scheduler_name == "constant":
            return 1.0
        if scheduler_name != "cosine":
            raise ValueError(f"unsupported scheduler: {scheduler_name}")
        progress = (update_index - warmup_updates) / max(1, total_updates - warmup_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start_step = 0
    resume_payload: dict[str, Any] | None = None
    if resume:
        header = torch.load(resume, map_location="cpu", weights_only=False)
        if header.get("stage") != stage:
            raise ValueError(
                f"resume checkpoint stage {header.get('stage')!r} does not match requested stage {stage!r}"
            )
        resume_payload = load_checkpoint(
            resume, models, device, optimizer, scheduler=scheduler, restore_rng=True
        )
        start_step = int(resume_payload.get("step", 0))

    clip = float(config["training"].get("gradient_clip_norm", 1.0))
    precision = str(config["training"].get("mixed_precision", "none"))
    amp_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and precision == "fp16")
    if resume_payload and resume_payload.get("amp_scaler"):
        scaler.load_state_dict(resume_payload["amp_scaler"])
    log_path = output_dir.parent / "reports" / "training.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_every = int(config["training"].get("checkpoint_every", 0))
    validate_every = int(config["training"].get("validate_every", checkpoint_every))
    best_validation = float(
        resume_payload.get("best_validation_loss", math.inf) if resume_payload else math.inf
    )
    best_step = int(
        resume_payload.get("best_step", resume_payload.get("step", 0))
        if resume_payload else 0
    )
    if not np.isfinite(best_validation):
        best_validation = math.inf
    latest_validation = float(
        resume_payload.get("validation_loss", math.inf) if resume_payload else math.inf
    )
    started_at = time.perf_counter()
    max_stage_seconds = float(config["training"].get("max_stage_seconds", 0.0))
    stopped_by_time_budget = False
    completed_step = start_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, float] = {}
    for step in range(start_step, max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        for model_name, model in models.items():
            model.train(model_name in names)
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
        with context:
            total, metrics = stage_loss(stage, models, batch, config)
            scaled_loss = total / accumulation
        scaler.scale(scaled_loss).backward()
        update = (step + 1) % accumulation == 0 or step + 1 == max_steps
        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        last_metrics = {key: float(value.detach().cpu()) for key, value in metrics.items()}
        completed_step = step + 1
        if step == start_step or (step + 1) % int(config["training"].get("log_every", 50)) == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            event = {
                "stage": stage,
                "step": step + 1,
                "seed": int(config.get("seed", -1)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
                "micro_batches_per_second": (step + 1 - start_step) / elapsed,
                **last_metrics,
            }
            if device.type == "cuda":
                event["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)

        if max_stage_seconds > 0 and update:
            elapsed_now = time.perf_counter() - started_at
            # Reserve a small margin for validation and atomic checkpoint I/O.
            if elapsed_now >= max(1.0, max_stage_seconds - 120.0):
                stopped_by_time_budget = True
                print(json.dumps({
                    "stage": stage,
                    "event": "wall_clock_budget_reached",
                    "step": completed_step,
                    "elapsed_seconds": elapsed_now,
                    "budget_seconds": max_stage_seconds,
                }), flush=True)
                break

        should_validate = validate_every > 0 and (step + 1) % validate_every == 0
        should_checkpoint = checkpoint_every > 0 and (step + 1) % checkpoint_every == 0
        if should_validate:
            latest_validation = validate(stage, models, val_loader, config, device)
        if should_checkpoint or should_validate:
            if not np.isfinite(latest_validation):
                latest_validation = validate(stage, models, val_loader, config, device)
            if latest_validation < best_validation:
                best_validation = latest_validation
                best_step = step + 1
                save_checkpoint(
                    output_dir / f"{stage}_best.pt", models, optimizer, config, stage,
                    step + 1, latest_validation, scaler, scheduler, best_validation, best_step,
                )
            stage_last = output_dir / f"{stage}_last.pt"
            save_checkpoint(
                stage_last, models, optimizer, config, stage, step + 1,
                latest_validation, scaler, scheduler, best_validation, best_step,
            )
            save_checkpoint(
                output_dir / "last.pt", models, optimizer, config, stage, step + 1,
                latest_validation, scaler, scheduler, best_validation, best_step,
            )

    validation_loss = validate(stage, models, val_loader, config, device)
    best_path = output_dir / f"{stage}_best.pt"
    if validation_loss <= best_validation or not best_path.is_file():
        best_validation = validation_loss
        best_step = completed_step
        save_checkpoint(
            best_path, models, optimizer, config, stage, completed_step,
            validation_loss, scaler, scheduler, best_validation, best_step,
        )
    save_checkpoint(
        output_dir / f"{stage}_last.pt", models, optimizer, config, stage, completed_step,
        validation_loss, scaler, scheduler, best_validation, best_step,
    )
    save_checkpoint(
        output_dir / "last.pt", models, optimizer, config, stage, completed_step,
        validation_loss, scaler, scheduler, best_validation, best_step,
    )
    # Rhythm rescue stages may explicitly request the final checkpoint.  Their
    # differentiable training loss is only a proxy for the released BAS metric,
    # so selecting an early checkpoint by total validation loss can discard the
    # rhythm learned later in fine-tuning.
    stage_path = output_dir / CHECKPOINT_NAMES[stage]
    checkpoint_policy = str(config["training"].get("checkpoint_policy", "best_validation"))
    source_path = output_dir / f"{stage}_last.pt" if checkpoint_policy == "final" else best_path
    shutil.copy2(source_path, stage_path)
    load_checkpoint(stage_path, models, device)
    elapsed = time.perf_counter() - started_at
    report = {
        "stage": stage,
        "steps": completed_step,
        "requested_steps": max_steps,
        "start_step": start_step,
        "optimizer_updates": math.ceil(max(0, completed_step - start_step) / accumulation),
        "stopped_by_time_budget": stopped_by_time_budget,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "checkpoint": str(stage_path),
        "checkpoint_policy": checkpoint_policy,
        "elapsed_seconds": elapsed,
        "micro_batches_per_second": max(0, completed_step - start_step) / max(elapsed, 1e-9),
        "last_metrics": last_metrics,
    }
    if device.type == "cuda":
        report["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    (output_dir.parent / "reports" / f"{stage}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
