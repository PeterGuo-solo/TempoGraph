"""Stage-specific PyTorch update functions.

These functions intentionally accept plain dictionaries so dataset-specific loaders remain outside
the models. They are executable once PyTorch and real canonical batches are available.
"""

from __future__ import annotations

import torch

from .losses import balanced_contact_loss, boundary_reconstruction_losses
from .models.boundary_bridge import flow_matching_loss
from .models.graph_planner import duration_coverage_loss
from .models.local_generator import rectified_flow_training_pair


def _step(total: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0
    )
    optimizer.step()
    return float(total.detach())


def train_autoencoder_batch(model, batch: dict[str, torch.Tensor], optimizer) -> dict[str, float]:
    reconstruction, _ = model(batch["motion"])
    losses = boundary_reconstruction_losses(reconstruction[:, :, 4:], batch["motion"][:, :, 4:])
    contact = balanced_contact_loss(reconstruction[:, :, :4], batch["motion"][:, :, :4])
    total = losses["pose"] + 0.5 * losses["velocity"] + 0.25 * losses["acceleration"] + 2.0 * contact
    value = _step(total, optimizer)
    return {**{key: float(item.detach()) for key, item in losses.items()}, "contact": float(contact), "total": value}


def train_graph_batch(model, batch: dict[str, torch.Tensor], optimizer) -> dict[str, float]:
    output = model(batch["music_nodes"], batch.get("padding_mask"))
    role = torch.nn.functional.cross_entropy(
        output["role_logits"].flatten(0, 1), batch["roles"].flatten()
    )
    semantic = torch.nn.functional.cross_entropy(
        output["semantic_logits"].flatten(0, 1), batch["semantics"].flatten()
    )
    duration = duration_coverage_loss(output["durations"], batch["durations"], batch["total_beats"])
    edge = torch.nn.functional.cross_entropy(
        output["edge_logits"].flatten(0, 1),
        batch["edge_types"].flatten(),
        reduction="none",
    )
    edge = (edge * batch["edge_confidence"].flatten()).mean()
    total = role + semantic + duration + edge
    value = _step(total, optimizer)
    return {"role": float(role), "semantic": float(semantic), "duration": float(duration), "edge": float(edge), "total": value}


def train_local_batch(model, batch: dict[str, torch.Tensor], optimizer) -> dict[str, float]:
    data = batch["motion"][:, :, 4:]
    interpolation, target_velocity, time = rectified_flow_training_pair(data)
    output = model(
        interpolation,
        batch["music"],
        batch["graph_condition"],
        time,
        batch.get("graph_durations"),
        batch.get("padding_mask"),
    )
    flow = torch.nn.functional.mse_loss(output["velocity"], target_velocity)
    contact = balanced_contact_loss(output["contact_logits"], batch["motion"][:, :, :4])
    total = flow + 2.0 * contact
    value = _step(total, optimizer)
    return {"flow": float(flow), "contact": float(contact), "total": value}


def train_boundary_planner_batch(model, batch: dict[str, torch.Tensor], optimizer) -> dict[str, float]:
    output = model(
        batch["left_summary"],
        batch["right_summary"],
        batch["music_summary"],
        batch["conflict"],
        batch["edge_one_hot"],
    )
    length = torch.nn.functional.cross_entropy(output["length_logits"], batch["length_class"])
    contact = balanced_contact_loss(output["contact_summary"], batch["contact_summary"])
    root = torch.nn.functional.smooth_l1_loss(output["root_residual"], batch["root_residual"])
    energy = torch.nn.functional.smooth_l1_loss(
        output["energy_parameters"], batch["energy_parameters"]
    )
    total = length + contact + root + energy
    value = _step(total, optimizer)
    return {"length": float(length), "contact": float(contact), "root": float(root), "energy": float(energy), "total": value}


def train_boundary_bridge_batch(model, batch: dict[str, torch.Tensor], optimizer) -> dict[str, float]:
    flow = flow_matching_loss(model, batch["transition_latent"], batch["condition"])
    value = _step(flow, optimizer)
    return {"flow": value, "total": value}
