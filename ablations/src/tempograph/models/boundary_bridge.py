"""Learned Boundary Planner and conditional Rectified Flow velocity field."""

from __future__ import annotations

import torch
from torch import nn

from .common import PositionalEncoding, SinusoidalTimeEmbedding


class BoundaryPlanNetwork(nn.Module):
    def __init__(self, boundary_dim: int = 139, music_dim: int = 35, hidden_dim: int = 512) -> None:
        super().__init__()
        input_dim = boundary_dim * 2 + music_dim + 5 + 4
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.length_logits = nn.Linear(hidden_dim, 3)
        self.contact_summary = nn.Linear(hidden_dim, 8)
        self.root_residual = nn.Linear(hidden_dim, 6)
        self.energy_parameters = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        left_summary: torch.Tensor,
        right_summary: torch.Tensor,
        music_summary: torch.Tensor,
        conflict: torch.Tensor,
        edge_one_hot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.backbone(
            torch.cat([left_summary, right_summary, music_summary, conflict, edge_one_hot], dim=-1)
        )
        return {
            "length_logits": self.length_logits(hidden),
            "contact_summary": self.contact_summary(hidden),
            "root_residual": self.root_residual(hidden),
            "energy_parameters": self.energy_parameters(hidden),
        }


class BoundaryVelocityField(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.position = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 2:
            condition = condition[:, None].expand(-1, latent.shape[1], -1)
        hidden = self.latent_projection(latent) + self.condition_projection(condition)
        hidden = self.input_norm(hidden + self.time_embedding(time)[:, None])
        return self.output(self.backbone(self.position(hidden)))


def rectified_flow_training_pair(
    data_latent: torch.Tensor,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(data_latent)
    if time is None:
        time = torch.rand(data_latent.shape[0], device=data_latent.device, dtype=data_latent.dtype)
    view = (data_latent.shape[0],) + (1,) * (data_latent.ndim - 1)
    interpolation = (1 - time.view(view)) * noise + time.view(view) * data_latent
    target_velocity = data_latent - noise
    return interpolation, target_velocity, time


def flow_matching_loss(
    field: BoundaryVelocityField,
    data_latent: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    interpolation, target, time = rectified_flow_training_pair(data_latent)
    prediction = field(interpolation, time, condition)
    return torch.nn.functional.mse_loss(prediction, target)


def integrate_boundary_flow(
    field: BoundaryVelocityField,
    noise: torch.Tensor,
    condition: torch.Tensor,
    steps: int = 20,
    *,
    method: str = "heun",
) -> torch.Tensor:
    """Differentiable boundary integration shared by training and inference."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    latent = noise
    step = 1.0 / steps
    for index in range(steps):
        time = torch.full((latent.shape[0],), index / steps, device=latent.device, dtype=latent.dtype)
        first = field(latent, time, condition)
        if method == "euler":
            latent = latent + step * first
            continue
        proposal = latent + step * first
        next_time = torch.clamp(time + step, max=1.0)
        second = field(proposal, next_time, condition)
        latent = latent + 0.5 * step * (first + second)
    return latent


@torch.no_grad()
def sample_flow(
    field: BoundaryVelocityField,
    noise: torch.Tensor,
    condition: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:
    """Heun integration of dz/dt=v(z,t,c) from t=0 to 1."""
    return integrate_boundary_flow(field, noise, condition, steps, method="heun")
