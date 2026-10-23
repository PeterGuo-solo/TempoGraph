"""Graph- and music-conditioned rectified-flow local motion generator."""

from __future__ import annotations

import torch
from torch import nn

from .common import PositionalEncoding, SinusoidalTimeEmbedding


def expand_graph_condition(
    graph_condition: torch.Tensor,
    frames: int,
    durations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand node-wise graph conditions to one condition per motion frame."""
    if graph_condition.ndim == 2:
        return graph_condition[:, None].expand(-1, frames, -1)
    if graph_condition.ndim != 3:
        raise ValueError("graph_condition must have shape [B, D] or [B, N, D]")
    if graph_condition.shape[1] == frames:
        return graph_condition
    batch, nodes, _ = graph_condition.shape
    if nodes == 1:
        return graph_condition.expand(-1, frames, -1)
    if durations is None:
        positions = torch.linspace(
            0, nodes, frames + 1, device=graph_condition.device, dtype=torch.float32
        )[:-1]
        indices = positions.floor().long().clamp_max(nodes - 1)
        return graph_condition[:, indices]
    if durations.shape != (batch, nodes):
        raise ValueError("durations must have shape [B, N]")
    result = []
    frame_positions = torch.arange(frames, device=graph_condition.device)
    for sample in range(batch):
        weights = durations[sample].detach().float().clamp_min(1e-4)
        boundaries = torch.cumsum(weights / weights.sum() * frames, dim=0)
        boundaries[-1] = float(frames)
        indices = torch.bucketize(frame_positions, boundaries, right=True).clamp_max(nodes - 1)
        result.append(graph_condition[sample, indices])
    return torch.stack(result, dim=0)


class LocalMotionGenerator(nn.Module):
    """Velocity field over continuous motion plus a contact classifier."""

    def __init__(
        self,
        motion_dim: int = 139,
        music_dim: int = 35,
        graph_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        contact_dim: int = 4,
    ) -> None:
        super().__init__()
        if motion_dim <= contact_dim:
            raise ValueError("motion_dim must be larger than contact_dim")
        self.motion_dim = motion_dim
        self.contact_dim = contact_dim
        self.continuous_dim = motion_dim - contact_dim
        if self.continuous_dim < 4:
            raise ValueError("continuous motion must contain root xyz and pose channels")
        # The flow operates on standardized root velocity + pose, not raw
        # absolute root positions.  These buffers are estimated from the
        # training split and travel with every checkpoint.
        self.register_buffer("feature_mean", torch.zeros(self.continuous_dim))
        self.register_buffer("feature_std", torch.ones(self.continuous_dim))
        self.register_buffer("contact_prior", torch.full((contact_dim,), 0.5))
        self.register_buffer("statistics_ready", torch.tensor(False))
        self.register_buffer("rest_joints", torch.zeros(24, 3))
        self.register_buffer("skeleton_ready", torch.tensor(False))
        self.motion_projection = nn.Linear(self.continuous_dim, hidden_dim)
        self.condition_projection = nn.Linear(music_dim + graph_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.position = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers)
        self.velocity_output = nn.Linear(hidden_dim, self.continuous_dim)
        self.contact_output = nn.Linear(hidden_dim, contact_dim)

    @staticmethod
    def to_relative_representation(raw_continuous: torch.Tensor) -> torch.Tensor:
        """Encode planar root velocity, absolute vertical height, and pose.

        Integrating the vertical channel as velocity caused the large upward/downward
        drift observed in V4.  FineDance is root-centred at the first frame, so the
        vertical displacement is bounded and can be modelled directly.
        """
        root = raw_continuous[..., :3]
        encoded_root = root.clone()
        encoded_root[..., 0] = 0.0
        encoded_root[..., 2] = 0.0
        encoded_root[..., 1:, 0] = root[..., 1:, 0] - root[..., :-1, 0]
        encoded_root[..., 1:, 2] = root[..., 1:, 2] - root[..., :-1, 2]
        return torch.cat([encoded_root, raw_continuous[..., 3:]], dim=-1)

    def configure_statistics(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        contact_prior: torch.Tensor,
    ) -> None:
        if feature_mean.shape != (self.continuous_dim,) or feature_std.shape != (self.continuous_dim,):
            raise ValueError("local-flow feature statistics have the wrong shape")
        if contact_prior.shape != (self.contact_dim,):
            raise ValueError("contact prior has the wrong shape")
        self.feature_mean.copy_(feature_mean.to(self.feature_mean).detach())
        self.feature_std.copy_(feature_std.to(self.feature_std).detach().clamp_min(1e-4))
        self.contact_prior.copy_(contact_prior.to(self.contact_prior).detach().clamp(1e-4, 1 - 1e-4))
        self.statistics_ready.fill_(True)

    def configure_skeleton(self, rest_joints: torch.Tensor) -> None:
        """Store the same neutral skeleton used by the released LODGE metric."""
        if rest_joints.ndim != 2 or rest_joints.shape[0] < 24 or rest_joints.shape[1] != 3:
            raise ValueError("rest_joints must contain at least 24 xyz joints")
        self.rest_joints.copy_(rest_joints[:24].to(self.rest_joints).detach())
        self.skeleton_ready.fill_(True)

    def encode_motion(self, raw_continuous: torch.Tensor) -> torch.Tensor:
        relative = self.to_relative_representation(raw_continuous)
        return (relative - self.feature_mean) / self.feature_std

    @staticmethod
    def project_rotation_6d(rotation: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Project two predicted matrix rows onto SO(3)'s 6D representation."""
        first_raw, second_raw = rotation[..., :3], rotation[..., 3:]
        first = torch.nn.functional.normalize(first_raw, dim=-1, eps=epsilon)
        second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
        second = torch.nn.functional.normalize(second, dim=-1, eps=epsilon)
        return torch.cat([first, second], dim=-1)

    def decode_motion(
        self, normalized: torch.Tensor, *, project_rotations: bool = False
    ) -> torch.Tensor:
        relative = normalized * self.feature_std + self.feature_mean
        root = relative[..., :3].clone()
        root[..., 0] = torch.cumsum(relative[..., 0], dim=-1)
        root[..., 2] = torch.cumsum(relative[..., 2], dim=-1)
        # Channel 1 is height, not velocity.
        root[..., 1] = relative[..., 1]
        pose = relative[..., 3:]
        if project_rotations:
            pose = self.project_rotation_6d(pose.reshape(*pose.shape[:-1], -1, 6)).flatten(-2)
        return torch.cat([root, pose], dim=-1)

    def forward(
        self,
        noisy_continuous: torch.Tensor,
        music: torch.Tensor,
        graph_condition: torch.Tensor,
        time: torch.Tensor,
        graph_durations: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if noisy_continuous.shape[-1] != self.continuous_dim:
            raise ValueError(
                f"expected {self.continuous_dim} continuous channels, "
                f"received {noisy_continuous.shape[-1]}"
            )
        frame_condition = expand_graph_condition(
            graph_condition, noisy_continuous.shape[1], graph_durations
        )
        condition = torch.cat([music, frame_condition], dim=-1)
        hidden = self.motion_projection(noisy_continuous) + self.condition_projection(condition)
        hidden = self.input_norm(hidden + self.time_embedding(time)[:, None])
        hidden = self.backbone(self.position(hidden), src_key_padding_mask=padding_mask)
        return {
            "velocity": self.velocity_output(hidden),
            "contact_logits": self.contact_output(hidden),
        }


def rectified_flow_training_pair(
    data: torch.Tensor,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(data)
    if time is None:
        time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
    view = (data.shape[0],) + (1,) * (data.ndim - 1)
    interpolation = (1 - time.view(view)) * noise + time.view(view) * data
    return interpolation, data - noise, time


def integrate_local_flow(
    model: LocalMotionGenerator,
    noise: torch.Tensor,
    music: torch.Tensor,
    graph_condition: torch.Tensor,
    graph_durations: torch.Tensor | None = None,
    steps: int = 20,
    *,
    method: str = "heun",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable flow integration shared by training and inference.

    Returning the normalized endpoint lets training regularize the unprojected
    rotation representation, while the decoded result exactly matches the
    SO(3) projection used by final generation.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    motion = noise
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (motion.shape[0],), index / steps, device=motion.device, dtype=motion.dtype
        )
        first = model(motion, music, graph_condition, time, graph_durations)["velocity"]
        if method == "euler":
            motion = motion + step_size * first
            continue
        proposal = motion + step_size * first
        next_time = torch.clamp(time + step_size, max=1.0)
        second = model(
            proposal, music, graph_condition, next_time, graph_durations
        )["velocity"]
        motion = motion + 0.5 * step_size * (first + second)
    final_time = torch.ones(motion.shape[0], device=motion.device, dtype=motion.dtype)
    contact_logits = model(
        motion, music, graph_condition, final_time, graph_durations
    )["contact_logits"]
    decoded = model.decode_motion(motion, project_rotations=True)
    return decoded, contact_logits, motion


@torch.no_grad()
def sample_local_flow(
    model: LocalMotionGenerator,
    noise: torch.Tensor,
    music: torch.Tensor,
    graph_condition: torch.Tensor,
    graph_durations: torch.Tensor | None = None,
    steps: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heun integration from Gaussian noise at t=0 to motion at t=1."""
    decoded, contact_logits, _ = integrate_local_flow(
        model, noise, music, graph_condition, graph_durations, steps, method="heun"
    )
    return decoded, contact_logits
