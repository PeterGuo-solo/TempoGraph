"""Temporal motion autoencoder used by local generation and Boundary Bridge."""

from __future__ import annotations

import torch
from torch import nn


class MotionAutoencoder(nn.Module):
    def __init__(self, feature_dim: int = 139, latent_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(latent_dim, latent_dim, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(latent_dim, hidden_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, feature_dim, 5, padding=2),
        )

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != self.feature_dim:
            raise ValueError("motion must have shape [B, T, feature_dim]")
        return self.encoder(motion.transpose(1, 2)).transpose(1, 2)

    def decode(self, latent: torch.Tensor, target_frames: int | None = None) -> torch.Tensor:
        result = self.decoder(latent.transpose(1, 2)).transpose(1, 2)
        if target_frames is not None:
            if target_frames <= 0:
                raise ValueError("target_frames must be positive")
            if result.shape[1] != target_frames:
                # A stride-2 encoder maps odd lengths such as the 15-frame
                # boundary target to floor(T/2); the transposed convolution
                # consequently reconstructs 14 frames.  Interpolate on the
                # temporal axis instead of silently returning a short tensor.
                result = torch.nn.functional.interpolate(
                    result.transpose(1, 2),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        return result

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(motion)
        return self.decode(latent, motion.shape[1]), latent
