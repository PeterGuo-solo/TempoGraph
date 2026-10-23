"""Shared PyTorch layers."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = time.reshape(-1, 1) * frequency.reshape(1, -1)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dimension, 2).float() * (-math.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[: values.shape[1]].unsqueeze(0).to(values.dtype)

