"""Music-conditioned global choreography topology planner."""

from __future__ import annotations

import torch
from torch import nn

from .common import PositionalEncoding


class GraphPlanner(nn.Module):
    def __init__(
        self,
        music_dim: int = 35,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        role_classes: int = 4,
        semantic_classes: int = 128,
        edge_classes: int = 4,
        graph_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.music_projection = nn.Linear(music_dim, hidden_dim)
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
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.role_head = nn.Linear(hidden_dim, role_classes)
        self.semantic_head = nn.Linear(hidden_dim, semantic_classes)
        self.duration_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.keyframe_head = nn.Linear(hidden_dim, 1)
        self.boundary_head = nn.Linear(hidden_dim, 2 * 139)
        self.edge_head = nn.Linear(hidden_dim * 2, edge_classes)
        self.graph_condition_head = nn.Linear(hidden_dim, graph_dim)
        self.role_condition = nn.Linear(role_classes, graph_dim, bias=False)
        self.keyframe_condition = nn.Linear(1, graph_dim, bias=False)
        self.boundary_condition = nn.Linear(2 * 139, graph_dim, bias=False)
        self.relation_condition = nn.Linear(hidden_dim, graph_dim, bias=False)

    def forward(
        self,
        music_nodes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(self.position(self.music_projection(music_nodes)), src_key_padding_mask=padding_mask)
        node_count = hidden.shape[1]
        left = hidden[:, :, None, :].expand(-1, -1, node_count, -1)
        right = hidden[:, None, :, :].expand(-1, node_count, -1, -1)
        dense_edges = self.edge_head(torch.cat([left, right], dim=-1))
        diagonal = torch.eye(node_count, device=hidden.device, dtype=torch.bool)[None, :, :]
        dense_edges = dense_edges.masked_fill(diagonal[..., None], 0.0)
        if node_count > 1:
            adjacent = torch.cat([hidden[:, :-1], hidden[:, 1:]], dim=-1)
            edges = self.edge_head(adjacent)
        else:
            edges = hidden.new_empty((hidden.shape[0], 0, self.edge_head.out_features))
        role_logits = self.role_head(hidden)
        keyframe_logits = self.keyframe_head(hidden).squeeze(-1)
        boundary_states = self.boundary_head(hidden)
        relation_weights = torch.softmax(dense_edges, dim=-1)[..., 0] + 0.5 * torch.softmax(dense_edges, dim=-1)[..., 1]
        relation_weights = relation_weights.masked_fill(diagonal, 0.0)
        relation_weights = relation_weights / relation_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        relation_message = torch.matmul(relation_weights, hidden)
        graph_condition = (
            self.graph_condition_head(hidden)
            + self.role_condition(torch.softmax(role_logits, dim=-1))
            + self.keyframe_condition(torch.sigmoid(keyframe_logits)[..., None])
            + self.boundary_condition(torch.tanh(boundary_states))
            + self.relation_condition(relation_message)
        )
        return {
            "role_logits": role_logits,
            "semantic_logits": self.semantic_head(hidden),
            "durations": self.duration_head(hidden).squeeze(-1),
            "keyframe_logits": keyframe_logits,
            "boundary_states": boundary_states,
            "edge_logits": edges,
            "dense_edge_logits": dense_edges,
            "node_hidden": hidden,
            # Preserve per-node conditions.  The local generator expands them
            # to individual frames using predicted node durations.
            "graph_condition": graph_condition,
        }


def duration_coverage_loss(predicted: torch.Tensor, target: torch.Tensor, total_beats: torch.Tensor) -> torch.Tensor:
    pointwise = torch.nn.functional.l1_loss(predicted, target)
    coverage = torch.nn.functional.l1_loss(predicted.sum(dim=-1), total_beats)
    return pointwise + coverage
