"""Lazy access to optional PyTorch models.

The NumPy data and boundary pipeline remains importable on machines without PyTorch.
"""

from __future__ import annotations

from typing import Any


def load_torch_models() -> dict[str, Any]:
    try:
        from .boundary_bridge import BoundaryPlanNetwork, BoundaryVelocityField
        from .graph_planner import GraphPlanner
        from .local_generator import LocalMotionGenerator
        from .motion_autoencoder import MotionAutoencoder
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError("PyTorch is required for neural TempoGraph models") from error
        raise
    return {
        "MotionAutoencoder": MotionAutoencoder,
        "GraphPlanner": GraphPlanner,
        "LocalMotionGenerator": LocalMotionGenerator,
        "BoundaryPlanNetwork": BoundaryPlanNetwork,
        "BoundaryVelocityField": BoundaryVelocityField,
    }


__all__ = ["load_torch_models"]

