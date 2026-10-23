"""Configuration helpers with no third-party parser dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a JSON configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    for required in ("motion", "music", "graph", "boundary", "model", "training"):
        if required not in config:
            raise ValueError(f"Missing configuration section: {required}")
    if config["motion"]["feature_dim"] != 4 + 3 + 6 * config["motion"]["joint_count"]:
        raise ValueError("motion.feature_dim is inconsistent with the canonical packing")
    return config


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively merged copy of two dictionaries."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

