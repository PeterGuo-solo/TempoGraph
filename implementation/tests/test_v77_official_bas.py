from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tempograph.generation import _impose_beat_velocity_valleys


PROJECT = Path(__file__).resolve().parents[1]


def _features(frames: int = 160) -> np.ndarray:
    result = np.zeros((frames, 139), dtype=np.float32)
    result[:, 4] = np.arange(frames, dtype=np.float32)
    result[:, 7:] = np.tile(np.asarray([1, 0, 0, 0, 1, 0], np.float32), (frames, 22))
    return result


def test_velocity_valleys_cover_every_beat_and_preserve_endpoints() -> None:
    original = _features()
    beats = np.asarray([16, 32, 48, 64, 80, 96, 112, 128, 144])
    warped, info = _impose_beat_velocity_valleys(original, beats, strength=0.75, width=2.0)
    speed = np.abs(np.diff(warped[:, 4], prepend=warped[0, 4]))
    assert info["used"] is True
    assert info["beat_count"] == len(beats)
    assert info["strictly_monotonic"] is True
    assert info["maximum_beat_phase_error"] < 1e-8
    assert np.allclose(warped[[0, -1], 4:7], original[[0, -1], 4:7])
    assert np.mean(speed[beats]) < 0.5 * np.median(speed)
    assert np.isfinite(warped).all()


def test_v77_uses_all_beats_final_checkpoint_and_quality_guards() -> None:
    config = json.loads((PROJECT / "configs" / "v77_official_bas_train.json").read_text())
    training = config["training"]
    weights = training["loss_weights"]
    assert training["stage_steps"]["joint_finetune"] == 15_000
    assert training["key_motion_downbeat_stride"] == 1
    assert training["checkpoint_policy"] == "final"
    assert weights["diversity_preservation"] > 0
    assert weights["kinetic_coverage"] > 0
    runner = (PROJECT / "scripts" / "run_v77_official_bas_autodl.sh").read_text()
    assert "--split test" not in runner
    assert "--max-steps 15000" in runner
    assert "v77_protected" in runner and "v77_rhythmic" in runner
