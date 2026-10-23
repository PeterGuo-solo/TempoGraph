from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tempograph.generation import _closed_loop_official_bas


PROJECT = Path(__file__).resolve().parents[1]


def _features(frames: int = 256) -> np.ndarray:
    result = np.zeros((frames, 139), dtype=np.float32)
    result[:, 4] = np.arange(frames, dtype=np.float32) * 0.01
    result[:, 7:] = np.tile(np.asarray([1, 0, 0, 0, 1, 0], np.float32), (frames, 22))
    return result


def test_closed_loop_uses_official_score_after_all_motion_edits() -> None:
    beats = np.arange(20, 240, 16, dtype=np.int64)
    result, info = _closed_loop_official_bas(
        _features(), np.zeros((24, 3), np.float32), beats,
        candidates=[[0.65, 3.0], [0.82, 4.5], [0.92, 5.5]],
        target_bas=0.80, maximum_frame_displacement=12.0, foot_tolerance=1.0,
    )
    assert result.shape == (256, 139)
    assert info["used"] is True
    assert info["selected"]["official_bas"] > info["baseline_bas"]
    assert info["selected"]["strictly_monotonic"] is True
    assert info["selected"]["maximum_frame_displacement"] <= 12.0
    assert info["selected"]["maximum_beat_phase_error"] < 1e-8


def test_selector_never_ranks_incomplete_group(tmp_path: Path) -> None:
    metrics = {"fid_k": 30.0, "fid_g": 20.0, "foot_skating_ratio": 0.02,
               "div_k": 8.3, "div_g": 7.0, "bas": 0.25}
    # An incomplete profile with excellent values must not beat a complete one.
    incomplete = tmp_path / "incomplete" / "seed_1234" / "reports"
    incomplete.mkdir(parents=True)
    (incomplete / "lodge_protocol_metrics.json").write_text(json.dumps({"aggregate": metrics}))
    for seed in (1234, 2025, 3407):
        report = tmp_path / "complete" / f"seed_{seed}" / "reports"
        report.mkdir(parents=True)
        (report / "lodge_protocol_metrics.json").write_text(json.dumps({"aggregate": metrics}))
    output = tmp_path / "selection.json"
    completed = subprocess.run([
        sys.executable, str(PROJECT / "scripts" / "select_complete_validation.py"),
        "--candidates-root", str(tmp_path), "--output", str(output),
    ], check=False)
    assert completed.returncode == 0
    assert json.loads(output.read_text())["selected_profile"] == "complete"


def test_v78_runner_is_inference_only_and_never_reads_test() -> None:
    runner = (PROJECT / "scripts" / "run_closed_loop_autodl.sh").read_text()
    assert "scripts/train.py" not in runner
    assert "--split test" not in runner
    assert "for seed in 1234 2025 3407" in runner
    assert "select_complete_validation.py" in runner
