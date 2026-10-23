from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _candidate(root: Path, tag: str, weight: float, fid_g: float) -> None:
    run = root / f"weight_{tag}"
    generated = run / "generated" / "tempograph"
    reports = run / "reports"
    generated.mkdir(parents=True)
    reports.mkdir(parents=True)
    (generated / "generation_index.json").write_text(
        json.dumps({
            "split": "val",
            "items": [{"motion_prior": {"blend_weight": weight}}],
        }), encoding="utf-8"
    )
    (reports / "lodge_protocol_metrics.json").write_text(
        json.dumps({"aggregate": {
            "fid_k": 45.56, "fid_g": fid_g,
            "foot_skating_ratio": 0.0276,
            "div_k": 8.0, "div_k_gt": 8.0,
            "div_g": 6.0, "div_g_gt": 6.0,
            "bas": 0.2397,
        }}), encoding="utf-8"
    )


def test_prior_weight_selection_uses_validation_candidates(tmp_path: Path) -> None:
    root = tmp_path / "tune"
    _candidate(root, "65", 0.65, 60.0)
    _candidate(root, "75", 0.75, 28.17)
    output = root / "selection.json"
    script = Path(__file__).parents[1] / "scripts" / "select_v61_prior_weight.py"
    subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["selection_split"] == "val"
    assert result["test_metrics_used"] is False
    assert result["chosen_weight"] == 0.75
