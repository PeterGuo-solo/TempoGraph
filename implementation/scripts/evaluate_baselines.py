from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = next(p for p in PROJECT.parents if (p / "reference_code" / "baselines" / "LODGE-main").is_dir())
sys.path.insert(0, str(PROJECT / "src"))

from tempograph.benchmarking import evaluate_predictions
from tempograph.data.dataset import read_manifest


COMPARABILITY = PROJECT / "configs" / "five_paper_comparability.json"
DISPLAY_NAMES = {
    "lodge": "LODGE",
    "edge": "EDGE",
    "sgmd": "SGMD",
    "infinitedance": "ChoreoLLaMA",
    "megadance": "MEGADance",
    "motioncraft": "MotionCraft",
}


def checkpoint_inventory(root: Path) -> dict[str, object]:
    global_ckpt = root / "exp" / "Global_Module" / "FineDance_Global" / "checkpoints" / "epoch=2999.ckpt"
    local_ckpt = root / "exp" / "Local_Module" / "FineDance_FineTuneV2_Local" / "checkpoints" / "epoch=299.ckpt"
    return {
        "global_checkpoint": str(global_ckpt),
        "global_ready": global_ckpt.is_file() and global_ckpt.stat().st_size > 1_000_000_000,
        "local_checkpoint": str(local_ckpt),
        "local_ready": local_ckpt.is_file() and local_ckpt.stat().st_size > 1_000_000_000,
    }


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_lodge_runtime(source: Path, checkpoint: Path) -> Path:
    """Build an isolated Linux runtime without changing the reference checkout."""
    runtime = PROJECT / "artifacts" / "baselines" / "lodge_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {".git", "configs", "data", "exp", "experiments"}:
            continue
        _link(item, runtime / item.name)
    configs = runtime / "configs"
    if configs.exists():
        shutil.rmtree(configs)
    shutil.copytree(source / "configs", configs)
    assets = configs / "data" / "assets.yaml"
    text = assets.read_text(encoding="utf-8")
    replacements = {
        "/data2/lrh/smpl_model/smplh": str((PROJECT / "models" / "smplh").resolve()).replace("\\", "/"),
        "/data2/lrh/smpl_model/smpl": str((PROJECT / "models" / "smpl").resolve()).replace("\\", "/"),
        "/data2/lrh/dataset/fine_dance/origin/label_json/": str((PROJECT / "data" / "finedance" / "label_json").resolve()).replace("\\", "/") + "/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    assets.write_text(text, encoding="utf-8")
    _link(checkpoint / "exp", runtime / "exp")
    fine = runtime / "data" / "finedance"
    _link(PROJECT / "data" / "finedance" / "label_json", fine / "label_json")
    _link(PROJECT / "data" / "finedance" / "music_wav", fine / "music")
    _link(PROJECT / "data" / "finedance" / "music_npynew", fine / "music_npynew")
    _link(PROJECT / "data" / "finedance" / "mofea319", fine / "mofea319")
    _link(PROJECT / "data" / "finedance" / "Normalizer.pth", runtime / "data" / "Normalizer.pth")
    return runtime


def collect_lodge_outputs(runtime: Path, destination: Path, manifest: Path, split: str) -> int:
    candidates = sorted(
        runtime.glob("experiments/**/concat/npy/*.npy"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        return 0
    records = [record for record in read_manifest(manifest) if record.split == split]
    by_id = {record.name: record for record in records}
    by_song: dict[str, list] = {}
    for record in records:
        by_song.setdefault(record.song_name or record.name, []).append(record)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in candidates:
        matched = [by_id[path.stem]] if path.stem in by_id else by_song.get(path.stem, [])
        for record in matched:
            shutil.copy2(path, destination / f"{record.name}.npy")
            written += 1
    return written


def write_benchmark(path: Path, reports: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Method", "Status", "Samples", "FeatureFrechetProxy", "Diversity", "BeatAlignment",
        "DiversityRatio", "FootSkating", "ContactBalancedAccuracy", "ContactMacroF1",
        "EnergyRatio", "RootVelocityRatio", "BoundaryPosition", "BoundaryVelocity",
        "BoundaryAcceleration",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, report in reports:
            metric = report.get("aggregate", {})
            writer.writerow({
                "Method": method,
                "Status": report.get("status", "not_run"),
                "Samples": report.get("sample_count", 0),
                "FeatureFrechetProxy": metric.get("feature_frechet_proxy"),
                "Diversity": metric.get("diversity"),
                "BeatAlignment": metric.get("beat_alignment"),
                "DiversityRatio": metric.get("diversity_ratio"),
                "FootSkating": metric.get("foot_skating"),
                "ContactBalancedAccuracy": metric.get("contact_balanced_accuracy"),
                "ContactMacroF1": metric.get("contact_macro_f1"),
                "EnergyRatio": metric.get("energy_ratio"),
                "RootVelocityRatio": metric.get("root_velocity_ratio"),
                "BoundaryPosition": metric.get("boundary_position_jump"),
                "BoundaryVelocity": metric.get("boundary_velocity_jump"),
                "BoundaryAcceleration": metric.get("boundary_acceleration_jump"),
            })


def _load_comparability(baseline: str) -> dict[str, object]:
    if baseline == "lodge" or not COMPARABILITY.is_file():
        return {}
    payload = json.loads(COMPARABILITY.read_text(encoding="utf-8"))
    return next((item for item in payload["papers"] if item["id"] == baseline), {})


def _report_inventory(report_dir: Path) -> list[tuple[str, dict]]:
    reports: list[tuple[str, dict]] = []
    tempograph = report_dir / "tempograph_metrics.json"
    if tempograph.is_file():
        reports.append(("TempoGraph", json.loads(tempograph.read_text(encoding="utf-8"))))
    for slug, display in DISPLAY_NAMES.items():
        path = report_dir / f"{slug}_metrics.json"
        if path.is_file():
            reports.append((display, json.loads(path.read_text(encoding="utf-8"))))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate converted music-to-dance baselines with one shared FineDance protocol"
    )
    parser.add_argument("--baseline", default="lodge", choices=sorted(DISPLAY_NAMES))
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "artifacts" / "baselines" / "lodge")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "finedance" / "manifest.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prediction-format",
        default=None,
        choices=["canonical_npz", "npy139", "npy151", "npy319"],
        help="all methods are converted to the shared 22-joint canonical representation before scoring",
    )
    parser.add_argument("--run-lodge", action="store_true", help="invoke the official LODGE script in an already prepared LODGE environment")
    args = parser.parse_args()
    args.predictions = args.predictions or PROJECT / "artifacts" / "generated" / args.baseline
    args.output = args.output or PROJECT / "artifacts" / "reports" / f"{args.baseline}_metrics.json"
    args.prediction_format = args.prediction_format or (
        "npy319" if args.baseline == "lodge" else "canonical_npz"
    )
    if args.run_lodge and args.baseline != "lodge":
        parser.error("--run-lodge is valid only with --baseline lodge")
    inventory = checkpoint_inventory(args.checkpoint.resolve()) if args.baseline == "lodge" else {}
    source = REPOSITORY / "reference_code" / "baselines" / "LODGE-main"
    run_result = None
    if args.run_lodge:
        if os.name == "nt":
            run_result = {
                "return_code": None,
                "reason": "Official LODGE inference is Linux/CUDA-only; run this same command on AutoDL.",
            }
        else:
            runtime = prepare_lodge_runtime(source, args.checkpoint.resolve())
            command = [
                sys.executable, "infer_lodge.py",
                "--cfg", "exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml",
                "--cfg_assets", "configs/data/assets.yaml", "--soft", "1.0",
            ]
            completed = subprocess.run(command, cwd=runtime, check=False)
            collected = collect_lodge_outputs(runtime, args.predictions, args.manifest, args.split)
            run_result = {"command": command, "return_code": completed.returncode, "collected_predictions": collected}
    prediction_suffix = ".npz" if args.prediction_format == "canonical_npz" else ".npy"
    if args.predictions.is_dir() and any(args.predictions.glob(f"*{prediction_suffix}")):
        report = evaluate_predictions(
            args.predictions, args.manifest, args.split, args.prediction_format
        )
    else:
        format_hint = prediction_suffix
        report = {
            "status": "not_run",
            "sample_count": 0,
            "aggregate": {},
            "reason": (
                f"No ID-named converted {format_hint} predictions were found. "
                "Generate outputs with the official method, convert them to the declared format, "
                "and rerun this evaluator."
            ),
        }
    comparability = _load_comparability(args.baseline)
    report.update(
        {
            "baseline": DISPLAY_NAMES[args.baseline],
            "baseline_id": args.baseline,
            "checkpoint_inventory": inventory,
            "official_run": run_result,
            "comparison_tier": comparability.get("comparison_tier"),
            "protocol_warning": comparability.get("protocol_warning"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_benchmark(args.output.parent / "benchmark.csv", _report_inventory(args.output.parent))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
