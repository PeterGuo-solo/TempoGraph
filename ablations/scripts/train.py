from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
# Must be set before importing torch or touching CUDA.  The shell runners also
# export it, but direct train.py invocations must be deterministic as well.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tempograph.config import deep_update, load_config
from tempograph.data.finedance import build_finedance_manifest


def load_run_config(path: Path) -> dict:
    try:
        return load_config(path)
    except ValueError:
        base = load_config(PROJECT / "configs" / "base.json")
        override = json.loads(path.read_text(encoding="utf-8"))
        merged = deep_update(base, override)
        expected = 4 + 3 + 6 * merged["motion"]["joint_count"]
        if merged["motion"]["feature_dim"] != expected:
            raise ValueError("motion.feature_dim is inconsistent with joint_count")
        return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TempoGraph on real FineDance data")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "base.json")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "motion_autoencoder", "graph_and_local", "boundary_planner", "boundary_bridge", "joint_finetune"],
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--max-hours", type=float,
        help="gracefully stop the selected stage after this wall-clock budget",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="load model weights only; start a fresh optimizer/scheduler at step zero",
    )
    parser.add_argument("--resume-auto", action="store_true", help="resume from output-dir/last.pt")
    parser.add_argument("--device")
    parser.add_argument(
        "--seed", type=int,
        help="training/init seed override; the dataset split keeps experiment.split_seed",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT / "artifacts" / "checkpoints",
        help="checkpoint directory; reports remain under artifacts/reports",
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        help="report directory; defaults to a reports sibling of output-dir",
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "finedance" / "manifest.json")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from tempograph.data.torch_data import FineDanceWindowDataset, compute_local_flow_statistics
        from tempograph.engine import STAGES, build_models, load_checkpoint, resolve_device, seed_everything, train_stage
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise SystemExit('PyTorch is required. Install with: python -m pip install -e ".[train]"')
        raise

    config = load_run_config(args.config.resolve())
    if args.seed is not None:
        config["seed"] = int(args.seed)
    training = config["training"]
    if args.max_hours is not None:
        if args.max_hours <= 0:
            raise SystemExit("--max-hours must be positive")
        training["max_stage_seconds"] = float(args.max_hours) * 3600.0
    if not args.manifest.is_file():
        split_seed = int(config.get("experiment", {}).get("split_seed", 1234))
        build_finedance_manifest(args.manifest.parent, args.manifest, seed=split_seed)
    device = resolve_device(args.device or str(training.get("device", "auto")))
    seed_everything(int(config["seed"]))

    checkpoints = args.output_dir.resolve()
    checkpoints.mkdir(parents=True, exist_ok=True)
    if sum(bool(value) for value in (args.resume, args.resume_auto, args.init_checkpoint)) > 1:
        raise SystemExit("choose only one of --resume, --resume-auto, or --init-checkpoint")
    if args.resume_auto:
        candidate = checkpoints / "last.pt"
        if not candidate.is_file():
            raise SystemExit(f"automatic resume checkpoint not found: {candidate}")
        args.resume = candidate

    reports_dir = (
        args.reports_dir.resolve()
        if args.reports_dir
        else checkpoints.parent / "reports"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(checkpoints),
        "requested_stage": args.stage,
        "max_steps_override": args.max_steps,
        "max_hours_override": args.max_hours,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": int(config["seed"]),
        "training": training,
    }
    (reports_dir / "run_manifest.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dataset_options = {
        "manifest_path": args.manifest,
        "window_frames": int(training["window_frames"]),
        "stride": int(training["window_stride"]),
        "max_records": training.get("max_records"),
        "max_windows_per_record": training.get("max_windows_per_record"),
    }
    train_data = FineDanceWindowDataset(split="train", **dataset_options)
    val_data = FineDanceWindowDataset(split="val", **dataset_options)
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    models = build_models(config, device)
    if args.init_checkpoint:
        if not args.init_checkpoint.is_file():
            raise SystemExit(f"initial checkpoint not found: {args.init_checkpoint}")
        load_checkpoint(args.init_checkpoint.resolve(), models, device)
    statistics = compute_local_flow_statistics(
        train_data, contact_dim=int(config["motion"].get("contact_dim", 4))
    )
    models["local_generator"].configure_statistics(
        statistics["feature_mean"], statistics["feature_std"], statistics["contact_prior"]
    )
    skeleton_path = args.manifest.parent / "smplx_neu_J_1.npy"
    if not skeleton_path.is_file():
        raise FileNotFoundError(
            f"neutral skeleton required for differentiable LODGE losses: {skeleton_path}"
        )
    models["local_generator"].configure_skeleton(
        torch.from_numpy(np.asarray(np.load(skeleton_path, allow_pickle=False), dtype=np.float32))
    )
    statistics_report = {
        "representation": "standardized planar root velocity, vertical height, and rotation-6D pose",
        "frame_count": int(statistics["frame_count"]),
        "feature_mean": statistics["feature_mean"].tolist(),
        "feature_std": statistics["feature_std"].tolist(),
        "contact_prior": statistics["contact_prior"].tolist(),
        "skeleton": str(skeleton_path.resolve()),
    }
    (reports_dir / "local_flow_statistics.json").write_text(
        json.dumps(statistics_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    resume_for_stage = args.resume.resolve() if args.resume else None
    if resume_for_stage and args.stage == "all":
        payload = torch.load(resume_for_stage, map_location="cpu", weights_only=False)
        resume_stage = payload.get("stage")
        if resume_stage in stages:
            stages = stages[stages.index(resume_stage) :]

    reports = []
    checkpoint_names = {
        "motion_autoencoder": "motion_autoencoder.pt",
        "graph_and_local": "graph_local.pt",
        "boundary_planner": "boundary_planner.pt",
        "boundary_bridge": "boundary_bridge.pt",
        "joint_finetune": "tempograph.pt",
    }
    for index, stage in enumerate(stages):
        if index == 0 and resume_for_stage is None and args.stage != "all":
            position = list(STAGES).index(stage)
            if position > 0:
                prior_stage = STAGES[position - 1]
                dependency = checkpoints / checkpoint_names[prior_stage]
                if dependency.is_file():
                    load_checkpoint(dependency, models, device)
        steps = int(args.max_steps or training.get("stage_steps", {}).get(stage, training.get("max_steps", 1000)))
        reports.append(
            train_stage(
                stage, models, train_loader, val_loader, config, device, checkpoints, steps,
                resume=resume_for_stage if index == 0 else None,
            )
        )
        resume_for_stage = None
    print(json.dumps({"device": str(device), "reports": reports}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
