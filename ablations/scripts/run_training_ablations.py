"""Matched training for Full / no relation supervision / randomized relations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from tempograph.config import deep_update, load_config
from tempograph.relation_supervision import MODES, configure_relation_mode

NAMES = {"motion_autoencoder": "motion_autoencoder.pt", "graph_and_local": "graph_local.pt",
         "boundary_planner": "boundary_planner.pt", "boundary_bridge": "boundary_bridge.pt",
         "joint_finetune": "tempograph.pt"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print("RUN:", " ".join(map(str, arguments)), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(list(map(str, arguments)), cwd=PROJECT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f"command failed; see {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def checkpoint_header(path, stage, steps=None):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != stage or payload.get("schema_version") not in (5, 6, 7):
        raise ValueError(f"incompatible checkpoint: {path}")
    if int(payload.get("step", 0)) < 1:
        raise ValueError(f"empty checkpoint: {path}")
    if steps is not None and int(payload["step"]) != steps:
        raise ValueError(f"incomplete training: {path}; expected {steps}, found {payload['step']}")
    return payload


def group_initialization(args, autoencoder, group_root):
    """Import only AE weights; all graph/local/boundary weights are newly initialized."""
    import torch
    from tempograph.engine import build_models, seed_everything
    destination = group_root / "initialization.pt"
    if destination.exists():
        return destination
    config = load_config(PROJECT / "configs/base.json")
    seed_everything(args.training_seed)
    models = build_models(config, torch.device("cpu"))
    previous = checkpoint_header(autoencoder, "motion_autoencoder")
    models["motion_autoencoder"].load_state_dict(previous["models"]["motion_autoencoder"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save({"schema_version": 7, "stage": "motion_autoencoder", "step": previous["step"],
                "config": config, "models": {k: v.state_dict() for k, v in models.items()},
                "initialization": "AE weights only; all other model weights freshly seeded",
                "ae_source_sha256": sha(autoencoder)}, temporary)
    temporary.replace(destination)
    return destination


def run_phase(args, root, phase, mode, initial):
    folder = root / phase["name"]
    stage, steps = phase["stage"], int(phase["steps"])
    base = load_config(PROJECT / "configs/base.json")
    config = deep_update(base, read(PROJECT / phase["config"]))
    config["seed"] = args.training_seed
    configure_relation_mode(config, mode, args.shuffle_seed)
    config["training"]["stage_steps"] = {stage: steps}
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    updates = math.ceil(steps / accumulation)
    hyper = config["training"].setdefault("stage_hyperparameters", {}).setdefault(stage, {})
    hyper["warmup_updates"] = min(int(hyper.get("warmup_updates", config["training"].get("warmup_updates", 0))),
                                  int(updates * 0.05))
    # The declared recipe preserves full data/model sizes. Never use a time-budget truncation.
    for key in ("max_records", "max_windows_per_record"):
        if config["training"].get(key):
            raise ValueError(f"partial-data option is not allowed: {key}")
    destination = folder / "checkpoints" / NAMES[stage]
    descriptor = {"stage": stage, "steps": steps, "config": config,
                  "initial": str(initial) if initial else None,
                  "initial_sha256": sha(initial) if initial else None}
    request = folder / "request.json"
    if request.exists():
        if not args.resume:
            raise ValueError(f"existing run: {folder}; pass --resume to continue it")
        if read(request) != descriptor:
            raise ValueError(f"changed initial weights/configuration: {folder}; choose a new output directory")
    else:
        write(request, descriptor)
    if (folder / "complete.json").exists():
        complete = read(folder / "complete.json")
        if not destination.is_file() or complete["sha256"] != sha(destination):
            raise ValueError(f"completed checkpoint changed or missing: {destination}")
        print(f"SKIP completed: {folder}", flush=True)
        return destination
    config_path = folder / "config.json"
    write(config_path, config)
    command = [sys.executable, PROJECT / "scripts/train.py", "--config", config_path,
               "--stage", stage, "--max-steps", steps, "--device", args.device,
               "--seed", args.training_seed, "--manifest", args.data_root / "manifest.json",
               "--output-dir", folder / "checkpoints", "--reports-dir", folder / "reports"]
    if args.resume and (folder / "checkpoints/last.pt").exists():
        command += ["--resume-auto"]
    elif initial:
        command += ["--init-checkpoint", initial]
    execute(command, folder / "train.log")
    checkpoint_header(destination, stage, steps)
    write(folder / "complete.json", {"checkpoint": str(destination), "sha256": sha(destination)})
    return destination


def evaluate(args, root, checkpoint):
    diagnostic = root / "relation_diagnostics.json"
    if not diagnostic.exists():
        execute([sys.executable, PROJECT / "scripts/diagnose_relations.py", "--checkpoint", checkpoint,
                 "--manifest", args.data_root / "manifest.json", "--device", args.device,
                 "--output", diagnostic], root / "relation_diagnostics.log")
    for seed in args.sampling_seeds:
        folder = root / "evaluation" / f"sampling_seed_{seed}"
        if args.resume and (folder / "complete.json").exists():
            continue
        npz, npy = folder / "npz", folder / "npy139"
        execute([sys.executable, PROJECT / "scripts/infer.py", "--checkpoint", checkpoint,
                 "--manifest", args.data_root / "manifest.json", "--split", "val",
                 "--device", args.device, "--seed", seed, "--max-frames", 1024,
                 "--motion-prior-weight", 0.90, "--generation-profile", "v78_rhythmic",
                 "--output", npz], folder / "infer.log")
        execute([sys.executable, PROJECT / "scripts/export_finedance139.py", "--input", npz,
                 "--output", npy, "--frames", 1024], folder / "export.log")
        execute([sys.executable, PROJECT / "scripts/evaluate_lodge_protocol.py", "--predictions", npy,
                 "--data-root", args.data_root, "--lodge-root", args.lodge_root,
                 "--output", folder / "metrics.json"], folder / "evaluate.log")
        report = read(folder / "metrics.json")
        if report.get("status") != "complete":
            raise ValueError(f"evaluation incomplete: {folder}")
        write(folder / "complete.json", {"checkpoint_sha256": sha(checkpoint), "sampling_seed": seed})


def validate_recipe(recipe):
    phases = recipe["phases"]
    expected = ["motion_autoencoder", "graph_and_local", "boundary_planner", "boundary_bridge",
                "joint_finetune", "joint_finetune"]
    if [p["stage"] for p in phases] != expected or len({p["name"] for p in phases}) != len(phases):
        raise ValueError("recipe must retain all declared phases with unique names")
    if any(not isinstance(p["steps"], int) or p["steps"] <= 0 for p in phases):
        raise ValueError("all phase step counts must be positive integers")
    if recipe.get("steps_per_group") != 25000 or sum(p["steps"] for p in phases) != 25000 or "autoencoder" in recipe:
        raise ValueError("each experiment must total exactly 25000 steps, including autoencoder; no extra shared training")
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lodge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=PROJECT / "configs/training_ablations.json")
    parser.add_argument("--group", choices=("all",) + MODES, default="all")
    parser.add_argument("--training-seed", type=int, default=1234)
    parser.add_argument("--shuffle-seed", type=int, default=77123)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[3407])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="print schedule; do not train or write outputs")
    args = parser.parse_args()
    for key in ("data_root", "lodge_root", "output", "recipe"):
        setattr(args, key, getattr(args, key).resolve())
    recipe = read(args.recipe)
    phases = validate_recipe(recipe)
    groups = list(MODES) if args.group == "all" else [args.group]
    print(json.dumps({"recipe": recipe, "groups": groups, "training_seed": args.training_seed,
                      "sampling_seeds": args.sampling_seeds,
                      "microsteps_per_group": 25000,
                      "extra_shared_training_steps": 0,
                      "total_microsteps": len(groups) * 25000}, indent=2), flush=True)
    if args.plan_only:
        return
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    manifest = args.data_root / "manifest.json"
    for required in (manifest, args.data_root / "smplx_neu_J_1.npy",
                     args.lodge_root / "metric/features/kinetic.py"):
        if not required.exists():
            raise FileNotFoundError(required)
    for features in ("kinetic_features", "manual_features_new"):
        if not any((args.data_root / "mofea319" / features).glob("*.npy")):
            raise FileNotFoundError(f"missing LODGE reference features: mofea319/{features}")
    records = read(manifest)["records"]
    for record in records:
        if record["split"] not in ("train", "val"):
            continue
        for key in ("motion_path", "music_path", "audio_path"):
            if not record.get(key) or not (args.data_root / record[key]).is_file():
                raise FileNotFoundError(f"{record['name']}: missing {key}")
        if not (args.data_root / "lodge_beats" / (record["name"] + ".npy")).is_file():
            raise FileNotFoundError("missing official beat cache; run scripts/cache_lodge_beats.py --manifest <manifest>")
    contract = {"recipe": recipe, "training_seed": args.training_seed,
                "shuffle_seed": args.shuffle_seed, "sampling_seeds": args.sampling_seeds,
                "manifest": str(manifest), "manifest_sha256": sha(manifest),
                "lodge_root": str(args.lodge_root),
                "source_hashes": {str(p.relative_to(PROJECT)): sha(p)
                                  for folder in ("src", "scripts", "configs")
                                  for p in sorted((PROJECT / folder).rglob("*"))
                                  if p.suffix in (".py", ".json")}}
    contract_path = args.output / "run_contract.json"
    if contract_path.exists():
        if not args.resume or read(contract_path) != contract:
            raise ValueError("existing output or changed protocol; use --resume with identical settings, or a new output path")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output directory is not empty")
        write(contract_path, contract)
    for group in groups:
        initial = None
        for phase in phases:
            initial = run_phase(args, args.output / group, phase, group, initial)
            if phase["stage"] == "motion_autoencoder":
                initial = group_initialization(args, initial, args.output / group)
        evaluate(args, args.output / group, initial)
    summary = {}
    for group in MODES:
        rows = []
        for seed in args.sampling_seeds:
            path = args.output / group / "evaluation" / f"sampling_seed_{seed}" / "metrics.json"
            if path.is_file():
                report = read(path)
                rows.append({"sampling_seed": seed, "sample_count": report.get("sample_count"),
                             "ids": report.get("ids"), "aggregate": report.get("aggregate")})
        summary[group] = rows
    for seed in args.sampling_seeds:
        comparable = [row for rows in summary.values() for row in rows if row["sampling_seed"] == seed]
        if comparable and any((row["ids"], row["sample_count"]) != (comparable[0]["ids"], comparable[0]["sample_count"])
                              for row in comparable):
            raise ValueError("evaluation song sets differ across groups")
    write(args.output / "comparison.json", {"training_seed": args.training_seed, "groups": summary})
    print(f"Finished. Results: {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
