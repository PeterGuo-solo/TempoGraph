"""Training schedule, dependency gates, and dry-run validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np

from .data.dataset import read_manifest


@dataclass(frozen=True)
class TrainingStage:
    name: str
    required_inputs: tuple[str, ...]
    output: str
    pass_condition: str


def build_training_schedule() -> list[TrainingStage]:
    return [
        TrainingStage(
            "motion_autoencoder",
            ("canonical motions", "train/validation split"),
            "artifacts/checkpoints/motion_autoencoder.pt",
            "rotation, root, contact, velocity, and acceleration validation losses are finite",
        ),
        TrainingStage(
            "graph_and_local",
            ("music hierarchy", "confidence-weighted pseudo graph", "autoencoder"),
            "artifacts/checkpoints/graph_local.pt",
            "durations cover every beat and local outputs satisfy schema",
        ),
        TrainingStage(
            "boundary_planner",
            ("masked transition samples", "contact/root/orientation/energy targets"),
            "artifacts/checkpoints/boundary_planner.pt",
            "all five plan heads improve over analytic constant baselines",
        ),
        TrainingStage(
            "boundary_bridge",
            ("frozen or weakly updated planner", "transition latents"),
            "artifacts/checkpoints/boundary_bridge.pt",
            "flow, endpoint, contact, energy, foot, and rigid losses are finite",
        ),
        TrainingStage(
            "joint_finetune",
            ("all previous checkpoints",),
            "artifacts/checkpoints/tempograph.pt",
            "full inference passes boundary and music-alignment validation",
        ),
    ]


def _valid_files(directory: Path, pattern: str, minimum: int = 1) -> tuple[bool, int]:
    count = sum(1 for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0)
    return count >= minimum, count


def _zip_signature(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"PK\x03\x04"


def inspect_training_assets(
    project_root: str | Path,
    require_cuda: bool = True,
) -> dict[str, object]:
    root = Path(project_root)
    data = root / "data" / "finedance"
    motion_ok, motion_count = _valid_files(data / "mofea319", "*.npy", 100)
    music_ok, music_count = _valid_files(data / "music_npynew", "*.npy", 100)
    label_ok, label_count = _valid_files(data / "label_json", "*.json", 100)
    audio_ok, audio_count = _valid_files(data / "music_wav", "*.wav", 100)
    paired_ids = (
        {path.stem for path in (data / "mofea319").glob("*.npy")}
        & {path.stem for path in (data / "music_npynew").glob("*.npy")}
        & {path.stem for path in (data / "label_json").glob("*.json")}
        & {path.stem for path in (data / "music_wav").glob("*.wav")}
    )
    sample_valid = False
    if paired_ids:
        sample = sorted(paired_ids)[0]
        motion = np.load(data / "mofea319" / f"{sample}.npy", mmap_mode="r", allow_pickle=False)
        music = np.load(data / "music_npynew" / f"{sample}.npy", mmap_mode="r", allow_pickle=False)
        sample_valid = motion.ndim == 2 and motion.shape[1] == 319 and music.ndim == 2 and music.shape[1] == 35

    manifest_path = data / "manifest.json"
    manifest_valid = False
    manifest_files_valid = False
    split_song_isolation = False
    manifest_counts: dict[str, int] = {}
    if manifest_path.is_file():
        try:
            records = read_manifest(manifest_path)
            manifest_counts = {
                split: sum(record.split == split for record in records)
                for split in ("train", "val", "test")
            }
            manifest_valid = bool(records) and all(manifest_counts.values())
            manifest_files_valid = all(
                (manifest_path.parent / record.motion_path).is_file()
                and bool(record.music_path)
                and (manifest_path.parent / str(record.music_path)).is_file()
                for record in records
            )
            songs = {
                split: {record.song_name for record in records if record.split == split}
                for split in ("train", "val", "test")
            }
            split_song_isolation = not (
                songs["train"] & songs["val"]
                or songs["train"] & songs["test"]
                or songs["val"] & songs["test"]
            )
        except (ValueError, OSError, json.JSONDecodeError, TypeError):
            manifest_valid = False

    required_smpl = ["SMPL_FEMALE.pkl", "SMPL_MALE.pkl", "SMPL_NEUTRAL.pkl"]
    required_smplh = ["SMPLH_FEMALE.pkl", "SMPLH_MALE.pkl"]
    smpl_paths = [root / "models" / "smpl" / name for name in required_smpl]
    smplh_paths = [root / "models" / "smplh" / name for name in required_smplh]
    smpl_valid = all(path.is_file() and path.stat().st_size > 1_000_000 for path in smpl_paths)
    smplh_valid = all(path.is_file() and path.stat().st_size > 1_000_000 for path in smplh_paths)

    lodge_global = root / "artifacts" / "baselines" / "lodge" / "exp" / "Global_Module" / "FineDance_Global" / "checkpoints" / "epoch=2999.ckpt"
    lodge_local = root / "artifacts" / "baselines" / "lodge" / "exp" / "Local_Module" / "FineDance_FineTuneV2_Local" / "checkpoints" / "epoch=299.ckpt"
    lodge_valid = _zip_signature(lodge_global) and _zip_signature(lodge_local)

    torch_installed = importlib.util.find_spec("torch") is not None
    cuda_available = False
    torch_version = None
    if torch_installed:
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, OSError):
            torch_installed = False

    checks = {
        "motion_319": motion_ok,
        "music_35": music_ok,
        "labels": label_ok,
        "audio": audio_ok,
        "paired_records": len(paired_ids) >= 100,
        "sample_shapes": sample_valid,
        "manifest": manifest_valid,
        "manifest_files": manifest_files_valid,
        "split_song_isolation": split_song_isolation,
        "normalizer": (data / "Normalizer.pth").is_file(),
        "joint_template": (data / "smplx_neu_J_1.npy").is_file(),
        "smpl": smpl_valid,
        "smplh": smplh_valid,
        "lodge_checkpoints": lodge_valid,
        "pytorch": torch_installed,
        "cuda": cuda_available,
    }
    # The released FineDance/LODGE benchmark split is defined by exact clip IDs.
    # Multiple clips can share a song name across splits, so song-level isolation
    # is diagnostic only and must not block the official split.  Published LODGE
    # checkpoints are also optional: TempoGraph training and the feature-based
    # paper metric evaluator do not load or run the LODGE model.
    optional_checks = {"split_song_isolation", "lodge_checkpoints"}
    required = [
        key
        for key in checks
        if key not in optional_checks and (key != "cuda" or require_cuda)
    ]
    ready = all(checks[key] for key in required)
    return {
        "ready_for_training": ready,
        "require_cuda": require_cuda,
        "optional_checks": sorted(optional_checks),
        "checks": checks,
        "counts": {
            "motion": motion_count,
            "music": music_count,
            "labels": label_count,
            "audio": audio_count,
            "paired": len(paired_ids),
            "manifest": manifest_counts,
        },
        "runtime": {"torch": torch_version, "cuda_available": cuda_available},
        "schedule": [asdict(stage) for stage in build_training_schedule()],
    }
