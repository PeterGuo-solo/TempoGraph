"""FineDance@Genre manifest construction and lazy aligned window loading."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetRecord, read_manifest, write_manifest


FINEDANCE_GENRE_TEST = {
    "063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
    "179", "065", "137", "161", "092", "120", "037", "109", "204", "144",
}
FINEDANCE_IGNORE = {"116", "117", "118", "119", "120", "121", "122", "123", "202"}


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _relative(path: Path, parent: Path) -> str:
    return path.resolve().relative_to(parent.resolve()).as_posix()


def _validation_groups(
    entries: list[dict[str, Any]], fraction: float, seed: int
) -> set[str]:
    by_genre: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        by_genre[entry["style_fine"]][entry["song_name"]].append(entry)
    selected: set[str] = set()
    for genre, groups in sorted(by_genre.items()):
        names = sorted(groups, key=lambda name: (_stable_fraction(f"{genre}:{name}", seed), name))
        count = max(1, int(round(len(names) * fraction))) if len(names) >= 3 else 0
        selected.update(names[:count])
    return selected


def build_finedance_manifest(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
    validation_fraction: float = 0.10,
    seed: int = 1234,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else root / "manifest.json"
    if not 0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in [0, 0.5)")

    motion_dir = root / "mofea319"
    music_dir = root / "music_npynew"
    label_dir = root / "label_json"
    audio_dir = root / "music_wav"
    structure_dir = root / "structure_allin1"

    ids = sorted(path.stem for path in motion_dir.glob("*.npy"))
    entries: list[dict[str, Any]] = []
    skipped: dict[str, list[str]] = defaultdict(list)
    for clip_id in ids:
        required = {
            "motion": motion_dir / f"{clip_id}.npy",
            "music": music_dir / f"{clip_id}.npy",
            "label": label_dir / f"{clip_id}.json",
            "audio": audio_dir / f"{clip_id}.wav",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            skipped[clip_id].extend(missing)
            continue
        if clip_id in FINEDANCE_IGNORE:
            skipped[clip_id].append("official_ignore")
            continue
        label = json.loads(required["label"].read_text(encoding="utf-8"))
        motion = np.load(required["motion"], mmap_mode="r", allow_pickle=False)
        music = np.load(required["music"], mmap_mode="r", allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 319:
            raise ValueError(f"{required['motion']} must have shape [T, 319]")
        if music.ndim != 2 or music.shape[1] != 35:
            raise ValueError(f"{required['music']} must have shape [T, 35]")
        aligned = min(int(motion.shape[0]), int(music.shape[0]))
        entries.append(
            {
                "id": clip_id,
                "motion": required["motion"],
                "music": required["music"],
                "label": required["label"],
                "audio": required["audio"],
                "structure": structure_dir / f"{clip_id}.json",
                "frames": int(motion.shape[0]),
                "aligned_frames": aligned,
                "style_coarse": str(label.get("style1", "unknown")),
                "style_fine": str(label.get("style2", "unknown")),
                "song_name": str(label.get("name", clip_id)),
            }
        )

    # Match the released FineDance/LODGE split by clip id.  Grouping every
    # recording with the same song title into the test set silently expands
    # the official split (25 clips in the local data instead of the released
    # id list) and makes paper-level metrics incomparable.
    train_candidates = [item for item in entries if item["id"] not in FINEDANCE_GENRE_TEST]
    val_songs = _validation_groups(train_candidates, validation_fraction, seed)

    records: list[DatasetRecord] = []
    for item in entries:
        if item["id"] in FINEDANCE_GENRE_TEST:
            split = "test"
        elif item["song_name"] in val_songs:
            split = "val"
        else:
            split = "train"
        structure_path = item["structure"] if item["structure"].is_file() else None
        records.append(
            DatasetRecord(
                name=item["id"],
                split=split,
                motion_path=_relative(item["motion"], manifest.parent),
                music_path=_relative(item["music"], manifest.parent),
                structure_path=(
                    _relative(structure_path, manifest.parent) if structure_path else None
                ),
                label_path=_relative(item["label"], manifest.parent),
                audio_path=_relative(item["audio"], manifest.parent),
                motion_format="finedance319",
                fps=30.0,
                frames=item["frames"],
                aligned_frames=item["aligned_frames"],
                style_coarse=item["style_coarse"],
                style_fine=item["style_fine"],
                song_name=item["song_name"],
                source_dataset="FineDance",
            )
        )
    write_manifest(manifest, records)

    split_counts = Counter(record.split for record in records)
    genre_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        genre_counts[record.split][record.style_fine or "unknown"] += 1
    summary = {
        "manifest": str(manifest),
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "genres": {split: dict(sorted(counts.items())) for split, counts in genre_counts.items()},
        "skipped": dict(sorted(skipped.items())),
        "alignment": "motion and music are cropped to their shared frame interval at load time",
        "split_policy": "released FineDance/LODGE test clip ids; genre-stratified song-group validation",
        "seed": seed,
        "validation_fraction": validation_fraction,
    }
    (manifest.parent / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def canonical_139(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] not in (139, 151, 319):
        raise ValueError("motion must have shape [T, 139], [T, 151], or [T, 319]")
    if array.shape[1] in (151, 319):
        # EDGE stores 24 SMPL joints (151D), while the shared FineDance
        # evaluator uses the first 22 body joints (139D). FineDance 319D
        # likewise begins with the same contacts/root/body rotation block.
        array = np.concatenate([array[:, :7], array[:, 7 : 7 + 22 * 6]], axis=-1)
    result = np.asarray(array, dtype=np.float32).copy()
    result[:, 4:7] -= result[:1, 4:7]
    return result


def load_aligned_record(manifest_path: str | Path, record: DatasetRecord) -> dict[str, Any]:
    parent = Path(manifest_path).resolve().parent
    motion = canonical_139(
        np.load(parent / record.motion_path, allow_pickle=False, mmap_mode="r")
    )
    if not record.music_path:
        raise ValueError(f"record {record.name} has no music_path")
    music = np.asarray(
        np.load(parent / record.music_path, allow_pickle=False, mmap_mode="r"), dtype=np.float32
    )
    frames = min(len(motion), len(music))
    return {
        "record": record,
        "motion": motion[:frames],
        "music": music[:frames],
        "frames": frames,
    }


def manifest_summary(manifest_path: str | Path) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    return {
        "records": len(records),
        "splits": dict(Counter(record.split for record in records)),
        "genres": len({record.style_fine for record in records}),
        "songs": len({record.song_name for record in records}),
    }
