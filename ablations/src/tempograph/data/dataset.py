"""Manifest-backed canonical dataset shared by every training stage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .adapters import load_motion_npz


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    split: str
    motion_path: str
    music_path: str | None = None
    structure_path: str | None = None
    label_path: str | None = None
    audio_path: str | None = None
    motion_format: str = "canonical_npz"
    fps: float = 30.0
    frames: int | None = None
    aligned_frames: int | None = None
    style_coarse: str | None = None
    style_fine: str | None = None
    song_name: str | None = None
    source_dataset: str = "unknown"


def write_manifest(path: str | Path, records: list[DatasetRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "records": [asdict(record) for record in records]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: str | Path) -> list[DatasetRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported dataset manifest version")
    return [DatasetRecord(**item) for item in payload["records"]]


class CanonicalDataset:
    def __init__(self, manifest_path: str | Path, split: str | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        records = read_manifest(self.manifest_path)
        self.records = records if split is None else [record for record in records if record.split == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        motion_path = (self.manifest_path.parent / record.motion_path).resolve()
        if record.motion_format != "canonical_npz":
            raise ValueError(
                "CanonicalDataset only loads canonical_npz records; "
                "use FineDanceWindowDataset for raw FineDance features"
            )
        return {"record": record, "motion": load_motion_npz(motion_path)}
