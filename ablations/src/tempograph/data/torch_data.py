"""PyTorch datasets kept separate so NumPy-only commands remain lightweight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import read_manifest
from .finedance import canonical_139


class FineDanceWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        window_frames: int,
        stride: int,
        max_records: int | None = None,
        max_windows_per_record: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.parent = self.manifest_path.parent
        records = [record for record in read_manifest(self.manifest_path) if record.split == split]
        self.records = records[:max_records] if max_records else records
        self.window_frames = int(window_frames)
        self.index: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            frames = int(record.aligned_frames or record.frames or 0)
            starts = list(range(0, max(1, frames - self.window_frames + 1), max(1, stride)))
            if frames >= self.window_frames and (not starts or starts[-1] + self.window_frames < frames):
                starts.append(frames - self.window_frames)
            if max_windows_per_record:
                starts = starts[:max_windows_per_record]
            self.index.extend((record_index, start) for start in starts if frames >= self.window_frames)
        if not self.index:
            raise ValueError(f"no {split} windows of length {window_frames} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, object]:
        record_index, start = self.index[index]
        record = self.records[record_index]
        motion_raw = np.load(
            self.parent / record.motion_path, allow_pickle=False, mmap_mode="r"
        )
        if not record.music_path:
            raise ValueError(f"record {record.name} has no music")
        music_raw = np.load(
            self.parent / record.music_path, allow_pickle=False, mmap_mode="r"
        )
        end = start + self.window_frames
        motion = canonical_139(motion_raw[start:end])
        music = np.asarray(music_raw[start:end], dtype=np.float32)
        if len(motion) != self.window_frames or len(music) != self.window_frames:
            raise ValueError(f"short window for {record.name} at {start}")
        # Train against the exact WAV/librosa beat detector used by the
        # released LODGE BAS evaluator.  Older runs used the final channel of
        # music_npynew, which is only a proxy and can disagree substantially
        # with the official metric.  The cache is prepared once by
        # scripts/cache_lodge_beats.py.  Falling back keeps generic/debug
        # datasets usable, while the V7.3 runner requires the cache.
        beat_evidence = np.zeros(self.window_frames, dtype=np.float32)
        beat_path = self.parent / "lodge_beats" / f"{record.name}.npy"
        if beat_path.is_file():
            beats = np.asarray(np.load(beat_path, allow_pickle=False), dtype=np.int64)
            local = beats[(beats >= start) & (beats < end)] - start
            beat_evidence[local] = 1.0
        else:
            beat_evidence[:] = np.clip(music[:, -1], 0.0, None)
        return {
            "motion": torch.from_numpy(motion),
            "music": torch.from_numpy(music.copy()),
            "beat_evidence": torch.from_numpy(beat_evidence),
            "name": record.name,
            "style_fine": record.style_fine or "unknown",
            "start": start,
        }


@torch.no_grad()
def compute_local_flow_statistics(
    dataset: Dataset,
    contact_dim: int = 4,
) -> dict[str, torch.Tensor]:
    """Streaming statistics for planar-root-velocity/height/pose features."""
    feature_sum: torch.Tensor | None = None
    feature_square_sum: torch.Tensor | None = None
    contact_sum: torch.Tensor | None = None
    frame_count = 0
    for index in range(len(dataset)):
        motion = dataset[index]["motion"].double()
        raw = motion[:, contact_dim:]
        encoded_root = raw[:, :3].clone()
        encoded_root[:, 0] = 0.0
        encoded_root[:, 2] = 0.0
        encoded_root[1:, 0] = raw[1:, 0] - raw[:-1, 0]
        encoded_root[1:, 2] = raw[1:, 2] - raw[:-1, 2]
        relative = torch.cat([encoded_root, raw[:, 3:]], dim=-1)
        current_sum = relative.sum(dim=0)
        current_square_sum = relative.square().sum(dim=0)
        current_contact_sum = motion[:, :contact_dim].sum(dim=0)
        feature_sum = current_sum if feature_sum is None else feature_sum + current_sum
        feature_square_sum = (
            current_square_sum
            if feature_square_sum is None
            else feature_square_sum + current_square_sum
        )
        contact_sum = (
            current_contact_sum if contact_sum is None else contact_sum + current_contact_sum
        )
        frame_count += len(motion)
    if frame_count == 0 or feature_sum is None or feature_square_sum is None or contact_sum is None:
        raise ValueError("cannot compute local-flow statistics from an empty dataset")
    mean = feature_sum / frame_count
    variance = (feature_square_sum / frame_count - mean.square()).clamp_min(1e-8)
    return {
        "feature_mean": mean.float(),
        "feature_std": variance.sqrt().float().clamp_min(1e-3),
        "contact_prior": (contact_sum / frame_count).float().clamp(1e-4, 1 - 1e-4),
        "frame_count": torch.tensor(frame_count, dtype=torch.long),
    }
