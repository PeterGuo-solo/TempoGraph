import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

from tempograph.data.adapters import (
    motion_from_edge151,
    motion_from_finedance319,
    motion_from_lodge139,
)
from tempograph.data.boundary_samples import extract_boundary_sample
from tempograph.data.dataset import CanonicalDataset, DatasetRecord, write_manifest
from tempograph.data.adapters import save_motion_npz
from tempograph.data.music import resample_frame_features
from tempograph.data.finedance import build_finedance_manifest
from tempograph.data.dataset import read_manifest
from tempograph.synthetic import synthetic_motion
from tempograph.types import MotionSequence


class DataTest(unittest.TestCase):
    def test_local_flow_statistics_use_root_velocity_not_absolute_position(self):
        import torch
        from tempograph.data.torch_data import compute_local_flow_statistics

        class TinyDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                motion = torch.zeros(4, 13)
                motion[:, :4] = torch.tensor([1.0, 1.0, 0.0, 0.0])
                motion[:, 4] = torch.tensor([100.0, 101.0, 102.0, 103.0])
                return {"motion": motion}

        statistics = compute_local_flow_statistics(TinyDataset(), contact_dim=4)
        self.assertAlmostEqual(float(statistics["feature_mean"][0]), 0.75)
        np.testing.assert_allclose(
            statistics["contact_prior"].numpy(),
            np.asarray([1.0 - 1e-4, 1.0 - 1e-4, 1e-4, 1e-4]),
            rtol=1e-4, atol=1e-6,
        )

    def test_canonical_motion_round_trip(self):
        motion = synthetic_motion(frames=60)
        rebuilt = MotionSequence.from_features(motion.to_features(), motion.fps)
        np.testing.assert_allclose(rebuilt.to_features(), motion.to_features())
        self.assertEqual(rebuilt.feature_dim, 139)

    def test_reference_dimensions(self):
        motion_from_lodge139(np.zeros((10, 139), dtype=np.float32))
        motion_from_edge151(np.zeros((10, 151), dtype=np.float32))
        motion_from_finedance319(np.zeros((10, 319), dtype=np.float32))

    def test_boundary_mask_has_no_leakage(self):
        motion = synthetic_motion(frames=90)
        sample = extract_boundary_sample(motion, 35, 15, 10)
        sample.assert_no_leakage()
        self.assertTrue(np.isnan(sample.masked_features[10:25]).all())

    def test_music_resampling(self):
        source = np.arange(100, dtype=np.float32).reshape(20, 5)
        result = resample_frame_features(source, 10, 20)
        self.assertEqual(result.shape, (40, 5))

    def test_manifest_dataset(self):
        motion = synthetic_motion(frames=30)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_motion_npz(root / "clip.npz", motion)
            write_manifest(
                root / "manifest.json",
                [DatasetRecord("clip", "train", "clip.npz", source_dataset="synthetic")],
            )
            dataset = CanonicalDataset(root / "manifest.json", split="train")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["motion"].frames, 30)

    def test_finedance_official_test_split_uses_clip_ids_not_song_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("mofea319", "music_npynew", "label_json", "music_wav"):
                (root / name).mkdir()
            for clip_id in ("063", "133"):
                np.save(root / "mofea319" / f"{clip_id}.npy", np.zeros((8, 319), np.float32))
                np.save(root / "music_npynew" / f"{clip_id}.npy", np.zeros((8, 35), np.float32))
                (root / "label_json" / f"{clip_id}.json").write_text(
                    json.dumps({"style1": "same", "style2": "same", "name": "same-song"}),
                    encoding="utf-8",
                )
                (root / "music_wav" / f"{clip_id}.wav").write_bytes(b"placeholder")
            manifest = root / "manifest.json"
            build_finedance_manifest(root, manifest, validation_fraction=0.0)
            splits = {record.name: record.split for record in read_manifest(manifest)}
            self.assertEqual(splits["063"], "test")
            self.assertEqual(splits["133"], "train")


if __name__ == "__main__":
    unittest.main()
