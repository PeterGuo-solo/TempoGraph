import json
from pathlib import Path

import numpy as np
import pytest

from tempograph.benchmarking import evaluate_predictions
from tempograph.data.adapters import save_motion_npz
from tempograph.data.dataset import DatasetRecord, write_manifest
from tempograph.data.finedance import canonical_139
from tempograph.lodge_protocol import forward_kinematics_139
from tempograph.types import MotionSequence
from tempograph.generation import _stabilize_generated_motion


def test_canonical_139_keeps_body_and_centers_root():
    source = np.zeros((5, 319), dtype=np.float32)
    source[:, 4:7] = np.asarray([3.0, 4.0, 5.0])
    source[:, 7:] = np.arange(312, dtype=np.float32)
    result = canonical_139(source)
    assert result.shape == (5, 139)
    np.testing.assert_allclose(result[0, 4:7], 0.0)
    np.testing.assert_allclose(result[:, 7:], source[:, 7:139])


def test_generation_stabilizer_scales_only_planar_root_velocity_and_smooths_pose():
    features = np.zeros((6, 139), dtype=np.float32)
    features[:, 4] = np.arange(6, dtype=np.float32)
    features[:, 5] = np.asarray([0.0, 0.2, 0.1, 0.3, 0.1, 0.0])
    features[:, 6] = 2.0 * np.arange(6, dtype=np.float32)
    features[::2, 7:] = 1.0
    result = _stabilize_generated_motion(
        features,
        {"generation": {"planar_root_velocity_scale": 0.5, "pose_temporal_smoothing": 0.35}},
    )
    np.testing.assert_allclose(result[:, 5], features[:, 5])
    np.testing.assert_allclose(np.diff(result[:, 4]), 0.5 * np.diff(features[:, 4]))
    np.testing.assert_allclose(np.diff(result[:, 6]), 0.5 * np.diff(features[:, 6]))
    assert np.abs(np.diff(result[:, 7])).mean() < np.abs(np.diff(features[:, 7])).mean()


def test_canonical_139_accepts_edge_151_and_drops_two_extra_joints():
    source = np.zeros((4, 151), dtype=np.float32)
    source[:, 4:7] = np.asarray([2.0, 3.0, 4.0])
    source[:, 7:] = np.arange(144, dtype=np.float32)
    result = canonical_139(source)
    assert result.shape == (4, 139)
    np.testing.assert_allclose(result[0, 4:7], 0.0)
    np.testing.assert_allclose(result[:, 7:], source[:, 7:139])


def test_shared_evaluator_reads_real_files(tmp_path: Path):
    data = tmp_path / "data"
    predictions = tmp_path / "predictions"
    data.mkdir()
    predictions.mkdir()
    features = np.zeros((8, 139), dtype=np.float32)
    features[:, :4] = 0.5
    features[:, 7::6] = 1.0
    features[:, 11::6] = 1.0
    np.save(data / "001.npy", features)
    music = np.zeros((8, 35), dtype=np.float32)
    music[[1, 5], -1] = 1.0
    np.save(data / "001_music.npy", music)
    write_manifest(
        data / "manifest.json",
        [DatasetRecord("001", "test", "001.npy", "001_music.npy", motion_format="finedance319")],
    )
    save_motion_npz(predictions / "001.npz", MotionSequence.from_features(features))
    (predictions / "generation_index.json").write_text(
        json.dumps({"items": [{"id": "001", "seam_frames": [4]}]}), encoding="utf-8"
    )
    report = evaluate_predictions(predictions, data / "manifest.json")
    assert report["status"] == "complete"
    assert report["sample_count"] == 1
    assert report["aggregate"]["l1"] == 0.0


def test_shared_evaluator_accepts_edge_npy151(tmp_path: Path):
    data = tmp_path / "data"
    predictions = tmp_path / "edge"
    data.mkdir()
    predictions.mkdir()
    reference = np.zeros((8, 139), dtype=np.float32)
    reference[:, 7::6] = 1.0
    reference[:, 11::6] = 1.0
    np.save(data / "063.npy", reference)
    music = np.zeros((8, 35), dtype=np.float32)
    music[[1, 5], -1] = 1.0
    np.save(data / "063_music.npy", music)
    write_manifest(
        data / "manifest.json",
        [DatasetRecord("063", "test", "063.npy", "063_music.npy", motion_format="finedance319")],
    )
    edge = np.zeros((8, 151), dtype=np.float32)
    edge[:, :139] = reference
    edge[:, 139:] = 123.0
    np.save(predictions / "063.npy", edge)
    report = evaluate_predictions(
        predictions, data / "manifest.json", prediction_format="npy151"
    )
    assert report["status"] == "complete"
    assert report["sample_count"] == 1
    assert report["aggregate"]["l1"] == 0.0


def test_lodge_forward_kinematics_identity_matches_rest_pose():
    features = np.zeros((3, 139), dtype=np.float32)
    identity_6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    features[:, 7:] = np.tile(identity_6d, 22)
    features[:, 4:7] = np.asarray([[0, 0, 0], [1, 2, 3], [-1, 0.5, 2]])
    rest = np.arange(55 * 3, dtype=np.float32).reshape(55, 3) / 100
    joints = forward_kinematics_139(features, rest)
    np.testing.assert_allclose(joints, rest[None, :24] + features[:, None, 4:7], atol=1e-5)


def test_foot_skating_pads_canonical_22_joints_for_smpl():
    torch = pytest.importorskip("torch")
    from tempograph.benchmarking import _foot_skating

    class FakeSMPL:
        parents = torch.arange(24)
        num_betas = 10

        def __call__(self, global_orient, body_pose, transl, betas, pose2rot):
            assert global_orient.shape == (6, 1, 3, 3)
            assert body_pose.shape == (6, 23, 3, 3)
            assert pose2rot is False
            joints = torch.zeros((6, 24, 3), dtype=torch.float32)
            return type("SMPLResult", (), {"joints": joints})()

    features = np.zeros((6, 139), dtype=np.float32)
    features[:, 7::6] = 1.0
    features[:, 11::6] = 1.0
    assert _foot_skating(features, FakeSMPL(), 30.0) == 0.0
