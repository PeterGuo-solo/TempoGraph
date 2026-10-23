"""Shared, transparent evaluation for TempoGraph and converted baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data.adapters import load_motion_npz
from .data.dataset import read_manifest
from .data.finedance import canonical_139
from .geometry.rotations import rotation_6d_to_matrix


def _velocity(value: np.ndarray) -> np.ndarray:
    return np.diff(value, axis=0) if len(value) > 1 else np.zeros_like(value)


def _energy(features: np.ndarray) -> np.ndarray:
    velocity = _velocity(features[:, 7:])
    result = np.zeros(len(features), dtype=np.float32)
    if len(velocity):
        result[1:] = np.linalg.norm(velocity, axis=-1)
    return result


def _beat_score(features: np.ndarray, music: np.ndarray, fps: float) -> float | None:
    beat_frames = np.flatnonzero(music[:, -1] > 0.5)
    if not len(beat_frames):
        return None
    energy = _energy(features)
    events = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:])) + 1
    if not len(events):
        return 0.0
    distances = np.min(np.abs(beat_frames[:, None] - events[None, :]), axis=1) / fps
    return float(np.mean(np.exp(-(distances**2) / (2 * 0.1**2))))


def _embedding(features: np.ndarray) -> np.ndarray:
    root_velocity = _velocity(features[:, 4:7])
    pose_velocity = _velocity(features[:, 7:])
    values = np.concatenate(
        [
            features[:, :7].mean(0),
            features[:, :7].std(0),
            root_velocity.mean(0),
            root_velocity.std(0),
            np.asarray([pose_velocity.mean(), pose_velocity.std(), np.abs(pose_velocity).mean()]),
        ]
    )
    return values.astype(np.float64)


def _frechet_proxy(real: np.ndarray, fake: np.ndarray) -> float | None:
    if len(real) < 2 or len(fake) < 2:
        return None
    mean_real, mean_fake = real.mean(0), fake.mean(0)
    cov_real = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * 1e-6
    cov_fake = np.cov(fake, rowvar=False) + np.eye(fake.shape[1]) * 1e-6
    product = cov_real @ cov_fake
    eigenvalues = np.linalg.eigvals(product).real
    trace_sqrt = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    return float(np.sum((mean_real - mean_fake) ** 2) + np.trace(cov_real) + np.trace(cov_fake) - 2 * trace_sqrt)


def _diversity(embeddings: np.ndarray) -> float | None:
    if len(embeddings) < 2:
        return None
    distances = np.linalg.norm(embeddings[:, None] - embeddings[None], axis=-1)
    upper = distances[np.triu_indices(len(embeddings), 1)]
    return float(upper.mean())


def _seam_metrics(features: np.ndarray, seams: list[int]) -> dict[str, float | None]:
    valid = [index for index in seams if 2 <= index < len(features) - 1]
    if not valid:
        return {"position_jump": None, "velocity_jump": None, "acceleration_jump": None}
    velocity = _velocity(features)
    acceleration = _velocity(velocity)
    return {
        "position_jump": float(np.mean([np.abs(features[i] - features[i - 1]).mean() for i in valid])),
        "velocity_jump": float(np.mean([np.abs(velocity[i] - velocity[i - 1]).mean() for i in valid])),
        "acceleration_jump": float(np.mean([np.abs(acceleration[i - 1] - acceleration[i - 2]).mean() for i in valid])),
    }


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def _contact_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    predicted = prediction[:, :4] >= 0.5
    target = reference[:, :4] >= 0.5
    tp = int(np.logical_and(predicted, target).sum())
    tn = int(np.logical_and(~predicted, ~target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    negative_precision = tn / max(tn + fn, 1)
    positive_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    negative_recall = specificity
    negative_f1 = (
        2 * negative_precision * negative_recall
        / max(negative_precision + negative_recall, 1e-12)
    )
    return {
        "contact_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "contact_balanced_accuracy": 0.5 * (recall + specificity),
        "contact_macro_f1": 0.5 * (positive_f1 + negative_f1),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_specificity": specificity,
        "contact_positive_rate": float(predicted.mean()),
        "reference_contact_positive_rate": float(target.mean()),
    }


def _load_body_model(project_root: Path):
    try:
        import smplx
    except (ImportError, OSError):
        return None
    model = project_root / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
    if not model.is_file():
        return None
    return smplx.create(
        str(project_root / "models"), model_type="smpl", gender="neutral", ext="pkl",
        use_pca=False, batch_size=1,
    ).eval()


def _foot_skating(features: np.ndarray, body_model: Any, fps: float) -> float | None:
    if body_model is None:
        return None
    import torch

    rotations = rotation_6d_to_matrix(features[:, 7:].reshape(len(features), 22, 6))
    # The canonical TempoGraph/FineDance body representation keeps 22 joints,
    # while the SMPL layer expects the complete 24-joint kinematic tree
    # (root + 23 body joints).  The two omitted terminal hand joints do not
    # affect the feet, so pad them with identity rotations before SMPL LBS.
    smpl_joint_count = int(body_model.parents.numel())
    if rotations.shape[1] < smpl_joint_count:
        identity = np.eye(3, dtype=np.float32)
        padding = np.broadcast_to(
            identity,
            (len(rotations), smpl_joint_count - rotations.shape[1], 3, 3),
        )
        rotations = np.concatenate([rotations, padding], axis=1)
    elif rotations.shape[1] > smpl_joint_count:
        rotations = rotations[:, :smpl_joint_count]
    joint_chunks = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            matrices = torch.as_tensor(rotations[start : start + 256], dtype=torch.float32)
            translation = torch.as_tensor(features[start : start + 256, 4:7], dtype=torch.float32)
            output = body_model(
                global_orient=matrices[:, :1], body_pose=matrices[:, 1:],
                transl=translation,
                betas=torch.zeros((len(matrices), body_model.num_betas), dtype=torch.float32),
                pose2rot=False,
            )
            joint_chunks.append(output.joints[:, :22].cpu().numpy())
    joints = np.concatenate(joint_chunks, axis=0)
    feet = joints[:, (7, 8, 10, 11)]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
    active = features[1:, :4] >= 0.5
    return float(speed[active].mean()) if np.any(active) else 0.0


def evaluate_predictions(
    predictions: str | Path,
    manifest: str | Path,
    split: str = "test",
    prediction_format: str = "canonical_npz",
) -> dict[str, Any]:
    predictions = Path(predictions).resolve()
    manifest = Path(manifest).resolve()
    manifest_parent = manifest.parent
    records = {record.name: record for record in read_manifest(manifest) if record.split == split}
    project_root = manifest.parents[2]
    body_model = _load_body_model(project_root)
    index_path = predictions / "generation_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"items": []}
    seams = {item["id"]: item.get("seam_frames", []) for item in index.get("items", [])}
    supported_formats = {"canonical_npz", "npy139", "npy151", "npy319"}
    if prediction_format not in supported_formats:
        raise ValueError(
            f"unsupported prediction format {prediction_format!r}; "
            f"expected one of {sorted(supported_formats)}"
        )
    suffix = ".npz" if prediction_format == "canonical_npz" else ".npy"
    files = {path.stem: path for path in predictions.glob(f"*{suffix}")}
    sequence_metrics: list[dict[str, Any]] = []
    real_embeddings: list[np.ndarray] = []
    fake_embeddings: list[np.ndarray] = []
    for clip_id in sorted(records.keys() & files.keys()):
        record = records[clip_id]
        reference = canonical_139(np.load(manifest_parent / record.motion_path, allow_pickle=False))
        music = np.asarray(np.load(manifest_parent / record.music_path, allow_pickle=False), dtype=np.float32)
        if prediction_format == "canonical_npz":
            prediction = load_motion_npz(files[clip_id]).to_features()
        else:
            raw = np.asarray(np.load(files[clip_id], allow_pickle=False))
            prediction = canonical_139(raw)
        frames = min(len(reference), len(prediction), len(music))
        reference, prediction, music = reference[:frames], prediction[:frames], music[:frames]
        if frames < 3:
            continue
        pred_velocity, ref_velocity = _velocity(prediction), _velocity(reference)
        pred_acceleration, ref_acceleration = _velocity(pred_velocity), _velocity(ref_velocity)
        pred_energy, ref_energy = _energy(prediction), _energy(reference)
        seam = _seam_metrics(prediction, seams.get(clip_id, []))
        predicted_root_velocity = float(np.abs(_velocity(prediction[:, 4:7])).mean())
        reference_root_velocity = float(np.abs(_velocity(reference[:, 4:7])).mean())
        predicted_energy_mean = float(pred_energy.mean())
        reference_energy_mean = float(ref_energy.mean())
        item = {
            "id": clip_id,
            "frames": frames,
            "l1": float(np.abs(prediction - reference).mean()),
            "mse": float(np.square(prediction - reference).mean()),
            "velocity_l1": float(np.abs(pred_velocity - ref_velocity).mean()),
            "acceleration_l1": float(np.abs(pred_acceleration - ref_acceleration).mean()),
            "root_l1": float(np.abs(prediction[:, 4:7] - reference[:, 4:7]).mean()),
            "energy_envelope_l1": float(np.abs(np.log1p(pred_energy) - np.log1p(ref_energy)).mean()),
            "energy_ratio": _safe_ratio(predicted_energy_mean, reference_energy_mean),
            "root_velocity_ratio": _safe_ratio(predicted_root_velocity, reference_root_velocity),
            "beat_alignment": _beat_score(prediction, music, record.fps),
            "foot_skating": _foot_skating(prediction, body_model, record.fps),
            **_contact_metrics(prediction, reference),
            **{f"boundary_{key}": value for key, value in seam.items()},
        }
        sequence_metrics.append(item)
        real_embeddings.append(_embedding(reference))
        fake_embeddings.append(_embedding(prediction))
    metric_names = [
        "l1", "mse", "velocity_l1", "acceleration_l1", "root_l1", "contact_accuracy",
        "contact_balanced_accuracy", "contact_macro_f1", "contact_precision",
        "contact_recall", "contact_specificity", "contact_positive_rate",
        "reference_contact_positive_rate", "energy_envelope_l1", "energy_ratio",
        "root_velocity_ratio", "beat_alignment", "boundary_position_jump",
        "boundary_velocity_jump", "boundary_acceleration_jump", "foot_skating",
    ]
    aggregate = {name: _mean_optional([item.get(name) for item in sequence_metrics]) for name in metric_names}
    real_array = np.asarray(real_embeddings)
    fake_array = np.asarray(fake_embeddings)
    aggregate["feature_frechet_proxy"] = _frechet_proxy(real_array, fake_array) if len(real_array) else None
    aggregate["diversity"] = _diversity(fake_array) if len(fake_array) else None
    aggregate["reference_diversity"] = _diversity(real_array) if len(real_array) else None
    aggregate["diversity_ratio"] = _safe_ratio(
        aggregate["diversity"] or 0.0, aggregate["reference_diversity"] or 0.0
    )
    return {
        "schema_version": 1,
        "status": "complete" if sequence_metrics else "not_run",
        "predictions": str(predictions),
        "manifest": str(manifest),
        "split": split,
        "prediction_format": prediction_format,
        "training_seed": index.get("training_seed"),
        "sampling_seed": index.get("seed"),
        "sampling_seed_policy": index.get("sampling_seed_policy"),
        "sample_count": len(sequence_metrics),
        "fps": 30.0,
        "skeleton": "SMPL body 22 joints, rotation-6D, canonical 139D",
        "normalization": "predictions and references evaluated in stored canonical feature space",
        "metric_notice": "feature_frechet_proxy is not an official FineDance FID; foot skating is null only when the optional smplx runtime or neutral SMPL model is unavailable",
        "aggregate": aggregate,
        "sequences": sequence_metrics,
    }
