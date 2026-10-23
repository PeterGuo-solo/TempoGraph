from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SMPL_SOURCES = {
    "SMPL_FEMALE.pkl": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
    "SMPL_MALE.pkl": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
    "SMPL_NEUTRAL.pkl": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
}

SMPLH_SOURCES = {
    "SMPLH_FEMALE.pkl": "female/model.npz",
    "SMPLH_MALE.pkl": "male/model.npz",
    "SMPLH_NEUTRAL.pkl": "neutral/model.npz",
}


def load_legacy_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream, encoding="latin1")
    if not hasattr(value, "items"):
        raise TypeError(f"Expected a mapping in {path}")
    return dict(value.items())


def remove_chumpy(value: Any) -> Any:
    if "chumpy" in type(value).__module__:
        return np.asarray(value)
    return value


def clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: remove_chumpy(value) for key, value in values.items()}


def save_model(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(clean_mapping(values), stream, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_smpl(raw_root: Path, output_root: Path) -> list[Path]:
    created = []
    source_root = raw_root / "smpl" / "models"
    for output_name, source_name in SMPL_SOURCES.items():
        source = source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_root / "smpl" / output_name
        save_model(target, load_legacy_pickle(source))
        created.append(target)
    return created


def prepare_smplh(raw_root: Path, output_root: Path) -> list[Path]:
    mano_root = raw_root / "mano_v1_2" / "models"
    left = clean_mapping(load_legacy_pickle(mano_root / "MANO_LEFT.pkl"))
    right = clean_mapping(load_legacy_pickle(mano_root / "MANO_RIGHT.pkl"))

    hand_fields = {
        "hands_componentsl": left["hands_components"],
        "hands_componentsr": right["hands_components"],
        "hands_coeffsl": left["hands_coeffs"],
        "hands_coeffsr": right["hands_coeffs"],
        "hands_meanl": left["hands_mean"],
        "hands_meanr": right["hands_mean"],
    }

    created = []
    source_root = raw_root / "smplh"
    for output_name, relative_source in SMPLH_SOURCES.items():
        source = source_root / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            model = {key: archive[key] for key in archive.files}
        model.update(hand_fields)
        target = output_root / "smplh" / output_name
        save_model(target, model)
        created.append(target)
    return created


def validate_model(path: Path, required: set[str]) -> None:
    values = load_legacy_pickle(path)
    missing = required.difference(values)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    remaining_chumpy = [
        key for key, value in values.items() if "chumpy" in type(value).__module__
    ]
    if remaining_chumpy:
        raise ValueError(f"{path} still contains Chumpy fields: {remaining_chumpy}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare licensed SMPL and SMPL+H models")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    args = parser.parse_args()

    models_root = args.models_root.resolve()
    raw_root = models_root / "raw"
    created = prepare_smpl(raw_root, models_root)
    created.extend(prepare_smplh(raw_root, models_root))

    smpl_required = {"v_template", "shapedirs", "posedirs", "weights", "kintree_table"}
    smplh_required = smpl_required | {
        "hands_componentsl",
        "hands_componentsr",
        "hands_meanl",
        "hands_meanr",
    }
    for path in created:
        validate_model(path, smplh_required if path.parent.name == "smplh" else smpl_required)
        print(f"READY {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
