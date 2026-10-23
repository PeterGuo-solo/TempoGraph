from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]

METRICS = (
    "feature_frechet_proxy",
    "diversity",
    "diversity_ratio",
    "beat_alignment",
    "foot_skating",
    "contact_balanced_accuracy",
    "contact_macro_f1",
    "contact_positive_rate",
    "reference_contact_positive_rate",
    "energy_ratio",
    "root_velocity_ratio",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_seed_root(root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for directory in sorted(root.glob("seed_*")):
        metrics_path = directory / "reports" / "tempograph_metrics.json"
        run_path = directory / "reports" / "run_manifest.json"
        generation_path = directory / "generated" / "tempograph" / "generation_index.json"
        if not (metrics_path.is_file() and run_path.is_file() and generation_path.is_file()):
            continue
        metrics = _read(metrics_path)
        runtime = _read(run_path)
        generation = _read(generation_path)
        row = {
            "training_seed": int(runtime["seed"]),
            "sampling_seed": int(generation["seed"]),
            "sample_count": int(metrics.get("sample_count", 0)),
            **{name: metrics.get("aggregate", {}).get(name) for name in METRICS},
        }
        official = directory / "reports" / "lodge_protocol_metrics.json"
        if official.is_file():
            for name, value in _read(official).get("aggregate", {}).items():
                row[f"lodge_{name}"] = value
        runs.append(row)
    if not runs:
        raise ValueError(f"no complete seed runs found under {root}")

    numeric_names = sorted(
        {
            key
            for row in runs
            for key, value in row.items()
            if key not in {"training_seed", "sampling_seed", "sample_count"}
            and isinstance(value, (int, float))
        }
    )
    summary = {}
    for name in numeric_names:
        values = [float(row[name]) for row in runs if isinstance(row.get(name), (int, float))]
        summary[name] = {
            "mean": mean(values),
            "sample_std": stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
    return {
        "schema_version": 1,
        "method": "TempoGraph",
        "training_seed_policy": "independent full training runs on one fixed data split",
        "seeds": [row["training_seed"] for row in runs],
        "runs": runs,
        "aggregate": summary,
    }


def _number(value: Any) -> str:
    return "NR" if value is None else f"{float(value):.4f}"


def _mean_std(summary: dict[str, Any], name: str) -> str:
    item = summary.get(name)
    if not item:
        return "TBD"
    if item["sample_std"] is None:
        return f"{item['mean']:.4f}"
    return f"{item['mean']:.4f} ± {item['sample_std']:.4f}"


def write_markdown(result: dict[str, Any], output: Path) -> None:
    paper = _read(PROJECT / "configs" / "paper_baselines.json")
    supplied = _read(PROJECT / "configs" / "five_paper_comparability.json")
    official_names = (
        "lodge_fid_k", "lodge_fid_g", "lodge_foot_skating_ratio",
        "lodge_div_k", "lodge_div_g", "lodge_bas",
    )
    lines = [
        "# TempoGraph multi-seed result and paper references",
        "",
        "TempoGraph values below are measured runs. Baseline values are cited from papers and were not rerun.",
        "",
        "## TempoGraph per-seed results",
        "",
        "| Training seed | Sampling seed | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Contact BA ↑ | Contact Macro-F1 ↑ | Energy/GT →1 | Diversity/GT →1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        lines.append(
            "| {training_seed} | {sampling_seed} | {fid_k} | {fid_g} | {fsr} | {div_k} | {div_g} | {bas} | {contact_ba} | {contact_f1} | {energy} | {diversity_ratio} |".format(
                training_seed=row["training_seed"], sampling_seed=row["sampling_seed"],
                fid_k=_number(row.get("lodge_fid_k")), fid_g=_number(row.get("lodge_fid_g")),
                fsr=_number(row.get("lodge_foot_skating_ratio")), div_k=_number(row.get("lodge_div_k")),
                div_g=_number(row.get("lodge_div_g")), bas=_number(row.get("lodge_bas")),
                contact_ba=_number(row.get("contact_balanced_accuracy")),
                contact_f1=_number(row.get("contact_macro_f1")),
                energy=_number(row.get("energy_ratio")),
                diversity_ratio=_number(row.get("diversity_ratio")),
            )
        )
    summary = result["aggregate"]
    lines.extend([
        "",
        "## TempoGraph mean ± sample standard deviation",
        "",
        "| Method | Seeds | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        "| TempoGraph | {seeds} | {values} |".format(
            seeds=", ".join(map(str, result["seeds"])),
            values=" | ".join(_mean_std(summary, name) for name in official_names),
        ),
        "",
        "## Published LODGE/FineDance table values (citation only)",
        "",
        f"Paper seed: NR. {paper['source']['seed_evidence']}",
        "",
        "| Method | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paper["rows"]:
        lines.append(
            f"| {row['method']} | {_number(row['fid_k'])} | {_number(row['fid_g'])} | "
            f"{_number(row['foot_skating_ratio'])} | {_number(row['div_k'])} | "
            f"{_number(row['div_g'])} | {_number(row['bas'])} |"
        )
    lines.extend([
        "",
        "## Seed reporting in the five supplied papers",
        "",
        "| Model | Paper seed | Evidence | Comparability |",
        "|---|---:|---|---|",
    ])
    for item in supplied["papers"]:
        lines.append(
            f"| {item['model']} | {_number(item.get('paper_seed'))} | "
            f"{item['seed_evidence']} | {item['comparison_tier']} |"
        )
    lines.extend([
        "",
        "`NR` means the paper did not report an explicit integer seed. It is not replaced by a guessed value. SGMD's 100 trials and EDGE's 1000 Elo trials are evaluation repetitions, not reported training seeds.",
        "",
        "The local diagnostic proxy metrics are not substituted for paper FID. Only the `lodge_*` fields use the released FineDance feature definitions.",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate TempoGraph seed runs")
    parser.add_argument("--root", type=Path, default=Path("artifacts/multiseed"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate_seed_root(args.root)
    output = args.output or args.root / "reports" / "tempograph_seed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = sorted({key for row in result["runs"] for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["runs"])
    markdown_path = output.with_suffix(".md")
    write_markdown(result, markdown_path)
    print(output)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
