from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "artifacts" / "reports"

SHARED_METRICS = [
    ("feature_frechet_proxy", "lower"),
    ("diversity", "higher"),
    ("diversity_ratio", "target_1"),
    ("beat_alignment", "higher"),
    ("foot_skating", "lower"),
    ("contact_balanced_accuracy", "higher"),
    ("contact_macro_f1", "higher"),
    ("energy_ratio", "target_1"),
    ("root_velocity_ratio", "target_1"),
    ("boundary_position_jump", "lower"),
    ("boundary_velocity_jump", "lower"),
    ("boundary_acceleration_jump", "lower"),
]

DISPLAY_NAMES = {
    "lodge": "LODGE",
    "edge": "EDGE",
    "sgmd": "SGMD",
    "infinitedance": "ChoreoLLaMA",
    "megadance": "MEGADance",
    "motioncraft": "MotionCraft",
}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_run", "sample_count": 0, "aggregate": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _shared_reports() -> list[tuple[str, dict[str, Any]]]:
    reports = [("TempoGraph", _read(REPORTS / "tempograph_metrics.json"))]
    reports.extend(
        (display, _read(REPORTS / f"{slug}_metrics.json"))
        for slug, display in DISPLAY_NAMES.items()
    )
    return reports


def write_shared_comparison(reports: list[tuple[str, dict[str, Any]]]) -> None:
    fields = ["method", "status", "samples", *[name for name, _ in SHARED_METRICS]]
    rows = []
    for method, report in reports:
        aggregate = report.get("aggregate", {})
        rows.append(
            {
                "method": method,
                "status": report.get("status", "not_run"),
                "samples": report.get("sample_count", 0),
                **{name: aggregate.get(name) for name, _ in SHARED_METRICS},
            }
        )
    with (REPORTS / "final_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_reference(reference: dict[str, Any]) -> None:
    fields = [
        "method", "fid_k", "fid_g", "foot_skating_ratio", "div_k", "div_g",
        "bas", "runtime_seconds", "source",
    ]
    with (REPORTS / "paper_reference.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in reference["rows"]:
            writer.writerow({**row, "source": reference["source"]["url"]})


def write_five_paper_tables(comparability: dict[str, Any]) -> None:
    screening_fields = [
        "model", "paper_dataset", "comparison_tier", "shared_prediction_evaluation",
        "primary_for_current_finedance_claim", "paper_protocol", "code_status",
        "protocol_warning", "paper_seed", "seed_evidence",
    ]
    with (REPORTS / "five_paper_screening.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=screening_fields)
        writer.writeheader()
        for paper in comparability["papers"]:
            writer.writerow({field: paper.get(field) for field in screening_fields})

    metric_fields = [
        "model", "variant", "paper_dataset", "paper_protocol", "metric", "value"
    ]
    with (REPORTS / "five_paper_reference.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        for paper in comparability["papers"]:
            for metric, value in paper.get("published_metrics", {}).items():
                writer.writerow(
                    {
                        "model": paper["model"],
                        "variant": paper["published_variant"],
                        "paper_dataset": paper["paper_dataset"],
                        "paper_protocol": paper["paper_protocol"],
                        "metric": metric,
                        "value": value,
                    }
                )


def write_markdown(
    reports: list[tuple[str, dict[str, Any]]],
    reference: dict[str, Any],
    comparability: dict[str, Any],
) -> None:
    lines = [
        "# TempoGraph final comparison",
        "",
        "## Reproduced comparison on the shared local protocol",
        "",
        "A row is directly comparable here only when its status is `complete`: every completed prediction folder is evaluated by the same code, manifest, FPS, skeleton, and preprocessing.",
        "",
        "| Method | Status | Samples | Feature Frechet proxy ↓ | Diversity ↑ | Diversity/GT →1 | Beat alignment ↑ | Foot skating ↓ | Contact balanced acc. ↑ | Contact macro-F1 ↑ | Energy/GT →1 | Root velocity/GT →1 | Boundary position ↓ | Boundary velocity ↓ | Boundary acceleration ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, report in reports:
        aggregate = report.get("aggregate", {})
        lines.append(
            f"| {method} | {report.get('status', 'not_run')} | {report.get('sample_count', 0)} | "
            + " | ".join(_format(aggregate.get(name)) for name, _ in SHARED_METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "`feature_frechet_proxy` is the local transparent proxy, not any paper's official FID.",
            "",
            "## Published FineDance reference (not directly merged with local proxy metrics)",
            "",
            f"Source: [{reference['source']['title']}]({reference['source']['url']}), {reference['source']['table']}. Protocol: {reference['source']['protocol']}.",
            "",
            "| Method | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Runtime (s) ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reference["rows"]:
        lines.append(
            "| {method} | {fid_k} | {fid_g} | {fsr} | {div_k} | {div_g} | {bas} | {runtime} |".format(
                method=row["method"],
                fid_k=_format(row["fid_k"]),
                fid_g=_format(row["fid_g"]),
                fsr=(
                    "TBD"
                    if row["foot_skating_ratio"] is None
                    else f"{100 * row['foot_skating_ratio']:.2f}%"
                ),
                div_k=_format(row["div_k"]),
                div_g=_format(row["div_g"]),
                bas=_format(row["bas"]),
                runtime=_format(row["runtime_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## Screening of the five newly supplied papers",
            "",
            "| Model | Paper dataset/protocol | Paper seed | Role for TempoGraph | Current primary claim? |",
            "|---|---|---|---|---|",
        ]
    )
    for paper in comparability["papers"]:
        primary = (
            "yes, after exact evaluator"
            if paper["primary_for_current_finedance_claim"]
            else "no"
        )
        lines.append(
            f"| {paper['model']} | {paper['paper_dataset']}: {paper['paper_protocol']} | "
            f"{'NR' if paper.get('paper_seed') is None else paper['paper_seed']} | "
            f"{paper['comparison_tier']} | {primary} |"
        )
    lines.extend(
        [
            "",
            "MEGADance is the closest paper-level primary baseline because it uses the same FineDance 20-clip, 1024-frame protocol. EDGE and SGMD are AIST++ paper baselines; MotionCraft changes the representation/evaluator; ChoreoLLaMA's FineDance row is cross-dataset transfer. All five can still be scored in the shared local table after exporting ID-named canonical predictions.",
            "",
            "Do not claim that TempoGraph beats a published paper row unless TempoGraph has also been evaluated with that paper's exact feature extraction and test protocol. Missing runs remain `TBD`.",
            "",
        ]
    )
    (REPORTS / "final_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    reports = _shared_reports()
    reference = _read(PROJECT / "configs" / "paper_baselines.json")
    comparability = _read(PROJECT / "configs" / "five_paper_comparability.json")
    write_shared_comparison(reports)
    write_paper_reference(reference)
    write_five_paper_tables(comparability)
    write_markdown(reports, reference, comparability)
    print(REPORTS / "final_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
