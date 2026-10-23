from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from tempograph.data.torch_data import FineDanceWindowDataset
from tempograph.engine import build_models, graph_targets, load_checkpoint, resolve_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Agreement with ORIGINAL pseudo labels on every validation window")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    device = resolve_device(args.device)
    models = build_models(config, device)
    load_checkpoint(args.checkpoint, models, device)
    graph = models["graph_planner"].eval()
    training = config["training"]
    data = FineDanceWindowDataset(args.manifest, "val", training["window_frames"], training["window_stride"])
    counts = torch.zeros(4, 4, dtype=torch.int64)
    message_sum = 0.0
    nodes_seen = 0
    hook = graph.relation_condition.register_forward_hook(
        lambda module, inputs, output: norms.append(output.norm(dim=-1).sum().cpu().item()))
    try:
        for batch in DataLoader(data, batch_size=4, shuffle=False, num_workers=0):
            targets = graph_targets(batch["motion"].to(device), batch["music"].to(device),
                                    list(batch["style_fine"]), training["graph_nodes"],
                                    batch["beat_evidence"].to(device))
            norms = []
            prediction = graph(targets["music_nodes"])["dense_edge_logits"].argmax(-1)
            valid = targets["dense_edge_confidence"] > 0
            indices = targets["dense_edge_types"][valid] * 4 + prediction[valid]
            counts += torch.bincount(indices.cpu(), minlength=16).reshape(4, 4)
            message_sum += sum(norms)
            nodes_seen += targets["music_nodes"].shape[0] * targets["music_nodes"].shape[1]
    finally:
        hook.remove()
    tp = counts.diag().float()
    precision = tp / counts.sum(0).clamp_min(1)
    recall = tp / counts.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    report = {"split": "val", "windows": len(data), "labels": ["repeat", "variation", "contrast", "transition"],
              "reference": "original motion-derived pseudo labels, NOT human ground truth",
              "confusion_true_rows_predicted_columns": counts.tolist(),
              "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
              "macro_f1_all_four_zero_division_0": f1.mean().item(),
              "target_counts": counts.sum(1).tolist(), "prediction_counts": counts.sum(0).tolist(),
              "mean_relation_projection_l2_per_node": message_sum / max(1, nodes_seen)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
