"""Training-only edge-supervision interventions; original boundary targets stay intact."""
from __future__ import annotations

import hashlib
import torch

MODES = ("full", "no_relation_supervision", "random_relation_labels")


def relation_targets(graph_batch, batch, config, training):
    mode = config["training"].get("relation_supervision", "full")
    if mode not in MODES:
        raise ValueError(f"unknown relation_supervision: {mode}")
    labels = graph_batch["dense_edge_types"]
    confidence = graph_batch["dense_edge_confidence"]
    if mode != "random_relation_labels" or not training:
        return labels, confidence, confidence.new_zeros(())
    labels_out, confidence_out = labels.clone(), confidence.clone()
    nodes = labels.shape[1]
    row, col = torch.triu_indices(nodes, nodes, offset=1, device=labels.device)
    changed = confidence.new_zeros(())
    total = 0
    for sample in range(labels.shape[0]):
        key = (f"{config['training'].get('relation_shuffle_seed', 77123)}|"
               f"{batch['name'][sample]}|{int(batch['start'][sample])}|{nodes}")
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        valid = confidence[sample, row, col] > 0
        r, c = row[valid], col[valid]
        permutation = torch.randperm(len(r), generator=generator).to(labels.device)
        y = labels[sample, r, c][permutation]
        w = confidence[sample, r, c][permutation]
        changed = changed + (y != labels[sample, r, c]).sum()
        total += len(r)
        labels_out[sample, r, c] = y
        labels_out[sample, c, r] = y
        confidence_out[sample, r, c] = w
        confidence_out[sample, c, r] = w
    return labels_out, confidence_out, changed / max(total, 1)


def configure_relation_mode(config, mode, shuffle_seed):
    if mode not in MODES:
        raise ValueError(mode)
    training = config["training"]
    training["relation_supervision"] = mode
    training["relation_shuffle_seed"] = shuffle_seed
    training["checkpoint_policy"] = "final"
    training.pop("max_stage_seconds", None)
    weights = training["loss_weights"]
    # Use the same explicit full weights in every phase; disable BOTH in ablation 3.
    weights["graph_edge"] = 0.0 if mode == "no_relation_supervision" else 1.0
    weights["graph_edge_coverage"] = 0.0 if mode == "no_relation_supervision" else 0.5
    return config
