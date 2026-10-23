from copy import deepcopy
from pathlib import Path
import json
from unittest.mock import patch
import torch
from tempograph.relation_supervision import relation_targets, configure_relation_mode
from tempograph.engine import build_models, stage_loss


def example():
    n = 8
    r, c = torch.triu_indices(n, n, 1)
    labels = torch.zeros(2, n, n, dtype=torch.long)
    weights = torch.zeros(2, n, n)
    for b in range(2):
        y = (torch.arange(len(r)) + b) % 4
        w = torch.linspace(0.1, 1, len(r))
        labels[b, r, c] = labels[b, c, r] = y
        weights[b, r, c] = weights[b, c, r] = w
    return {"dense_edge_types": labels, "dense_edge_confidence": weights,
            "edge_types": labels[:, torch.arange(n-1), torch.arange(1, n)].clone()}, {
            "name": ["song1", "song2"], "start": torch.tensor([0, 120])}, {
            "training": {"relation_supervision": "random_relation_labels", "relation_shuffle_seed": 77123}}


def test_randomization_preserves_supervision_mass_symmetry_boundary_and_rng():
    graph, batch, config = example()
    original = deepcopy(graph)
    state = torch.get_rng_state().clone()
    y, w, changed = relation_targets(graph, batch, config, True)
    assert torch.equal(state, torch.get_rng_state())
    assert 0 < changed < 1
    assert torch.equal(y, y.transpose(1, 2))
    assert torch.equal(w, w.transpose(1, 2))
    assert torch.count_nonzero(w.diagonal(dim1=1, dim2=2)) == 0
    for key in graph:
        assert torch.equal(graph[key], original[key])
    for sample in range(2):
        for cls in range(4):
            assert ((y[sample] == cls) & (w[sample] > 0)).sum() == ((original['dense_edge_types'][sample] == cls) & (original['dense_edge_confidence'][sample] > 0)).sum()
            assert torch.allclose(w[sample][y[sample] == cls].sum(), original['dense_edge_confidence'][sample][original['dense_edge_types'][sample] == cls].sum())
    assert torch.equal(y, relation_targets(graph, batch, config, True)[0])
    # Different batch order preserves per-window permutation.
    flipped = {k: v.flip(0) for k, v in graph.items()}
    batch2 = {"name": list(reversed(batch["name"])), "start": batch["start"].flip(0)}
    assert torch.equal(y.flip(0), relation_targets(flipped, batch2, config, True)[0])
    assert torch.equal(relation_targets(graph, batch, config, False)[0], original["dense_edge_types"])


def small_config():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/base.json").read_text())
    config["model"].update(hidden_dim=16, latent_dim=8, num_layers=1, num_heads=4, dropout=0)
    config["training"].update(graph_nodes=4, training_rollout_steps=1, boundary_training_rollout_steps=1)
    return config


def small_batch():
    torch.manual_seed(33)
    motion = torch.randn(2, 48, 139) * 0.1
    motion[:, :, :4] = torch.randint(0, 2, (2, 48, 4)).float()
    motion[:, :, 7:] = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(22) + torch.randn(2, 48, 132) * 0.1
    return {"motion": motion, "music": torch.randn(2, 48, 35), "style_fine": ["a", "b"],
            "beat_evidence": torch.ones(2, 48), "name": ["one", "two"], "start": torch.tensor([0, 120])}


def test_no_supervision_actual_motion_loss_reaches_edge_head():
    config = configure_relation_mode(small_config(), "no_relation_supervision", 77123)
    assert config["training"]["loss_weights"]["graph_edge"] == 0
    assert config["training"]["loss_weights"]["graph_edge_coverage"] == 0
    models = build_models(config, torch.device("cpu"))
    rest = torch.zeros(24, 3)
    rest[:, 1] = torch.linspace(0., -1., 24)
    models["local_generator"].configure_skeleton(rest)
    total, metrics = stage_loss("graph_and_local", models, small_batch(), config)
    assert torch.isfinite(total)
    total.backward()
    gradient = models["graph_planner"].edge_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # With the motion objective isolated away, relation-head gradients vanish.
    for model in models.values():
        model.zero_grad(set_to_none=True)
    with patch("tempograph.engine._local_flow_total", return_value=(torch.tensor(0.), {})):
        total, _ = stage_loss("graph_and_local", models, small_batch(), config)
        total.backward()
    gradient = models["graph_planner"].edge_head.weight.grad
    assert gradient is None or torch.count_nonzero(gradient) == 0


def test_random_labels_do_not_change_boundary_training_inputs_or_objective():
    config = small_config()
    models = build_models(config, torch.device("cpu"))
    for model in models.values():
        model.eval()
    batch = small_batch()
    full, _ = stage_loss("boundary_planner", models, batch, configure_relation_mode(deepcopy(config), "full", 77123))
    random, _ = stage_loss("boundary_planner", models, batch, configure_relation_mode(deepcopy(config), "random_relation_labels", 77123))
    assert torch.equal(full, random)


def test_runner_resume_reuses_completed_phase_and_rejects_changed_recipe(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    import pytest
    path = Path(__file__).resolve().parents[1] / "scripts/run_training_ablations.py"
    spec = importlib.util.spec_from_file_location("ablation_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = SimpleNamespace(training_seed=1234, shuffle_seed=77123, resume=False,
                           device="cpu", data_root=tmp_path)
    phase = {"name": "graph", "stage": "graph_and_local", "steps": 2, "config": "configs/base.json"}
    calls = []
    def fake_execute(command, log):
        # Runner-only fixture: no training result is inferred from this synthetic checkpoint.
        calls.append(command)
        destination = Path(command[command.index("--output-dir") + 1]) / "graph_local.pt"
        destination.parent.mkdir(parents=True)
        torch.save({"stage": "graph_and_local", "step": 2, "schema_version": 7}, destination)
    monkeypatch.setattr(runner, "execute", fake_execute)
    first = runner.run_phase(args, tmp_path, phase, "full", None)
    args.resume = True
    assert runner.run_phase(args, tmp_path, phase, "full", None) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="changed initial weights/configuration"):
        runner.run_phase(args, tmp_path, dict(phase, steps=3), "full", None)


def test_group_initialization_imports_only_autoencoder_weights(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "scripts/run_training_ablations.py"
    spec = importlib.util.spec_from_file_location("ablation_runner_init", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = small_config()
    models = build_models(config, torch.device("cpu"))
    with torch.no_grad():
        models['graph_planner'].edge_head.weight.fill_(999.)
    source = tmp_path / 'external_ae.pt'
    torch.save({'schema_version': 7, 'stage': 'motion_autoencoder', 'step': 10,
                'models': {k: v.state_dict() for k, v in models.items()}}, source)
    monkeypatch.setattr(runner, 'load_config', lambda path: deepcopy(config))
    args = SimpleNamespace(output=tmp_path / 'run', training_seed=1234)
    destination = runner.group_initialization(args, source, args.output / 'full')
    payload = torch.load(destination, map_location='cpu', weights_only=False)
    assert not torch.any(payload['models']['graph_planner']['edge_head.weight'] == 999.)
    for key, expected in models['motion_autoencoder'].state_dict().items():
        assert torch.equal(payload['models']['motion_autoencoder'][key], expected)


def test_recipe_enforces_25k_including_autoencoder_and_no_extra_training():
    import importlib.util
    import pytest
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('ablation_budget_runner', root / 'scripts/run_training_ablations.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    recipe = json.loads((root / 'configs/training_ablations.json').read_text())
    phases = runner.validate_recipe(recipe)
    assert sum(p['steps'] for p in phases) == 25000
    assert phases[0]['stage'] == 'motion_autoencoder' and phases[0]['steps'] == 3000
    for change in ('excess', 'shortfall', 'extra_shared'):
        invalid = deepcopy(recipe)
        if change == 'extra_shared':
            invalid['autoencoder'] = {'steps': 100000}
        else:
            invalid['phases'][-1]['steps'] += 1 if change == 'excess' else -1
        with pytest.raises(ValueError, match='exactly 25000'):
            runner.validate_recipe(invalid)
