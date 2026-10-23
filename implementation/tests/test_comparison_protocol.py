import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_five_paper_registry_is_complete_and_conservative():
    registry = json.loads(
        (PROJECT / "configs" / "five_paper_comparability.json").read_text(encoding="utf-8")
    )
    papers = {paper["id"]: paper for paper in registry["papers"]}
    assert set(papers) == {"sgmd", "edge", "infinitedance", "megadance", "motioncraft"}
    assert all(paper["shared_prediction_evaluation"] for paper in papers.values())
    primary = {
        paper_id
        for paper_id, paper in papers.items()
        if paper["primary_for_current_finedance_claim"]
    }
    assert primary == {"megadance"}
    assert "exact feature extractors" in papers["megadance"]["protocol_warning"]


def test_every_paper_reference_has_a_protocol_and_real_metrics():
    registry = json.loads(
        (PROJECT / "configs" / "five_paper_comparability.json").read_text(encoding="utf-8")
    )
    for paper in registry["papers"]:
        assert paper["paper_protocol"]
        assert paper["published_variant"]
        assert paper["published_metrics"]
        assert all(isinstance(value, (int, float)) for value in paper["published_metrics"].values())
        assert "paper_seed" in paper
        assert paper["paper_seed"] is None
        assert "No explicit integer" in paper["seed_evidence"]
