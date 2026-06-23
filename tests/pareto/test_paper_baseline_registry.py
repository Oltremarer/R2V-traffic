from __future__ import annotations

import pytest

from pareto.rl.paper_baseline_registry import (
    BASELINE_REGISTRY,
    REQUIRED_PAPER_BASELINES,
    registry_with_learned_artifact_inventory,
    baseline_blockers,
    validate_baseline_registry,
)


def test_registry_contains_every_paper_baseline_and_no_stage_a_diagnostic_method():
    registry = validate_baseline_registry(BASELINE_REGISTRY)

    assert set(registry) == set(REQUIRED_PAPER_BASELINES)
    assert "EnvReward-QueuePenalty-PPO" not in registry
    assert "env_reward" not in registry


def test_registry_reports_missing_implementation_blockers():
    blockers = baseline_blockers(BASELINE_REGISTRY)

    assert not any("PressLight" in item for item in blockers)
    assert not any("MPLight" in item for item in blockers)
    assert not any("CoLight" in item for item in blockers)
    assert any("C2T-scalar" in item for item in blockers)
    assert any("Weighted-RL" in item for item in blockers)


def test_registry_rejects_unknown_status():
    bad = dict(BASELINE_REGISTRY)
    bad["Random"] = dict(bad["Random"], status="maybe")

    with pytest.raises(ValueError, match="unknown baseline status"):
        validate_baseline_registry(bad)


def test_registry_rejects_missing_city_support_for_implemented_baseline():
    bad = dict(BASELINE_REGISTRY)
    bad["VectorQ-PPO"] = dict(
        bad["VectorQ-PPO"],
        status="implemented",
        city_support=["jinan"],
    )

    with pytest.raises(ValueError, match="missing city support"):
        validate_baseline_registry(bad, require_all_cities_for_implemented=True)


def _complete_learned_artifact_inventory() -> dict:
    rows = []
    for baseline, family in (("Cond-Scalar-RL", "cond_scalar"), ("VectorQ-PPO", "pareto_quality")):
        for city in ("jinan", "hangzhou", "newyork_28x7"):
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "status": "implemented_guarded_preview",
                    "model_path": f"model_weights/{family}/{city}/paper_final/run/model.pt",
                    "model_hash": "a" * 64,
                    "objective_normalizer_path": f"model_weights/{family}/{city}/paper_final/run/objective_normalizer.json",
                    "objective_normalizer_hash": "b" * 64,
                    "executes_training_now": False,
                }
            )
    return {
        "packet_type": "paper_learned_artifact_inventory",
        "coverage_status": "complete",
        "rows": rows,
        "executes_training_now": False,
    }


def test_registry_closes_only_learned_artifact_blockers_from_complete_inventory():
    registry = registry_with_learned_artifact_inventory(
        BASELINE_REGISTRY,
        _complete_learned_artifact_inventory(),
    )
    blockers = baseline_blockers(registry)

    assert not any("Cond-Scalar-RL" in item for item in blockers)
    assert not any("VectorQ-PPO" in item for item in blockers)
    assert any("C2T-scalar" in item for item in blockers)
    assert any("Weighted-RL" in item for item in blockers)
    assert registry["Cond-Scalar-RL"]["status"] == "implemented"
    assert registry["VectorQ-PPO"]["status"] == "implemented"
