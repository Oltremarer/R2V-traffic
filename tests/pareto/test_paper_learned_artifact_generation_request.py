from __future__ import annotations

import pytest

from pareto.rl.paper_learned_artifact_generation_request import (
    build_learned_artifact_generation_request,
    validate_learned_artifact_generation_request,
)


def test_learned_artifact_generation_request_covers_all_city_model_families_without_claiming_hashes():
    request = build_learned_artifact_generation_request(run_id="paper_final_request_v1")

    assert request["status"] == "request_only"
    assert request["executes_training_now"] is False
    assert len(request["rows"]) == 6
    assert {row["baseline"] for row in request["rows"]} == {"Cond-Scalar-RL", "VectorQ-PPO"}
    assert {row["city"] for row in request["rows"]} == {"jinan", "hangzhou", "newyork_28x7"}
    assert all("/paper_final/" in row["model_path"] for row in request["rows"])
    assert all(row["model_hash"] is None for row in request["rows"])
    assert all(row["objective_normalizer_hash"] is None for row in request["rows"])
    validate_learned_artifact_generation_request(request)


def test_learned_artifact_generation_request_rejects_hash_claims_before_artifacts_exist():
    request = build_learned_artifact_generation_request(run_id="paper_final_request_v1")
    request["rows"][0] = dict(request["rows"][0], model_hash="a" * 64)

    with pytest.raises(ValueError, match="must not claim model_hash"):
        validate_learned_artifact_generation_request(request)


def test_learned_artifact_generation_request_rejects_non_paper_final_root():
    request = build_learned_artifact_generation_request(run_id="paper_final_request_v1")
    request["rows"][0] = dict(
        request["rows"][0],
        model_path="model_weights/cond_scalar/jinan/preformal_final/run/model.pt",
    )

    with pytest.raises(ValueError, match="paper_final"):
        validate_learned_artifact_generation_request(request)
