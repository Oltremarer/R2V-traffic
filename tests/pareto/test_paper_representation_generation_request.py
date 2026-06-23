from __future__ import annotations

import pytest

from pareto.eval.paper_representation_generation_request import (
    build_representation_generation_request,
    validate_representation_generation_request,
)


def test_representation_generation_request_covers_all_city_model_family_sources():
    request = build_representation_generation_request(request_id="paper_final_representation_sources_v1")

    assert request["status"] == "request_only"
    assert request["executes_generation_now"] is False
    assert len(request["rows"]) == 6
    assert {row["city"] for row in request["rows"]} == {"jinan", "hangzhou", "newyork_28x7"}
    assert {row["model_family"] for row in request["rows"]} == {"VectorQ-PPO", "Cond-Scalar-RL"}
    assert all(row["reads_final_traffic_result_values"] is False for row in request["rows"])
    assert all(row["packet_path"].startswith("docs/pro_reviews/") for row in request["rows"])
    validate_representation_generation_request(request)


def test_representation_generation_request_rejects_missing_dpr_packet_key():
    request = build_representation_generation_request(request_id="paper_final_representation_sources_v1")
    bad_keys = dict(request["rows"][0]["packet_keys"])
    bad_keys["dpr"] = ["dpr_head"]
    request["rows"][0] = dict(request["rows"][0], packet_keys=bad_keys)

    with pytest.raises(ValueError, match="dpr"):
        validate_representation_generation_request(request)


def test_representation_generation_request_rejects_final_result_reading():
    request = build_representation_generation_request(request_id="paper_final_representation_sources_v1")
    request["rows"][0] = dict(request["rows"][0], reads_final_traffic_result_values=True)

    with pytest.raises(ValueError, match="final traffic result"):
        validate_representation_generation_request(request)
