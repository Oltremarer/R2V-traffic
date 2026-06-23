from __future__ import annotations

import pytest

from pareto.rl.paper_reference_baseline_guard import (
    CONVENTIONAL_BASELINES,
    build_reference_baseline_smoke_request,
    validate_reference_baseline_smoke_matrix,
)


def test_reference_baseline_smoke_matrix_covers_all_cities_without_execution():
    rows = validate_reference_baseline_smoke_matrix()

    assert len(rows) == 3 * len(CONVENTIONAL_BASELINES)
    assert {row["city"] for row in rows} == {"jinan", "hangzhou", "newyork_28x7"}
    assert all(row["executes_now"] is False for row in rows)
    assert all(row["reads_result_values"] is False for row in rows)


def test_reference_baseline_smoke_request_uses_registered_traffic_and_preflight_root():
    request = build_reference_baseline_smoke_request(
        baseline="PressLight",
        city="hangzhou",
        traffic_file="anon_4_4_hangzhou_real.json",
        seed=2,
    )

    assert request["baseline"] == "PressLight"
    assert request["seed_binding"] == "cityflow_seed=policy_seed=model_seed=seed_id"
    assert "records/paper_final/preflight_20260602_v1" in request["out_dir"]


def test_reference_baseline_smoke_rejects_unregistered_traffic_file():
    with pytest.raises(ValueError, match="not registered"):
        build_reference_baseline_smoke_request(
            baseline="MaxPressure",
            city="hangzhou",
            traffic_file="anon_3_4_jinan_real.json",
            seed=0,
        )
