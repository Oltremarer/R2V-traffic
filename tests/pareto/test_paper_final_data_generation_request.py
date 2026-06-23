from __future__ import annotations

import pytest

from pareto.data.paper_final_data_generation_request import (
    PAPER_FINAL_DATA_POLICIES,
    build_paper_final_data_generation_request,
    data_generation_request_blockers,
    validate_paper_final_data_generation_request,
)
from pareto.rl.paper_final_experiment_manifest import PAPER_FINAL_SEEDS, REQUIRED_CITY_TRAFFIC


def test_request_is_non_executing_and_covers_all_cities_policies_seeds():
    request = build_paper_final_data_generation_request(run_id="paper_final_20260602_v1")

    assert request["packet_type"] == "paper_final_data_generation_request"
    assert request["status"] == "ready_request"
    assert request["execution_allowed_now"] is False
    assert request["executes_generation_now"] is False
    assert request["reads_final_traffic_result_values"] is False
    assert request["writes_records_paper_final"] is False

    rows = request["collection_rows"]
    expected = {
        (city, policy, seed)
        for city in REQUIRED_CITY_TRAFFIC
        for policy in PAPER_FINAL_DATA_POLICIES
        for seed in PAPER_FINAL_SEEDS
    }
    observed = {(row["city"], row["policy"], row["seed"]) for row in rows}
    assert observed == expected

    for row in rows:
        assert "python pareto/data/collect_pareto_buffer.py" in row["command_preview"]
        assert "--overwrite" not in row["command_preview"]
        assert "records/paper_final/" not in row["command_preview"]
        assert row["executes_generation_now"] is False


def test_request_builds_split_normalizer_pair_validation_and_audit_commands():
    request = build_paper_final_data_generation_request(run_id="paper_final_20260602_v1")

    assert len(request["city_rows"]) == len(REQUIRED_CITY_TRAFFIC)
    for row in request["city_rows"]:
        city = row["city"]
        assert row["raw_split_root"] == f"data/pareto_records_split/{city}/paper_final"
        assert row["normalized_records_root"] == f"data/pareto_records_split_norm/{city}/paper_final"
        assert row["pairs_root"] == f"data/pareto_pairs/{city}/paper_final"
        assert "python pareto/data/split_records.py" in row["split_records_command_preview"]
        assert "--group_key time_block" in row["split_records_command_preview"]
        assert "python pareto/data/fit_objective_normalizer.py" in row["fit_normalizer_command_preview"]
        assert f"{row['raw_split_root']}/train_raw.jsonl" in row["fit_normalizer_command_preview"]
        assert "val_raw.jsonl" not in row["fit_normalizer_command_preview"]
        assert "test_raw.jsonl" not in row["fit_normalizer_command_preview"]
        assert "python pareto/data/apply_objective_normalizer.py" in row["apply_normalizer_command_preview"]
        assert row["final_audit_command_preview"] == "python -m pareto.data.paper_final_data_roots"

        pair_rows = row["pair_rows"]
        assert {pair_row["split"] for pair_row in pair_rows} == {"train", "val", "test"}
        for pair_row in pair_rows:
            assert "python pareto/data/build_pairs.py" in pair_row["build_pairs_command_preview"]
            assert "python pareto/data/validate_pairs.py" in pair_row["validate_pairs_command_preview"]
            assert "--strict" in pair_row["validate_pairs_command_preview"]
            assert pair_row["executes_generation_now"] is False


def test_request_validation_rejects_execution_or_final_result_claims():
    request = build_paper_final_data_generation_request(run_id="paper_final_20260602_v1")

    bad = dict(request, executes_generation_now=True)
    with pytest.raises(ValueError, match="non-executing"):
        validate_paper_final_data_generation_request(bad)

    bad = dict(request, reads_final_traffic_result_values=True)
    with pytest.raises(ValueError, match="must not read final traffic result values"):
        validate_paper_final_data_generation_request(bad)

    bad = dict(request, writes_records_paper_final=True)
    with pytest.raises(ValueError, match="must not write records/paper_final"):
        validate_paper_final_data_generation_request(bad)


def test_request_blockers_are_empty_only_for_ready_non_executing_packet():
    request = build_paper_final_data_generation_request(run_id="paper_final_20260602_v1")
    assert data_generation_request_blockers(request) == []

    blocked = dict(request, status="missing_blocker", blocker="reviewer has not accepted data source policy")
    assert data_generation_request_blockers(blocked) == [
        "data generation request: reviewer has not accepted data source policy"
    ]
