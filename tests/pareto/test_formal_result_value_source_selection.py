from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_result_value_source_selection import (
    FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    generate_result_value_source_selection,
)
from pareto.rl.formal_result_value_source_selection_validator import (
    validate_result_value_source_selection_packet,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_policy(tmp_path: Path) -> Path:
    path = tmp_path / "formal_jinan_result_metric_source_policy.json"
    fields = {
        "sim_time": {
            "status": "candidate_independent_result_metric_requires_pro_review",
            "source_files": ["train_metrics.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "arrived_vehicle_count": {
            "status": "candidate_independent_result_metric_requires_pro_review",
            "source_files": ["eval_metrics.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "env_reward": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "potential_reward": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "weighted_proxy_reward": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "scalar_quality_score_t": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "scalar_quality_score_tp1": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "proxy_objectives_norm_tp1": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "total_reward": {
            "status": "forbidden_proxy_reward_metric",
            "source_files": ["reward_components.jsonl"],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
    }
    _write_json(
        path,
        {
            "report_status": "FORMAL_JINAN_RESULT_METRIC_SOURCE_POLICY_PASS",
            "scope": "result_metric_source_policy_only_no_value_reading",
            "candidate_sources": {
                "metadata.json": {
                    "status": "allowed_guard_metadata",
                    "present_in_inventory": True,
                    "schema_key_names_recorded": True,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
                "status.json": {
                    "status": "allowed_guard_metadata",
                    "present_in_inventory": True,
                    "schema_key_names_recorded": True,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
                "loss_debug.jsonl": {
                    "status": "allowed_training_stability_only",
                    "present_in_inventory": True,
                    "schema_key_names_recorded": True,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
                "reward_components.jsonl": {
                    "status": "forbidden_proxy_reward_metric",
                    "present_in_inventory": True,
                    "schema_key_names_recorded": True,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
                "train_metrics.jsonl": {
                    "status": "candidate_independent_result_metric_requires_pro_review",
                    "present_in_inventory": True,
                    "schema_key_names_recorded": True,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
                "eval_metrics.jsonl": {
                    "status": "candidate_independent_result_metric_requires_pro_review",
                    "present_in_inventory": False,
                    "schema_key_names_recorded": False,
                    "value_reading_allowed": False,
                    "numeric_aggregation_allowed": False,
                },
            },
            "fields": fields,
            "policy": {
                "value_reading_allowed": False,
                "numeric_aggregation_allowed": False,
                "method_level_aggregate_allowed": False,
                "seed_level_table_allowed": False,
                "significance_testing_allowed": False,
                "confidence_interval_allowed": False,
                "formal_result_analysis_allowed": False,
                "formal_result_table_allowed": False,
            },
        },
    )
    return path


def test_result_value_source_selection_selects_no_sources(tmp_path: Path):
    out_dir = tmp_path / "out"
    report = generate_result_value_source_selection(
        result_metric_source_policy_json=_source_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_RESULT_VALUE_SOURCE_SELECTION_PASS"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "formal_jinan_result_value_source_selection.json",
        "formal_jinan_result_value_source_selection.md",
    ]
    payload = json.loads((out_dir / "formal_jinan_result_value_source_selection.json").read_text(encoding="utf-8"))
    assert payload["policy"]["result_value_sources_selected"] is False
    assert payload["policy"]["formal_result_analysis_allowed"] is False
    assert payload["source_selection"]["train_metrics.jsonl"]["selection_status"] == (
        "candidate_independent_result_metric_requires_future_pro_review"
    )
    assert payload["source_selection"]["eval_metrics.jsonl"]["selection_status"] == (
        "candidate_independent_result_metric_requires_future_pro_review"
    )
    assert payload["source_selection"]["reward_components.jsonl"]["selection_status"] == "forbidden_proxy_reward_metric"
    assert payload["future_candidate_fields"]["sim_time"]["value_reading_allowed"] is False
    assert payload["future_candidate_fields"]["arrived_vehicle_count"]["value_reading_allowed"] is False
    assert payload["forbidden_proxy_reward_fields"]["env_reward"]["selection_status"] == "forbidden_proxy_reward_metric"
    validate_result_value_source_selection_packet(out_dir)


def test_result_value_source_selection_rejects_wrong_phrase(tmp_path: Path):
    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        generate_result_value_source_selection(
            result_metric_source_policy_json=_source_policy(tmp_path),
            out_dir=tmp_path / "out",
            approval_phrase="wrong",
        )


def test_result_value_source_selection_validator_rejects_numeric_value_carrier(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_value_source_selection(
        result_metric_source_policy_json=_source_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_value_source_selection.json").read_text(encoding="utf-8"))
    payload["mean_values"] = {"sim_time": 123.0}
    _write_json(out_dir / "formal_jinan_result_value_source_selection.json", payload)

    with pytest.raises(ValueError, match="mean_values"):
        validate_result_value_source_selection_packet(out_dir)


def test_result_value_source_selection_validator_rejects_reward_source_allowed(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_value_source_selection(
        result_metric_source_policy_json=_source_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_value_source_selection.json").read_text(encoding="utf-8"))
    payload["source_selection"]["reward_components.jsonl"]["selection_status"] = (
        "candidate_independent_result_metric_requires_future_pro_review"
    )
    _write_json(out_dir / "formal_jinan_result_value_source_selection.json", payload)

    with pytest.raises(ValueError, match="reward_components"):
        validate_result_value_source_selection_packet(out_dir)


def test_result_value_source_selection_validator_rejects_train_eval_allowed(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_value_source_selection(
        result_metric_source_policy_json=_source_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_value_source_selection.json").read_text(encoding="utf-8"))
    payload["source_selection"]["train_metrics.jsonl"]["source_policy_status"] = "allowed_result_value_source"
    _write_json(out_dir / "formal_jinan_result_value_source_selection.json", payload)

    with pytest.raises(ValueError, match="train_metrics"):
        validate_result_value_source_selection_packet(out_dir)

    payload["source_selection"]["train_metrics.jsonl"]["source_policy_status"] = (
        "candidate_independent_result_metric_requires_pro_review"
    )
    payload["source_selection"]["eval_metrics.jsonl"]["selected_for_result_value_reading"] = True
    _write_json(out_dir / "formal_jinan_result_value_source_selection.json", payload)

    with pytest.raises(ValueError, match="eval_metrics"):
        validate_result_value_source_selection_packet(out_dir)


def test_result_value_source_selection_validator_rejects_forbidden_output(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_value_source_selection(
        result_metric_source_policy_json=_source_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE,
    )
    (out_dir / "main_results.csv").write_text("forbidden\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected|forbidden"):
        validate_result_value_source_selection_packet(out_dir)
