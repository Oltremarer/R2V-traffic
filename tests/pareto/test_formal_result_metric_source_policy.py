from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_result_metric_source_policy import (
    FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    generate_result_metric_source_policy,
)
from pareto.rl.formal_result_metric_source_policy_validator import validate_result_metric_source_policy_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _inventory(tmp_path: Path) -> Path:
    path = tmp_path / "formal_jinan_result_field_inventory.json"
    _write_json(
        path,
        {
            "report_status": "FORMAL_JINAN_RESULT_FIELD_INVENTORY_PASS",
            "scope": "field_inventory_only_no_metric_values",
            "files": {
                "metadata.json": {"keys": ["method", "cityflow_seed", "formal_experiment"]},
                "status.json": {"keys": ["status", "policy_update_count"]},
                "train_metrics.jsonl": {"keys": ["episode", "step", "sim_time", "reward_finite"]},
                "reward_components.jsonl": {
                    "keys": [
                        "env_reward",
                        "potential_reward",
                        "weighted_proxy_reward",
                        "scalar_quality_score_t",
                        "scalar_quality_score_tp1",
                        "proxy_objectives_norm_tp1",
                        "action",
                    ]
                },
                "loss_debug.jsonl": {"keys": ["approx_kl", "grad_norm", "ratio_max", "total_loss"]},
            },
            "field_categories": {
                "method": "diagnostic",
                "cityflow_seed": "diagnostic",
                "formal_experiment": "diagnostic",
                "status": "diagnostic",
                "policy_update_count": "training_stability",
                "episode": "diagnostic",
                "step": "diagnostic",
                "sim_time": "unknown_requires_review",
                "reward_finite": "traffic_like_forbidden",
                "env_reward": "traffic_like_forbidden",
                "potential_reward": "traffic_like_forbidden",
                "weighted_proxy_reward": "traffic_like_forbidden",
                "scalar_quality_score_t": "traffic_like_forbidden",
                "scalar_quality_score_tp1": "traffic_like_forbidden",
                "proxy_objectives_norm_tp1": "unknown_requires_review",
                "action": "unknown_requires_review",
                "approx_kl": "training_stability",
                "grad_norm": "training_stability",
                "ratio_max": "training_stability",
                "total_loss": "training_stability",
            },
        },
    )
    return path


def _safe_policy(tmp_path: Path) -> Path:
    path = tmp_path / "formal_jinan_safe_field_policy.json"
    fields = {
        "method": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "cityflow_seed": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "formal_experiment": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "status": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "policy_update_count": {
            "status": "allowed_training_stability_sanity_only",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "episode": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "step": {"status": "allowed_guard_metadata", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "sim_time": {"status": "unknown_requires_new_pro_review", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "reward_finite": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "env_reward": {"status": "forbidden_proxy_result_metric", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "potential_reward": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "weighted_proxy_reward": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "scalar_quality_score_t": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "scalar_quality_score_tp1": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "proxy_objectives_norm_tp1": {
            "status": "forbidden_proxy_result_metric",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "action": {"status": "unknown_requires_new_pro_review", "value_reading_allowed": False, "numeric_aggregation_allowed": False},
        "approx_kl": {
            "status": "allowed_training_stability_sanity_only",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "grad_norm": {
            "status": "allowed_training_stability_sanity_only",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "ratio_max": {
            "status": "allowed_training_stability_sanity_only",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
        "total_loss": {
            "status": "allowed_training_stability_sanity_only",
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
        },
    }
    _write_json(
        path,
        {
            "report_status": "FORMAL_JINAN_SAFE_FIELD_POLICY_PASS",
            "scope": "safe_field_policy_only_no_numeric_aggregation",
            "fields": fields,
            "policy": {
                "method_comparison_allowed": False,
                "formal_result_values_allowed": False,
                "formal_result_table_allowed": False,
                "numeric_aggregation_allowed": False,
            },
        },
    )
    return path


def test_result_metric_source_policy_locks_all_value_reading(tmp_path: Path):
    out_dir = tmp_path / "out"
    report = generate_result_metric_source_policy(
        inventory_json=_inventory(tmp_path),
        safe_field_policy_json=_safe_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_RESULT_METRIC_SOURCE_POLICY_PASS"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "formal_jinan_result_metric_source_policy.json",
        "formal_jinan_result_metric_source_policy.md",
    ]
    payload = json.loads((out_dir / "formal_jinan_result_metric_source_policy.json").read_text(encoding="utf-8"))
    assert payload["policy"]["value_reading_allowed"] is False
    assert payload["policy"]["numeric_aggregation_allowed"] is False
    assert payload["policy"]["method_level_aggregate_allowed"] is False
    assert payload["policy"]["seed_level_table_allowed"] is False
    assert payload["policy"]["formal_result_analysis_allowed"] is False

    sources = payload["candidate_sources"]
    assert sources["metadata.json"]["status"] == "allowed_guard_metadata"
    assert sources["status.json"]["status"] == "allowed_guard_metadata"
    assert sources["loss_debug.jsonl"]["status"] == "allowed_training_stability_only"
    assert sources["reward_components.jsonl"]["status"] == "forbidden_proxy_reward_metric"
    assert sources["train_metrics.jsonl"]["status"] == "candidate_independent_result_metric_requires_pro_review"
    assert sources["eval_metrics.jsonl"]["status"] == "candidate_independent_result_metric_requires_pro_review"

    fields = payload["fields"]
    assert fields["env_reward"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["potential_reward"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["weighted_proxy_reward"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["scalar_quality_score_t"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["scalar_quality_score_tp1"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["proxy_objectives_norm_tp1"]["status"] == "forbidden_proxy_reward_metric"
    assert fields["method"]["status"] == "allowed_guard_metadata"
    assert fields["approx_kl"]["status"] == "allowed_training_stability_only"
    assert fields["sim_time"]["status"] == "candidate_independent_result_metric_requires_pro_review"
    for entry in fields.values():
        assert entry["value_reading_allowed"] is False
        assert entry["numeric_aggregation_allowed"] is False

    validate_result_metric_source_policy_packet(out_dir)


def test_result_metric_source_policy_rejects_wrong_phrase(tmp_path: Path):
    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        generate_result_metric_source_policy(
            inventory_json=_inventory(tmp_path),
            safe_field_policy_json=_safe_policy(tmp_path),
            out_dir=tmp_path / "out",
            approval_phrase="wrong",
        )


def test_result_metric_source_policy_validator_rejects_value_reading(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_metric_source_policy(
        inventory_json=_inventory(tmp_path),
        safe_field_policy_json=_safe_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_metric_source_policy.json").read_text(encoding="utf-8"))
    payload["fields"]["env_reward"]["value_reading_allowed"] = True
    _write_json(out_dir / "formal_jinan_result_metric_source_policy.json", payload)

    with pytest.raises(ValueError, match="value reading"):
        validate_result_metric_source_policy_packet(out_dir)


def test_result_metric_source_policy_validator_rejects_reward_components_allowed(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_metric_source_policy(
        inventory_json=_inventory(tmp_path),
        safe_field_policy_json=_safe_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_metric_source_policy.json").read_text(encoding="utf-8"))
    payload["candidate_sources"]["reward_components.jsonl"]["status"] = "candidate_independent_result_metric_requires_pro_review"
    _write_json(out_dir / "formal_jinan_result_metric_source_policy.json", payload)

    with pytest.raises(ValueError, match="reward_components"):
        validate_result_metric_source_policy_packet(out_dir)


def test_result_metric_source_policy_validator_rejects_forbidden_field_status(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_metric_source_policy(
        inventory_json=_inventory(tmp_path),
        safe_field_policy_json=_safe_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_metric_source_policy.json").read_text(encoding="utf-8"))
    payload["fields"]["env_reward"]["status"] = "unknown_requires_new_pro_review"
    _write_json(out_dir / "formal_jinan_result_metric_source_policy.json", payload)

    with pytest.raises(ValueError, match="env_reward"):
        validate_result_metric_source_policy_packet(out_dir)


def test_result_metric_source_policy_validator_rejects_raw_value_carrier(tmp_path: Path):
    out_dir = tmp_path / "out"
    generate_result_metric_source_policy(
        inventory_json=_inventory(tmp_path),
        safe_field_policy_json=_safe_policy(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_result_metric_source_policy.json").read_text(encoding="utf-8"))
    payload["raw_values"] = {"env_reward": 1.23}
    _write_json(out_dir / "formal_jinan_result_metric_source_policy.json", payload)

    with pytest.raises(ValueError, match="raw_values"):
        validate_result_metric_source_policy_packet(out_dir)
