from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_jinan_3seed_reference_eval_remediation import (
    build_reference_eval_remediation_packet,
    validate_reference_eval_remediation_packet,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_failure_audit(path: Path) -> None:
    _write_json(
        path,
        {
            "packet_type": "formal_jinan_3seed_reference_eval_execution_audit",
            "audit_status": "FAIL",
            "failures": [
                "seed0/vector_quality_potential: execution failed: ValueError: non-finite common metric: average_travel_time"
            ],
            "reference_eval_run_in_this_packet": False,
            "cityflow_run_in_this_packet": True,
            "model_rollout_in_this_packet": True,
            "traffic_result_value_reading_in_this_packet": False,
            "numeric_traffic_aggregation_in_this_packet": False,
            "method_ranking_in_this_packet": False,
            "performance_table_in_this_packet": False,
            "best_method_claim_in_this_packet": False,
            "traffic_improvement_claim_in_this_packet": False,
            "paper_result_claim_in_this_packet": False,
            "provenance": {
                "binding_commit": "bc594a8",
                "adapter_commit": "e04d7ce",
                "runner_commit": "c980523",
            },
            "runs": [
                {
                    "method": "vector_quality_potential",
                    "seed": 0,
                    "status": "FAIL",
                    "failures": ["execution failed: ValueError: non-finite common metric: average_travel_time"],
                    "checkpoint_sha_rechecked": False,
                }
            ],
        },
    )


def test_reference_eval_remediation_packet_is_guard_only_and_consumes_prior_go(tmp_path: Path):
    audit = tmp_path / "execution_audit.json"
    _write_failure_audit(audit)

    packet = build_reference_eval_remediation_packet(
        out_dir=tmp_path / "remediation",
        failure_audit_packet=audit,
        remediation_commit="local-remediation",
        remote_targeted_tests_result="25 passed in 0.73s",
        remote_full_tests_result="296 passed in 15.55s",
    )

    validate_reference_eval_remediation_packet(packet)
    assert packet["packet_type"] == "formal_jinan_3seed_reference_eval_remediation"
    assert packet["overall_pass"] is True
    assert packet["reference_eval_run_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["method_ranking_in_this_packet"] is False
    assert packet["performance_table_in_this_packet"] is False
    assert packet["prior_reference_eval_go_consumed"] is True
    assert packet["prior_go_reusable"] is False
    assert packet["new_go_required_before_reference_eval"] is True
    assert packet["verify_report"]["overall_pass"] is True
    assert packet["verify_report"]["verification_evidence"]["remote_targeted_tests_result"] == "25 passed in 0.73s"
    assert packet["verify_report"]["verification_evidence"]["remote_full_tests_result"] == "296 passed in 15.55s"
    assert packet["verify_report"]["reference_eval_execution_allowed_now"] is False
    assert packet["closure_summary"]["closed_count"] >= 10
    assert packet["closure_summary"]["open_blockers"] == []
    assert packet["protocol"]["output_root_strategy"]["must_be_empty_before_next_run"] is True
    assert packet["executor_protocol"]["common_metric_policy"]["nonfinite_metric_action"] == "hard_fail"
    assert packet["executor_protocol"]["common_metric_policy"]["nan_leave_time_action"] == "skip_and_count_incomplete"
    assert packet["executor_protocol"]["common_metric_policy"]["incomplete_vehicle_count_recorded"] is True
    assert packet["executor_protocol"]["common_metric_policy"]["incomplete_vehicle_count_metadata_path"] == (
        "metadata.json.common_metric_debug.incomplete_vehicle_count"
    )
    assert packet["executor_protocol"]["obs_feature_policy"]["schema_hash_drift_action"] == "hard_fail"
    assert packet["executor_protocol"]["metric_window_policy"]["queue_observations_reset_on_env_reset"] is True
    assert packet["executor_protocol"]["ppo_preference_policy"]["default_preference_locked"] is True
    assert packet["executor_protocol"]["reference_policy_policy"]["encoder_required"] is False
    assert packet["chain_manifest"]["upstream_commits"]["binding_commit"] == "local-remediation"

    output_dir = tmp_path / "remediation"
    assert (output_dir / "verify_report.json").exists()
    assert (output_dir / "REVIEW.md").exists()
    assert (output_dir / "protocol.json").exists()
    assert (output_dir / "executor_protocol.json").exists()
    assert (output_dir / "chain_manifest.json").exists()
    assert (output_dir / "command.txt").exists()


def test_reference_eval_remediation_validator_rejects_reusable_prior_go(tmp_path: Path):
    audit = tmp_path / "execution_audit.json"
    _write_failure_audit(audit)
    packet = build_reference_eval_remediation_packet(
        out_dir=tmp_path / "remediation",
        failure_audit_packet=audit,
        remediation_commit="local-remediation",
    )
    packet["prior_go_reusable"] = True

    try:
        validate_reference_eval_remediation_packet(packet)
    except ValueError as exc:
        assert "prior_go_reusable must be false" in str(exc)
    else:
        raise AssertionError("validator accepted reusable prior GO")
