from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
    build_reference_eval_proposal_packet,
    validate_reference_eval_proposal_packet,
)


def _write_analysis_packet(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "packet_type": "formal_jinan_3seed_descriptive_analysis",
                "analysis_status": "PASS",
                "permissions": {
                    "method_ranking_executed": False,
                    "performance_table_generated": False,
                    "best_method_claim_generated": False,
                    "traffic_improvement_claim_generated": False,
                    "paper_ready_claim_generated": False,
                    "comparison_requires_new_gate": True,
                    "seed_expansion_executed": False,
                    "city_expansion_executed": False,
                    "not_for_main_results": True,
                    "exclude_from_paper": True,
                },
                "provenance": {
                    "guard_packet_sha256": "guard-sha",
                    "verification_packet_sha256": "verification-sha",
                    "request_packet_sha256": "request-sha",
                    "execution_audit_packet_sha256": "execution-audit-sha",
                    "execution_audit_commit": "d08ecc9",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_reference_eval_proposal_locks_scope_without_execution(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_packet(analysis_path)

    packet = build_reference_eval_proposal_packet(out_dir=tmp_path / "proposal", analysis_packet=analysis_path)

    assert packet["packet_type"] == "formal_jinan_3seed_reference_eval_proposal"
    assert packet["formal_reference_eval_execution_allowed_now"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["model_rollout_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["numeric_traffic_aggregation_in_this_packet"] is False
    assert packet["method_ranking_in_this_packet"] is False
    assert packet["performance_table_in_this_packet"] is False
    assert packet["paper_result_claim_in_this_packet"] is False
    assert packet["next_gate"]["required_exact_phrase"] == FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE

    scope = packet["locked_future_eval_scope"]
    assert scope["scenario"] == "jinan"
    assert scope["traffic_file"] == "anon_3_4_jinan_real.json"
    assert scope["seed_ids"] == [0, 1, 2]
    assert scope["methods"] == [
        "vector_quality_potential",
        "film_scalar_potential",
        "weighted_proxy",
        "env_reward",
    ]
    assert scope["reference_baselines"] == ["MaxPressure", "AdvancedMaxPressure"]
    assert scope["min_action_time"] == 30
    assert scope["same_protocol_for_methods_and_references"] is True

    metrics = packet["locked_future_eval_metrics"]
    assert metrics["primary_metric"] == "average_travel_time"
    assert metrics["training_reward_excluded"] is True
    assert "training_reward" not in metrics["secondary_metrics"]
    assert "total_reward" not in metrics["secondary_metrics"]

    artifacts = packet["future_eval_artifact_policy"]
    assert artifacts["eval_output_root"] == "records/formal_jinan_3seed_eval_20260601"
    assert "performance_table.csv" in artifacts["forbidden_outputs"]
    assert "ranking.csv" in artifacts["forbidden_outputs"]
    assert "eval_metrics.jsonl" in artifacts["allowed_raw_outputs"]
    assert not (set(artifacts["allowed_raw_outputs"]) & set(artifacts["forbidden_outputs"]))

    provenance = packet["provenance"]
    assert provenance["analysis_packet_sha256"] == sha256_file(analysis_path)
    assert provenance["execution_audit_commit"] == "d08ecc9"
    assert packet["overall_pass"] is True
    assert packet["failures"] == []
    assert (tmp_path / "proposal" / "formal_jinan_3seed_reference_eval_proposal.json").exists()
    assert (tmp_path / "proposal" / "formal_jinan_3seed_reference_eval_proposal.md").exists()


def test_reference_eval_proposal_validator_rejects_execution_or_result_scope(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_packet(analysis_path)
    packet = build_reference_eval_proposal_packet(out_dir=tmp_path / "proposal", analysis_packet=analysis_path)

    packet["cityflow_run_in_this_packet"] = True
    with pytest.raises(ValueError, match="cityflow_run_in_this_packet"):
        validate_reference_eval_proposal_packet(packet)

    packet["cityflow_run_in_this_packet"] = False
    packet["locked_future_eval_metrics"]["primary_metric"] = "total_reward"
    with pytest.raises(ValueError, match="primary_metric"):
        validate_reference_eval_proposal_packet(packet)


def test_reference_eval_proposal_rejects_unapproved_analysis_packet(tmp_path: Path):
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_packet(analysis_path)
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload["permissions"]["method_ranking_executed"] = True
    analysis_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    packet = build_reference_eval_proposal_packet(out_dir=tmp_path / "proposal", analysis_packet=analysis_path)

    assert packet["overall_pass"] is False
    assert any("analysis packet" in failure for failure in packet["failures"])
