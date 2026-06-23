from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.rl.formal_readiness_proposal import (
    APPROVED_PILOT_METHODS,
    METHOD_DISPLAY_NAMES,
    REFERENCE_ONLY_METHODS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_STOP_CONDITIONS,
    load_formal_readiness_proposal,
    FormalReadinessProposal,
)


def _proposal_payload() -> dict:
    return {
        "proposal_type": "formal_experiment_proposal_gate_packet",
        "commit_provenance": {
            "run_code_commit": "b7b9906",
            "dry_run_report_commit": "01cf9aa",
            "packet_commit": "next_packet_commit",
        },
        "scope": {
            "current_stage": "formal_readiness_proposal_only",
            "runs_new_cityflow_training": False,
            "generates_ranking_or_performance_table": False,
        },
        "permissions": {
            "formal_experiment_allowed": False,
            "performance_claim_allowed": False,
            "seed_expansion_allowed": False,
            "city_expansion_allowed": False,
            "method_ranking_allowed": False,
            "performance_table_allowed": False,
        },
        "methods": {
            "ppo_methods": list(APPROVED_PILOT_METHODS),
            "reference_only_methods": list(REFERENCE_ONLY_METHODS),
            "display_names": dict(METHOD_DISPLAY_NAMES),
        },
        "reward_policy": {
            "env_reward": {
                "method_display_name": "EnvReward-QueuePenalty-PPO",
                "reward_adapter_semantics": "queue_length_penalty_proxy",
                "allowed_role": "diagnostic_ablation_only",
                "may_be_called_llmlight_original_reward": False,
            }
        },
        "required_metadata_fields": sorted(REQUIRED_METADATA_FIELDS),
        "forbidden_artifacts": [
            "best_method.json",
            "best_method.txt",
            "leaderboard.csv",
            "main_results.csv",
            "method_ranking.csv",
            "paper_results.csv",
            "performance_table.csv",
            "performance_table.json",
            "performance_table.md",
            "performance_table.tex",
            "preference_response_plot.pdf",
            "preference_response_plot.png",
            "preference_sweep.csv",
            "ranking.csv",
            "traffic_metrics.csv",
        ],
        "stop_conditions": sorted(REQUIRED_STOP_CONDITIONS),
    }


def test_formal_readiness_proposal_accepts_strict_no_run_packet(tmp_path: Path):
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(_proposal_payload()), encoding="utf-8")

    proposal = load_formal_readiness_proposal(path)

    assert proposal.payload["permissions"]["formal_experiment_allowed"] is False


def test_formal_readiness_proposal_rejects_single_commit_field():
    payload = _proposal_payload()
    payload["commit"] = "ambiguous"

    with pytest.raises(ValueError, match="split provenance"):
        FormalReadinessProposal.from_dict(payload)


def test_formal_readiness_proposal_rejects_formal_permission():
    payload = _proposal_payload()
    payload["permissions"]["formal_experiment_allowed"] = True

    with pytest.raises(ValueError, match="formal_experiment_allowed"):
        FormalReadinessProposal.from_dict(payload)


def test_formal_readiness_proposal_rejects_env_reward_semantic_regression():
    payload = _proposal_payload()
    payload["reward_policy"]["env_reward"]["method_display_name"] = "EnvReward-PPO"

    with pytest.raises(ValueError, match="EnvReward-QueuePenalty-PPO"):
        FormalReadinessProposal.from_dict(payload)


def test_formal_readiness_proposal_rejects_missing_artifact_ban():
    payload = _proposal_payload()
    payload["forbidden_artifacts"].remove("leaderboard.csv")

    with pytest.raises(ValueError, match="leaderboard.csv"):
        FormalReadinessProposal.from_dict(payload)


def test_artifact_guard_blocks_new_ranking_and_table_names(tmp_path: Path):
    for name in ("performance_table.md", "paper_results.csv", "leaderboard.csv", "ranking.csv"):
        path = tmp_path / name
        path.write_text("forbidden\n", encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            assert_no_forbidden_performance_artifacts(tmp_path)
        path.unlink()

