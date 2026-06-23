from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_readiness_validator import (
    EXPECTED_METHODS,
    FORMAL_PILOT_READINESS_PACKET_STATUS,
    validate_formal_pilot_readiness_packet,
)


def _method_audit(method: str) -> dict:
    semantics = "queue_length_penalty_proxy" if method == "env_reward" else method
    audit = {
        "scope": {
            "city": "jinan",
            "seed": 0,
            "traffic_file": "anon_3_4_jinan_real.json",
            "run_type": "exploratory_pilot_only",
        },
        "checks": {
            "closed_loop_executed": {"status": "PASS"},
            "reward_finite_nonzero": {"status": "PASS", "reward_row_count": 1440},
            "loss_finite": {"status": "PASS", "loss_row_count": 96},
            "ppo_update": {"status": "PASS", "policy_update_count": 96},
            "action_non_collapse": {
                "status": "PASS",
                "unique_actions": 4,
                "global_single_action_rate": 0.3,
                "threshold": 0.95,
            },
            "checkpoint_roundtrip": {"status": "PASS", "checkpoint_load_verified": True},
            "artifact_allowlist": {"status": "PASS", "forbidden_artifacts_found": []},
            "formal_flags": {
                "status": "PASS",
                "formal_experiment": False,
                "performance_claim": False,
                "method_ranking_allowed": False,
                "seed_expansion_allowed": False,
                "city_expansion_allowed": False,
                "paper_result_allowed": False,
            },
        },
        "reward_adapter_semantics": semantics,
    }
    if method == "env_reward":
        audit["allowed_role"] = "diagnostic_ablation_only"
    return audit


def _packet() -> dict:
    return {
        "packet_type": "pareto_ppo_formal_pilot_readiness_packet",
        "packet_status": FORMAL_PILOT_READINESS_PACKET_STATUS,
        "scope": "formal_pilot_readiness_packet_only_no_new_run",
        "permissions": {
            "formal_ppo_allowed": False,
            "formal_experiment_allowed": False,
            "performance_claim_allowed": False,
            "method_comparison_allowed": False,
            "method_ranking_allowed": False,
            "performance_table_allowed": False,
            "seed_expansion_allowed": False,
            "city_expansion_allowed": False,
            "traffic_metric_reading_allowed": False,
            "paper_result_allowed": False,
        },
        "forbidden_contents": {
            "traffic_performance_values_included": False,
            "method_comparison_included": False,
            "ranking_included": False,
            "formal_table_included": False,
            "winner_claim_included": False,
            "improvement_claim_included": False,
            "paper_result_included": False,
        },
        "exploratory_budget": {"episodes": 1, "max_decision_steps_per_episode": 120},
        "closed_loop_engineering_gate": {"status": "PASS", "methods": list(EXPECTED_METHODS)},
        "method_guard_audits": {method: _method_audit(method) for method in EXPECTED_METHODS},
        "offline_representation_gate": {"status": "partial_pass", "formal_pass": False},
        "next_gate_request": {"formal_experiment_permission_requested": False},
    }


def _write_packet(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "formal_pilot_readiness_packet.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "formal_pilot_readiness_packet.md").write_text(
        "# Formal Pilot Readiness Packet\n\n"
        "Closed-loop engineering gate: PASS.\n\n"
        "Formal PPO remains NO-GO. No ranking, no result table, no winner claim.\n",
        encoding="utf-8",
    )


def test_formal_pilot_readiness_validator_accepts_packet(tmp_path: Path):
    out_dir = tmp_path / "packet"
    _write_packet(out_dir, _packet())

    validate_formal_pilot_readiness_packet(out_dir)


def test_formal_pilot_readiness_validator_rejects_formal_permission(tmp_path: Path):
    out_dir = tmp_path / "packet"
    payload = _packet()
    payload["permissions"]["formal_ppo_allowed"] = True
    _write_packet(out_dir, payload)

    with pytest.raises(ValueError, match="formal_ppo_allowed"):
        validate_formal_pilot_readiness_packet(out_dir)


def test_formal_pilot_readiness_validator_rejects_missing_method(tmp_path: Path):
    out_dir = tmp_path / "packet"
    payload = _packet()
    del payload["method_guard_audits"]["weighted_proxy"]
    _write_packet(out_dir, payload)

    with pytest.raises(ValueError, match="expected methods"):
        validate_formal_pilot_readiness_packet(out_dir)


def test_formal_pilot_readiness_validator_rejects_env_reward_semantic_regression(tmp_path: Path):
    out_dir = tmp_path / "packet"
    payload = _packet()
    payload["method_guard_audits"]["env_reward"]["reward_adapter_semantics"] = "original_llmlight_reward"
    _write_packet(out_dir, payload)

    with pytest.raises(ValueError, match="queue_length_penalty_proxy"):
        validate_formal_pilot_readiness_packet(out_dir)


def test_formal_pilot_readiness_validator_rejects_representation_formal_pass(tmp_path: Path):
    out_dir = tmp_path / "packet"
    payload = _packet()
    payload["offline_representation_gate"]["status"] = "formal_pass"
    payload["offline_representation_gate"]["formal_pass"] = True
    _write_packet(out_dir, payload)

    with pytest.raises(ValueError, match="partial_pass"):
        validate_formal_pilot_readiness_packet(out_dir)
