from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS
from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
from pareto.rl.formal_readiness_proposal import (
    APPROVED_PILOT_METHODS,
    METHOD_DISPLAY_NAMES,
    REFERENCE_ONLY_METHODS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_STOP_CONDITIONS,
)
from pareto.rl.formal_run_plan import (
    FORMAL_EXECUTION_APPROVAL_PHRASE,
    REQUIRED_ALLOWED_OUTPUTS,
    FormalRunPlan,
    load_formal_run_plan,
)


def _plan_payload() -> dict:
    return {
        "plan_type": "formal_jinan_3seed_run_plan",
        "approval": {
            "execution_allowed_now": False,
            "required_exact_phrase": FORMAL_EXECUTION_APPROVAL_PHRASE,
            "received_exact_phrase": False,
        },
        "scope": {
            "stage": "formal_run_plan_only",
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "cities": ["jinan"],
            "traffic_files": ["anon_3_4_jinan_real.json"],
            "runs_new_cityflow_training": False,
            "generates_ranking_or_performance_table": False,
        },
        "methods": {
            "ppo_methods": list(APPROVED_PILOT_METHODS),
            "reference_only_methods": list(REFERENCE_ONLY_METHODS),
            "display_names": dict(METHOD_DISPLAY_NAMES),
        },
        "seed_binding": {
            "candidate_seeds": [0, 1, 2],
            "cityflow_seed": "seed_id",
            "policy_seed": "seed_id",
            "model_seed": "seed_id",
        },
        "budget": {
            "episodes_per_method_seed": 5,
            "max_decision_steps_per_episode": 120,
            "min_action_time": 30,
            "rollout_steps": 120,
            "policy_update_schedule": "fixed_rollout_steps",
            "checkpoint_cadence": "last_only_plus_status",
            "eval_cadence": "post_run_guard_audit_only",
            "adaptive_early_stop": False,
            "safety_stop_only": True,
        },
        "outputs": {
            "allowed": sorted(REQUIRED_ALLOWED_OUTPUTS | {"formal_run_plan.md", "formal_run_plan.json"}),
            "forbidden": sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS),
        },
        "stop_conditions": sorted(
            REQUIRED_STOP_CONDITIONS
            | {"seed_mismatch", "city_or_traffic_file_mismatch", "unapproved_method", "ranking_or_performance_text"}
        ),
        "metadata_schema": {"required_fields": sorted(REQUIRED_METADATA_FIELDS)},
        "reward_policy": {
            "env_reward": {
                "method_display_name": "EnvReward-QueuePenalty-PPO",
                "reward_adapter_semantics": "queue_length_penalty_proxy",
                "allowed_role": "diagnostic_ablation_only",
                "may_be_called_llmlight_original_reward": False,
            }
        },
    }


def test_formal_run_plan_accepts_no_execution_plan(tmp_path: Path):
    assert FORMAL_EXECUTION_APPROVAL_PHRASE == FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan_payload()), encoding="utf-8")

    plan = load_formal_run_plan(path)

    assert plan.payload["approval"]["execution_allowed_now"] is False


def test_formal_run_plan_rejects_missing_exact_phrase():
    payload = _plan_payload()
    payload["approval"]["required_exact_phrase"] = "GO"

    with pytest.raises(ValueError, match="required_exact_phrase"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_execution_permission_in_plan_packet():
    payload = _plan_payload()
    payload["approval"]["execution_allowed_now"] = True

    with pytest.raises(ValueError, match="execution_allowed_now"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_city_expansion():
    payload = _plan_payload()
    payload["scope"]["cities"] = ["jinan", "hangzhou"]

    with pytest.raises(ValueError, match="cities"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_seed_binding_mismatch():
    payload = _plan_payload()
    payload["seed_binding"]["model_seed"] = "0"

    with pytest.raises(ValueError, match="model_seed"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_adaptive_early_stop():
    payload = _plan_payload()
    payload["budget"]["adaptive_early_stop"] = True

    with pytest.raises(ValueError, match="adaptive_early_stop"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_forbidden_output_gap():
    payload = _plan_payload()
    payload["outputs"]["forbidden"].remove("method_ranking.csv")

    with pytest.raises(ValueError, match="method_ranking.csv"):
        FormalRunPlan.from_dict(payload)


def test_formal_run_plan_rejects_env_reward_role_regression():
    payload = _plan_payload()
    payload["reward_policy"]["env_reward"]["allowed_role"] = "main_baseline"

    with pytest.raises(ValueError, match="diagnostic_ablation_only"):
        FormalRunPlan.from_dict(payload)
