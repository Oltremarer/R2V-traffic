from __future__ import annotations

from typing import Any

from pareto.rl.paper_final_experiment_manifest import (
    PAPER_FINAL_SEEDS,
    REQUIRED_CITY_TRAFFIC,
    REQUIRED_PREFERENCE_TEMPLATES,
)
from pareto.rl.paper_weighted_mapping_policy import REQUIRED_WEIGHTED_MAPPING_APPROVAL


ALLOWED_WEIGHTED_RL_STATUSES = {"request_only", "mapping_approved_preview", "missing_blocker"}
PAPER_SCALE_PPO_BUDGET = {
    "rollout_steps": 3600,
    "minibatch_size": 2048,
    "update_epochs": 10,
    "total_env_steps_per_seed": 1_000_000,
}


def build_weighted_rl_policy_family_request(
    *,
    policy_family_type: str = "discrete_policy_family",
    mapping_approval_phrase: str | None = None,
    output_root: str = "records/paper_final/train_20260602_v1",
) -> dict[str, Any]:
    if policy_family_type == "weighted_proxy_mapping":
        if mapping_approval_phrase != REQUIRED_WEIGHTED_MAPPING_APPROVAL:
            return validate_weighted_rl_policy_family_request(
                {
                    "baseline": "Weighted-RL",
                    "status": "missing_blocker",
                    "blocker": "Weighted-RL mapping requires exact reviewer approval",
                    "mapping_approval_phrase": mapping_approval_phrase or "",
                    "executes_training_now": False,
                }
            )
        return validate_weighted_rl_policy_family_request(
            {
                "baseline": "Weighted-RL",
                "status": "mapping_approved_preview",
                "mapping_approval_phrase": mapping_approval_phrase,
                "method_id": "weighted_proxy",
                "executes_training_now": False,
                "paper_claim_limitation": "Weighted-RL is represented by reviewer-approved weighted_proxy mapping.",
            }
        )
    if policy_family_type != "discrete_policy_family":
        raise ValueError("true Weighted-RL requires one controller per preference")

    controllers = []
    for city, traffic_file in REQUIRED_CITY_TRAFFIC.items():
        for seed in PAPER_FINAL_SEEDS:
            for preference_name, weights in REQUIRED_PREFERENCE_TEMPLATES.items():
                row_output_root = f"{output_root}/{city}/seed{int(seed)}/Weighted-RL/{preference_name}"
                controllers.append(
                    {
                        "city": city,
                        "traffic_file": traffic_file,
                        "seed": int(seed),
                        "fixed_preference_template": preference_name,
                        "preference_weights": list(weights),
                        "controller_scope": "single_fixed_preference",
                        "reward_adapter": "weighted_proxy_fixed_preference",
                        "policy_conditioned_on_w": False,
                        "critic_conditioned_on_w": False,
                        "output_root": row_output_root,
                        "ppo_budget": dict(PAPER_SCALE_PPO_BUDGET),
                        "executes_training_now": False,
                    }
                )
    return validate_weighted_rl_policy_family_request(
        {
            "baseline": "Weighted-RL",
            "status": "request_only",
            "policy_family_type": "discrete_policy_family",
            "controllers": controllers,
            "executes_training_now": False,
            "artifact_hash_claimed": False,
        }
    )


def validate_weighted_rl_policy_family_request(request: dict[str, Any]) -> dict[str, Any]:
    status = request.get("status")
    if request.get("baseline") not in {None, "Weighted-RL"}:
        raise ValueError("Weighted-RL request must target Weighted-RL")
    if status not in ALLOWED_WEIGHTED_RL_STATUSES:
        raise ValueError(f"unknown Weighted-RL request status: {status}")
    if request.get("executes_training_now") is not False:
        raise ValueError("Weighted-RL request must be non-executing")
    if status == "missing_blocker":
        if not request.get("blocker"):
            raise ValueError("missing Weighted-RL request must include blocker")
        return dict(request)
    if status == "mapping_approved_preview":
        if request.get("mapping_approval_phrase") != REQUIRED_WEIGHTED_MAPPING_APPROVAL:
            raise ValueError("Weighted-RL mapping requires exact reviewer approval")
        if request.get("method_id") != "weighted_proxy":
            raise ValueError("Weighted-RL mapping preview must target weighted_proxy")
        return dict(request)

    controllers = request.get("controllers") or []
    expected = {
        (city, seed, preference)
        for city in REQUIRED_CITY_TRAFFIC
        for seed in PAPER_FINAL_SEEDS
        for preference in REQUIRED_PREFERENCE_TEMPLATES
    }
    observed = {
        (row.get("city"), int(row.get("seed")), row.get("fixed_preference_template"))
        for row in controllers
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Weighted-RL request missing controller rows: {missing}")
    for row in controllers:
        if row.get("executes_training_now") is not False:
            raise ValueError("Weighted-RL controller row must be non-executing")
        if row.get("policy_conditioned_on_w") is not False or row.get("critic_conditioned_on_w") is not False:
            raise ValueError("true Weighted-RL discrete controllers must not be conditioned on w")
        if row.get("controller_scope") != "single_fixed_preference":
            raise ValueError("true Weighted-RL requires one controller per preference")
        if not str(row.get("output_root") or "").startswith("records/paper_final/"):
            raise ValueError("Weighted-RL output_root must be under records/paper_final")
        budget = row.get("ppo_budget") or {}
        for key, value in PAPER_SCALE_PPO_BUDGET.items():
            if int(budget.get(key, -1)) != value:
                raise ValueError(f"Weighted-RL controller row requires {key}={value}")
    return dict(request)


def weighted_rl_policy_family_blockers(request: dict[str, Any]) -> list[str]:
    validated = validate_weighted_rl_policy_family_request(request)
    if validated["status"] == "missing_blocker":
        return [f"Weighted-RL: {validated['blocker']}"]
    return []
