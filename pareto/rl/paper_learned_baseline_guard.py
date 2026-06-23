from __future__ import annotations

from typing import Any

from pareto.rl.paper_weighted_mapping_policy import REQUIRED_WEIGHTED_MAPPING_APPROVAL
from pareto.rl.paper_weighted_rl_policy_family import validate_weighted_rl_policy_family_request


LEARNED_BASELINES = {"C2T-scalar", "Cond-Scalar-RL", "Weighted-RL", "VectorQ-PPO"}
DEFAULT_PAPER_SCALE_CONFIG = {
    "model": {"hidden_dim": 256},
    "ppo": {
        "rollout_steps": 3600,
        "minibatch_size": 2048,
        "update_epochs": 10,
        "total_env_steps_per_seed": 1_000_000,
    },
}


def _validate_paper_scale_config(config: dict[str, Any]) -> None:
    model = config.get("model") or {}
    ppo = config.get("ppo") or {}
    if int(model.get("hidden_dim", -1)) != 256:
        raise ValueError("paper-scale learned baseline requires hidden_dim=256")
    expected = {
        "rollout_steps": 3600,
        "minibatch_size": 2048,
        "update_epochs": 10,
        "total_env_steps_per_seed": 1_000_000,
    }
    for key, value in expected.items():
        if int(ppo.get(key, -1)) != value:
            raise ValueError(f"paper-scale learned baseline requires {key}={value}")


def build_learned_baseline_readiness(
    *,
    baseline: str,
    city: str,
    config: dict[str, Any] | None = None,
    model_hash: str | None = None,
    objective_normalizer_hash: str | None = None,
    mapping_approved: bool | None = None,
    mapping_approval_phrase: str | None = None,
    weighted_family_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if baseline not in LEARNED_BASELINES:
        raise ValueError(f"baseline {baseline} is not a learned paper baseline")
    config = config or DEFAULT_PAPER_SCALE_CONFIG
    _validate_paper_scale_config(config)
    if baseline == "C2T-scalar":
        return {
            "baseline": baseline,
            "city": city,
            "status": "missing_blocker",
            "blocker": "unresolved_c2t_scalar_command",
        }
    if baseline == "Weighted-RL" and weighted_family_request is not None:
        weighted_request = validate_weighted_rl_policy_family_request(weighted_family_request)
        if weighted_request["status"] == "request_only":
            return {
                "baseline": baseline,
                "city": city,
                "status": "missing_blocker",
                "blocker": "Weighted-RL true-family request-only preview is not executed and not final-ready",
                "weighted_family_request_status": weighted_request["status"],
            }
        if weighted_request["status"] == "missing_blocker":
            return {
                "baseline": baseline,
                "city": city,
                "status": "missing_blocker",
                "blocker": weighted_request.get("blocker") or "Weighted-RL true-family request is blocked",
                "weighted_family_request_status": weighted_request["status"],
            }
    if baseline == "Weighted-RL" and mapping_approval_phrase != REQUIRED_WEIGHTED_MAPPING_APPROVAL:
        return {
            "baseline": baseline,
            "city": city,
            "status": "missing_blocker",
            "blocker": "Weighted-RL mapping exact reviewer approval required",
            "mapping_approved": bool(mapping_approved),
        }
    if not model_hash:
        return {"baseline": baseline, "city": city, "status": "missing_blocker", "blocker": "model hash missing"}
    if not objective_normalizer_hash:
        return {"baseline": baseline, "city": city, "status": "missing_blocker", "blocker": "objective normalizer hash missing"}
    return {
        "baseline": baseline,
        "city": city,
        "status": "implemented_guarded_preview",
        "model_hash": str(model_hash),
        "objective_normalizer_hash": str(objective_normalizer_hash),
        "executes_now": False,
    }


def validate_learned_baseline_readiness(readiness: dict[str, Any]) -> dict[str, Any]:
    if readiness.get("baseline") not in LEARNED_BASELINES:
        raise ValueError("unknown learned baseline readiness row")
    if readiness.get("status") not in {"implemented_guarded_preview", "missing_blocker"}:
        raise ValueError("unknown learned baseline readiness status")
    if readiness.get("status") == "implemented_guarded_preview":
        if len(str(readiness.get("model_hash") or "")) != 64:
            raise ValueError("implemented learned baseline missing model hash")
        if len(str(readiness.get("objective_normalizer_hash") or "")) != 64:
            raise ValueError("implemented learned baseline missing objective normalizer hash")
    return dict(readiness)
