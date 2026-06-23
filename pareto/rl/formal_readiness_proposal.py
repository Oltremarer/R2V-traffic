from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS


APPROVED_PILOT_METHODS = ("film_scalar_potential", "weighted_proxy", "env_reward")
METHOD_DISPLAY_NAMES = {
    "film_scalar_potential": "FiLMScalar-PPO",
    "weighted_proxy": "WeightedProxy-PPO",
    "env_reward": "EnvReward-QueuePenalty-PPO",
}
REFERENCE_ONLY_METHODS = ("MaxPressure", "AdvancedMaxPressure")
REQUIRED_METADATA_FIELDS = {
    "pilot_only",
    "formal_experiment",
    "performance_claim",
    "not_for_main_results",
    "cityflow_seed",
    "policy_seed",
    "model_seed",
    "episodes",
    "max_decision_steps_per_episode",
    "min_action_time",
    "reward_adapter",
    "reward_adapter_semantics",
    "method_display_name",
    "checkpoint_use",
}
REQUIRED_STOP_CONDITIONS = {
    "nonfinite_loss",
    "nonfinite_reward",
    "nonfinite_logits_or_grad",
    "checkpoint_save_or_load_fail",
    "budget_mismatch",
    "missing_metadata",
    "forbidden_artifact",
    "direct_inner_step_call",
    "env_reward_all_zero",
    "env_reward_wrong_source",
    "env_reward_name_regression",
    "preference_not_conditioning_actor_critic",
}


@dataclass(frozen=True)
class FormalReadinessProposal:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalReadinessProposal":
        proposal = cls(payload=dict(payload))
        proposal.validate()
        return proposal

    def validate(self) -> None:
        payload = self.payload
        if payload.get("proposal_type") != "formal_experiment_proposal_gate_packet":
            raise ValueError("proposal_type must be formal_experiment_proposal_gate_packet")
        if "commit" in payload:
            raise ValueError("use split provenance fields, not a single commit field")

        provenance = payload.get("commit_provenance") or {}
        for key in ("run_code_commit", "dry_run_report_commit", "packet_commit"):
            value = str(provenance.get(key, "")).strip()
            if len(value) < 7:
                raise ValueError(f"commit_provenance.{key} is required")

        permissions = payload.get("permissions") or {}
        for key in (
            "formal_experiment_allowed",
            "performance_claim_allowed",
            "seed_expansion_allowed",
            "city_expansion_allowed",
            "method_ranking_allowed",
            "performance_table_allowed",
        ):
            if permissions.get(key) is not False:
                raise ValueError(f"permissions.{key} must be false until Pro explicitly approves")

        methods = payload.get("methods") or {}
        if tuple(methods.get("ppo_methods") or ()) != APPROVED_PILOT_METHODS:
            raise ValueError("methods.ppo_methods must match the approved pilot method list")
        if tuple(methods.get("reference_only_methods") or ()) != REFERENCE_ONLY_METHODS:
            raise ValueError("methods.reference_only_methods must be MaxPressure/AdvancedMaxPressure")
        display_names = methods.get("display_names") or {}
        for method, expected in METHOD_DISPLAY_NAMES.items():
            if display_names.get(method) != expected:
                raise ValueError(f"display_names.{method} must be {expected}")

        reward_policy = payload.get("reward_policy") or {}
        env_reward = reward_policy.get("env_reward") or {}
        if env_reward.get("method_display_name") != "EnvReward-QueuePenalty-PPO":
            raise ValueError("env_reward.method_display_name must be EnvReward-QueuePenalty-PPO")
        if env_reward.get("reward_adapter_semantics") != "queue_length_penalty_proxy":
            raise ValueError("env_reward.reward_adapter_semantics must be queue_length_penalty_proxy")
        if env_reward.get("allowed_role") != "diagnostic_ablation_only":
            raise ValueError("env_reward.allowed_role must be diagnostic_ablation_only")
        if env_reward.get("may_be_called_llmlight_original_reward") is not False:
            raise ValueError("env_reward must not be called LLMLight original reward")

        required_metadata = set(payload.get("required_metadata_fields") or [])
        missing_metadata = sorted(REQUIRED_METADATA_FIELDS - required_metadata)
        if missing_metadata:
            raise ValueError(f"required_metadata_fields missing: {missing_metadata}")

        forbidden = set(payload.get("forbidden_artifacts") or [])
        missing_forbidden = sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS - forbidden)
        if missing_forbidden:
            raise ValueError(f"forbidden_artifacts missing: {missing_forbidden}")

        stop_conditions = set(payload.get("stop_conditions") or [])
        missing_stop_conditions = sorted(REQUIRED_STOP_CONDITIONS - stop_conditions)
        if missing_stop_conditions:
            raise ValueError(f"stop_conditions missing: {missing_stop_conditions}")

        scope = payload.get("scope") or {}
        if scope.get("current_stage") != "formal_readiness_proposal_only":
            raise ValueError("scope.current_stage must be formal_readiness_proposal_only")
        if scope.get("runs_new_cityflow_training") is not False:
            raise ValueError("proposal must not run new CityFlow training")
        if scope.get("generates_ranking_or_performance_table") is not False:
            raise ValueError("proposal must not generate ranking or performance tables")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_formal_readiness_proposal(path: str | Path) -> FormalReadinessProposal:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalReadinessProposal.from_dict(payload)

