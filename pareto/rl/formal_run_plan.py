from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS
from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
from pareto.rl.formal_readiness_proposal import (
    APPROVED_PILOT_METHODS,
    METHOD_DISPLAY_NAMES,
    REFERENCE_ONLY_METHODS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_STOP_CONDITIONS,
)


FORMAL_EXECUTION_APPROVAL_PHRASE = FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
REQUIRED_ALLOWED_OUTPUTS = {
    "command.txt",
    "metadata.json",
    "status.json",
    "stdout.txt",
    "stderr.txt",
    "train_metrics.jsonl",
    "reward_components.jsonl",
    "loss_debug.jsonl",
    "checkpoint_last.pt",
    "training_checkpoint_last.pt",
}
REQUIRED_PLAN_FIELDS = {
    "plan_type",
    "approval",
    "scope",
    "methods",
    "seed_binding",
    "budget",
    "outputs",
    "stop_conditions",
    "metadata_schema",
    "reward_policy",
}


@dataclass(frozen=True)
class FormalRunPlan:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalRunPlan":
        plan = cls(payload=dict(payload))
        plan.validate()
        return plan

    def validate(self) -> None:
        payload = self.payload
        missing = sorted(REQUIRED_PLAN_FIELDS - set(payload))
        if missing:
            raise ValueError(f"formal run plan missing fields: {missing}")
        if payload.get("plan_type") != "formal_jinan_3seed_run_plan":
            raise ValueError("plan_type must be formal_jinan_3seed_run_plan")

        approval = payload.get("approval") or {}
        if approval.get("execution_allowed_now") is not False:
            raise ValueError("approval.execution_allowed_now must be false until Pro explicitly approves")
        if approval.get("required_exact_phrase") != FORMAL_EXECUTION_APPROVAL_PHRASE:
            raise ValueError("approval.required_exact_phrase is not the Pro-required phrase")
        if approval.get("received_exact_phrase") is not False:
            raise ValueError("approval.received_exact_phrase must be false in the plan packet")

        scope = payload.get("scope") or {}
        if scope.get("stage") != "formal_run_plan_only":
            raise ValueError("scope.stage must be formal_run_plan_only")
        if scope.get("scenario") != "jinan":
            raise ValueError("formal run plan is limited to Jinan")
        if scope.get("traffic_file") != "anon_3_4_jinan_real.json":
            raise ValueError("formal run plan is limited to anon_3_4_jinan_real.json")
        if scope.get("cities") != ["jinan"]:
            raise ValueError("scope.cities must be ['jinan']")
        if scope.get("traffic_files") != ["anon_3_4_jinan_real.json"]:
            raise ValueError("scope.traffic_files must contain only anon_3_4_jinan_real.json")
        if scope.get("runs_new_cityflow_training") is not False:
            raise ValueError("run plan packet must not execute new CityFlow training")
        if scope.get("generates_ranking_or_performance_table") is not False:
            raise ValueError("run plan packet must not generate ranking or performance tables")

        methods = payload.get("methods") or {}
        if tuple(methods.get("ppo_methods") or ()) != APPROVED_PILOT_METHODS:
            raise ValueError("methods.ppo_methods must match approved PPO methods")
        if tuple(methods.get("reference_only_methods") or ()) != REFERENCE_ONLY_METHODS:
            raise ValueError("methods.reference_only_methods must match approved reference-only methods")
        display_names = methods.get("display_names") or {}
        for method, expected in METHOD_DISPLAY_NAMES.items():
            if display_names.get(method) != expected:
                raise ValueError(f"display_names.{method} must be {expected}")

        seed_binding = payload.get("seed_binding") or {}
        if seed_binding.get("candidate_seeds") != [0, 1, 2]:
            raise ValueError("candidate_seeds must be [0, 1, 2]")
        if seed_binding.get("cityflow_seed") != "seed_id":
            raise ValueError("cityflow_seed must bind to seed_id")
        if seed_binding.get("policy_seed") != "seed_id":
            raise ValueError("policy_seed must bind to seed_id")
        if seed_binding.get("model_seed") != "seed_id":
            raise ValueError("model_seed must bind to seed_id")

        budget = payload.get("budget") or {}
        for key in (
            "episodes_per_method_seed",
            "max_decision_steps_per_episode",
            "min_action_time",
            "rollout_steps",
            "policy_update_schedule",
            "checkpoint_cadence",
            "eval_cadence",
        ):
            if key not in budget:
                raise ValueError(f"budget.{key} is required")
        if budget.get("adaptive_early_stop") is not False:
            raise ValueError("adaptive_early_stop must be false")
        if budget.get("safety_stop_only") is not True:
            raise ValueError("safety_stop_only must be true")

        outputs = payload.get("outputs") or {}
        allowed_outputs = set(outputs.get("allowed") or [])
        missing_allowed = sorted(REQUIRED_ALLOWED_OUTPUTS - allowed_outputs)
        if missing_allowed:
            raise ValueError(f"outputs.allowed missing: {missing_allowed}")
        forbidden_outputs = set(outputs.get("forbidden") or [])
        missing_forbidden = sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS - forbidden_outputs)
        if missing_forbidden:
            raise ValueError(f"outputs.forbidden missing: {missing_forbidden}")

        stop_conditions = set(payload.get("stop_conditions") or [])
        missing_stop_conditions = sorted(REQUIRED_STOP_CONDITIONS - stop_conditions)
        if missing_stop_conditions:
            raise ValueError(f"stop_conditions missing: {missing_stop_conditions}")
        for extra in ("seed_mismatch", "city_or_traffic_file_mismatch", "unapproved_method", "ranking_or_performance_text"):
            if extra not in stop_conditions:
                raise ValueError(f"stop_conditions missing: {extra}")

        metadata_fields = set((payload.get("metadata_schema") or {}).get("required_fields") or [])
        missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata_fields)
        if missing_metadata:
            raise ValueError(f"metadata_schema.required_fields missing: {missing_metadata}")

        env_reward = (payload.get("reward_policy") or {}).get("env_reward") or {}
        if env_reward.get("method_display_name") != "EnvReward-QueuePenalty-PPO":
            raise ValueError("EnvReward method display name regression")
        if env_reward.get("reward_adapter_semantics") != "queue_length_penalty_proxy":
            raise ValueError("EnvReward reward_adapter_semantics regression")
        if env_reward.get("allowed_role") != "diagnostic_ablation_only":
            raise ValueError("EnvReward must remain diagnostic_ablation_only")
        if env_reward.get("may_be_called_llmlight_original_reward") is not False:
            raise ValueError("EnvReward must not be called LLMLight original reward")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_formal_run_plan(path: str | Path) -> FormalRunPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalRunPlan.from_dict(payload)
