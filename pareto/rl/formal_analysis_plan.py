from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS
from pareto.rl.formal_pilot_runner import FINAL_JINAN_PILOT_METHODS, FINAL_JINAN_REFERENCE_ONLY_METHODS


FORMAL_ANALYSIS_APPROVAL_PHRASE = "FORMAL JINAN NO-RANKING ANALYSIS GO"
REQUIRED_PLAN_FIELDS = {
    "plan_type",
    "approval",
    "scope",
    "inputs",
    "permissions",
    "allowed_future_outputs",
    "forbidden_outputs",
    "forbidden_wording",
    "allowed_future_metrics",
    "statistical_policy",
    "method_policy",
}
REQUIRED_FORBIDDEN_WORDING = {
    "best method",
    "beats",
    "leaderboard",
    "outperforms",
    "ranked",
    "traffic improvement",
}
PERFORMANCE_LIKE_METRIC_TOKENS = {
    "average_travel_time",
    "delay",
    "queue",
    "throughput",
    "traffic",
    "travel_time",
    "waiting_time",
}


@dataclass(frozen=True)
class FormalAnalysisPlan:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalAnalysisPlan":
        plan = cls(payload=dict(payload))
        plan.validate()
        return plan

    def validate(self) -> None:
        payload = self.payload
        missing = sorted(REQUIRED_PLAN_FIELDS - set(payload))
        if missing:
            raise ValueError(f"formal analysis plan missing fields: {missing}")
        if payload.get("plan_type") != "formal_jinan_postrun_analysis_plan":
            raise ValueError("plan_type must be formal_jinan_postrun_analysis_plan")

        approval = payload.get("approval") or {}
        if approval.get("analysis_allowed_now") is not False:
            raise ValueError("approval.analysis_allowed_now must be false until Pro approves analysis")
        if approval.get("required_exact_phrase") != FORMAL_ANALYSIS_APPROVAL_PHRASE:
            raise ValueError("approval.required_exact_phrase mismatch")
        if approval.get("received_exact_phrase") is not False:
            raise ValueError("approval.received_exact_phrase must be false in the plan packet")

        scope = payload.get("scope") or {}
        if scope.get("stage") != "analysis_plan_only":
            raise ValueError("scope.stage must be analysis_plan_only")
        if scope.get("scenario") != "jinan":
            raise ValueError("analysis plan is limited to Jinan")
        if scope.get("traffic_file") != "anon_3_4_jinan_real.json":
            raise ValueError("analysis plan traffic file mismatch")
        if scope.get("consumes_run_outputs") is not False:
            raise ValueError("analysis plan packet must not consume run outputs")
        if scope.get("generates_analysis_outputs") is not False:
            raise ValueError("analysis plan packet must not generate analysis outputs")
        if scope.get("generates_ranking_or_performance_table") is not False:
            raise ValueError("analysis plan packet must not generate ranking or performance tables")

        permissions = payload.get("permissions") or {}
        for key in (
            "ranking_allowed",
            "performance_table_allowed",
            "best_method_claim_allowed",
            "traffic_control_improvement_claim_allowed",
            "city_expansion_allowed",
            "seed_expansion_allowed",
            "extra_methods_allowed",
        ):
            if permissions.get(key) is not False:
                raise ValueError(f"permissions.{key} must be false")

        forbidden_outputs = set(payload.get("forbidden_outputs") or [])
        missing_forbidden = sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS - forbidden_outputs)
        if missing_forbidden:
            raise ValueError(f"forbidden_outputs missing: {missing_forbidden}")
        allowed_outputs = set(payload.get("allowed_future_outputs") or [])
        leaked = sorted(allowed_outputs & FORBIDDEN_PREFLIGHT_ARTIFACTS)
        if leaked:
            raise ValueError(f"allowed_future_outputs includes forbidden outputs: {leaked}")

        forbidden_wording = {str(value).lower() for value in payload.get("forbidden_wording") or []}
        missing_wording = sorted(REQUIRED_FORBIDDEN_WORDING - forbidden_wording)
        if missing_wording:
            raise ValueError(f"forbidden_wording missing: {missing_wording}")

        metrics = [str(metric).lower() for metric in payload.get("allowed_future_metrics") or []]
        for metric in metrics:
            if any(token in metric for token in PERFORMANCE_LIKE_METRIC_TOKENS):
                raise ValueError(f"performance-like metric is not allowed before analysis approval: {metric}")

        stats = payload.get("statistical_policy") or {}
        if stats.get("ranking") != "forbidden":
            raise ValueError("statistical_policy.ranking must be forbidden")
        if stats.get("mean_std_performance_table") != "forbidden":
            raise ValueError("statistical_policy.mean_std_performance_table must be forbidden")
        if stats.get("method_comparison_claim") != "forbidden":
            raise ValueError("statistical_policy.method_comparison_claim must be forbidden")

        methods = payload.get("method_policy") or {}
        if tuple(methods.get("ppo_methods") or ()) != FINAL_JINAN_PILOT_METHODS:
            raise ValueError("method_policy.ppo_methods mismatch")
        if tuple(methods.get("reference_only_methods") or ()) != FINAL_JINAN_REFERENCE_ONLY_METHODS:
            raise ValueError("method_policy.reference_only_methods mismatch")
        if methods.get("env_reward_role") != "diagnostic_ablation_only":
            raise ValueError("EnvReward must remain diagnostic_ablation_only")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_formal_analysis_plan(path: str | Path) -> FormalAnalysisPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalAnalysisPlan.from_dict(payload)
