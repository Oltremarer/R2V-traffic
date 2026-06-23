from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS


FORMAL_RESULT_ANALYSIS_PLAN_APPROVAL_PHRASE = "FORMAL JINAN RESULT-ANALYSIS PLAN GO"
PLAN_TYPE = "formal_jinan_result_analysis_plan_packet"
REQUIRED_FIELDS = {
    "plan_type",
    "approval",
    "scope",
    "allowed_inputs_now",
    "allowed_outputs_now",
    "permissions",
    "future_analysis_policy",
    "forbidden_metrics",
    "forbidden_outputs",
    "forbidden_wording",
}
REQUIRED_FORBIDDEN_METRICS = {
    "travel_time",
    "queue",
    "delay",
    "throughput",
    "waiting_time",
    "traffic metrics",
    "reward total as performance",
    "score",
    "mean/std performance",
    "improvement rate",
    "win/loss count",
}
REQUIRED_FORBIDDEN_WORDING = {
    "best method",
    "beats",
    "outperforms",
    "ranked",
    "leaderboard",
    "traffic improvement",
    "state-of-the-art",
    "better than",
    "wins",
    "performance gain",
    "main result",
    "paper result",
}
FORBIDDEN_RUN_LOG_MARKERS = {
    "checkpoint",
    "eval",
    "loss_debug",
    "metrics",
    "reward_components",
    "stderr",
    "stdout",
    "train_metrics",
    "training_checkpoint",
}
ALLOWED_PLAN_OUTPUTS = {
    "formal_jinan_result_analysis_plan.md",
    "formal_jinan_result_analysis_plan.json",
    "formal_result_analysis_plan_validator.py",
    "test_formal_result_analysis_plan_validator.py",
}


@dataclass(frozen=True)
class FormalResultAnalysisPlan:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalResultAnalysisPlan":
        plan = cls(payload=dict(payload))
        plan.validate()
        return plan

    def validate(self) -> None:
        payload = self.payload
        missing = sorted(REQUIRED_FIELDS - set(payload))
        if missing:
            raise ValueError(f"formal result-analysis plan missing fields: {missing}")
        if payload.get("plan_type") != PLAN_TYPE:
            raise ValueError(f"plan_type must be {PLAN_TYPE}")

        approval = payload.get("approval") or {}
        if approval.get("plan_creation_exact_phrase") != FORMAL_RESULT_ANALYSIS_PLAN_APPROVAL_PHRASE:
            raise ValueError("approval.plan_creation_exact_phrase mismatch")
        if approval.get("result_analysis_allowed_now") is not False:
            raise ValueError("approval.result_analysis_allowed_now must be false")
        if approval.get("received_future_result_analysis_phrase") is not False:
            raise ValueError("approval.received_future_result_analysis_phrase must be false")

        scope = payload.get("scope") or {}
        if scope.get("stage") != "result_analysis_plan_only":
            raise ValueError("scope.stage must be result_analysis_plan_only")
        if scope.get("scenario") != "jinan":
            raise ValueError("result-analysis plan is limited to Jinan")
        if scope.get("traffic_file") != "anon_3_4_jinan_real.json":
            raise ValueError("result-analysis plan traffic file mismatch")
        for key in (
            "consumes_run_logs_now",
            "generates_formal_result_values_now",
            "generates_method_ordering_now",
            "generates_result_table_now",
        ):
            if scope.get(key) is not False:
                raise ValueError(f"scope.{key} must be false")

        for input_path in payload.get("allowed_inputs_now") or []:
            self._validate_allowed_input_now(str(input_path))

        allowed_outputs = {Path(str(item)).name for item in payload.get("allowed_outputs_now") or []}
        extra_outputs = sorted(allowed_outputs - ALLOWED_PLAN_OUTPUTS)
        if extra_outputs:
            raise ValueError(f"forbidden output in allowed_outputs_now: {extra_outputs}")
        if not ALLOWED_PLAN_OUTPUTS.issubset(allowed_outputs):
            raise ValueError(f"allowed_outputs_now missing plan artifacts: {sorted(ALLOWED_PLAN_OUTPUTS - allowed_outputs)}")
        leaked_outputs = sorted(allowed_outputs & FORBIDDEN_PREFLIGHT_ARTIFACTS)
        if leaked_outputs:
            raise ValueError(f"forbidden output in allowed_outputs_now: {leaked_outputs}")

        permissions = payload.get("permissions") or {}
        for key in (
            "read_run_logs_for_result_analysis_allowed",
            "result_table_allowed",
            "method_ordering_allowed",
            "method_comparison_claim_allowed",
            "best_method_claim_allowed",
            "traffic_control_improvement_claim_allowed",
            "city_expansion_allowed",
            "seed_expansion_allowed",
            "extra_methods_allowed",
        ):
            if permissions.get(key) is not False:
                raise ValueError(f"permissions.{key} must be false")

        future = payload.get("future_analysis_policy") or {}
        if future.get("requires_new_pro_phrase_before_reading_run_logs") is not True:
            raise ValueError("future_analysis_policy.requires_new_pro_phrase_before_reading_run_logs must be true")
        if future.get("actual_log_value_extraction_now") is not False:
            raise ValueError("future_analysis_policy.actual_log_value_extraction_now must be false")
        if future.get("formal_result_table_now") is not False:
            raise ValueError("future_analysis_policy.formal_result_table_now must be false")

        forbidden_metrics = {str(item).lower() for item in payload.get("forbidden_metrics") or []}
        missing_metrics = sorted(REQUIRED_FORBIDDEN_METRICS - forbidden_metrics)
        if missing_metrics:
            raise ValueError(f"forbidden_metrics missing: {missing_metrics}")

        forbidden_outputs = {Path(str(item)).name for item in payload.get("forbidden_outputs") or []}
        missing_outputs = sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS - forbidden_outputs)
        if missing_outputs:
            raise ValueError(f"forbidden_outputs missing: {missing_outputs}")

        forbidden_wording = {str(item).lower() for item in payload.get("forbidden_wording") or []}
        missing_wording = sorted(REQUIRED_FORBIDDEN_WORDING - forbidden_wording)
        if missing_wording:
            raise ValueError(f"forbidden_wording missing: {missing_wording}")

    @staticmethod
    def _validate_allowed_input_now(input_path: str) -> None:
        lowered = input_path.lower()
        if "records/formal_jinan_3seed_guarded_20260531/seed" in lowered:
            raise ValueError(f"run log input is not allowed in plan-only stage: {input_path}")
        if lowered.startswith("records/") and any(marker in lowered for marker in FORBIDDEN_RUN_LOG_MARKERS):
            raise ValueError(f"run log input is not allowed in plan-only stage: {input_path}")
        if lowered.endswith((".csv", ".jsonl", ".pt", ".pth")) and "docs/pro_reviews" not in lowered:
            raise ValueError(f"run log input is not allowed in plan-only stage: {input_path}")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_formal_result_analysis_plan(path: str | Path) -> FormalResultAnalysisPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalResultAnalysisPlan.from_dict(payload)
