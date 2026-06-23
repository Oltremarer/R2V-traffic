from __future__ import annotations

from typing import Any

from pareto.eval.paper_metric_schema import REQUIRED_METRIC_KEYS, validate_metric_value
from pareto.eval.paper_representation_artifact_sources import representation_artifact_blockers


IMPLEMENTED_COMMON_EVAL_METRICS = {
    "average_travel_time",
    "average_waiting_time",
    "average_queue_length",
    "ttc_p10",
    "ttc_p50",
    "ttc_violation_rate",
    "harsh_brake_rate",
    "waiting_time_imbalance",
    "phase_switch_rate",
    "oscillation_index",
    "hypervolume",
    "coverage",
    "dominance_violation",
    "utility",
    "alignment",
    "monotonicity",
    "smoothness",
    "calibration_error",
    "heldout_preference_utility",
    "shifted_traffic_att",
}
REPRESENTATION_DIAGNOSTIC_METRICS = {"obj_acc", "pref_acc", "rev_acc", "dpr"}
METRIC_SOURCE_POLICY: dict[str, dict[str, Any]] = {
    key: {
        "status": "implemented" if key in IMPLEMENTED_COMMON_EVAL_METRICS else "missing_blocker",
        "allowed_sources": ["eval_metrics.json", "eval_metrics.jsonl"] if key in IMPLEMENTED_COMMON_EVAL_METRICS else ["representation_diagnostics.json"],
        "forbidden_sources": ["train_metrics.jsonl"],
        "blocker": None if key in IMPLEMENTED_COMMON_EVAL_METRICS else "representation diagnostic artifact source is not implemented",
    }
    for key in REQUIRED_METRIC_KEYS
}


def build_metric_source_policy(
    *,
    representation_artifact_audit: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    policy = {metric: dict(row) for metric, row in METRIC_SOURCE_POLICY.items()}
    if representation_artifact_audit is None:
        return validate_metric_source_policy(policy)
    blockers = representation_artifact_blockers(representation_artifact_audit)
    if blockers:
        for metric in REPRESENTATION_DIAGNOSTIC_METRICS:
            policy[metric] = dict(
                policy[metric],
                status="missing_blocker",
                allowed_sources=["representation_formal_gate_packet.json"],
                blocker="representation artifact source coverage is incomplete",
                source_blockers=blockers,
            )
        return validate_metric_source_policy(policy)
    for metric in REPRESENTATION_DIAGNOSTIC_METRICS:
        policy[metric] = {
            "status": "implemented",
            "allowed_sources": ["representation_formal_gate_packet.json"],
            "forbidden_sources": ["train_metrics.jsonl", "eval_metrics.json", "eval_metrics.jsonl"],
            "blocker": None,
        }
    return validate_metric_source_policy(policy)


def validate_metric_source_policy(policy: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    missing = sorted(set(REQUIRED_METRIC_KEYS) - set(policy))
    if missing:
        raise ValueError(f"missing metric source policy: {missing}")
    for metric, row in policy.items():
        if metric not in REQUIRED_METRIC_KEYS:
            raise ValueError(f"unknown metric source policy: {metric}")
        if row.get("status") not in {"implemented", "missing_blocker"}:
            raise ValueError(f"unknown metric source status for {metric}: {row.get('status')}")
        if row.get("status") == "implemented" and not row.get("allowed_sources"):
            raise ValueError(f"implemented metric {metric} must declare allowed sources")
    return {metric: dict(row) for metric, row in policy.items()}


def metric_source_blockers(policy: dict[str, dict[str, Any]]) -> list[str]:
    validated = validate_metric_source_policy(policy)
    return [
        f"{metric}: {row.get('blocker') or 'source missing'}"
        for metric, row in validated.items()
        if row.get("status") == "missing_blocker"
    ]


def validate_metric_source(
    metric_key: str,
    source_file: str,
    value: float | int,
    *,
    policy: dict[str, dict[str, Any]] = METRIC_SOURCE_POLICY,
) -> float:
    validated = validate_metric_source_policy(policy)
    if metric_key not in validated:
        raise ValueError(f"unknown metric source: {metric_key}")
    row = validated[metric_key]
    if row.get("status") != "implemented":
        raise ValueError(f"metric {metric_key} source is not implemented")
    if source_file in set(row.get("forbidden_sources") or []):
        raise ValueError(f"{source_file} is not an approved source for {metric_key}")
    if source_file not in set(row.get("allowed_sources") or []):
        raise ValueError(f"{source_file} is not an approved source for {metric_key}")
    return validate_metric_value(metric_key, value)
