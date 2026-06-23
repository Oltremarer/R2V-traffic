from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULT_METRIC_SOURCE_POLICY_PASS = "FORMAL_JINAN_RESULT_METRIC_SOURCE_POLICY_PASS"
ALLOWED_RESULT_METRIC_SOURCE_POLICY_OUTPUTS = {
    "formal_jinan_result_metric_source_policy.json",
    "formal_jinan_result_metric_source_policy.md",
}
ALLOWED_SOURCE_STATUSES = {
    "allowed_guard_metadata",
    "allowed_training_stability_only",
    "forbidden_proxy_reward_metric",
    "candidate_independent_result_metric_requires_pro_review",
    "unknown_requires_new_pro_review",
}
REQUIRED_CANDIDATE_SOURCES = {
    "metadata.json",
    "status.json",
    "train_metrics.jsonl",
    "reward_components.jsonl",
    "loss_debug.jsonl",
    "eval_metrics.jsonl",
}
REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS = {
    "env_reward",
    "potential_reward",
    "proxy_objectives_norm_tp1",
    "scalar_quality_score_t",
    "scalar_quality_score_tp1",
    "total_reward",
    "weighted_proxy_reward",
}
FORBIDDEN_VALUE_KEYS = {
    "field_values",
    "max_values",
    "mean_values",
    "metric_values",
    "min_values",
    "raw_values",
    "sample_values",
    "std_values",
    "values",
}
FORBIDDEN_OUTPUT_NAMES = {
    "best_method.csv",
    "best_method.json",
    "leaderboard.csv",
    "main_results.csv",
    "method_ranking.csv",
    "paper_results.csv",
    "performance_table.csv",
    "ranking.csv",
    "traffic_metrics.csv",
}
FORBIDDEN_PACKET_WORDING = {
    "best method",
    "beats",
    "better than",
    "leaderboard",
    "main result",
    "outperforms",
    "paper result",
    "performance gain",
    "ranked",
    "state-of-the-art",
    "traffic improvement",
    "wins",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def validate_result_metric_source_policy_packet(out_dir: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(existing - ALLOWED_RESULT_METRIC_SOURCE_POLICY_OUTPUTS)
    forbidden_outputs = sorted(existing & FORBIDDEN_OUTPUT_NAMES)
    if unexpected:
        raise ValueError(f"unexpected result-metric source policy outputs: {unexpected}")
    if forbidden_outputs:
        raise ValueError(f"forbidden result outputs present: {forbidden_outputs}")
    missing_outputs = sorted(ALLOWED_RESULT_METRIC_SOURCE_POLICY_OUTPUTS - existing)
    if missing_outputs:
        raise ValueError(f"missing result-metric source policy outputs: {missing_outputs}")

    policy = _load_json(root / "formal_jinan_result_metric_source_policy.json")
    if policy.get("report_status") != RESULT_METRIC_SOURCE_POLICY_PASS:
        raise ValueError("result-metric source policy did not pass")
    if policy.get("scope") != "result_metric_source_policy_only_no_value_reading":
        raise ValueError("result-metric source policy scope mismatch")

    leaked_keys = sorted({key for key in _walk_keys(policy) if key.lower() in FORBIDDEN_VALUE_KEYS})
    if leaked_keys:
        raise ValueError(f"result-metric source policy contains forbidden value carrier keys: {leaked_keys}")

    lock_policy = policy.get("policy")
    if not isinstance(lock_policy, dict):
        raise ValueError("result-metric source policy missing policy locks")
    for key in (
        "value_reading_allowed",
        "numeric_aggregation_allowed",
        "method_level_aggregate_allowed",
        "seed_level_table_allowed",
        "significance_testing_allowed",
        "confidence_interval_allowed",
        "formal_result_analysis_allowed",
        "formal_result_table_allowed",
    ):
        if lock_policy.get(key) is not False:
            raise ValueError(f"{key} must remain false")

    sources = policy.get("candidate_sources")
    if not isinstance(sources, dict):
        raise ValueError("result-metric source policy missing candidate_sources")
    missing_sources = sorted(REQUIRED_CANDIDATE_SOURCES - set(sources))
    if missing_sources:
        raise ValueError(f"missing candidate sources: {missing_sources}")
    for source_name, entry in sources.items():
        if entry.get("status") not in ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"invalid source status for {source_name}: {entry.get('status')}")
        if entry.get("value_reading_allowed") is not False:
            raise ValueError(f"value reading must remain false for {source_name}")
        if entry.get("numeric_aggregation_allowed") is not False:
            raise ValueError(f"numeric aggregation must remain false for {source_name}")
    if sources["reward_components.jsonl"].get("status") != "forbidden_proxy_reward_metric":
        raise ValueError("reward_components.jsonl must remain forbidden_proxy_reward_metric")
    if sources["loss_debug.jsonl"].get("status") != "allowed_training_stability_only":
        raise ValueError("loss_debug.jsonl must remain allowed_training_stability_only")
    for source_name in ("metadata.json", "status.json"):
        if sources[source_name].get("status") != "allowed_guard_metadata":
            raise ValueError(f"{source_name} must remain allowed_guard_metadata")

    fields = policy.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("result-metric source policy missing fields")
    for field, entry in fields.items():
        if entry.get("status") not in ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"invalid field status for {field}: {entry.get('status')}")
        if entry.get("value_reading_allowed") is not False:
            raise ValueError(f"value reading must remain false for {field}")
        if entry.get("numeric_aggregation_allowed") is not False:
            raise ValueError(f"numeric aggregation must remain false for {field}")
    for field in REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS & set(fields):
        if fields[field].get("status") != "forbidden_proxy_reward_metric":
            raise ValueError(f"{field} must remain forbidden_proxy_reward_metric")

    packet = (root / "formal_jinan_result_metric_source_policy.md").read_text(encoding="utf-8").lower()
    wording_hits = sorted(word for word in FORBIDDEN_PACKET_WORDING if word in packet)
    if wording_hits:
        raise ValueError(f"result-metric source policy packet contains forbidden wording: {wording_hits}")
