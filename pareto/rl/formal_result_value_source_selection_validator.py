from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pareto.rl.formal_result_metric_source_policy_validator import (
    FORBIDDEN_OUTPUT_NAMES,
    FORBIDDEN_PACKET_WORDING,
    FORBIDDEN_VALUE_KEYS,
    REQUIRED_CANDIDATE_SOURCES,
    REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS,
)


RESULT_VALUE_SOURCE_SELECTION_PASS = "FORMAL_JINAN_RESULT_VALUE_SOURCE_SELECTION_PASS"
ALLOWED_RESULT_VALUE_SOURCE_SELECTION_OUTPUTS = {
    "formal_jinan_result_value_source_selection.json",
    "formal_jinan_result_value_source_selection.md",
}
ALLOWED_SELECTION_STATUSES = {
    "guard_metadata_not_result_value_source",
    "training_stability_not_result_value_source",
    "forbidden_proxy_reward_metric",
    "candidate_independent_result_metric_requires_future_pro_review",
    "unknown_requires_new_pro_review",
}
REQUIRED_CANDIDATE_RESULT_SOURCES = {
    "train_metrics.jsonl",
    "eval_metrics.jsonl",
}
REQUIRED_FORBIDDEN_RESULT_SOURCES = {
    "reward_components.jsonl",
}
REQUIRED_TRAINING_STABILITY_ONLY_SOURCES = {
    "loss_debug.jsonl",
}
REQUIRED_GUARD_METADATA_SOURCES = {
    "metadata.json",
    "status.json",
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


def _require_false(payload: dict[str, Any], key: str, context: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"{context}.{key} must remain false")


def validate_result_value_source_selection_packet(out_dir: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(existing - ALLOWED_RESULT_VALUE_SOURCE_SELECTION_OUTPUTS)
    forbidden_outputs = sorted(existing & FORBIDDEN_OUTPUT_NAMES)
    if unexpected:
        raise ValueError(f"unexpected result-value source selection outputs: {unexpected}")
    if forbidden_outputs:
        raise ValueError(f"forbidden result outputs present: {forbidden_outputs}")
    missing_outputs = sorted(ALLOWED_RESULT_VALUE_SOURCE_SELECTION_OUTPUTS - existing)
    if missing_outputs:
        raise ValueError(f"missing result-value source selection outputs: {missing_outputs}")

    selection = _load_json(root / "formal_jinan_result_value_source_selection.json")
    if selection.get("report_status") != RESULT_VALUE_SOURCE_SELECTION_PASS:
        raise ValueError("result-value source selection did not pass")
    if selection.get("scope") != "result_value_source_selection_only_no_value_reading":
        raise ValueError("result-value source selection scope mismatch")

    leaked_keys = sorted({key for key in _walk_keys(selection) if key.lower() in FORBIDDEN_VALUE_KEYS})
    if leaked_keys:
        raise ValueError(f"result-value source selection contains forbidden value carrier keys: {leaked_keys}")

    lock_policy = selection.get("policy")
    if not isinstance(lock_policy, dict):
        raise ValueError("result-value source selection missing policy locks")
    for key in (
        "value_reading_allowed",
        "numeric_aggregation_allowed",
        "method_level_aggregate_allowed",
        "seed_level_table_allowed",
        "significance_testing_allowed",
        "confidence_interval_allowed",
        "formal_result_analysis_allowed",
        "formal_result_table_allowed",
        "result_value_sources_selected",
    ):
        _require_false(lock_policy, key, "policy")

    source_selection = selection.get("source_selection")
    if not isinstance(source_selection, dict):
        raise ValueError("result-value source selection missing source_selection")
    missing_sources = sorted(REQUIRED_CANDIDATE_SOURCES - set(source_selection))
    if missing_sources:
        raise ValueError(f"missing candidate sources: {missing_sources}")

    for source_name, entry in source_selection.items():
        if entry.get("selection_status") not in ALLOWED_SELECTION_STATUSES:
            raise ValueError(f"invalid selection status for {source_name}: {entry.get('selection_status')}")
        _require_false(entry, "value_reading_allowed", f"source_selection.{source_name}")
        _require_false(entry, "numeric_aggregation_allowed", f"source_selection.{source_name}")
        _require_false(entry, "selected_for_result_value_reading", f"source_selection.{source_name}")
        _require_false(entry, "selected_for_result_aggregation", f"source_selection.{source_name}")

    for source_name in REQUIRED_CANDIDATE_RESULT_SOURCES:
        entry = source_selection[source_name]
        if entry.get("source_policy_status") != "candidate_independent_result_metric_requires_pro_review":
            raise ValueError(f"{source_name} must remain candidate_independent_result_metric_requires_pro_review")
        if entry.get("selection_status") != "candidate_independent_result_metric_requires_future_pro_review":
            raise ValueError(f"{source_name} must not be selected as an allowed result value source")

    for source_name in REQUIRED_FORBIDDEN_RESULT_SOURCES:
        entry = source_selection[source_name]
        if entry.get("source_policy_status") != "forbidden_proxy_reward_metric":
            raise ValueError(f"{source_name} must remain forbidden_proxy_reward_metric")
        if entry.get("selection_status") != "forbidden_proxy_reward_metric":
            raise ValueError(f"{source_name} must not be selected")

    for source_name in REQUIRED_TRAINING_STABILITY_ONLY_SOURCES:
        entry = source_selection[source_name]
        if entry.get("source_policy_status") != "allowed_training_stability_only":
            raise ValueError(f"{source_name} must remain training-stability only")
        if entry.get("selection_status") != "training_stability_not_result_value_source":
            raise ValueError(f"{source_name} must not be selected as a result value source")

    for source_name in REQUIRED_GUARD_METADATA_SOURCES:
        entry = source_selection[source_name]
        if entry.get("source_policy_status") != "allowed_guard_metadata":
            raise ValueError(f"{source_name} must remain guard metadata only")
        if entry.get("selection_status") != "guard_metadata_not_result_value_source":
            raise ValueError(f"{source_name} must not be selected as a result value source")

    future_fields = selection.get("future_candidate_fields")
    if not isinstance(future_fields, dict):
        raise ValueError("result-value source selection missing future_candidate_fields")
    for field, entry in future_fields.items():
        _require_false(entry, "value_reading_allowed", f"future_candidate_fields.{field}")
        _require_false(entry, "numeric_aggregation_allowed", f"future_candidate_fields.{field}")
        if entry.get("selection_status") != "candidate_independent_result_metric_requires_future_pro_review":
            raise ValueError(f"{field} must remain a future candidate, not an allowed field")
        source_files = set(entry.get("source_files") or [])
        if not source_files <= REQUIRED_CANDIDATE_RESULT_SOURCES:
            raise ValueError(f"{field} future candidate has non-result source files: {sorted(source_files)}")

    forbidden_fields = selection.get("forbidden_proxy_reward_fields")
    if not isinstance(forbidden_fields, dict):
        raise ValueError("result-value source selection missing forbidden_proxy_reward_fields")
    missing_forbidden = sorted(REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS - set(forbidden_fields))
    if missing_forbidden:
        raise ValueError(f"missing required forbidden proxy/reward fields: {missing_forbidden}")
    for field, entry in forbidden_fields.items():
        if entry.get("selection_status") != "forbidden_proxy_reward_metric":
            raise ValueError(f"{field} must remain forbidden_proxy_reward_metric")
        _require_false(entry, "value_reading_allowed", f"forbidden_proxy_reward_fields.{field}")
        _require_false(entry, "numeric_aggregation_allowed", f"forbidden_proxy_reward_fields.{field}")

    packet = (root / "formal_jinan_result_value_source_selection.md").read_text(encoding="utf-8").lower()
    wording_hits = sorted(word for word in FORBIDDEN_PACKET_WORDING if word in packet)
    if wording_hits:
        raise ValueError(f"result-value source selection packet contains forbidden wording: {wording_hits}")
