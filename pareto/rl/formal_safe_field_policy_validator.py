from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAFE_FIELD_POLICY_PASS = "FORMAL_JINAN_SAFE_FIELD_POLICY_PASS"
ALLOWED_SAFE_FIELD_OUTPUTS = {
    "formal_jinan_safe_field_policy.json",
    "formal_jinan_safe_field_policy.md",
}
ALLOWED_STATUSES = {
    "allowed_guard_metadata",
    "allowed_training_stability_sanity_only",
    "forbidden_result_metric",
    "forbidden_proxy_result_metric",
    "unknown_requires_new_pro_review",
}
REQUIRED_FORBIDDEN_PROXY_FIELDS = {
    "env_reward",
    "potential_reward",
    "proxy_objectives_norm_tp1",
    "scalar_quality_score_t",
    "scalar_quality_score_tp1",
    "total_reward",
    "weighted_proxy_reward",
}
REQUIRED_GUARD_METADATA_FIELDS = {
    "env_reward_info_source",
    "env_reward_source",
    "reward_adapter_semantics",
}
FORBIDDEN_PACKET_WORDING = {
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory_fields(path: Path) -> set[str]:
    inventory = _load_json(path)
    return set((inventory.get("field_categories") or {}).keys())


def validate_safe_field_policy_packet(out_dir: str | Path, *, inventory_json: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(existing - ALLOWED_SAFE_FIELD_OUTPUTS)
    if unexpected:
        raise ValueError(f"unexpected safe-field policy outputs: {unexpected}")
    missing_outputs = sorted(ALLOWED_SAFE_FIELD_OUTPUTS - existing)
    if missing_outputs:
        raise ValueError(f"missing safe-field policy outputs: {missing_outputs}")

    policy = _load_json(root / "formal_jinan_safe_field_policy.json")
    if policy.get("report_status") != SAFE_FIELD_POLICY_PASS:
        raise ValueError("safe-field policy did not pass")
    if policy.get("scope") != "safe_field_policy_only_no_numeric_aggregation":
        raise ValueError("safe-field policy scope mismatch")
    fields = policy.get("fields") or {}
    inventory_fields = _inventory_fields(Path(inventory_json))
    missing_fields = sorted(inventory_fields - set(fields))
    if missing_fields:
        raise ValueError(f"missing policy fields: {missing_fields}")

    for field, entry in fields.items():
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status for {field}: {status}")
        if entry.get("value_reading_allowed") is not False:
            raise ValueError(f"value reading must remain false for {field}")
        if entry.get("numeric_aggregation_allowed") is not False:
            raise ValueError(f"numeric aggregation must remain false for {field}")

    for field in REQUIRED_FORBIDDEN_PROXY_FIELDS & inventory_fields:
        if fields[field].get("status") != "forbidden_proxy_result_metric":
            raise ValueError(f"{field} must be forbidden_proxy_result_metric")
    for field in REQUIRED_GUARD_METADATA_FIELDS & inventory_fields:
        if fields[field].get("status") != "allowed_guard_metadata":
            raise ValueError(f"{field} must be allowed_guard_metadata")

    packet = (root / "formal_jinan_safe_field_policy.md").read_text(encoding="utf-8").lower()
    wording_hits = sorted(word for word in FORBIDDEN_PACKET_WORDING if word in packet)
    if wording_hits:
        raise ValueError(f"safe-field policy packet contains forbidden wording: {wording_hits}")
