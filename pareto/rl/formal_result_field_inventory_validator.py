from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIELD_INVENTORY_PASS = "FORMAL_JINAN_RESULT_FIELD_INVENTORY_PASS"
ALLOWED_FIELD_INVENTORY_OUTPUTS = {
    "formal_jinan_result_field_inventory.json",
    "formal_jinan_result_field_inventory_packet.md",
}
FORBIDDEN_VALUE_KEYS = {
    "metric_values",
    "numeric_values",
    "raw_values",
    "sample_metric_values",
    "sample_values",
    "values",
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


def validate_field_inventory_packet(out_dir: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(existing - ALLOWED_FIELD_INVENTORY_OUTPUTS)
    if unexpected:
        raise ValueError(f"unexpected field inventory outputs: {unexpected}")
    missing = sorted(ALLOWED_FIELD_INVENTORY_OUTPUTS - existing)
    if missing:
        raise ValueError(f"missing field inventory outputs: {missing}")

    inventory = json.loads((root / "formal_jinan_result_field_inventory.json").read_text(encoding="utf-8"))
    if inventory.get("report_status") != FIELD_INVENTORY_PASS:
        raise ValueError("field inventory did not pass")
    if inventory.get("scope") != "field_inventory_only_no_metric_values":
        raise ValueError("field inventory scope mismatch")
    leaked_keys = sorted({key for key in _walk_keys(inventory) if key.lower() in FORBIDDEN_VALUE_KEYS})
    if leaked_keys:
        raise ValueError(f"field inventory contains forbidden value carrier keys: {leaked_keys}")

    packet = (root / "formal_jinan_result_field_inventory_packet.md").read_text(encoding="utf-8").lower()
    wording_hits = sorted(word for word in FORBIDDEN_PACKET_WORDING if word in packet)
    if wording_hits:
        raise ValueError(f"field inventory packet contains forbidden wording: {wording_hits}")
