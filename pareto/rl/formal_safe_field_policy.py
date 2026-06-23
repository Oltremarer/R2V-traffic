#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.rl.formal_safe_field_policy_validator import (
    REQUIRED_FORBIDDEN_PROXY_FIELDS,
    REQUIRED_GUARD_METADATA_FIELDS,
    SAFE_FIELD_POLICY_PASS,
    validate_safe_field_policy_packet,
)


FORMAL_SAFE_FIELD_POLICY_APPROVAL_PHRASE = "FORMAL JINAN SAFE-FIELD POLICY GO"
TRAINING_STABILITY_FIELDS = {
    "approx_kl",
    "clip_fraction",
    "entropy_bonus",
    "grad_norm",
    "loss_debug_finite",
    "policy_loss",
    "ratio_max",
    "ratio_mean",
    "ratio_min",
    "total_loss",
    "value_loss",
}
GUARD_METADATA_FIELDS = {
    "cityflow_seed",
    "env_reward_info_source",
    "env_reward_source",
    "episodes",
    "formal_experiment",
    "formal_jinan_3seed_execution",
    "method",
    "method_display_name",
    "method_ranking_allowed",
    "model_seed",
    "performance_claim",
    "performance_table_allowed",
    "policy_seed",
    "pro_approval_phrase_verified",
    "reference_only_methods",
    "reward_adapter_semantics",
    "scenario",
    "status",
    "traffic_file",
}
PROXY_OR_REWARD_MARKERS = (
    "objective",
    "potential",
    "proxy",
    "quality_score",
    "reward",
    "score",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_field(field: str, source_category: str) -> tuple[str, str]:
    lowered = field.lower()
    if field in REQUIRED_FORBIDDEN_PROXY_FIELDS:
        return "forbidden_proxy_result_metric", "required forbidden proxy/result field from Pro review"
    if field in REQUIRED_GUARD_METADATA_FIELDS or field in GUARD_METADATA_FIELDS:
        return "allowed_guard_metadata", "guard metadata only; no method comparison"
    if field in TRAINING_STABILITY_FIELDS or source_category == "training_stability":
        return "allowed_training_stability_sanity_only", "training stability sanity only"
    if any(marker in lowered for marker in PROXY_OR_REWARD_MARKERS):
        return "forbidden_proxy_result_metric", "proxy/reward/score-like field; value reading forbidden"
    if source_category == "traffic_like_forbidden":
        return "forbidden_result_metric", "traffic-like field; value reading forbidden"
    if source_category == "diagnostic":
        return "allowed_guard_metadata", "diagnostic guard metadata only"
    return "unknown_requires_new_pro_review", "requires a future Pro review before value reading"


def generate_safe_field_policy(
    *,
    inventory_json: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    if approval_phrase != FORMAL_SAFE_FIELD_POLICY_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for formal safe-field policy")
    inventory_path = Path(inventory_json)
    inventory = _read_json(inventory_path)
    field_categories = inventory.get("field_categories") or {}
    fields: dict[str, dict[str, Any]] = {}
    for field, source_category in sorted(field_categories.items()):
        status, rationale = _classify_field(str(field), str(source_category))
        fields[str(field)] = {
            "source_category": str(source_category),
            "status": status,
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "rationale": rationale,
        }
    status_counts: dict[str, int] = {}
    for entry in fields.values():
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

    report = {
        "report_status": SAFE_FIELD_POLICY_PASS,
        "scope": "safe_field_policy_only_no_numeric_aggregation",
        "approval_phrase_verified": True,
        "source_inventory_status": inventory.get("report_status"),
        "source_field_count": len(field_categories),
        "status_counts": dict(sorted(status_counts.items())),
        "fields": fields,
        "policy": {
            "method_comparison_allowed": False,
            "formal_result_values_allowed": False,
            "formal_result_table_allowed": False,
            "numeric_aggregation_allowed": False,
            "requires_new_pro_phrase_before_any_value_reading": True,
        },
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "formal_jinan_safe_field_policy.json", report)
    _write_packet(out_path / "formal_jinan_safe_field_policy.md", report)
    validate_safe_field_policy_packet(out_path, inventory_json=inventory_path)
    return report


def _write_packet(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Formal Jinan Safe-Field Policy",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "Scope: field policy only. No numeric aggregation or value reading is allowed by this packet.",
        "",
        "Status counts:",
    ]
    for status, count in sorted(report["status_counts"].items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "Required policy locks:",
            "- reward/proxy fields remain forbidden for value reading",
            "- source and adapter semantics are guard metadata only",
            "- loss, KL, ratio, entropy, and gradient fields are training-stability sanity only",
            "- unknown fields require a new Pro review before value reading",
            "",
            "No formal result values, method comparison, formal table, or traffic-control claim is produced.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--approval_phrase", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_safe_field_policy(
        inventory_json=args.inventory_json,
        out_dir=args.out_dir,
        approval_phrase=args.approval_phrase,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
