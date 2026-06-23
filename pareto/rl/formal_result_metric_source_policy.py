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
from pareto.rl.formal_result_metric_source_policy_validator import (
    REQUIRED_CANDIDATE_SOURCES,
    REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS,
    RESULT_METRIC_SOURCE_POLICY_PASS,
    validate_result_metric_source_policy_packet,
)


FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE = "FORMAL JINAN RESULT-METRIC SOURCE POLICY GO"
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
    "episode",
    "episodes",
    "formal_experiment",
    "formal_jinan_3seed_execution",
    "method",
    "method_display_name",
    "model_seed",
    "policy_seed",
    "scenario",
    "seed",
    "status",
    "step",
    "traffic_file",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_status(source_name: str) -> tuple[str, str]:
    if source_name in {"metadata.json", "status.json"}:
        return "allowed_guard_metadata", "guard metadata source only; no result value reading"
    if source_name == "loss_debug.jsonl":
        return "allowed_training_stability_only", "training stability guard source only"
    if source_name == "reward_components.jsonl":
        return "forbidden_proxy_reward_metric", "reward/proxy source; value reading remains forbidden"
    if source_name in {"train_metrics.jsonl", "eval_metrics.jsonl"}:
        return (
            "candidate_independent_result_metric_requires_pro_review",
            "candidate source name only; values require a future Pro review",
        )
    return "unknown_requires_new_pro_review", "unknown source requires a future Pro review"


def _safe_status_to_source_status(status: str) -> str:
    if status == "forbidden_proxy_result_metric":
        return "forbidden_proxy_reward_metric"
    if status == "allowed_training_stability_sanity_only":
        return "allowed_training_stability_only"
    return status


def _field_status(field: str, safe_status: str | None, source_names: set[str]) -> tuple[str, str]:
    if field in REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS:
        return "forbidden_proxy_reward_metric", "required forbidden proxy/reward field from Pro gate"
    if safe_status:
        mapped = _safe_status_to_source_status(safe_status)
        if mapped in {"allowed_guard_metadata", "allowed_training_stability_only", "forbidden_proxy_reward_metric"}:
            return mapped, "inherited from reviewed safe-field policy"
    if field in TRAINING_STABILITY_FIELDS or source_names == {"loss_debug.jsonl"}:
        return "allowed_training_stability_only", "training stability field only"
    if field in GUARD_METADATA_FIELDS:
        return "allowed_guard_metadata", "guard metadata field only"
    if any(source in {"train_metrics.jsonl", "eval_metrics.jsonl"} for source in source_names):
        return (
            "candidate_independent_result_metric_requires_pro_review",
            "candidate result field name only; values require a future Pro review",
        )
    return "unknown_requires_new_pro_review", "requires a future Pro review before any value reading"


def _inventory_field_sources(inventory: dict[str, Any]) -> dict[str, set[str]]:
    field_sources: dict[str, set[str]] = {}
    for source_name, summary in (inventory.get("files") or {}).items():
        for key in summary.get("keys") or []:
            field_sources.setdefault(str(key), set()).add(str(source_name))
    return field_sources


def generate_result_metric_source_policy(
    *,
    inventory_json: str | Path,
    safe_field_policy_json: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    if approval_phrase != FORMAL_RESULT_METRIC_SOURCE_POLICY_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for result-metric source policy")
    inventory = _read_json(inventory_json)
    safe_policy = _read_json(safe_field_policy_json)
    safe_fields = safe_policy.get("fields") or {}
    inventory_files = inventory.get("files") or {}
    field_sources = _inventory_field_sources(inventory)

    candidate_sources: dict[str, dict[str, Any]] = {}
    for source_name in sorted(REQUIRED_CANDIDATE_SOURCES | set(inventory_files)):
        status, rationale = _source_status(source_name)
        candidate_sources[source_name] = {
            "status": status,
            "present_in_inventory": source_name in inventory_files,
            "schema_key_names_recorded": bool((inventory_files.get(source_name) or {}).get("keys")),
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "rationale": rationale,
        }

    fields: dict[str, dict[str, Any]] = {}
    for field in sorted(set(field_sources) | set(safe_fields)):
        safe_status = None
        if isinstance(safe_fields.get(field), dict):
            safe_status = safe_fields[field].get("status")
        status, rationale = _field_status(field, safe_status, field_sources.get(field, set()))
        fields[field] = {
            "status": status,
            "source_files": sorted(field_sources.get(field, set())),
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "rationale": rationale,
        }

    report = {
        "report_status": RESULT_METRIC_SOURCE_POLICY_PASS,
        "scope": "result_metric_source_policy_only_no_value_reading",
        "approval_phrase_verified": True,
        "source_artifacts": {
            "inventory_status": inventory.get("report_status"),
            "safe_field_policy_status": safe_policy.get("report_status"),
        },
        "policy": {
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "method_level_aggregate_allowed": False,
            "seed_level_table_allowed": False,
            "significance_testing_allowed": False,
            "confidence_interval_allowed": False,
            "formal_result_analysis_allowed": False,
            "formal_result_table_allowed": False,
            "requires_future_pro_review_before_any_result_value_reading": True,
        },
        "candidate_sources": candidate_sources,
        "fields": fields,
        "notes": [
            "This packet is source-policy only and records file/key eligibility, not values.",
            "Reward/proxy/objective values remain forbidden.",
            "Candidate independent result metrics require a future Pro review before value reading.",
        ],
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "formal_jinan_result_metric_source_policy.json", report)
    _write_packet(out_path / "formal_jinan_result_metric_source_policy.md", report)
    validate_result_metric_source_policy_packet(out_path)
    return report


def _write_packet(path: Path, report: dict[str, Any]) -> None:
    source_lines = [
        f"- `{name}`: `{entry['status']}`"
        for name, entry in sorted(report["candidate_sources"].items())
    ]
    lines = [
        "# Formal Jinan Result-Metric Source Policy",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "Scope: source policy only. No result value reading or numeric aggregation is allowed by this packet.",
        "",
        "Candidate source statuses:",
        *source_lines,
        "",
        "Policy locks:",
        "- reward/proxy/objective value reading remains forbidden",
        "- candidate independent result metrics require a future Pro review",
        "- method-level aggregate, seed-level table, statistical test, confidence interval, and result table remain forbidden",
        "",
        "No formal result values, method comparison, formal table, or traffic-control claim is produced.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_json", required=True)
    parser.add_argument("--safe_field_policy_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--approval_phrase", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_result_metric_source_policy(
        inventory_json=args.inventory_json,
        safe_field_policy_json=args.safe_field_policy_json,
        out_dir=args.out_dir,
        approval_phrase=args.approval_phrase,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
