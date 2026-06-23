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
)
from pareto.rl.formal_result_value_source_selection_validator import (
    RESULT_VALUE_SOURCE_SELECTION_PASS,
    validate_result_value_source_selection_packet,
)


FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE = "FORMAL JINAN RESULT-VALUE SOURCE SELECTION GO"
RESULT_METRIC_SOURCE_POLICY_PASS = "FORMAL_JINAN_RESULT_METRIC_SOURCE_POLICY_PASS"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selection_status(source_policy_status: str) -> tuple[str, str]:
    if source_policy_status == "allowed_guard_metadata":
        return "guard_metadata_not_result_value_source", "guard metadata may support guards only"
    if source_policy_status == "allowed_training_stability_only":
        return "training_stability_not_result_value_source", "training-stability source is not a result value source"
    if source_policy_status == "forbidden_proxy_reward_metric":
        return "forbidden_proxy_reward_metric", "reward/proxy source remains forbidden"
    if source_policy_status == "candidate_independent_result_metric_requires_pro_review":
        return (
            "candidate_independent_result_metric_requires_future_pro_review",
            "candidate source name only; value reading requires a future Pro review",
        )
    return "unknown_requires_new_pro_review", "unknown source requires a future Pro review"


def _field_source_files(field_entry: dict[str, Any]) -> list[str]:
    return sorted(str(item) for item in (field_entry.get("source_files") or []))


def _is_future_candidate_field(field_entry: dict[str, Any]) -> bool:
    if field_entry.get("status") != "candidate_independent_result_metric_requires_pro_review":
        return False
    return any(source in {"train_metrics.jsonl", "eval_metrics.jsonl"} for source in _field_source_files(field_entry))


def generate_result_value_source_selection(
    *,
    result_metric_source_policy_json: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    if approval_phrase != FORMAL_RESULT_VALUE_SOURCE_SELECTION_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for result-value source selection")
    source_policy = _read_json(result_metric_source_policy_json)
    if source_policy.get("report_status") != RESULT_METRIC_SOURCE_POLICY_PASS:
        raise ValueError("result-metric source policy must pass before source selection")

    source_selection: dict[str, dict[str, Any]] = {}
    candidate_sources = source_policy.get("candidate_sources") or {}
    for source_name in sorted(REQUIRED_CANDIDATE_SOURCES | set(candidate_sources)):
        source_entry = candidate_sources.get(source_name) or {}
        source_policy_status = str(source_entry.get("status") or "unknown_requires_new_pro_review")
        selection_status, rationale = _selection_status(source_policy_status)
        source_selection[source_name] = {
            "source_policy_status": source_policy_status,
            "selection_status": selection_status,
            "present_in_inventory": bool(source_entry.get("present_in_inventory", False)),
            "schema_key_names_recorded": bool(source_entry.get("schema_key_names_recorded", False)),
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "selected_for_result_value_reading": False,
            "selected_for_result_aggregation": False,
            "rationale": rationale,
        }

    future_candidate_fields: dict[str, dict[str, Any]] = {}
    forbidden_proxy_reward_fields: dict[str, dict[str, Any]] = {}
    source_policy_fields = source_policy.get("fields") or {}
    for field, field_entry in sorted(source_policy_fields.items()):
        if field in REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS or field_entry.get("status") == "forbidden_proxy_reward_metric":
            forbidden_proxy_reward_fields[field] = {
                "source_policy_status": field_entry.get("status"),
                "selection_status": "forbidden_proxy_reward_metric",
                "source_files": _field_source_files(field_entry),
                "value_reading_allowed": False,
                "numeric_aggregation_allowed": False,
                "rationale": "proxy/reward/objective field remains forbidden",
            }
            continue
        if _is_future_candidate_field(field_entry):
            future_candidate_fields[field] = {
                "source_policy_status": field_entry.get("status"),
                "selection_status": "candidate_independent_result_metric_requires_future_pro_review",
                "source_files": [
                    source
                    for source in _field_source_files(field_entry)
                    if source in {"train_metrics.jsonl", "eval_metrics.jsonl"}
                ],
                "value_reading_allowed": False,
                "numeric_aggregation_allowed": False,
                "selected_for_result_value_reading": False,
                "selected_for_result_aggregation": False,
                "rationale": "future independent-result candidate field name only; values are still locked",
            }

    for field in sorted(REQUIRED_FORBIDDEN_PROXY_REWARD_FIELDS - set(forbidden_proxy_reward_fields)):
        forbidden_proxy_reward_fields[field] = {
            "source_policy_status": "forbidden_proxy_reward_metric",
            "selection_status": "forbidden_proxy_reward_metric",
            "source_files": [],
            "value_reading_allowed": False,
            "numeric_aggregation_allowed": False,
            "rationale": "required forbidden proxy/reward field from Pro gate",
        }

    report = {
        "report_status": RESULT_VALUE_SOURCE_SELECTION_PASS,
        "scope": "result_value_source_selection_only_no_value_reading",
        "approval_phrase_verified": True,
        "source_artifacts": {
            "result_metric_source_policy_status": source_policy.get("report_status"),
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
            "result_value_sources_selected": False,
            "requires_future_pro_review_before_any_result_value_reading": True,
        },
        "source_selection": source_selection,
        "future_candidate_fields": future_candidate_fields,
        "forbidden_proxy_reward_fields": forbidden_proxy_reward_fields,
        "notes": [
            "This packet selects no result-value source.",
            "train_metrics.jsonl and eval_metrics.jsonl remain future candidates only.",
            "reward_components.jsonl and proxy/reward/objective fields remain forbidden.",
        ],
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "formal_jinan_result_value_source_selection.json", report)
    _write_packet(out_path / "formal_jinan_result_value_source_selection.md", report)
    validate_result_value_source_selection_packet(out_path)
    return report


def _write_packet(path: Path, report: dict[str, Any]) -> None:
    source_lines = [
        f"- `{name}`: `{entry['selection_status']}`"
        for name, entry in sorted(report["source_selection"].items())
    ]
    future_fields = ", ".join(f"`{name}`" for name in sorted(report["future_candidate_fields"])) or "none"
    lines = [
        "# Formal Jinan Result-Value Source Selection",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "Scope: source selection guard only. No value reading or numeric aggregation is allowed.",
        "",
        "Source selection statuses:",
        *source_lines,
        "",
        f"Future candidate field names: {future_fields}",
        "",
        "Policy locks:",
        "- no result-value source is selected",
        "- train/eval metric sources remain future candidates only",
        "- reward/proxy/objective fields remain forbidden",
        "- method aggregate, seed table, statistical test, confidence interval, and result table remain forbidden",
        "",
        "No value reading, method comparison, formal table, or traffic-control claim is produced.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_metric_source_policy_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--approval_phrase", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_result_value_source_selection(
        result_metric_source_policy_json=args.result_metric_source_policy_json,
        out_dir=args.out_dir,
        approval_phrase=args.approval_phrase,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
