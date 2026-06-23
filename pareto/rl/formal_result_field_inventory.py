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
from pareto.rl.formal_result_field_inventory_validator import (
    FIELD_INVENTORY_PASS,
    validate_field_inventory_packet,
)


FORMAL_RESULT_FIELD_INVENTORY_APPROVAL_PHRASE = "FORMAL JINAN RESULT-FIELD INVENTORY GO"
ALLOWED_RUN_LOG_FILES = (
    "metadata.json",
    "status.json",
    "train_metrics.jsonl",
    "reward_components.jsonl",
    "loss_debug.jsonl",
)
TRAFFIC_LIKE_MARKERS = (
    "avg_delay",
    "avg_queue",
    "avg_travel_time",
    "delay",
    "improvement",
    "mean_reward",
    "pressure",
    "queue",
    "return",
    "reward",
    "score",
    "throughput",
    "total_reward",
    "total_travel_time",
    "traffic",
    "travel_time",
    "waiting",
    "win",
)
TRAINING_STABILITY_MARKERS = (
    "entropy",
    "grad",
    "kl",
    "loss",
    "lr",
    "policy_update",
)
DIAGNOSTIC_MARKERS = (
    "cityflow_seed",
    "done",
    "episode",
    "method",
    "model_seed",
    "policy_seed",
    "seed",
    "status",
    "step",
    "traffic_file",
)
REWARD_COMPONENT_MARKERS = (
    "component",
    "source",
)


def _read_json_shape(path: Path) -> tuple[int, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 1, {"<root>": type(payload).__name__}
    return 1, {str(key): _type_shape(value) for key, value in payload.items()}


def _read_jsonl_shape(path: Path) -> tuple[int, dict[str, str]]:
    row_count = 0
    shapes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row_count += 1
        payload = json.loads(line)
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_str = str(key)
                shape = _type_shape(value)
                prior = shapes.get(key_str)
                if prior is None:
                    shapes[key_str] = shape
                elif prior != shape:
                    shapes[key_str] = "mixed"
        else:
            shapes["<root>"] = type(payload).__name__
    return row_count, shapes


def _type_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _category_for_key(key: str) -> str:
    lowered = key.lower()
    if any(marker in lowered for marker in TRAFFIC_LIKE_MARKERS):
        return "traffic_like_forbidden"
    if any(marker in lowered for marker in TRAINING_STABILITY_MARKERS):
        return "training_stability"
    if any(marker in lowered for marker in REWARD_COMPONENT_MARKERS):
        return "reward_component"
    if any(marker in lowered for marker in DIAGNOSTIC_MARKERS):
        return "diagnostic"
    return "unknown_requires_review"


def _run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("seed*/*") if path.is_dir())


def _merge_file_shape(summary: dict[str, Any], *, run_dir: Path, filename: str, row_count: int, shape: dict[str, str]) -> None:
    summary["present_run_count"] += 1
    summary["row_count"] += row_count
    summary["runs_with_file"].append(str(run_dir.relative_to(run_dir.parents[1])))
    for key, value_shape in shape.items():
        summary["keys"].add(key)
        prior = summary["type_shapes"].get(key)
        if prior is None:
            summary["type_shapes"][key] = value_shape
        elif prior != value_shape:
            summary["type_shapes"][key] = "mixed"


def run_formal_result_field_inventory(
    *,
    root: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    if approval_phrase != FORMAL_RESULT_FIELD_INVENTORY_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for formal result-field inventory")
    root_path = Path(root)
    out_path = Path(out_dir)
    run_dirs = _run_dirs(root_path)

    files: dict[str, dict[str, Any]] = {
        filename: {
            "present_run_count": 0,
            "missing_run_count": 0,
            "row_count": 0,
            "keys": set(),
            "type_shapes": {},
            "runs_with_file": [],
        }
        for filename in ALLOWED_RUN_LOG_FILES
    }
    field_categories: dict[str, str] = {}

    for run_dir in run_dirs:
        for filename in ALLOWED_RUN_LOG_FILES:
            path = run_dir / filename
            if not path.exists():
                files[filename]["missing_run_count"] += 1
                continue
            if filename.endswith(".jsonl"):
                row_count, shape = _read_jsonl_shape(path)
            else:
                row_count, shape = _read_json_shape(path)
            _merge_file_shape(files[filename], run_dir=run_dir, filename=filename, row_count=row_count, shape=shape)
            for key in shape:
                existing = field_categories.get(key)
                category = _category_for_key(key)
                if existing is None:
                    field_categories[key] = category
                elif existing != category:
                    field_categories[key] = "mixed_requires_review"

    serializable_files: dict[str, dict[str, Any]] = {}
    for filename, summary in files.items():
        serializable_files[filename] = {
            "present_run_count": summary["present_run_count"],
            "missing_run_count": summary["missing_run_count"],
            "row_count": summary["row_count"],
            "keys": sorted(summary["keys"]),
            "type_shapes": dict(sorted(summary["type_shapes"].items())),
            "runs_with_file": sorted(summary["runs_with_file"]),
        }

    report = {
        "report_status": FIELD_INVENTORY_PASS,
        "scope": "field_inventory_only_no_metric_values",
        "approval_phrase_verified": True,
        "run_count": len(run_dirs),
        "allowed_run_log_files": list(ALLOWED_RUN_LOG_FILES),
        "files": serializable_files,
        "field_categories": dict(sorted(field_categories.items())),
        "notes": [
            "Only file existence, row counts, key names, and JSON type shapes are reported.",
            "Metric values are not written to this inventory.",
        ],
    }

    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "formal_jinan_result_field_inventory.json", report)
    _write_packet(out_path / "formal_jinan_result_field_inventory_packet.md", report)
    validate_field_inventory_packet(out_path)
    return report


def _write_packet(path: Path, report: dict[str, Any]) -> None:
    file_lines = [
        f"- `{filename}`: present runs `{summary['present_run_count']}`, rows `{summary['row_count']}`, keys `{len(summary['keys'])}`"
        for filename, summary in sorted(report["files"].items())
    ]
    category_counts: dict[str, int] = {}
    for category in report["field_categories"].values():
        category_counts[category] = category_counts.get(category, 0) + 1
    category_lines = [f"- `{category}`: `{count}`" for category, count in sorted(category_counts.items())]
    lines = [
        "# Formal Jinan Result-Field Inventory Packet",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "Scope: field inventory only. This packet reports file existence, row counts, key names, and JSON type shapes.",
        "",
        "Run-log files inspected for schema only:",
        *file_lines,
        "",
        "Field category counts:",
        *category_lines,
        "",
        "No metric values, method comparison, formal table, or traffic-control claim is produced.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--approval_phrase", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_formal_result_field_inventory(
        root=args.root,
        out_dir=args.out_dir,
        approval_phrase=args.approval_phrase,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
