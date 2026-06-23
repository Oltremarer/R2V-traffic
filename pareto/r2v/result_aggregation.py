from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PERFORMANCE_METRICS = (
    "average_travel_time",
    "queue_length",
    "delay",
    "throughput",
    "reward",
)
PERFORMANCE_METRIC_ALIASES = {
    "average_travel_time": ("average_travel_time", "test_avg_travel_time_over", "test_avg_travel_time", "avg_travel_time"),
    "queue_length": ("queue_length", "mean_queue_length", "average_queue_length", "test_avg_queue_len_over", "test_avg_queue_len", "avg_queue"),
    "delay": ("delay", "average_waiting_time", "test_avg_waiting_time_over", "test_avg_waiting_time", "avg_delay"),
    "throughput": ("throughput", "completed_vehicle_count", "completed_vehicles"),
    "reward": ("reward", "test_reward_over", "test_reward", "total_reward"),
}


def aggregate_r2v_results(
    *,
    performance_paths: Iterable[str | Path],
    integrity_paths: Iterable[str | Path],
) -> dict[str, Any]:
    performance_path_list = tuple(performance_paths)
    integrity_path_list = tuple(integrity_paths)
    performance_rows = _load_json_rows(performance_path_list)
    integrity_rows = _load_json_documents(integrity_path_list)
    metric_table = _aggregate_performance(performance_rows)
    integrity_summary = _aggregate_integrity(integrity_rows)
    return {
        "schema_version": "r2v-traffic-result-aggregation-v1",
        "performance": metric_table,
        "integrity": integrity_summary,
        "input_artifacts": {
            "performance": [_artifact_metadata(path) for path in performance_path_list],
            "integrity": [_artifact_metadata(path) for path in integrity_path_list],
        },
        "claim_boundary": (
            "performance metrics are aggregated separately from R2V integrity/status artifacts"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate R2V-Traffic performance and integrity artifacts separately.")
    parser.add_argument("--performance_path", action="append", default=[], help="Performance JSON/JSONL path. Repeatable.")
    parser.add_argument("--integrity_path", action="append", default=[], help="Integrity/status JSON path. Repeatable.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate_r2v_results(
        performance_paths=args.performance_path,
        integrity_paths=args.integrity_path,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(output_text, end="")


def _load_json_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(str(path_obj))
        if path_obj.suffix == ".jsonl":
            for line in path_obj.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend(dict(row) for row in payload)
            elif isinstance(payload, dict):
                rows.append(payload)
            else:
                raise ValueError(f"unsupported JSON payload in {path_obj}")
    return rows


def _load_json_documents(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(str(path_obj))
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"integrity artifact must be a JSON object: {path_obj}")
        docs.append(payload)
    return docs


def _artifact_metadata(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    return {
        "path": str(path_obj),
        "sha256": _sha256_file(path_obj),
        "size_bytes": path_obj.stat().st_size,
        "line_count": _line_count(path_obj),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    return len(text.splitlines())


def _aggregate_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_values: dict[str, dict[str, list[float]]] = {}
    by_method_row_count: dict[str, int] = {}
    observed_metrics: set[str] = set()
    performance_row_count = 0
    metric_value_count = 0
    for idx, row in enumerate(rows):
        metric_values = _canonical_metric_values(row, row_idx=idx)
        metrics_in_row = list(metric_values)
        if not metrics_in_row:
            continue
        method = str(row.get("method", "unknown"))
        by_method_values.setdefault(method, {name: [] for name in PERFORMANCE_METRICS})
        by_method_row_count[method] = by_method_row_count.get(method, 0) + 1
        performance_row_count += 1
        for name, value in metric_values.items():
            by_method_values[method][name].append(value)
            observed_metrics.add(name)
            metric_value_count += 1
    if not observed_metrics:
        raise ValueError("no performance metrics found; status/integrity rows cannot be aggregated as performance")
    by_method: dict[str, dict[str, dict[str, float]]] = {}
    for method, metric_values in sorted(by_method_values.items()):
        by_method[method] = {}
        for name in PERFORMANCE_METRICS:
            values = metric_values[name]
            if values:
                by_method[method][name] = _stats(values)
    return {
        "metrics": {name: name for name in PERFORMANCE_METRICS if name in observed_metrics},
        "by_method": by_method,
        "row_count": performance_row_count,
        "metric_value_count": metric_value_count,
        "by_method_row_count": dict(sorted(by_method_row_count.items())),
    }


def _aggregate_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_candidate_count = 0
    total_admitted_count = 0
    gate_counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        total_candidate_count += int(row.get("candidate_count", row.get("record_count", 0)) or 0)
        total_admitted_count += int(row.get("admitted_count", 0) or 0)
        for name, count in dict(row.get("gate_counts") or {}).items():
            gate_counts[name] = gate_counts.get(name, 0) + int(count)
        status = row.get("status")
        if status is not None:
            statuses[str(status)] = statuses.get(str(status), 0) + 1
    return {
        "artifact_count": len(rows),
        "total_candidate_count": total_candidate_count,
        "total_admitted_count": total_admitted_count,
        "gate_counts": dict(sorted(gate_counts.items())),
        "statuses": dict(sorted(statuses.items())),
    }


def _canonical_metric_values(row: dict[str, Any], *, row_idx: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for canonical, aliases in PERFORMANCE_METRIC_ALIASES.items():
        for alias in aliases:
            if alias in row:
                values[canonical] = _finite_metric(row[alias], name=alias, row_idx=row_idx)
                break
    return values


def _finite_metric(value: Any, *, name: str, row_idx: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid performance metric {name!r} at row {row_idx}: {value!r}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite performance metric {name!r} at row {row_idx}: {value!r}")
    return parsed


def _stats(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    if len(values) <= 1:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
