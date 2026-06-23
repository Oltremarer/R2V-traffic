#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.paper_final_experiment_manifest import (
    PAPER_FINAL_SEEDS,
    REQUIRED_CITY_TRAFFIC,
    REQUIRED_PREFERENCE_TEMPLATES,
)
from pareto.rl.paper_final_learned_eval_runner import (
    LEARNED_PPO_METHOD_IDS,
    PAPER_FINAL_LEARNED_EVAL_DONE,
    PAPER_FINAL_LEARNED_EVAL_ROOT,
    default_learned_eval_out_dir,
)
from pareto.rl.paper_final_reference_runner import (
    REFERENCE_BASELINE_SCRIPTS,
    REFERENCE_METRIC_KEYS,
    reference_output_is_complete,
)


DEFAULT_REFERENCE_ROOT = "records/paper_final/train_20260602_v1"
DEFAULT_AGGREGATION_OUT_DIR = "records/paper_final/aggregation_20260602_v1/no_newyork"
DEFAULT_CITIES = ("jinan", "hangzhou")
EXPECTED_LEARNED_PREFERENCES = {
    "Cond-Scalar-RL": ("balanced",),
    "VectorQ-PPO": ("balanced",),
    "Weighted-RL": tuple(REQUIRED_PREFERENCE_TEMPLATES.keys()),
}
AGGREGATION_FILES = (
    "paper_final_eval_raw_rows.jsonl",
    "paper_final_eval_summary.json",
    "paper_final_eval_summary.csv",
    "paper_final_eval_missing_rows.json",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite_metric_payload(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    metrics: dict[str, float] = {}
    missing = [key for key in REFERENCE_METRIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing paper-final metric keys in {path}: {missing}")
    for key in REFERENCE_METRIC_KEYS:
        value = float(payload[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric in {path}: {key}={payload[key]}")
        metrics[key] = value
    return metrics


def _city_list(cities: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(city).strip() for city in cities if str(city).strip())
    if not normalized:
        raise ValueError("at least one paper-final city is required")
    unknown = sorted(set(normalized) - set(REQUIRED_CITY_TRAFFIC))
    if unknown:
        raise ValueError(f"unknown paper-final cities: {unknown}")
    return normalized


def _reference_candidates(reference_root: Path, *, city: str, traffic_file: str, method: str, seed: int) -> list[Path]:
    return [
        reference_root / city / Path(traffic_file).stem / method / f"seed{seed}" / "paper_final_reference_metrics.json",
        reference_root / city / traffic_file / method / f"seed{seed}" / "paper_final_reference_metrics.json",
        reference_root / city / method / f"seed{seed}" / "paper_final_reference_metrics.json",
    ]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _learned_candidates(
    learned_eval_root: Path,
    *,
    city: str,
    traffic_file: str,
    method: str,
    seed: int,
    preference: str,
) -> list[Path]:
    default_path = Path(
        default_learned_eval_out_dir(
            eval_root=learned_eval_root.as_posix(),
            city=city,
            traffic_file=traffic_file,
            method=method,
            seed_id=seed,
            fixed_preference_template=preference,
        )
    )
    return [
        default_path / "paper_final_learned_eval_metrics.json",
        learned_eval_root / city / method / f"seed{seed}" / preference / "paper_final_learned_eval_metrics.json",
    ]


def _reference_complete(metrics_path: Path) -> bool:
    return reference_output_is_complete(metrics_path.parent)


def _learned_complete(metrics_path: Path) -> bool:
    status_path = metrics_path.parent / "paper_final_learned_eval_status.json"
    if not status_path.is_file():
        return False
    status = _read_json(status_path)
    return status.get("status") == PAPER_FINAL_LEARNED_EVAL_DONE


def collect_paper_final_eval_rows(
    *,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    learned_eval_root: str | Path = PAPER_FINAL_LEARNED_EVAL_ROOT,
    cities: Iterable[str] = DEFAULT_CITIES,
    require_status: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_tuple = _city_list(cities)
    reference_base = ROOT / str(reference_root)
    learned_base = ROOT / str(learned_eval_root)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for city in city_tuple:
        traffic_file = REQUIRED_CITY_TRAFFIC[city]
        for method in REFERENCE_BASELINE_SCRIPTS:
            for seed in PAPER_FINAL_SEEDS:
                metrics_path = _first_existing(
                    _reference_candidates(reference_base, city=city, traffic_file=traffic_file, method=method, seed=seed)
                )
                expected = {
                    "phase": "reference",
                    "city": city,
                    "traffic_file": traffic_file,
                    "method": method,
                    "seed": int(seed),
                    "preference_template": "not_applicable",
                }
                if metrics_path is None:
                    missing.append({**expected, "reason": "missing_reference_metrics"})
                    continue
                if require_status and not _reference_complete(metrics_path):
                    missing.append({**expected, "reason": "reference_status_not_done", "path": metrics_path.as_posix()})
                    continue
                rows.append({**expected, **_finite_metric_payload(metrics_path), "source_path": metrics_path.as_posix()})

        for method in LEARNED_PPO_METHOD_IDS:
            for seed in PAPER_FINAL_SEEDS:
                for preference in EXPECTED_LEARNED_PREFERENCES[method]:
                    metrics_path = _first_existing(
                        _learned_candidates(
                            learned_base,
                            city=city,
                            traffic_file=traffic_file,
                            method=method,
                            seed=seed,
                            preference=preference,
                        )
                    )
                    expected = {
                        "phase": "learned",
                        "city": city,
                        "traffic_file": traffic_file,
                        "method": method,
                        "seed": int(seed),
                        "preference_template": preference,
                    }
                    if metrics_path is None:
                        missing.append({**expected, "reason": "missing_learned_eval_metrics"})
                        continue
                    if require_status and not _learned_complete(metrics_path):
                        missing.append({**expected, "reason": "learned_eval_status_not_done", "path": metrics_path.as_posix()})
                        continue
                    rows.append({**expected, **_finite_metric_payload(metrics_path), "source_path": metrics_path.as_posix()})

    return rows, missing


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate empty values")
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def aggregate_paper_final_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["phase"]),
            str(row["city"]),
            str(row["method"]),
            str(row["preference_template"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        phase, city, method, preference = key
        group_rows = grouped[key]
        summary: dict[str, Any] = {
            "phase": phase,
            "city": city,
            "method": method,
            "preference_template": preference,
            "n": len(group_rows),
            "seeds": sorted(int(row["seed"]) for row in group_rows),
        }
        for metric in REFERENCE_METRIC_KEYS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
        summary_rows.append(summary)
    return summary_rows


def build_paper_final_eval_aggregation(
    *,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    learned_eval_root: str | Path = PAPER_FINAL_LEARNED_EVAL_ROOT,
    cities: Iterable[str] = DEFAULT_CITIES,
    require_status: bool = True,
) -> dict[str, Any]:
    city_tuple = _city_list(cities)
    rows, missing = collect_paper_final_eval_rows(
        reference_root=reference_root,
        learned_eval_root=learned_eval_root,
        cities=city_tuple,
        require_status=require_status,
    )
    summary_rows = aggregate_paper_final_eval_rows(rows)
    reference_count = sum(1 for row in rows if row["phase"] == "reference")
    learned_count = sum(1 for row in rows if row["phase"] == "learned")
    expected_reference_count = len(city_tuple) * len(REFERENCE_BASELINE_SCRIPTS) * len(PAPER_FINAL_SEEDS)
    expected_learned_count = len(city_tuple) * (
        len(PAPER_FINAL_SEEDS) * len(EXPECTED_LEARNED_PREFERENCES["Cond-Scalar-RL"])
        + len(PAPER_FINAL_SEEDS) * len(EXPECTED_LEARNED_PREFERENCES["VectorQ-PPO"])
        + len(PAPER_FINAL_SEEDS) * len(EXPECTED_LEARNED_PREFERENCES["Weighted-RL"])
    )
    return {
        "packet_type": "paper_final_eval_aggregation",
        "scope": "no_newyork" if city_tuple == DEFAULT_CITIES else "custom",
        "cities": list(city_tuple),
        "reference_root": str(reference_root),
        "learned_eval_root": str(learned_eval_root),
        "reference_metric_keys": list(REFERENCE_METRIC_KEYS),
        "same_metric_schema_for_reference_and_learned": True,
        "status_required": bool(require_status),
        "counts": {
            "reference_observed": reference_count,
            "reference_expected": expected_reference_count,
            "learned_observed": learned_count,
            "learned_expected": expected_learned_count,
            "missing": len(missing),
        },
        "aggregation_ready": len(missing) == 0,
        "raw_rows": rows,
        "summary_rows": summary_rows,
        "missing_rows": missing,
    }


def write_paper_final_eval_aggregation_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    output = ROOT / str(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "paper_final_eval_raw_rows.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["raw_rows"]),
        encoding="utf-8",
    )
    _write_json(output / "paper_final_eval_summary.json", {key: value for key, value in payload.items() if key != "raw_rows"})
    _write_json(output / "paper_final_eval_missing_rows.json", {"missing_rows": payload["missing_rows"]})
    csv_path = output / "paper_final_eval_summary.csv"
    fieldnames = [
        "phase",
        "city",
        "method",
        "preference_template",
        "n",
        "seeds",
    ]
    for metric in REFERENCE_METRIC_KEYS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["summary_rows"]:
            csv_row = dict(row)
            csv_row["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            writer.writerow(csv_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--learned_eval_root", default=PAPER_FINAL_LEARNED_EVAL_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_AGGREGATION_OUT_DIR)
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--allow_missing_status", action="store_true")
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cities = tuple(part.strip() for part in args.cities.split(",") if part.strip())
    payload = build_paper_final_eval_aggregation(
        reference_root=args.reference_root,
        learned_eval_root=args.learned_eval_root,
        cities=cities,
        require_status=not args.allow_missing_status,
    )
    write_paper_final_eval_aggregation_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "aggregation_ready": payload["aggregation_ready"],
                "counts": payload["counts"],
                "out_dir": args.out_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not payload["aggregation_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
