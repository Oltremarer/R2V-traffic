#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.constants import OBJECTIVE_NAMES


def load_records(path: str | Path) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _corr(records: List[Dict], a: str, b: str) -> float | None:
    xs = []
    ys = []
    for record in records:
        valid = record.get("objective_valid_mask", {})
        if valid.get(a, False) and valid.get(b, False):
            values = record.get("objective_values_norm") or record["objective_values_raw"]
            xs.append(float(values[a]))
            ys.append(float(values[b]))
    if len(xs) < 2:
        return None
    corr = np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1]
    return None if np.isnan(corr) else float(corr)


def summarize_records(records: Iterable[Dict]) -> Dict:
    records = list(records)
    safety_valid = [bool(record["objective_valid_mask"].get("safety", False)) for record in records]
    pair_counts = [
        float(record.get("metadata", {}).get("objective_debug", {}).get("local_ttc_pair_count", 0.0))
        for record in records
    ]
    violation_rates = [
        -float(record["objective_values_raw"]["safety"])
        for record in records
        if record["objective_valid_mask"].get("safety", False)
    ]
    by_time = defaultdict(list)
    for record in records:
        by_time[record["sim_time_sec"]].append(float(record["objective_values_raw"]["safety"]))
    safety_stds = [float(np.std(values)) for values in by_time.values() if len(values) > 1]

    correlations = {}
    for a in OBJECTIVE_NAMES:
        for b in OBJECTIVE_NAMES:
            if a < b:
                correlations[f"{a}__{b}"] = _corr(records, a, b)

    zero_rate = float(sum(count == 0 for count in pair_counts) / max(len(pair_counts), 1))
    return {
        "record_count": len(records),
        "safety_valid_rate": float(sum(safety_valid) / max(len(safety_valid), 1)),
        "ttc_pair_count_mean": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "ttc_pair_count_zero_rate": zero_rate,
        "ttc_violation_rate_mean": float(np.mean(violation_rates)) if violation_rates else 0.0,
        "same_time_cross_intersection_safety_std_mean": float(np.mean(safety_stds)) if safety_stds else 0.0,
        "objective_correlations": correlations,
        "warnings": [],
    }


def strict_failures(report: Dict) -> List[str]:
    failures = []
    if report["safety_valid_rate"] < 0.3:
        failures.append("safety_valid_rate below 0.3")
    if report["ttc_pair_count_zero_rate"] > 0.8:
        failures.append("ttc_pair_count_zero_rate above 0.8")
    if report["same_time_cross_intersection_safety_std_mean"] <= 1e-8:
        failures.append("same_time_cross_intersection_safety_std_mean is too small")
    for key in ("efficiency__fairness", "efficiency__stability"):
        value = report.get("objective_correlations", {}).get(key)
        if value is not None and abs(float(value)) > 0.95:
            failures.append(f"{key} correlation magnitude above 0.95")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--check", default="local_safety")
    parser.add_argument("--out", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = summarize_records(load_records(args.buffer))
    if report["safety_valid_rate"] < 0.3:
        report["warnings"].append("safety_valid_rate below 0.3; do not build safety pairs")
    if report["same_time_cross_intersection_safety_std_mean"] <= 0:
        report["warnings"].append("safety appears identical across intersections at each timestep")
    if args.strict:
        report["strict_failures"] = strict_failures(report)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["strict_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
