#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.data.schema import TrajectoryRecord, TransitionRecord


def _feature_hash(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]


def validate_file(input_path: str | Path, schema: str = "trajectory", require_next_links: bool = False) -> Dict:
    input_path = Path(input_path)
    ids = set()
    next_ids = []
    feature_dims = set()
    feature_schema_hashes = set()
    feature_schema_names = None
    objective_valid_counts = None
    policy_ids = set()
    nan_count = 0
    total = 0
    errors = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                if schema == "trajectory":
                    record = TrajectoryRecord.from_json(line)
                    record.validate()
                    record_id = record.sample_id
                    features = record.obs_features
                    feature_names = record.obs_feature_names
                    if record.next_sample_id:
                        next_ids.append(record.next_sample_id)
                    if objective_valid_counts is None:
                        objective_valid_counts = {key: 0 for key in record.objective_valid_mask}
                    for key, valid in record.objective_valid_mask.items():
                        objective_valid_counts[key] += int(bool(valid))
                elif schema == "transition":
                    record = TransitionRecord.from_json(line)
                    record.validate()
                    state_hash = record.metadata.get("state_feature_names_hash")
                    next_hash = record.metadata.get("next_state_feature_names_hash")
                    if not state_hash or not next_hash:
                        raise ValueError("transition missing state/next_state feature_names_hash")
                    if state_hash != next_hash:
                        raise ValueError("feature_names_hash mismatch between state and next_state")
                    record_id = record.transition_id
                    features = record.obs_features + record.next_obs_features
                    feature_names = []
                else:
                    raise ValueError(f"unknown schema {schema}")

                if record_id in ids:
                    raise ValueError(f"duplicate id {record_id}")
                ids.add(record_id)
                feature_dims.add(len(record.obs_features))
                if feature_names:
                    feature_hash = _feature_hash(feature_names)
                    feature_schema_hashes.add(feature_hash)
                    if feature_schema_names is None:
                        feature_schema_names = list(feature_names)
                    elif feature_schema_names != feature_names:
                        raise ValueError("feature schema changed within file")
                nan_count += sum(1 for value in features if not isinstance(value, (int, float)) or value != value)
                policy_ids.add(record.policy_id)
                total += 1
            except Exception as exc:
                errors.append({"line": line_no, "error": str(exc)})

    missing_next_link_count = 0
    if schema == "trajectory" and require_next_links:
        missing = [next_id for next_id in next_ids if next_id not in ids]
        missing_next_link_count = len(missing)
        for next_id in missing[:20]:
            errors.append({"line": None, "error": f"next_sample_id not found: {next_id}"})

    valid_rates = {
        key: value / max(total, 1)
        for key, value in (objective_valid_counts or {}).items()
    }
    return {
        "input": str(input_path),
        "record_count": total,
        "feature_dim": next(iter(feature_dims)) if len(feature_dims) == 1 else None,
        "feature_dim_set": sorted(feature_dims),
        "feature_schema_hashes": sorted(feature_schema_hashes),
        "unique_sample_ids": len(ids),
        "sample_id_unique": len(ids) == total,
        "objective_valid_rates": valid_rates,
        "nan_count": nan_count,
        "policy_ids": sorted(policy_ids),
        "missing_next_link_count": missing_next_link_count,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def validate_files(
    input_paths: List[str | Path],
    schema: str = "trajectory",
    check_feature_schema_same: bool = False,
    require_next_links: bool = False,
) -> Dict:
    file_reports = [
        validate_file(path, schema=schema, require_next_links=require_next_links)
        for path in input_paths
    ]
    errors = []
    if check_feature_schema_same and schema == "trajectory":
        hashes = [
            tuple(report["feature_schema_hashes"])
            for report in file_reports
            if report["record_count"] > 0
        ]
        if len(set(hashes)) > 1:
            errors.append({"line": None, "error": "feature schema differs across inputs"})

    return {
        "schema": schema,
        "inputs": [str(path) for path in input_paths],
        "files": file_reports,
        "record_count": sum(report["record_count"] for report in file_reports),
        "feature_dim_set": sorted({dim for report in file_reports for dim in report["feature_dim_set"]}),
        "feature_schema_hashes": sorted({hash_ for report in file_reports for hash_ in report["feature_schema_hashes"]}),
        "nan_count": sum(report["nan_count"] for report in file_reports),
        "policy_ids": sorted({policy for report in file_reports for policy in report["policy_ids"]}),
        "error_count": sum(report["error_count"] for report in file_reports) + len(errors),
        "errors": errors + [error for report in file_reports for error in report["errors"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--schema", choices=["trajectory", "transition"], default="trajectory")
    parser.add_argument("--check_feature_schema_same", action="store_true")
    parser.add_argument("--require_next_links", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    input_paths = args.inputs or ([args.input] if args.input else [])
    if not input_paths:
        parser.error("--input or --inputs is required")
    report = validate_files(
        input_paths,
        schema=args.schema,
        check_feature_schema_same=args.check_feature_schema_same,
        require_next_links=args.require_next_links,
    )
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
