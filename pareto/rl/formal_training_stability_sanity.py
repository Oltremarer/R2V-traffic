#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.rl.formal_training_stability_sanity_validator import (
    REQUIRED_ZERO_TOTAL_KEYS,
    TRAINING_STABILITY_FAIL,
    TRAINING_STABILITY_PASS,
    training_stability_coverage_errors,
    validate_training_stability_packet,
)


FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE = "FORMAL JINAN TRAINING-STABILITY SANITY GO"
ALLOWED_LOSS_FIELDS = {
    "approx_kl",
    "grad_norm",
    "policy_loss",
    "value_loss",
    "total_loss",
    "entropy_bonus",
    "clip_fraction",
    "ratio_min",
    "ratio_mean",
    "ratio_max",
    "loss_debug_finite",
}
FIELD_ABS_LIMITS = {
    "approx_kl": 1.0,
    "grad_norm": 1_000_000.0,
    "policy_loss": 1_000_000_000_000.0,
    "value_loss": 1_000_000_000_000.0,
    "total_loss": 1_000_000_000_000.0,
    "entropy_bonus": 1_000_000_000_000.0,
    "clip_fraction": 10.0,
    "ratio_min": 100.0,
    "ratio_mean": 100.0,
    "ratio_max": 100.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("seed*/*") if path.is_dir())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_threshold(field: str, value: Any) -> bool:
    if field == "loss_debug_finite":
        return False
    if not _is_finite_number(value):
        return False
    limit = FIELD_ABS_LIMITS.get(field)
    if limit is None:
        return False
    return abs(float(value)) > limit


def _analyze_run(run_dir: Path) -> dict[str, Any]:
    metadata = _read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    status = _read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
    rows = _read_jsonl(run_dir / "loss_debug.jsonl")
    nonfinite_count = 0
    threshold_violation_count = 0
    missing_allowed_field_count = 0
    observed_rows = 0

    for row in rows:
        if not isinstance(row, dict):
            nonfinite_count += 1
            continue
        observed_rows += 1
        for field in ALLOWED_LOSS_FIELDS:
            if field not in row:
                if field != "loss_debug_finite":
                    missing_allowed_field_count += 1
                continue
            value = row[field]
            if field == "loss_debug_finite":
                if value is not True:
                    nonfinite_count += 1
                continue
            if not _is_finite_number(value):
                nonfinite_count += 1
                continue
            if _check_threshold(field, value):
                threshold_violation_count += 1

    pass_fail = "PASS"
    if not rows or nonfinite_count or missing_allowed_field_count:
        pass_fail = "FAIL"
    elif threshold_violation_count:
        pass_fail = "WARN"

    return {
        "seed": metadata.get("cityflow_seed"),
        "method": metadata.get("method"),
        "status": status.get("status"),
        "pass_fail": pass_fail,
        "loss_debug_rows": len(rows),
        "observed_row_count": observed_rows,
        "missing_allowed_field_count": missing_allowed_field_count,
        "nonfinite_count": nonfinite_count,
        "threshold_violation_count": threshold_violation_count,
        "explosion_flag": bool(threshold_violation_count),
    }


def run_training_stability_sanity(
    *,
    root: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    if approval_phrase != FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for formal training-stability sanity")
    root_path = Path(root)
    runs = [_analyze_run(run_dir) for run_dir in _run_dirs(root_path)]
    totals = {
        "run_count": len(runs),
        "failed_run_count": sum(1 for run in runs if run["pass_fail"] == "FAIL"),
        "warn_run_count": sum(1 for run in runs if run["pass_fail"] == "WARN"),
        "nonfinite_count": sum(int(run["nonfinite_count"]) for run in runs),
        "threshold_violation_count": sum(int(run["threshold_violation_count"]) for run in runs),
        "missing_allowed_field_count": sum(int(run["missing_allowed_field_count"]) for run in runs),
    }
    report = {
        "report_status": TRAINING_STABILITY_PASS,
        "scope": "training_stability_sanity_only_no_method_comparison",
        "approval_phrase_verified": True,
        "allowed_input_files": ["metadata.json", "status.json", "loss_debug.jsonl"],
        "allowed_field_count": len(ALLOWED_LOSS_FIELDS),
        "output_policy": {
            "raw_values_written": False,
            "min_max_mean_std_written": False,
            "method_comparison_allowed": False,
            "formal_result_table_allowed": False,
        },
        "totals": totals,
        "runs": runs,
    }
    all_zero_guard_counts = all(totals[key] == 0 for key in REQUIRED_ZERO_TOTAL_KEYS)
    all_runs_pass = all(run.get("pass_fail") == "PASS" for run in runs)
    coverage_ok = not training_stability_coverage_errors(report)
    report["report_status"] = (
        TRAINING_STABILITY_PASS if all_zero_guard_counts and all_runs_pass and coverage_ok else TRAINING_STABILITY_FAIL
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "formal_jinan_training_stability_sanity.json", report)
    _write_packet(out_path / "formal_jinan_training_stability_sanity_packet.md", report)
    validate_training_stability_packet(out_path)
    return report


def _write_packet(path: Path, report: dict[str, Any]) -> None:
    totals = report["totals"]
    lines = [
        "# Formal Jinan Training-Stability Sanity",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "Scope: training-stability sanity only. Outputs are pass/fail and count based.",
        "",
        "Summary counts:",
        f"- runs checked: `{totals['run_count']}`",
        f"- failed runs: `{totals['failed_run_count']}`",
        f"- warning runs: `{totals['warn_run_count']}`",
        f"- nonfinite entries: `{totals['nonfinite_count']}`",
        f"- threshold violations: `{totals['threshold_violation_count']}`",
        f"- missing allowed fields: `{totals['missing_allowed_field_count']}`",
        "",
        "No raw values, min/max/mean/std summaries, method comparison, formal table, or traffic-control claim is produced.",
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
    report = run_training_stability_sanity(root=args.root, out_dir=args.out_dir, approval_phrase=args.approval_phrase)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("report_status") != TRAINING_STABILITY_PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
