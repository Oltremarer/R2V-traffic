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

from pareto.constants import OBJECTIVE_NAMES
from pareto.eval.bootstrap_metrics import bootstrap_mean_ci


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bernoulli_values(mean: float, count: int) -> list[int]:
    count = int(count)
    if count <= 0:
        return []
    positives = max(0, min(count, int(round(float(mean) * count))))
    return [1] * positives + [0] * (count - positives)


def _ci_from_metric(mean: float, count: int, n_boot: int, seed: int) -> dict[str, float | int] | None:
    values = _bernoulli_values(mean, count)
    if not values:
        return None
    return bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)


def build_offline_ci_report(
    diagnostics: dict[str, Any],
    pair_report: dict[str, Any],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    objective_counts = pair_report.get("objective_counts", {})
    report: dict[str, Any] = {
        "method": "aggregate_bernoulli_bootstrap",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "notes": [
            "CIs are reconstructed from aggregate accuracy and pair counts.",
            "Use pair-level correctness bootstrap for paper-grade reporting.",
        ],
        "metrics": {},
        "not_bootstrapped": {
            "head_leakage_diag_offdiag_gap": diagnostics.get("head_leakage_diag_offdiag_gap"),
        },
    }
    metric_specs = {
        "pref_acc": pair_report.get("preference_pairs", 0),
        "rev_acc": pair_report.get("reversal_pairs", 0),
        "dpr_head": pair_report.get("dominance_pairs", 0),
        "dpr_utility": int(pair_report.get("dominance_pairs", 0)) * 5,
    }
    for key, count in metric_specs.items():
        if key in diagnostics:
            ci = _ci_from_metric(float(diagnostics[key]), int(count), n_boot, seed)
            if ci is not None:
                report["metrics"][key] = ci
    obj_ci = {}
    for objective in OBJECTIVE_NAMES:
        value = diagnostics.get("obj_acc", {}).get(objective)
        count = objective_counts.get(objective, 0)
        if value is not None:
            ci = _ci_from_metric(float(value), int(count), n_boot, seed)
            if ci is not None:
                obj_ci[objective] = ci
    if obj_ci:
        report["metrics"]["obj_acc"] = obj_ci
    return report


def run(
    diagnostics_path: str | Path,
    pair_report_path: str | Path,
    out: str | Path,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    report = build_offline_ci_report(
        _read_json(diagnostics_path),
        _read_json(pair_report_path),
        n_boot=n_boot,
        seed=seed,
    )
    _write_json(out, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--pair_report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(
        run(args.diagnostics, args.pair_report, args.out, n_boot=args.n_boot, seed=args.seed),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
