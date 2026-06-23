#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS: dict[str, float] = {
    "rev_acc": 0.30,
    "dpr_head": 0.20,
    "dpr_utility": 0.15,
    "pref_acc": 0.15,
    "obj_acc_mean": 0.10,
    "head_leakage_diag_offdiag_gap": 0.10,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def composite_val_score(
    metrics: dict[str, Any],
    weights: dict[str, float] | None = None,
    return_components: bool = False,
) -> float | tuple[float, dict[str, dict[str, float]]]:
    weights = weights or DEFAULT_WEIGHTS
    components: dict[str, dict[str, float]] = {}
    score = 0.0
    for key, weight in weights.items():
        value = float(metrics.get(key, 0.0) or 0.0)
        contribution = weight * value
        components[key] = {
            "value": value,
            "weight": float(weight),
            "contribution": float(contribution),
        }
        score += contribution
    if return_components:
        return float(score), components
    return float(score)


def load_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    return {
        "run_id": metadata.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "metadata": metadata,
        "val_metrics": _read_json(run_dir / "diagnostics_val.json"),
        "test_metrics": _read_json(run_dir / "diagnostics_test.json"),
    }


def select_offline_model(
    run_dirs: list[str | Path],
    out_path: str | Path | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run_dir is required")
    weights = weights or DEFAULT_WEIGHTS
    runs = []
    for run_dir in run_dirs:
        payload = load_run_metrics(run_dir)
        score, components = composite_val_score(payload["val_metrics"], weights, return_components=True)
        runs.append({
            "run_id": payload["run_id"],
            "run_dir": payload["run_dir"],
            "val_score": score,
            "components": components,
            "val_metrics": payload["val_metrics"],
            "test_metrics": payload["test_metrics"],
        })
    runs.sort(key=lambda row: row["val_score"], reverse=True)
    selected = runs[0]
    report = {
        "selected_by": "composite_val_score",
        "weights": weights,
        "selected_run_id": selected["run_id"],
        "selected_run_dir": selected["run_dir"],
        "selected_val_score": selected["val_score"],
        "runs": runs,
    }
    if out_path is not None:
        _write_json(out_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(select_offline_model(args.run_dirs, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
