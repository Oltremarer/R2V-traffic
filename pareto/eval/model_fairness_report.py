#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BUDGET_KEYS = (
    "epochs",
    "batch_size",
    "lr",
    "records_root",
    "pairs_root",
    "training_schedule",
    "reversal_sampler",
    "pref_margin_loss_weight",
    "rev_margin_loss_weight",
    "pref_hinge_loss_weight",
    "rev_hinge_loss_weight",
    "classification_margin",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_model_fairness_report(
    vector_metadata: dict[str, Any],
    scalar_metadata: dict[str, Any],
    max_param_gap: float = 0.30,
) -> dict[str, Any]:
    vector_params = int(vector_metadata.get("param_count", 0))
    scalar_params = int(scalar_metadata.get("param_count", 0))
    denom = max(vector_params, scalar_params, 1)
    relative_gap = abs(vector_params - scalar_params) / denom
    budget_comparison = {}
    for key in BUDGET_KEYS:
        vector_value = vector_metadata.get(key)
        scalar_value = scalar_metadata.get(key)
        budget_comparison[key] = {
            "vector": vector_value,
            "scalar": scalar_value,
            "match": vector_value == scalar_value,
        }
    return {
        "vector_param_count": vector_params,
        "scalar_param_count": scalar_params,
        "relative_param_gap": float(relative_gap),
        "max_param_gap": float(max_param_gap),
        "param_gap_status": "pass" if relative_gap <= max_param_gap else "warn",
        "budget_comparison": budget_comparison,
    }


def run(
    vector_model_dir: str | Path,
    scalar_model_dir: str | Path,
    out: str | Path,
    max_param_gap: float = 0.30,
) -> dict[str, Any]:
    vector_metadata = _read_json(Path(vector_model_dir) / "metadata.json")
    scalar_metadata = _read_json(Path(scalar_model_dir) / "metadata.json")
    report = build_model_fairness_report(vector_metadata, scalar_metadata, max_param_gap=max_param_gap)
    _write_json(out, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_model_dir", required=True)
    parser.add_argument("--scalar_model_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_param_gap", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(
        run(args.vector_model_dir, args.scalar_model_dir, args.out, args.max_param_gap),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
