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

import torch

from pareto.constants import OBJECTIVE_INDEX, OBJECTIVE_NAMES
from pareto.data.offline_dataset import load_split_pairs, load_split_records, make_feature_tensor
from pareto.eval.offline_metrics import EVAL_PREFERENCES
from pareto.eval.run_offline_diagnostics import _load_vector_model
from pareto.train_common import resolve_device, write_json


MARGIN_BINS = (
    (0.0, 0.05, "0.00-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.50, "0.20-0.50"),
    (0.50, math.inf, "0.50-inf"),
)


def _bin_name(value: float) -> str:
    for low, high, name in MARGIN_BINS:
        if low <= value < high:
            return name
    return MARGIN_BINS[-1][2]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    alpha = pos - lo
    return float(ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha)


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _dominance_direction(row: dict[str, Any]) -> tuple[str, str, float]:
    if row["dominates"] == "a":
        return row["a_id"], row["b_id"], 1.0
    if row["dominates"] == "b":
        return row["b_id"], row["a_id"], -1.0
    raise ValueError(f"invalid dominates field: {row['dominates']}")


def _objective_margins(row: dict[str, Any]) -> dict[str, float]:
    _, _, sign = _dominance_direction(row)
    raw = row.get("objective_margins_norm", {})
    return {name: sign * float(raw.get(name, 0.0)) for name in OBJECTIVE_NAMES}


def dominance_error_audit(
    records_by_id: dict[str, dict],
    dominance_pairs: list[dict],
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
    utility_preferences: torch.Tensor = EVAL_PREFERENCES,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    ids = list(records_by_id)
    with torch.no_grad():
        q_all = model(make_feature_tensor(records_by_id, ids).to(torch_device)).detach().cpu()
    q_by_id = {sample_id: q_all[idx] for idx, sample_id in enumerate(ids)}

    per_objective_pass = {name: [] for name in OBJECTIVE_NAMES}
    per_objective_margins = {name: [] for name in OBJECTIVE_NAMES}
    margin_bins: dict[str, dict[str, int]] = {}
    utility_pass_head_fail = 0
    utility_all_pass_count = 0
    all_head_pass_count = 0
    utility_preferences = utility_preferences.detach().cpu().float()

    for row in dominance_pairs:
        dom_id, sub_id, _ = _dominance_direction(row)
        diff = q_by_id[dom_id] - q_by_id[sub_id]
        objective_margins = _objective_margins(row)
        head_pass_values = []
        for name in OBJECTIVE_NAMES:
            idx = OBJECTIVE_INDEX[name]
            passed = bool(float(diff[idx]) >= 0.0)
            per_objective_pass[name].append(passed)
            per_objective_margins[name].append(float(objective_margins[name]))
            head_pass_values.append(passed)

        all_head_pass = all(head_pass_values)
        all_head_pass_count += int(all_head_pass)
        utility_values = torch.mv(utility_preferences, diff.float())
        utility_all_pass = bool((utility_values >= 0.0).all().item())
        utility_all_pass_count += int(utility_all_pass)
        utility_pass_head_fail += int(utility_all_pass and not all_head_pass)

        min_margin = min(abs(value) for value in objective_margins.values())
        bucket = margin_bins.setdefault(_bin_name(min_margin), {"count": 0, "head_fail": 0, "utility_pass_head_fail": 0})
        bucket["count"] += 1
        bucket["head_fail"] += int(not all_head_pass)
        bucket["utility_pass_head_fail"] += int(utility_all_pass and not all_head_pass)

    total = len(dominance_pairs)
    return {
        "dominance_pairs": total,
        "DPR_head": float(all_head_pass_count / total) if total else 0.0,
        "DPR_utility_all_templates": float(utility_all_pass_count / total) if total else 0.0,
        "utility_pass_head_fail_rate": float(utility_pass_head_fail / total) if total else 0.0,
        "DPR_head_by_objective": {
            name: float(sum(values) / len(values)) if values else 0.0
            for name, values in per_objective_pass.items()
        },
        "violation_rate_by_objective": {
            name: 1.0 - (float(sum(values) / len(values)) if values else 0.0)
            for name, values in per_objective_pass.items()
        },
        "dominance_margin_stats": {
            name: _stats([abs(value) for value in values])
            for name, values in per_objective_margins.items()
        },
        "violation_by_margin_bin": {
            name: {
                "count": bucket["count"],
                "head_fail_rate": float(bucket["head_fail"] / bucket["count"]) if bucket["count"] else 0.0,
                "utility_pass_head_fail_rate": (
                    float(bucket["utility_pass_head_fail"] / bucket["count"]) if bucket["count"] else 0.0
                ),
            }
            for name, bucket in sorted(margin_bins.items())
        },
    }


def run(
    records_root: str | Path,
    pairs_root: str | Path,
    model_dir: str | Path,
    out: str | Path,
    split: str = "test",
    device: str = "cuda",
) -> dict[str, Any]:
    torch_device = resolve_device(device)
    model, _, _ = _load_vector_model(model_dir, torch_device)
    records = load_split_records(records_root, split)
    pairs = load_split_pairs(pairs_root, split)
    payload = {
        "records_root": str(records_root),
        "pairs_root": str(pairs_root),
        "model_dir": str(model_dir),
        "split": split,
        "device": str(torch_device),
        "audit": dominance_error_audit(records, pairs["dominance"], model, device=torch_device),
    }
    write_json(out, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        args.records_root,
        args.pairs_root,
        args.model_dir,
        args.out,
        split=args.split,
        device=args.device,
    )
    print(json.dumps(payload["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
