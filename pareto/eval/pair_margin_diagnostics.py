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
from pareto.models.vector_quality import build_vector_quality_model, score_with_preference
from pareto.train_common import load_checkpoint, resolve_device, write_json


MARGIN_BINS = (
    (0.0, 0.05, "0.00-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.50, "0.20-0.50"),
    (0.50, math.inf, "0.50-inf"),
)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _bin_name(value: float) -> str:
    for low, high, name in MARGIN_BINS:
        if low <= value < high:
            return name
    return MARGIN_BINS[-1][2]


def _accuracy(correct: list[bool]) -> float:
    if not correct:
        return 0.0
    return float(sum(1 for item in correct if item) / len(correct))


def _template_key(row: dict[str, Any]) -> str:
    return f"{row.get('w_1_name', 'unknown')}__{row.get('w_2_name', 'unknown')}"


def _q_by_id(
    model: torch.nn.Module | None,
    records_by_id: dict[str, dict],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if model is None:
        return {}
    ids = list(records_by_id)
    with torch.no_grad():
        q_all = model(make_feature_tensor(records_by_id, ids).to(device)).detach().cpu()
    return {sample_id: q_all[idx] for idx, sample_id in enumerate(ids)}


def _load_vector_model(model_dir: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(Path(model_dir) / "model.pt", device)
    config = checkpoint["config"]
    model = build_vector_quality_model(
        config.get("architecture", "shared_mlp"),
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        trunk_layers=config.get("trunk_layers", 2),
        head_layers=config.get("head_layers", 2),
        tower_residual_alpha=config.get("tower_residual_alpha", 0.5),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def diagnose_pair_margins(
    records_by_id: dict[str, dict],
    pairs: dict[str, list[dict]],
    model: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    if model is not None:
        model.to(device)
        model.eval()
    q_by_id = _q_by_id(model, records_by_id, device)

    objective_margin_values = {name: [] for name in OBJECTIVE_NAMES}
    objective_correct = {name: [] for name in OBJECTIVE_NAMES}
    objective_correct_by_bin: dict[str, dict[str, list[bool]]] = {name: {} for name in OBJECTIVE_NAMES}
    strategy_correct: dict[str, list[bool]] = {}
    strategy_counts: dict[str, int] = {}

    for row in pairs.get("objective", []):
        objective = row["objective"]
        margin_abs = abs(float(row.get("margin_norm", 0.0)))
        objective_margin_values[objective].append(margin_abs)
        strategy = row.get("sampling_strategy", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        if q_by_id:
            idx = OBJECTIVE_INDEX[objective]
            logit = float(q_by_id[row["a_id"]][idx] - q_by_id[row["b_id"]][idx])
            correct = int(logit >= 0.0) == int(row["label"])
            objective_correct[objective].append(correct)
            objective_correct_by_bin[objective].setdefault(_bin_name(margin_abs), []).append(correct)
            strategy_correct.setdefault(strategy, []).append(correct)

    reversal_templates: dict[str, dict[str, Any]] = {}
    reversal_correct_by_margin_bin: dict[str, list[bool]] = {}
    for row in pairs.get("reversal", []):
        key = _template_key(row)
        margin_1_abs = abs(float(row.get("margin_1", 0.0)))
        margin_2_abs = abs(float(row.get("margin_2", 0.0)))
        min_abs = min(margin_1_abs, margin_2_abs)
        bucket = reversal_templates.setdefault(
            key,
            {
                "count": 0,
                "margin_1_abs": [],
                "margin_2_abs": [],
                "min_abs_margin": [],
                "correct_both": 0,
                "correct_one": 0,
                "wrong_both": 0,
                "same_sign": 0,
            },
        )
        bucket["count"] += 1
        bucket["margin_1_abs"].append(margin_1_abs)
        bucket["margin_2_abs"].append(margin_2_abs)
        bucket["min_abs_margin"].append(min_abs)
        if q_by_id:
            q_a = q_by_id[row["a_id"]].reshape(1, -1)
            q_b = q_by_id[row["b_id"]].reshape(1, -1)
            w_1 = torch.tensor(row["w_1"], dtype=torch.float32).reshape(1, -1)
            w_2 = torch.tensor(row["w_2"], dtype=torch.float32).reshape(1, -1)
            logit_1 = float(score_with_preference(q_a, w_1) - score_with_preference(q_b, w_1))
            logit_2 = float(score_with_preference(q_a, w_2) - score_with_preference(q_b, w_2))
            correct_1 = int(logit_1 >= 0.0) == int(row["label_1"])
            correct_2 = int(logit_2 >= 0.0) == int(row["label_2"])
            if correct_1 and correct_2:
                bucket["correct_both"] += 1
            elif correct_1 or correct_2:
                bucket["correct_one"] += 1
            else:
                bucket["wrong_both"] += 1
            same_sign = (logit_1 >= 0.0) == (logit_2 >= 0.0)
            bucket["same_sign"] += int(same_sign)
            reversal_correct_by_margin_bin.setdefault(_bin_name(min_abs), []).append(correct_1 and correct_2)

    reversal_report = {}
    for key, bucket in reversal_templates.items():
        count = int(bucket["count"])
        correct_both = int(bucket["correct_both"])
        reversal_report[key] = {
            "count": count,
            "margin_1_abs_stats": _stats(bucket["margin_1_abs"]),
            "margin_2_abs_stats": _stats(bucket["margin_2_abs"]),
            "min_abs_margin_stats": _stats(bucket["min_abs_margin"]),
            "accuracy": float(correct_both / count) if count and q_by_id else None,
            "correct_both": correct_both,
            "correct_one": int(bucket["correct_one"]),
            "wrong_both": int(bucket["wrong_both"]),
            "same_sign_rate": float(bucket["same_sign"] / count) if count and q_by_id else None,
        }

    dominance_violations = {name: 0 for name in OBJECTIVE_NAMES}
    dominance_count = 0
    if q_by_id:
        for row in pairs.get("dominance", []):
            if row["dominates"] == "a":
                q_dom = q_by_id[row["a_id"]]
                q_sub = q_by_id[row["b_id"]]
            else:
                q_dom = q_by_id[row["b_id"]]
                q_sub = q_by_id[row["a_id"]]
            diff = q_dom - q_sub
            dominance_count += 1
            for idx, name in enumerate(OBJECTIVE_NAMES):
                dominance_violations[name] += int(float(diff[idx]) < 0.0)

    return {
        "record_count": len(records_by_id),
        "model_evaluated": bool(q_by_id),
        "objective_margin_stats": {
            name: _stats(values)
            for name, values in objective_margin_values.items()
        },
        "objective_accuracy_by_objective": {
            name: _accuracy(values)
            for name, values in objective_correct.items()
            if values
        },
        "objective_accuracy_by_margin_bin": {
            name: {
                bucket: _accuracy(values)
                for bucket, values in buckets.items()
            }
            for name, buckets in objective_correct_by_bin.items()
            if buckets
        },
        "strategy_counts": strategy_counts,
        "strategy_accuracy": {
            name: _accuracy(values)
            for name, values in strategy_correct.items()
        },
        "reversal_by_template_pair": reversal_report,
        "reversal_accuracy_by_min_margin_bin": {
            bucket: _accuracy(values)
            for bucket, values in reversal_correct_by_margin_bin.items()
        },
        "dominance_pairs": dominance_count,
        "dominance_violations_by_head": dominance_violations,
        "dominance_violation_rate_by_head": {
            name: (float(count / dominance_count) if dominance_count else 0.0)
            for name, count in dominance_violations.items()
        },
    }


def run(
    records_root: str | Path,
    pairs_root: str | Path,
    vector_model_dir: str | Path | None,
    out: str | Path,
    device: str = "cuda",
    splits: list[str] | None = None,
) -> dict[str, Any]:
    torch_device = resolve_device(device)
    model = _load_vector_model(vector_model_dir, torch_device) if vector_model_dir else None
    splits = splits or ["train", "val", "test"]
    payload = {
        "records_root": str(records_root),
        "pairs_root": str(pairs_root),
        "vector_model_dir": str(vector_model_dir) if vector_model_dir else None,
        "device": str(torch_device),
        "splits": {},
    }
    for split in splits:
        records = load_split_records(records_root, split)
        pairs = load_split_pairs(pairs_root, split)
        payload["splits"][split] = diagnose_pair_margins(records, pairs, model=model, device=torch_device)
    write_json(out, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--vector_model_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    payload = run(
        args.records_root,
        args.pairs_root,
        vector_model_dir=args.vector_model_dir,
        out=args.out,
        device=args.device,
        splits=splits,
    )
    summary = {
        split: {
            "record_count": report["record_count"],
            "reversal_templates": {
                key: value["count"]
                for key, value in report["reversal_by_template_pair"].items()
            },
            "strategy_counts": report["strategy_counts"],
        }
        for split, report in payload["splits"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
