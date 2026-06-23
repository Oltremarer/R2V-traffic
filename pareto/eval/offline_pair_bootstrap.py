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

import torch

from pareto.constants import OBJECTIVE_INDEX, OBJECTIVE_NAMES
from pareto.data.offline_dataset import (
    dominance_pair_tensors,
    load_split_pairs,
    load_split_records,
    make_feature_tensor,
    objective_pair_tensors,
    preference_pair_tensors,
    reversal_pair_tensors,
)
from pareto.eval.bootstrap_metrics import bootstrap_mean_ci
from pareto.eval.diagnostics import pairwise_head_leakage_matrix
from pareto.eval.offline_metrics import EVAL_PREFERENCES, head_leakage_gap
from pareto.eval.run_offline_diagnostics import _load_scalar_model, _load_vector_model
from pareto.models.vector_quality import score_with_preference
from pareto.train_common import resolve_device, write_json


def _bool_list(tensor: torch.Tensor) -> list[bool]:
    return [bool(value) for value in tensor.detach().cpu().tolist()]


def vector_correctness_lists(
    model: torch.nn.Module,
    records: dict[str, dict],
    pairs: dict[str, list[dict]],
    device: torch.device,
    scorer: torch.nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()
    model.to(device)
    if scorer is not None:
        scorer.eval()
        scorer.to(device)
    result: dict[str, Any] = {}
    with torch.no_grad():
        objective_pairs = pairs.get("objective", [])
        if objective_pairs:
            batch = {k: v.to(device) for k, v in objective_pair_tensors(objective_pairs, records).items()}
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            row_idx = torch.arange(q_a.shape[0], device=device)
            logits = q_a[row_idx, batch["objective_idx"]] - q_b[row_idx, batch["objective_idx"]]
            correct = (logits >= 0) == (batch["labels"] >= 0.5)
            result["obj_acc_mean"] = _bool_list(correct)
            by_objective = {}
            for idx, name in enumerate(OBJECTIVE_NAMES):
                mask = batch["objective_idx"] == idx
                if mask.any():
                    by_objective[name] = _bool_list(correct[mask])
            result["obj_acc"] = by_objective

            ids = list(records)
            q_all = model(make_feature_tensor(records, ids).to(device)).detach().cpu()
            q_by_id = {sample_id: q_all[row] for row, sample_id in enumerate(ids)}
            # Bootstrap the leakage gap by resampling objective pairs.
            result["head_leakage_gap_items"] = [
                {
                    "q_by_id": q_by_id,
                    "pair": pair,
                }
                for pair in objective_pairs
            ]

        preference_pairs = pairs.get("preference", [])
        if preference_pairs:
            batch = {k: v.to(device) for k, v in preference_pair_tensors(preference_pairs, records).items()}
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            score_fn = scorer if scorer is not None else score_with_preference
            logits = score_fn(q_a, batch["w"]) - score_fn(q_b, batch["w"])
            result["pref_acc"] = _bool_list((logits >= 0) == (batch["labels"] >= 0.5))

        reversal_pairs = pairs.get("reversal", [])
        if reversal_pairs:
            batch = {k: v.to(device) for k, v in reversal_pair_tensors(reversal_pairs, records).items()}
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            score_fn = scorer if scorer is not None else score_with_preference
            logits_1 = score_fn(q_a, batch["w_1"]) - score_fn(q_b, batch["w_1"])
            logits_2 = score_fn(q_a, batch["w_2"]) - score_fn(q_b, batch["w_2"])
            correct_1 = (logits_1 >= 0) == (batch["labels_1"] >= 0.5)
            correct_2 = (logits_2 >= 0) == (batch["labels_2"] >= 0.5)
            result["rev_acc"] = _bool_list(correct_1 & correct_2)

        dominance_pairs = pairs.get("dominance", [])
        if dominance_pairs:
            batch = {k: v.to(device) for k, v in dominance_pair_tensors(dominance_pairs, records).items()}
            q_dom = model(batch["x_dom"])
            q_sub = model(batch["x_sub"])
            diff = q_dom - q_sub
            result["dpr_head"] = _bool_list((diff >= 0).all(dim=-1))
            utility_ok = []
            for w in EVAL_PREFERENCES.to(device):
                utility_ok.extend(_bool_list((diff @ w) >= 0))
            result["dpr_utility"] = utility_ok
    return result


def scalar_correctness_lists(
    model: torch.nn.Module,
    records: dict[str, dict],
    pairs: dict[str, list[dict]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.to(device)
    result: dict[str, Any] = {}
    with torch.no_grad():
        preference_pairs = pairs.get("preference", [])
        if preference_pairs:
            batch = {k: v.to(device) for k, v in preference_pair_tensors(preference_pairs, records).items()}
            logits = model(batch["x_a"], batch["w"]) - model(batch["x_b"], batch["w"])
            result["pref_acc"] = _bool_list((logits >= 0) == (batch["labels"] >= 0.5))

        reversal_pairs = pairs.get("reversal", [])
        if reversal_pairs:
            batch = {k: v.to(device) for k, v in reversal_pair_tensors(reversal_pairs, records).items()}
            logits_1 = model(batch["x_a"], batch["w_1"]) - model(batch["x_b"], batch["w_1"])
            logits_2 = model(batch["x_a"], batch["w_2"]) - model(batch["x_b"], batch["w_2"])
            correct_1 = (logits_1 >= 0) == (batch["labels_1"] >= 0.5)
            correct_2 = (logits_2 >= 0) == (batch["labels_2"] >= 0.5)
            result["rev_acc"] = _bool_list(correct_1 & correct_2)

        dominance_pairs = pairs.get("dominance", [])
        if dominance_pairs:
            batch = {k: v.to(device) for k, v in dominance_pair_tensors(dominance_pairs, records).items()}
            utility_ok = []
            for w in EVAL_PREFERENCES.to(device):
                w_batch = w.reshape(1, -1).repeat(batch["x_dom"].shape[0], 1)
                diff = model(batch["x_dom"], w_batch) - model(batch["x_sub"], w_batch)
                utility_ok.extend(_bool_list(diff >= 0))
            result["dpr_utility"] = utility_ok
    return result


def _leakage_gap_from_items(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    q_by_id = items[0]["q_by_id"]
    matrix = pairwise_head_leakage_matrix(q_by_id, [item["pair"] for item in items])
    return float(head_leakage_gap(matrix))


def bootstrap_correctness_report(
    correctness: dict[str, Any],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, values in correctness.items():
        if key == "obj_acc":
            metrics[key] = {
                name: bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
                for name, values in values.items()
                if values
            }
        elif key == "head_leakage_gap_items":
            from pareto.eval.bootstrap_metrics import bootstrap_metric_ci

            if values:
                metrics["head_leakage_diag_offdiag_gap"] = bootstrap_metric_ci(
                    values,
                    _leakage_gap_from_items,
                    n_boot=n_boot,
                    seed=seed,
                )
        elif values:
            metrics[key] = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {
        "method": "pair_level_bootstrap",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "metrics": metrics,
    }


def run(
    records_root: str | Path,
    pairs_root: str | Path,
    split: str,
    out: str | Path,
    vector_model_dir: str | Path | None = None,
    scalar_model_dir: str | Path | None = None,
    device: str = "cuda",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    torch_device = resolve_device(device)
    records = load_split_records(records_root, split)
    pairs = load_split_pairs(pairs_root, split)
    payload: dict[str, Any] = {
        "records_root": str(records_root),
        "pairs_root": str(pairs_root),
        "split": split,
        "device": str(torch_device),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
    if vector_model_dir:
        vector_model, vector_scorer, _ = _load_vector_model(vector_model_dir, torch_device)
        payload["vector"] = bootstrap_correctness_report(
            vector_correctness_lists(vector_model, records, pairs, torch_device, scorer=vector_scorer),
            n_boot=n_boot,
            seed=seed,
        )
        payload["vector_model_dir"] = str(vector_model_dir)
    if scalar_model_dir:
        scalar_model, _ = _load_scalar_model(scalar_model_dir, torch_device)
        payload["cond_scalar"] = bootstrap_correctness_report(
            scalar_correctness_lists(scalar_model, records, pairs, torch_device),
            n_boot=n_boot,
            seed=seed,
        )
        payload["scalar_model_dir"] = str(scalar_model_dir)
    write_json(out, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vector_model_dir")
    parser.add_argument("--scalar_model_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(
        run(
            args.records_root,
            args.pairs_root,
            args.split,
            args.out,
            vector_model_dir=args.vector_model_dir,
            scalar_model_dir=args.scalar_model_dir,
            device=args.device,
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
