from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.offline_dataset import (
    dominance_pair_tensors,
    load_split_pairs,
    load_split_records,
    make_feature_tensor,
    objective_pair_tensors,
    objective_target_tensor,
    preference_pair_tensors,
    reversal_pair_tensors,
)
from pareto.eval.diagnostics import (
    binary_accuracy_from_logits,
    dominance_preservation_rate,
    dominance_utility_preservation_rate,
    head_target_correlation_matrix,
    output_correlation_matrix,
    pairwise_head_leakage_matrix,
    reversal_accuracy,
)
from pareto.models.vector_quality import score_with_preference


EVAL_PREFERENCES = torch.tensor(
    [
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70],
        [0.25, 0.25, 0.25, 0.25],
    ],
    dtype=torch.float32,
)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _matrix_to_list(matrix: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.detach().cpu()]


def head_leakage_gap(matrix: torch.Tensor) -> float:
    matrix = matrix.detach().cpu().float()
    diag = matrix.diag().mean()
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    offdiag = matrix[mask].mean()
    return float((diag - offdiag).item())


def evaluate_vector_model(
    model: torch.nn.Module,
    records_by_id: dict[str, dict],
    pairs: dict[str, list[dict]],
    device: torch.device | str = "cpu",
    scorer: torch.nn.Module | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    model.to(device)
    if scorer is not None:
        scorer.eval()
        scorer.to(device)
    metrics: dict[str, Any] = {}

    with torch.no_grad():
        objective_pairs = pairs.get("objective", [])
        if objective_pairs:
            batch = _to_device(objective_pair_tensors(objective_pairs, records_by_id), device)
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            row_idx = torch.arange(q_a.shape[0], device=device)
            logits = q_a[row_idx, batch["objective_idx"]] - q_b[row_idx, batch["objective_idx"]]
            metrics["obj_acc_mean"] = binary_accuracy_from_logits(logits, batch["labels"])
            per_objective = {}
            for idx, name in enumerate(OBJECTIVE_NAMES):
                mask = batch["objective_idx"] == idx
                if mask.any():
                    per_objective[name] = binary_accuracy_from_logits(logits[mask], batch["labels"][mask])
            metrics["obj_acc"] = per_objective

        preference_pairs = pairs.get("preference", [])
        if preference_pairs:
            batch = _to_device(preference_pair_tensors(preference_pairs, records_by_id), device)
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            score_fn = scorer if scorer is not None else score_with_preference
            logits = score_fn(q_a, batch["w"]) - score_fn(q_b, batch["w"])
            metrics["pref_acc"] = binary_accuracy_from_logits(logits, batch["labels"])

        reversal_pairs = pairs.get("reversal", [])
        if reversal_pairs:
            batch = _to_device(reversal_pair_tensors(reversal_pairs, records_by_id), device)
            q_a = model(batch["x_a"])
            q_b = model(batch["x_b"])
            score_fn = scorer if scorer is not None else score_with_preference
            logits_1 = score_fn(q_a, batch["w_1"]) - score_fn(q_b, batch["w_1"])
            logits_2 = score_fn(q_a, batch["w_2"]) - score_fn(q_b, batch["w_2"])
            metrics["rev_acc"] = reversal_accuracy(logits_1, logits_2, batch["labels_1"], batch["labels_2"])
            metrics["reversal_same_sign_rate"] = float(((logits_1 >= 0) == (logits_2 >= 0)).float().mean().item())

        dominance_pairs = pairs.get("dominance", [])
        if dominance_pairs:
            batch = _to_device(dominance_pair_tensors(dominance_pairs, records_by_id), device)
            q_dom = model(batch["x_dom"])
            q_sub = model(batch["x_sub"])
            metrics["dpr_head"] = dominance_preservation_rate(q_dom, q_sub)
            metrics["dpr_utility"] = dominance_utility_preservation_rate(q_dom, q_sub, EVAL_PREFERENCES.to(device))

        ids = list(records_by_id)
        x_all = make_feature_tensor(records_by_id, ids).to(device)
        q_all = model(x_all).detach().cpu()
        targets = objective_target_tensor(records_by_id, ids)
        q_by_id = {sample_id: q_all[idx] for idx, sample_id in enumerate(ids)}
        if scorer is not None:
            utility_by_w = torch.stack(
                [
                    scorer(q_all.to(device), w.reshape(1, -1).repeat(q_all.shape[0], 1).to(device)).detach().cpu()
                    for w in EVAL_PREFERENCES
                ],
                dim=1,
            )
        else:
            utility_by_w = q_all @ EVAL_PREFERENCES.T
        metrics["utility_w_sensitivity_mean_std"] = float(utility_by_w.std(dim=1, unbiased=False).mean().item())
        metrics["utility_eff_safety_abs_diff_mean"] = float((utility_by_w[:, 0] - utility_by_w[:, 1]).abs().mean().item())
        leakage = pairwise_head_leakage_matrix(q_by_id, objective_pairs)
        metrics["pairwise_head_leakage_matrix"] = _matrix_to_list(leakage)
        metrics["head_leakage_diag_offdiag_gap"] = head_leakage_gap(leakage)
        metrics["head_target_correlation_matrix"] = _matrix_to_list(head_target_correlation_matrix(q_all, targets))
        metrics["output_correlation_matrix"] = _matrix_to_list(output_correlation_matrix(q_all))
        metrics["output_std_per_head"] = [float(value) for value in q_all.std(dim=0, unbiased=False)]

    return metrics


def evaluate_scalar_model(
    model: torch.nn.Module,
    records_by_id: dict[str, dict],
    pairs: dict[str, list[dict]],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    model.to(device)
    metrics: dict[str, Any] = {}

    with torch.no_grad():
        preference_pairs = pairs.get("preference", [])
        preference_scores: list[torch.Tensor] = []
        if preference_pairs:
            batch = _to_device(preference_pair_tensors(preference_pairs, records_by_id), device)
            score_a = model(batch["x_a"], batch["w"])
            score_b = model(batch["x_b"], batch["w"])
            logits = score_a - score_b
            preference_scores.extend([score_a.detach().cpu(), score_b.detach().cpu()])
            metrics["pref_acc"] = binary_accuracy_from_logits(logits, batch["labels"])

        reversal_pairs = pairs.get("reversal", [])
        if reversal_pairs:
            batch = _to_device(reversal_pair_tensors(reversal_pairs, records_by_id), device)
            logits_1 = model(batch["x_a"], batch["w_1"]) - model(batch["x_b"], batch["w_1"])
            logits_2 = model(batch["x_a"], batch["w_2"]) - model(batch["x_b"], batch["w_2"])
            metrics["rev_acc"] = reversal_accuracy(logits_1, logits_2, batch["labels_1"], batch["labels_2"])
            metrics["reversal_same_sign_rate"] = float(((logits_1 >= 0) == (logits_2 >= 0)).float().mean().item())

        dominance_pairs = pairs.get("dominance", [])
        if dominance_pairs:
            batch = _to_device(dominance_pair_tensors(dominance_pairs, records_by_id), device)
            rates = []
            for w in EVAL_PREFERENCES.to(device):
                w_batch = w.reshape(1, -1).repeat(batch["x_dom"].shape[0], 1)
                diff = model(batch["x_dom"], w_batch) - model(batch["x_sub"], w_batch)
                rates.append((diff >= 0).float().mean())
            metrics["dpr_utility"] = float(torch.stack(rates).mean().item())

        ids = list(records_by_id)
        x_all = make_feature_tensor(records_by_id, ids).to(device)
        scores = []
        for w in EVAL_PREFERENCES.to(device):
            w_batch = w.reshape(1, -1).repeat(x_all.shape[0], 1)
            scores.append(model(x_all, w_batch).detach().cpu())
        scores_by_w = torch.stack(scores, dim=1)
        metrics["w_sensitivity_mean_std"] = float(scores_by_w.std(dim=1, unbiased=False).mean().item())
        metrics["w_sensitivity_eff_safety_abs_diff_mean"] = float((scores_by_w[:, 0] - scores_by_w[:, 1]).abs().mean().item())
        all_scores = torch.cat(scores)
        if preference_scores:
            all_scores = torch.cat([all_scores, *preference_scores])
        metrics["output_std"] = float(all_scores.std(unbiased=False).item())

    return metrics


def load_split_bundle(records_root: str | Path, pairs_root: str | Path, split: str) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    return load_split_records(records_root, split), load_split_pairs(pairs_root, split)
