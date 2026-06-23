from __future__ import annotations

import torch

from pareto.constants import OBJECTIVE_INDEX, OBJECTIVE_NAMES


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits >= 0).float()
    return float((preds == labels.float()).float().mean().item())


def reversal_accuracy(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    labels_1: torch.Tensor,
    labels_2: torch.Tensor,
) -> float:
    pred_1 = (logits_1 >= 0).float()
    pred_2 = (logits_2 >= 0).float()
    correct = (pred_1 == labels_1.float()) & (pred_2 == labels_2.float())
    return float(correct.float().mean().item())


def dominance_preservation_rate(q_dom: torch.Tensor, q_sub: torch.Tensor, margin: float = 0.0) -> float:
    preserved = (q_dom - q_sub >= float(margin)).all(dim=-1)
    return float(preserved.float().mean().item())


def dominance_utility_preservation_rate(
    q_dom: torch.Tensor,
    q_sub: torch.Tensor,
    w_eval: torch.Tensor,
    margin: float = 0.0,
) -> float:
    diff = q_dom.float() - q_sub.float()
    weights = w_eval.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    utility_diff = diff @ weights.T
    return float((utility_diff >= float(margin)).float().mean().item())


def _safe_corrcoef(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    denom = torch.sqrt((x.pow(2).sum(dim=0, keepdim=True).T @ x.pow(2).sum(dim=0, keepdim=True))).clamp_min(1e-8)
    return (x.T @ x) / denom


def head_target_correlation_matrix(q: torch.Tensor, objective_targets: torch.Tensor) -> torch.Tensor:
    q = q.float()
    objective_targets = objective_targets.float()
    q_centered = q - q.mean(dim=0, keepdim=True)
    target_centered = objective_targets - objective_targets.mean(dim=0, keepdim=True)
    numerator = q_centered.T @ target_centered
    denom = torch.sqrt(
        q_centered.pow(2).sum(dim=0, keepdim=True).T
        @ target_centered.pow(2).sum(dim=0, keepdim=True)
    ).clamp_min(1e-8)
    return numerator / denom


def pairwise_head_leakage_matrix(q_by_id: dict[str, torch.Tensor], objective_pairs: list[dict]) -> torch.Tensor:
    totals = torch.zeros(len(OBJECTIVE_NAMES), len(OBJECTIVE_NAMES), dtype=torch.float32)
    correct = torch.zeros_like(totals)
    for pair in objective_pairs:
        objective = pair["objective"]
        if objective not in OBJECTIVE_INDEX:
            continue
        a_id = pair["a_id"]
        b_id = pair["b_id"]
        if a_id not in q_by_id or b_id not in q_by_id:
            continue
        target_idx = OBJECTIVE_INDEX[objective]
        label = float(pair["label"])
        diffs = (q_by_id[a_id].float() - q_by_id[b_id].float()).reshape(-1)
        preds = (diffs >= 0).float()
        totals[:, target_idx] += 1.0
        correct[:, target_idx] += (preds == label).float()
    return correct / totals.clamp_min(1.0)


def output_correlation_matrix(q: torch.Tensor) -> torch.Tensor:
    return _safe_corrcoef(q)
