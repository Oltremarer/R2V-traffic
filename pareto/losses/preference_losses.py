from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from pareto.models.vector_quality import score_with_preference


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _score(q: torch.Tensor, w: torch.Tensor, scorer: ScoreFn | None = None) -> torch.Tensor:
    if scorer is None:
        return score_with_preference(q, w)
    return scorer(q, w)


def objective_pair_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    objective_idx: torch.Tensor,
    labels: torch.Tensor,
    objective_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_idx = torch.arange(q_a.shape[0], device=q_a.device)
    logits = q_a[batch_idx, objective_idx.long()] - q_b[batch_idx, objective_idx.long()]
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    if objective_weights is not None:
        loss = loss * objective_weights.to(q_a.device)[objective_idx.long()]
    return loss.mean()


def preference_pair_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w: torch.Tensor,
    labels: torch.Tensor,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    logits = _score(q_a, w, scorer) - _score(q_b, w, scorer)
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def preference_margin_regression_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w: torch.Tensor,
    target_margin: torch.Tensor,
    clip: float = 2.0,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    pred_margin = _score(q_a, w, scorer) - _score(q_b, w, scorer)
    target = target_margin.float().to(pred_margin.device).clamp(-float(clip), float(clip))
    return F.smooth_l1_loss(pred_margin, target)


def preference_hinge_margin_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    logits = _score(q_a, w, scorer) - _score(q_b, w, scorer)
    signed = (2.0 * labels.float().to(logits.device) - 1.0) * logits
    return torch.relu(float(margin) - signed).pow(2).mean()


def reversal_pair_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w_1: torch.Tensor,
    w_2: torch.Tensor,
    labels_1: torch.Tensor,
    labels_2: torch.Tensor,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    logits_1 = _score(q_a, w_1, scorer) - _score(q_b, w_1, scorer)
    logits_2 = _score(q_a, w_2, scorer) - _score(q_b, w_2, scorer)
    loss_1 = F.binary_cross_entropy_with_logits(logits_1, labels_1.float())
    loss_2 = F.binary_cross_entropy_with_logits(logits_2, labels_2.float())
    return loss_1 + loss_2


def reversal_margin_regression_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w_1: torch.Tensor,
    w_2: torch.Tensor,
    margin_1: torch.Tensor,
    margin_2: torch.Tensor,
    clip: float = 2.0,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    loss_1 = preference_margin_regression_loss(q_a, q_b, w_1, margin_1, clip=clip, scorer=scorer)
    loss_2 = preference_margin_regression_loss(q_a, q_b, w_2, margin_2, clip=clip, scorer=scorer)
    return loss_1 + loss_2


def reversal_hinge_margin_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    w_1: torch.Tensor,
    w_2: torch.Tensor,
    labels_1: torch.Tensor,
    labels_2: torch.Tensor,
    margin: float = 0.5,
    scorer: ScoreFn | None = None,
) -> torch.Tensor:
    loss_1 = preference_hinge_margin_loss(q_a, q_b, w_1, labels_1, margin=margin, scorer=scorer)
    loss_2 = preference_hinge_margin_loss(q_a, q_b, w_2, labels_2, margin=margin, scorer=scorer)
    return loss_1 + loss_2


def dominance_loss(q_dom: torch.Tensor, q_sub: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    violations = torch.relu(float(margin) - (q_dom - q_sub))
    return (violations ** 2).mean()


def dominance_utility_loss(
    q_dom: torch.Tensor,
    q_sub: torch.Tensor,
    w_eval: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    weights = w_eval.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    utility_diff = (q_dom - q_sub) @ weights.to(q_dom.device).T
    violations = torch.relu(float(margin) - utility_diff)
    return (violations ** 2).mean()


def isotonic_dominance_loss(
    q_dom: torch.Tensor,
    q_sub: torch.Tensor,
    objective_margins: torch.Tensor | None = None,
    margin_floor: float = 0.05,
) -> torch.Tensor:
    diff = q_dom.float() - q_sub.float()
    if objective_margins is None:
        target_margin = torch.full_like(diff, float(margin_floor))
    else:
        target_margin = objective_margins.float().to(diff.device).clamp_min(float(margin_floor))
    violations = torch.relu(target_margin - diff)
    return violations.pow(2).mean()


def calibration_loss(q: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    mean_loss = q.mean(dim=0).pow(2).mean()
    std = q.std(dim=0, unbiased=False)
    std_loss = (std - float(target_std)).pow(2).mean()
    return mean_loss + std_loss
