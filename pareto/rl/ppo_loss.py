from __future__ import annotations

import torch


def clipped_ppo_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> dict[str, torch.Tensor]:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy_bonus = entropy.mean()
    approx_kl = (old_log_probs - new_log_probs.detach()).mean()
    clip_fraction = (ratio.detach() - 1.0).abs().gt(float(clip_eps)).float().mean()
    total_loss = policy_loss + float(value_loss_coef) * value_loss - float(entropy_coef) * entropy_bonus
    if not torch.isfinite(total_loss):
        raise ValueError("non-finite PPO total loss")
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_bonus": entropy_bonus,
        "ratio": ratio.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio.detach().mean(),
        "ratio_min": ratio.detach().min(),
        "ratio_max": ratio.detach().max(),
    }
