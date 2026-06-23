from __future__ import annotations

import torch

from pareto.rl.ppo_loss import clipped_ppo_loss


def test_clipped_ppo_loss_uses_old_log_probs_and_backpropagates():
    new_log_probs = torch.tensor([-0.1, -0.4, -0.2], requires_grad=True)
    old_log_probs = torch.tensor([-0.2, -0.2, -0.2])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    values = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
    returns = torch.tensor([1.0, 0.0, 0.5])
    entropy = torch.tensor([0.5, 0.6, 0.7])

    output = clipped_ppo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
        entropy=entropy,
        clip_eps=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    ratio = torch.exp(new_log_probs.detach() - old_log_probs)
    assert torch.allclose(output["ratio"], ratio)
    assert output["approx_kl"].item() == torch.mean(old_log_probs - new_log_probs.detach()).item()
    assert output["clip_fraction"].item() == torch.mean((ratio - 1.0).abs().gt(0.2).float()).item()
    assert torch.allclose(output["ratio_mean"], ratio.mean())
    assert torch.allclose(output["ratio_min"], ratio.min())
    assert torch.allclose(output["ratio_max"], ratio.max())
    assert torch.isfinite(output["total_loss"])
    output["total_loss"].backward()
    assert new_log_probs.grad is not None
    assert values.grad is not None
    assert torch.isfinite(new_log_probs.grad).all()
    assert torch.isfinite(values.grad).all()
