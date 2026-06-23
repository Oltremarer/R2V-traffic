from __future__ import annotations

import torch

from pareto.rl.ppo_buffer import PPORolloutBuffer


def test_ppo_buffer_computes_finite_normalized_advantages_with_done_mask():
    buffer = PPORolloutBuffer()
    for idx in range(4):
        buffer.add(
            obs=torch.ones(3) * idx,
            w=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            action=idx % 2,
            reward=1.0,
            done=idx == 2,
            value=0.5,
            log_prob=-0.2,
        )

    batch = buffer.compute_returns_and_advantages(
        last_value=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
    )

    assert batch["obs"].shape == (4, 3)
    assert batch["w"].shape == (4, 4)
    assert batch["old_log_probs"].shape == (4,)
    assert torch.isfinite(batch["returns"]).all()
    assert torch.isfinite(batch["advantages"]).all()
    assert abs(float(batch["advantages"].mean())) < 1e-5
