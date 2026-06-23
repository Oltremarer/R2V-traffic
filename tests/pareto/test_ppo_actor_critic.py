from __future__ import annotations

import torch

from pareto.rl.ppo_actor_critic import PreferenceConditionedActorCritic


def test_preference_conditioned_actor_critic_uses_obs_and_w():
    model = PreferenceConditionedActorCritic(obs_dim=8, preference_dim=4, action_dim=4, hidden_dim=16)
    obs = torch.randn(5, 8)
    w_eff = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(5, -1)
    w_safety = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).expand(5, -1)

    logits_eff, values_eff = model(obs, w_eff)
    logits_safety, values_safety = model(obs, w_safety)
    actions, log_probs, values = model.act(obs, w_eff)
    eval_log_probs, entropy, eval_values = model.evaluate_actions(obs, w_eff, actions)

    assert model.actor_input_dim == 12
    assert model.critic_input_dim == 12
    assert logits_eff.shape == (5, 4)
    assert values_eff.shape == (5,)
    assert torch.isfinite(logits_eff).all()
    assert torch.isfinite(values_eff).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(eval_log_probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(eval_values).all()
    assert not torch.allclose(logits_eff, logits_safety)
    assert not torch.allclose(values_eff, values_safety)
