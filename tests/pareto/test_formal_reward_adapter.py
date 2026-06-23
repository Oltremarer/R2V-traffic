from __future__ import annotations

import torch

from pareto.rl.formal_reward_adapter import (
    EnvRewardAdapter,
    FiLMScalarPotentialRewardAdapter,
    VectorQDiagnosticRewardAdapter,
    WeightedProxyRewardAdapter,
)


class PreferenceSensitiveScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x[:, :4] * w).sum(dim=-1)


class FeatureVectorQ(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :4]


def test_film_reward_changes_when_preference_changes():
    obs_t = torch.tensor([0.0, 0.0, 0.0, 0.0])
    obs_tp1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    eff_adapter = FiLMScalarPotentialRewardAdapter(PreferenceSensitiveScalar(), [1.0, 0.0, 0.0, 0.0])
    safety_adapter = FiLMScalarPotentialRewardAdapter(PreferenceSensitiveScalar(), [0.0, 1.0, 0.0, 0.0])

    eff_reward, eff_debug = eff_adapter.compute(obs_t, obs_tp1, {}, {}, done=False)
    safety_reward, safety_debug = safety_adapter.compute(obs_t, obs_tp1, {}, {}, done=False)

    assert eff_reward != safety_reward
    assert eff_debug["adapter"] == "film_scalar_potential"
    assert safety_debug["adapter"] == "film_scalar_potential"


def test_weighted_proxy_and_env_reward_are_finite():
    weighted = WeightedProxyRewardAdapter([0.5, 0.5, 0.0, 0.0])
    reward, debug = weighted.compute(
        torch.zeros(4),
        torch.ones(4),
        {},
        {"efficiency": 1.0, "safety": 3.0, "fairness": 0.0, "stability": 0.0},
        done=False,
    )
    assert reward == 2.0
    assert debug["adapter"] == "weighted_proxy"

    env = EnvRewardAdapter()
    env_reward, env_debug = env.compute(torch.zeros(4), torch.ones(4), {}, {}, done=False, env_reward=1.25)
    assert env_reward == 1.25
    assert env_debug["adapter"] == "env_reward"


def test_vectorq_adapter_is_diagnostic_only():
    adapter = VectorQDiagnosticRewardAdapter(FeatureVectorQ(), [0.25, 0.25, 0.25, 0.25])
    reward, debug = adapter.compute(torch.zeros(4), torch.ones(4), {}, {}, done=False)

    assert isinstance(reward, float)
    assert debug["adapter"] == "vectorq_diagnostic_potential"
    assert debug["diagnostic_only"] is True
