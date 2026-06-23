from __future__ import annotations

from typing import Mapping, Sequence

import torch

from pareto.rl.reward_adapter import (
    ScalarQualityRewardAdapter,
    VectorQRewardAdapter,
    WeightedProxyRewardAdapter,
)


class FiLMScalarPotentialRewardAdapter(ScalarQualityRewardAdapter):
    name = "film_scalar_potential"


class VectorQDiagnosticRewardAdapter(VectorQRewardAdapter):
    name = "vectorq_diagnostic_potential"
    diagnostic_only = True

    def compute(
        self,
        obs_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        objectives_t: Mapping[str, float],
        objectives_tp1: Mapping[str, float],
        done: bool,
    ) -> tuple[float, dict]:
        reward, debug = super().compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
        debug["diagnostic_only"] = True
        return reward, debug


class EnvRewardAdapter:
    name = "env_reward"

    def compute(
        self,
        obs_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        objectives_t: Mapping[str, float],
        objectives_tp1: Mapping[str, float],
        done: bool,
        env_reward: float = 0.0,
    ) -> tuple[float, dict]:
        del obs_t, obs_tp1, objectives_t, objectives_tp1, done
        reward = float(env_reward)
        return reward, {
            "adapter": self.name,
            "env_reward": reward,
            "total_reward": reward,
        }


def build_weighted_proxy_adapter(w: Sequence[float] | torch.Tensor) -> WeightedProxyRewardAdapter:
    return WeightedProxyRewardAdapter(w)
