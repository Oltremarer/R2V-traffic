from __future__ import annotations

import torch

from pareto.rl.formal_pilot_runner import compute_pilot_reward


def _objectives(value: float = 0.0) -> dict[str, float]:
    return {
        "efficiency": value,
        "safety": value + 0.1,
        "fairness": value + 0.2,
        "stability": value + 0.3,
    }


def test_pilot_reward_adapters_use_method_specific_reward_sources():
    obs_t = torch.zeros(193)
    obs_tp1 = torch.ones(193)
    w = [0.25, 0.25, 0.25, 0.25]

    film_reward, film_debug = compute_pilot_reward(
        "film_scalar_potential", obs_t, obs_tp1, _objectives(0.0), _objectives(1.0), w, False, env_reward=2.5
    )
    assert film_debug["adapter"] == "film_scalar_potential"
    assert "scalar_quality_score_t" in film_debug
    assert "scalar_quality_score_tp1" in film_debug
    assert film_reward == film_debug["total_reward"]

    weighted_reward, weighted_debug = compute_pilot_reward(
        "weighted_proxy", obs_t, obs_tp1, _objectives(0.0), _objectives(1.0), w, False, env_reward=2.5
    )
    assert weighted_debug["adapter"] == "weighted_proxy"
    assert weighted_debug["proxy_objectives_norm_tp1"]["efficiency"] == 1.0
    assert weighted_reward != 2.5

    env_reward, env_debug = compute_pilot_reward(
        "env_reward", obs_t, obs_tp1, _objectives(0.0), _objectives(1.0), w, False, env_reward=2.5
    )
    assert env_debug["adapter"] == "env_reward"
    assert env_reward == 2.5
