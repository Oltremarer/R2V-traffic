from __future__ import annotations

import pytest

from pareto.rl.mock_llmlight_env import MockLLMLightEnv


def test_mock_llmlight_env_requires_multi_intersection_action_list():
    env = MockLLMLightEnv(num_intersections=12, obs_dim=193, min_action_time=30, max_steps=2)
    state = env.reset()

    assert state.obs.shape == (12, 193)
    assert state.sim_time == 0
    assert len(state.objectives_norm) == 12

    with pytest.raises(ValueError, match="action_list length"):
        env.step([0, 1])

    next_state, rewards, done, info = env.step([0] * 12)
    assert next_state.obs.shape == (12, 193)
    assert len(rewards) == 12
    assert done is False
    assert info["sim_time"] == 30
    assert info["min_action_time"] == 30
