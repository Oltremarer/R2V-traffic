from __future__ import annotations

import math

import pytest

from pareto.rl.env_reward_sanity import summarize_env_rewards, validate_env_reward_sanity_limits


def test_env_reward_sanity_limits_are_no_learning_and_bounded():
    validate_env_reward_sanity_limits(episodes=1, max_decision_steps_per_episode=20)

    with pytest.raises(ValueError, match="episodes must be exactly 1"):
        validate_env_reward_sanity_limits(episodes=2, max_decision_steps_per_episode=20)

    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        validate_env_reward_sanity_limits(episodes=1, max_decision_steps_per_episode=21)


def test_summarize_env_rewards_marks_all_zero_as_warning_not_claim():
    rows = [
        {"env_reward": 0.0, "total_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
        {"env_reward": 0.0, "total_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
    ]

    summary = summarize_env_rewards(rows)

    assert summary["row_count"] == 2
    assert summary["finite"] is True
    assert summary["all_zero_reward"] is True
    assert summary["nonzero_reward_count"] == 0
    assert summary["env_reward_nonzero_rate"] == 0.0
    assert summary["env_reward_sources"] == ["cityflow_average_reward"]
    assert summary["recommendation"] == "env_reward_source_has_no_signal"
    assert "all_zero_reward" in summary["warnings"]


def test_summarize_env_rewards_reports_nonzero_rate_and_source():
    rows = [
        {"env_reward": -1.0, "total_reward": -1.0, "env_reward_source": "cityflow_average_reward"},
        {"env_reward": 0.0, "total_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
    ]

    summary = summarize_env_rewards(rows)

    assert summary["finite"] is True
    assert summary["all_zero_reward"] is False
    assert summary["nonzero_reward_count"] == 1
    assert summary["env_reward_nonzero_rate"] == 0.5
    assert summary["env_reward_sources"] == ["cityflow_average_reward"]
    assert summary["recommendation"] == "env_reward_source_has_signal"


def test_summarize_env_rewards_rejects_non_finite_values():
    rows = [{"env_reward": math.inf, "total_reward": math.inf}]

    summary = summarize_env_rewards(rows)

    assert summary["finite"] is False
    assert "non_finite_reward" in summary["warnings"]
