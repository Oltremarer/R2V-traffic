from __future__ import annotations

from collections import Counter

import pytest

from pareto.rl.action_diagnostics import summarize_action_distribution_guard


def test_action_guard_rejects_empty_actions():
    with pytest.raises(ValueError, match="produced no actions"):
        summarize_action_distribution_guard(Counter(), [])


def test_action_guard_rejects_global_single_action_collapse():
    with pytest.raises(ValueError, match="global single-action rate"):
        summarize_action_distribution_guard(Counter({1: 100}), [Counter({1: 50}), Counter({1: 50})])


def test_action_guard_rejects_per_intersection_collapse():
    with pytest.raises(ValueError, match="per-intersection single-action rate"):
        summarize_action_distribution_guard(
            Counter({0: 50, 1: 50}),
            [Counter({0: 50}), Counter({1: 50})],
            max_global_single_action_rate=0.95,
            max_intersection_single_action_rate=0.98,
        )


def test_action_guard_accepts_mixed_actions_below_thresholds():
    summary = summarize_action_distribution_guard(
        Counter({0: 45, 1: 35, 2: 20}),
        [Counter({0: 20, 1: 20, 2: 10}), Counter({0: 25, 1: 15, 2: 10})],
    )

    assert summary["total_actions"] == 100
    assert summary["global_single_action_rate"] == 0.45
    assert summary["per_intersection_single_action_rate_max"] == 0.5
    assert summary["action_guard_pass"] is True
