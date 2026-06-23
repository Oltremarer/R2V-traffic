from __future__ import annotations

from collections import Counter

import pytest

from pareto.rl.action_diagnostics import summarize_paper_final_action_diversity


def test_paper_action_diagnostics_rejects_learned_policy_sequence_collapse():
    with pytest.raises(ValueError, match="learned policy action-sequence collapse"):
        summarize_paper_final_action_diversity(
            action_counts=Counter({0: 100}),
            action_sequence_hashes=["same"] * 5,
            method_kind="learned_policy",
            city="jinan",
            method="VectorQ-PPO",
            preference_id="balanced",
            max_single_action_rate=0.95,
        )


def test_paper_action_diagnostics_records_reference_determinism_without_failing():
    summary = summarize_paper_final_action_diversity(
        action_counts=Counter({1: 100}),
        action_sequence_hashes=["same"] * 5,
        method_kind="reference_policy",
        city="jinan",
        method="MaxPressure",
        preference_id="not_applicable",
        max_single_action_rate=1.0,
    )

    assert summary["deterministic_reference_repeat"] is True
    assert summary["action_guard_pass"] is True


def test_paper_action_diagnostics_accepts_diverse_learned_policy():
    summary = summarize_paper_final_action_diversity(
        action_counts=Counter({0: 40, 1: 30, 2: 20, 3: 10}),
        action_sequence_hashes=["a", "b", "c", "d", "e"],
        method_kind="learned_policy",
        city="hangzhou",
        method="Weighted-RL",
        preference_id="safety_focused",
    )

    assert summary["unique_actions_used"] == 4
    assert summary["repeated_sequence_hash_count"] == 0
