from __future__ import annotations

import pytest

from pareto.rl.paper_weighted_mapping_policy import (
    REQUIRED_WEIGHTED_MAPPING_APPROVAL,
    build_weighted_mapping_policy,
    validate_weighted_mapping_policy,
)


def test_weighted_mapping_stays_blocked_without_exact_reviewer_phrase():
    row = build_weighted_mapping_policy(approval_phrase="close but not exact")

    assert row["status"] == "missing_blocker"
    assert row["mapping_candidate"] == "Weighted-RL -> weighted_proxy"
    assert "exact reviewer approval" in row["blocker"]


def test_weighted_mapping_exact_phrase_creates_non_executing_preview():
    row = build_weighted_mapping_policy(approval_phrase=REQUIRED_WEIGHTED_MAPPING_APPROVAL)

    assert row["status"] == "implemented_guarded_preview"
    assert row["executes_training_now"] is False
    assert row["mapping_candidate"] == "Weighted-RL -> weighted_proxy"
    validate_weighted_mapping_policy(row)


def test_weighted_mapping_rejects_implemented_status_without_phrase():
    row = {
        "status": "implemented_guarded_preview",
        "mapping_candidate": "Weighted-RL -> weighted_proxy",
        "approval_phrase": "not enough",
        "executes_training_now": False,
    }

    with pytest.raises(ValueError, match="exact reviewer approval"):
        validate_weighted_mapping_policy(row)
