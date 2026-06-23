from __future__ import annotations

import pytest

from pareto.rl.paper_weighted_mapping_policy import REQUIRED_WEIGHTED_MAPPING_APPROVAL
from pareto.rl.paper_weighted_rl_policy_family import (
    build_weighted_rl_policy_family_request,
    validate_weighted_rl_policy_family_request,
    weighted_rl_policy_family_blockers,
)


def test_weighted_rl_discrete_family_enumerates_city_seed_preference_controllers():
    request = build_weighted_rl_policy_family_request()

    assert request["status"] == "request_only"
    assert request["executes_training_now"] is False
    assert len(request["controllers"]) == 75
    assert {row["city"] for row in request["controllers"]} == {"jinan", "hangzhou", "newyork_28x7"}
    assert {row["seed"] for row in request["controllers"]} == {0, 1, 2, 3, 4}
    assert len({row["fixed_preference_template"] for row in request["controllers"]}) == 5
    assert all(row["policy_conditioned_on_w"] is False for row in request["controllers"])
    assert weighted_rl_policy_family_blockers(request) == []
    validate_weighted_rl_policy_family_request(request)


def test_weighted_rl_rejects_single_conditioned_policy_without_mapping_phrase():
    with pytest.raises(ValueError, match="true Weighted-RL requires one controller per preference"):
        build_weighted_rl_policy_family_request(policy_family_type="single_conditioned_policy")


def test_weighted_rl_mapping_path_requires_exact_phrase():
    request = build_weighted_rl_policy_family_request(
        policy_family_type="weighted_proxy_mapping",
        mapping_approval_phrase="not exact",
    )

    assert request["status"] == "missing_blocker"
    assert any("exact reviewer approval" in blocker for blocker in weighted_rl_policy_family_blockers(request))


def test_weighted_rl_mapping_path_accepts_exact_phrase_as_non_executing_preview():
    request = build_weighted_rl_policy_family_request(
        policy_family_type="weighted_proxy_mapping",
        mapping_approval_phrase=REQUIRED_WEIGHTED_MAPPING_APPROVAL,
    )

    assert request["status"] == "mapping_approved_preview"
    assert request["executes_training_now"] is False
    validate_weighted_rl_policy_family_request(request)
