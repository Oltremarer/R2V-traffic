from __future__ import annotations

import pytest

from pareto.rl.paper_c2t_scalar_command_request import (
    C2T_SCALAR_COMMAND_FAMILY,
    build_c2t_scalar_command_request,
    c2t_scalar_command_blockers,
    validate_c2t_scalar_command_request,
)


def test_c2t_command_request_rejects_env_reward_mapping():
    with pytest.raises(ValueError, match="env_reward is not C2T-scalar"):
        build_c2t_scalar_command_request(command_family="env_reward")


def test_c2t_command_request_is_blocked_without_concrete_command():
    request = build_c2t_scalar_command_request()

    assert request["status"] == "missing_blocker"
    assert c2t_scalar_command_blockers(request) == ["C2T-scalar: concrete scalar reward-learning command missing"]


def test_c2t_command_request_builds_non_executing_all_city_seed_preview():
    request = build_c2t_scalar_command_request(command_family=C2T_SCALAR_COMMAND_FAMILY)

    assert request["status"] == "request_only"
    assert request["executes_training_now"] is False
    assert len(request["rows"]) == 15
    assert {row["city"] for row in request["rows"]} == {"jinan", "hangzhou", "newyork_28x7"}
    assert {row["seed"] for row in request["rows"]} == {0, 1, 2, 3, 4}
    assert all(row["output_root"].startswith("records/paper_final/") for row in request["rows"])
    validate_c2t_scalar_command_request(request)


def test_c2t_command_request_rejects_executing_preview():
    request = build_c2t_scalar_command_request(command_family=C2T_SCALAR_COMMAND_FAMILY)
    request["rows"][0] = dict(request["rows"][0], executes_training_now=True)

    with pytest.raises(ValueError, match="non-executing"):
        validate_c2t_scalar_command_request(request)
