from __future__ import annotations

import pytest

from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
from pareto.rl.formal_pilot_runner import validate_formal_jinan_3seed_execution_limits
from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config


FORMAL_EXECUTION_SPEC = "configs/formal/jinan_3seed_ppo_formal_execution_locked.json"


def _config():
    return load_formal_ppo_dryrun_config(FORMAL_EXECUTION_SPEC)


def test_formal_jinan_3seed_execution_limits_accept_locked_budget():
    request = validate_formal_jinan_3seed_execution_limits(
        _config(),
        "vector_quality_potential",
        seed_id=2,
        approval_phrase=FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
        episodes=5,
        max_decision_steps_per_episode=120,
        rollout_steps=120,
    )

    assert request["seed_id"] == 2
    assert request["method"] == "vector_quality_potential"
    assert request["rollout_steps"] == 120


def test_formal_jinan_3seed_execution_limits_accept_locked_film_hash():
    request = validate_formal_jinan_3seed_execution_limits(
        _config(),
        "film_scalar_potential",
        seed_id=1,
        approval_phrase=FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
        episodes=5,
        max_decision_steps_per_episode=120,
        rollout_steps=120,
    )

    assert request["seed_id"] == 1
    assert request["method"] == "film_scalar_potential"


def test_formal_jinan_3seed_execution_limits_reject_scope_drift():
    with pytest.raises(ValueError, match="exact external approval phrase"):
        validate_formal_jinan_3seed_execution_limits(
            _config(),
            "vector_quality_potential",
            seed_id=0,
            approval_phrase="FORMAL JINAN 3-SEED EXECUTION GO",
            episodes=5,
            max_decision_steps_per_episode=120,
            rollout_steps=120,
        )

    with pytest.raises(ValueError, match="exact external approval phrase"):
        validate_formal_jinan_3seed_execution_limits(
            _config(),
            "vector_quality_potential",
            seed_id=0,
            approval_phrase="wrong",
            episodes=5,
            max_decision_steps_per_episode=120,
            rollout_steps=120,
        )

    with pytest.raises(ValueError, match="rollout_steps"):
        validate_formal_jinan_3seed_execution_limits(
            _config(),
            "vector_quality_potential",
            seed_id=0,
            approval_phrase=FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
            episodes=5,
            max_decision_steps_per_episode=120,
            rollout_steps=24,
        )

    with pytest.raises(ValueError, match="not approved"):
        validate_formal_jinan_3seed_execution_limits(
            _config(),
            "unapproved_method",
            seed_id=0,
            approval_phrase=FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
            episodes=5,
            max_decision_steps_per_episode=120,
            rollout_steps=120,
        )
