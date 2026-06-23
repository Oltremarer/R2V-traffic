from __future__ import annotations

import copy

import pytest

from pareto.rl.paper_learned_baseline_guard import (
    build_learned_baseline_readiness,
    validate_learned_baseline_readiness,
)
from pareto.rl.paper_weighted_mapping_policy import REQUIRED_WEIGHTED_MAPPING_APPROVAL
from pareto.rl.paper_weighted_rl_policy_family import build_weighted_rl_policy_family_request


def test_learned_baseline_guard_rejects_stage_a_budget():
    with pytest.raises(ValueError, match="paper-scale"):
        build_learned_baseline_readiness(
            baseline="VectorQ-PPO",
            city="jinan",
            config={
                "model": {"hidden_dim": 64},
                "ppo": {"rollout_steps": 120, "minibatch_size": 12, "update_epochs": 1, "total_env_steps_per_seed": 120},
            },
            model_hash="a" * 64,
            objective_normalizer_hash="b" * 64,
        )


def test_learned_baseline_guard_requires_weighted_mapping_approval():
    readiness = build_learned_baseline_readiness(
        baseline="Weighted-RL",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
        mapping_approved=False,
    )

    assert readiness["status"] == "missing_blocker"
    assert "exact reviewer approval" in readiness["blocker"]


def test_learned_baseline_guard_ignores_boolean_mapping_approval_without_phrase():
    readiness = build_learned_baseline_readiness(
        baseline="Weighted-RL",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
        mapping_approved=True,
    )

    assert readiness["status"] == "missing_blocker"


def test_learned_baseline_guard_accepts_weighted_mapping_exact_phrase():
    readiness = build_learned_baseline_readiness(
        baseline="Weighted-RL",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
        mapping_approval_phrase=REQUIRED_WEIGHTED_MAPPING_APPROVAL,
    )

    assert validate_learned_baseline_readiness(readiness)["status"] == "implemented_guarded_preview"


def test_learned_baseline_guard_blocks_weighted_true_family_request_only():
    readiness = build_learned_baseline_readiness(
        baseline="Weighted-RL",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
        weighted_family_request=build_weighted_rl_policy_family_request(),
    )

    assert readiness["status"] == "missing_blocker"
    assert "request-only" in readiness["blocker"]
    assert "not final-ready" in readiness["blocker"]


def test_learned_baseline_guard_accepts_paper_scale_hash_locked_vectorq():
    readiness = build_learned_baseline_readiness(
        baseline="VectorQ-PPO",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
    )

    assert validate_learned_baseline_readiness(readiness)["status"] == "implemented_guarded_preview"


def test_learned_baseline_guard_keeps_c2t_scalar_unresolved():
    readiness = build_learned_baseline_readiness(
        baseline="C2T-scalar",
        city="jinan",
        model_hash="a" * 64,
        objective_normalizer_hash="b" * 64,
    )

    assert readiness["status"] == "missing_blocker"
    assert "unresolved_c2t_scalar_command" in readiness["blocker"]
