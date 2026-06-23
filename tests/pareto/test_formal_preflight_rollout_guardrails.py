from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_preflight_rollout import (
    build_preflight_rollout_metadata,
    validate_preflight_rollout_limits,
)


def test_preflight_rollout_requires_explicit_preflight_only():
    with pytest.raises(ValueError, match="--preflight_only"):
        validate_preflight_rollout_limits(
            preflight_only=False,
            episodes=2,
            max_decision_steps_per_episode=10,
        )


def test_preflight_rollout_rejects_large_episode_or_step_budget():
    with pytest.raises(ValueError, match="episodes"):
        validate_preflight_rollout_limits(
            preflight_only=True,
            episodes=3,
            max_decision_steps_per_episode=10,
        )
    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        validate_preflight_rollout_limits(
            preflight_only=True,
            episodes=2,
            max_decision_steps_per_episode=11,
        )


def test_preflight_metadata_forbids_performance_claims(tmp_path: Path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False, "representation_gate_pass": False}), encoding="utf-8")

    metadata = build_preflight_rollout_metadata(
        method="film_scalar_potential",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        cityflow_seed=0,
        policy_seed=0,
        model_seed=0,
        episodes=2,
        max_decision_steps_per_episode=10,
        min_action_time=30,
        state_encoder_id="hybrid_v1",
        state_encoder_hash="4d1c2b4e276043ac",
        objective_norm_path="data/normalizers/jinan/objective_norm_smoke3600.json",
        objective_normalizer_hash="b2c55e7d2c42856a",
        reward_adapter="film_scalar_potential",
        formal_gate_decision_path=str(gate),
    )

    assert metadata["preflight_only"] is True
    assert metadata["formal_experiment"] is False
    assert metadata["performance_claim"] is False
    assert metadata["not_for_main_results"] is True
    assert metadata["env_rollout"] is True
    assert metadata["ppo_training"] is True
    assert metadata["policy_update"] is True
