from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config


def _payload() -> dict:
    return {
        "pilot": {
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "cityflow_seed": 0,
            "policy_seed": 0,
            "model_seed": 0,
            "methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
            "formal_gate_decision_path": "records/preformal_final/formal_gate_decision.json",
            "pilot_spec_path": "docs/pro_reviews/jinan_1seed_film_pilot_spec_2026-05-30.md",
        },
        "ppo": {
            "algorithm_label": "PPO",
            "requires_clipped_objective": True,
            "rollout_steps": 120,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "update_epochs": 4,
            "minibatch_size": 64,
            "lr": 0.0003,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantages": True,
        },
        "model": {"obs_dim": 193, "preference_dim": 4, "action_dim": 4, "hidden_dim": 64},
    }


def test_formal_ppo_config_requires_clipped_objective(tmp_path: Path):
    payload = _payload()
    payload["ppo"]["requires_clipped_objective"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="clipped"):
        load_formal_ppo_dryrun_config(path)


def test_formal_ppo_config_hash_changes_when_budget_changes(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload_a = _payload()
    payload_b = _payload()
    payload_b["ppo"]["rollout_steps"] = 60
    first.write_text(json.dumps(payload_a), encoding="utf-8")
    second.write_text(json.dumps(payload_b), encoding="utf-8")

    config_a = load_formal_ppo_dryrun_config(first)
    config_b = load_formal_ppo_dryrun_config(second)

    assert config_a.ppo_config_hash() != config_b.ppo_config_hash()
    assert config_a.ppo["clip_eps"] == 0.2


def test_formal_ppo_config_rejects_missing_required_hyperparameter(tmp_path: Path):
    payload = _payload()
    del payload["ppo"]["clip_eps"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="clip_eps"):
        load_formal_ppo_dryrun_config(path)
