from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_experiment_spec import FormalExperimentSpec, load_formal_experiment_spec


def _base_spec_payload(tmp_path: Path) -> dict:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False, "wiring_smoke_allowed": True}), encoding="utf-8")
    return {
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "cityflow_seed": 0,
        "policy_seed": 0,
        "model_seed": 0,
        "state_encoder_id": "hybrid_v1",
        "state_encoder_hash": "state_hash",
        "feature_norm_path": None,
        "objective_norm_path": "objective_norm.json",
        "objective_normalizer_hash": "objective_hash",
        "film_model_dir": "model_weights/film",
        "film_model_hash": "film_hash",
        "film_model_selection_report": "records/model_selection.json",
        "film_training_commit": "abc123",
        "reward_adapter": "film_scalar_potential",
        "reward_scale": 1.0,
        "reward_clip": None,
        "reward_normalization": "none",
        "potential_gamma": 0.99,
        "mix_env_reward": False,
        "policy_conditioned_on_w": True,
        "critic_conditioned_on_w": True,
        "preference_sampling": "episode_fixed",
        "train_preferences": ["efficiency", "safety", "fairness", "stability", "balanced"],
        "eval_preferences": ["efficiency", "safety", "fairness", "stability", "balanced"],
        "min_action_time": 30,
        "action_interval_source": "LLMLight MIN_ACTION_TIME",
        "forbid_inner_step_direct_call": True,
        "ppo_budget": {"total_env_steps": 100, "rollout_steps": 10},
        "eval_protocol": "preference_sweep",
        "formal_gate_decision_path": str(gate),
        "approved_formal_spec": False,
    }


def test_formal_experiment_spec_requires_film_hash_for_film_adapter(tmp_path: Path):
    payload = _base_spec_payload(tmp_path)
    payload["film_model_hash"] = None

    with pytest.raises(ValueError, match="film_model_hash"):
        FormalExperimentSpec.from_dict(payload)


def test_formal_experiment_spec_requires_policy_conditioning_for_film(tmp_path: Path):
    payload = _base_spec_payload(tmp_path)
    payload["policy_conditioned_on_w"] = False

    with pytest.raises(ValueError, match="policy_conditioned_on_w"):
        FormalExperimentSpec.from_dict(payload)


def test_load_formal_experiment_spec_roundtrip_and_hash(tmp_path: Path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_base_spec_payload(tmp_path)), encoding="utf-8")

    spec = load_formal_experiment_spec(path)

    assert spec.reward_adapter == "film_scalar_potential"
    assert spec.approved_formal_spec is False
    assert spec.spec_hash() == load_formal_experiment_spec(path).spec_hash()
