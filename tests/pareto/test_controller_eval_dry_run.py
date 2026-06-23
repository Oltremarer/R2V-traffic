from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from pareto.eval.evaluate_controller import run_dry_run as run_eval_dry_run
from pareto.eval.preference_sweep import run_dry_run as run_sweep_dry_run


def _write_spec(tmp_path: Path) -> Path:
    gate = tmp_path / "formal_gate_decision.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False}), encoding="utf-8")
    spec = {
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
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_evaluate_controller_dry_run_writes_metadata(tmp_path: Path):
    payload = run_eval_dry_run(Namespace(spec=str(_write_spec(tmp_path)), out_dir=str(tmp_path / "eval")))

    assert payload["dry_run"] is True
    assert payload["env_rollout"] is False
    assert payload["performance_claim"] is False
    assert (tmp_path / "eval" / "EVAL_DRY_RUN_DONE").exists()


def test_preference_sweep_dry_run_writes_preferences(tmp_path: Path):
    payload = run_sweep_dry_run(Namespace(spec=str(_write_spec(tmp_path)), out_dir=str(tmp_path / "sweep")))

    assert payload["dry_run"] is True
    assert payload["env_rollout"] is False
    assert payload["performance_claim"] is False
    assert payload["eval_preferences"] == ["efficiency", "safety", "fairness", "stability", "balanced"]
    assert (tmp_path / "sweep" / "SWEEP_DRY_RUN_DONE").exists()
