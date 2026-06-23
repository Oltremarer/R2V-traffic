from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.rl.formal_experiment_spec import FormalExperimentSpec
from pareto.rl.formal_preflight_checks import (
    run_preflight_checks,
    validate_shared_budget,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _model_dir(tmp_path: Path) -> tuple[str, str]:
    model_dir = tmp_path / "film_model"
    torch.manual_seed(123)
    model = build_conditioned_scalar_model(
        "film",
        input_dim=6,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        film_layers=1,
        head_layers=1,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "architecture": "film",
                "input_dim": 6,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "film_layers": 1,
                "head_layers": 1,
            },
        },
        model_dir / "model.pt",
    )
    digest = hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()
    return str(model_dir), digest


def _spec_payload(tmp_path: Path, reward_adapter: str = "film_scalar_potential") -> dict:
    objective_norm = tmp_path / "objective_norm.json"
    _write_json(objective_norm, {"hash": "objective_hash", "version": "test"})
    gate = tmp_path / "formal_gate_decision.json"
    _write_json(gate, {"ppo_formal_allowed": False, "representation_gate_pass": False})
    selection = tmp_path / "model_selection_report.json"
    _write_json(selection, {"selected": "film"})
    film_model_dir, film_hash = _model_dir(tmp_path)
    return {
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "cityflow_seed": 0,
        "policy_seed": 0,
        "model_seed": 0,
        "state_encoder_id": "hybrid_v1",
        "state_encoder_hash": "4d1c2b4e276043ac",
        "feature_norm_path": None,
        "objective_norm_path": str(objective_norm),
        "objective_normalizer_hash": "objective_hash",
        "film_model_dir": film_model_dir if reward_adapter == "film_scalar_potential" else None,
        "film_model_hash": film_hash if reward_adapter == "film_scalar_potential" else None,
        "film_model_selection_report": str(selection) if reward_adapter == "film_scalar_potential" else None,
        "film_training_commit": "8f0c9d4" if reward_adapter == "film_scalar_potential" else None,
        "reward_adapter": reward_adapter,
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
        "ppo_budget": {
            "batch_size": 128,
            "clip_eps": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "lr": 0.0003,
            "rollout_steps": 128,
            "total_env_steps": 1000,
            "update_epochs": 4,
        },
        "eval_protocol": "preference_sweep",
        "formal_gate_decision_path": str(gate),
        "approved_formal_spec": False,
    }


def test_preflight_stage_rejects_placeholder_hashes(tmp_path: Path):
    payload = _spec_payload(tmp_path)
    payload["state_encoder_hash"] = "hybrid_v1_hash_placeholder"
    spec = FormalExperimentSpec.from_dict(payload)

    with pytest.raises(ValueError, match="placeholder"):
        spec.validate_for_stage("preflight")


def test_run_preflight_checks_validates_paths_hashes_and_reward_sensitivity(tmp_path: Path):
    specs = [
        FormalExperimentSpec.from_dict(_spec_payload(tmp_path, "film_scalar_potential")),
        FormalExperimentSpec.from_dict(_spec_payload(tmp_path, "weighted_proxy")),
        FormalExperimentSpec.from_dict(_spec_payload(tmp_path, "env_reward")),
    ]

    report = run_preflight_checks(specs, root=tmp_path, device="cpu")

    assert report["passed"] is True
    assert report["env_rollout"] is False
    assert report["ppo_training"] is False
    assert report["policy_update"] is False
    assert report["performance_claim"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["stage_validation"]["passed"] is True
    assert checks["path_and_hash_validation"]["passed"] is True
    assert checks["shared_budget"]["passed"] is True
    assert checks["film_reward_sensitivity"]["passed"] is True
    assert checks["reward_components_finite"]["passed"] is True
    assert checks["actor_critic_preference_conditioning"]["passed"] is True


def test_shared_budget_rejects_budget_mismatch(tmp_path: Path):
    first = FormalExperimentSpec.from_dict(_spec_payload(tmp_path, "film_scalar_potential"))
    second_payload = _spec_payload(tmp_path, "weighted_proxy")
    second_payload["ppo_budget"] = dict(second_payload["ppo_budget"], total_env_steps=2000)
    second = FormalExperimentSpec.from_dict(second_payload)

    with pytest.raises(ValueError, match="ppo_budget"):
        validate_shared_budget([first, second])
