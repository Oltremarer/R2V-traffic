from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_pilot_runner import run_formal_pilot_mock_dry_run
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig


def _config(tmp_path: Path) -> FormalPPODryRunConfig:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False}), encoding="utf-8")
    spec = tmp_path / "pilot.md"
    spec.write_text("pilot spec\n", encoding="utf-8")
    return FormalPPODryRunConfig.from_dict(
        {
            "pilot": {
                "scenario": "jinan",
                "traffic_file": "anon_3_4_jinan_real.json",
                "cityflow_seed": 0,
                "policy_seed": 0,
                "model_seed": 7,
                "methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
                "formal_gate_decision_path": str(gate),
                "pilot_spec_path": str(spec),
                "state_encoder_hash": "4d1c2b4e276043ac",
                "objective_normalizer_hash": "b2c55e7d2c42856a",
                "film_model_hash": "08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642",
            },
            "ppo": {
                "algorithm_label": "PPO",
                "requires_clipped_objective": True,
                "rollout_steps": 24,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "update_epochs": 1,
                "minibatch_size": 12,
                "lr": 0.0003,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
                "normalize_advantages": True,
            },
            "model": {"obs_dim": 193, "preference_dim": 4, "action_dim": 4, "hidden_dim": 16},
        }
    )


def test_formal_pilot_runner_mock_resume_loads_optimizer_state(tmp_path: Path):
    config = _config(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_formal_pilot_mock_dry_run(config, "weighted_proxy", first_dir, episodes=1, max_decision_steps_per_episode=3)
    resume_path = first_dir / "training_checkpoint_last.pt"

    run_formal_pilot_mock_dry_run(
        config,
        "weighted_proxy",
        second_dir,
        episodes=1,
        max_decision_steps_per_episode=3,
        resume_from=resume_path,
    )

    metadata = json.loads((second_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resume_loaded"] is True
    assert metadata["resume_from"] == str(resume_path)
    assert metadata["ppo_config_hash"] == config.ppo_config_hash()
