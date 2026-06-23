from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_ppo_config import FormalPPODryRunConfig
from pareto.rl.formal_ppo_trainer import run_synthetic_ppo_dry_run


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
                "model_seed": 123,
                "methods": ["film_scalar_potential"],
                "formal_gate_decision_path": str(gate),
                "pilot_spec_path": str(spec),
            },
            "ppo": {
                "algorithm_label": "PPO",
                "requires_clipped_objective": True,
                "rollout_steps": 16,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "update_epochs": 2,
                "minibatch_size": 8,
                "lr": 0.0003,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
                "normalize_advantages": True,
            },
            "model": {"obs_dim": 8, "preference_dim": 4, "action_dim": 4, "hidden_dim": 16},
        }
    )


def test_synthetic_ppo_dry_run_is_deterministic_for_same_seed(tmp_path: Path):
    config = _config(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    run_synthetic_ppo_dry_run(config, "film_scalar_potential", out_a)
    run_synthetic_ppo_dry_run(config, "film_scalar_potential", out_b)

    assert (out_a / "loss_debug.jsonl").read_text(encoding="utf-8") == (
        out_b / "loss_debug.jsonl"
    ).read_text(encoding="utf-8")
