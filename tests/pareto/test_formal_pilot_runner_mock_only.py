from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pareto.rl.formal_pilot_runner import run_formal_pilot_mock_dry_run
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig


def _config(tmp_path: Path) -> tuple[FormalPPODryRunConfig, str]:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False}), encoding="utf-8")
    spec = tmp_path / "pilot.md"
    spec.write_text("pilot spec\n", encoding="utf-8")
    spec_hash = hashlib.sha256(spec.read_bytes()).hexdigest()[:16]
    return (
        FormalPPODryRunConfig.from_dict(
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
        ),
        spec_hash,
    )


def test_formal_pilot_runner_mock_only_outputs_hash_lock_metadata(tmp_path: Path):
    config, spec_hash = _config(tmp_path)
    out_dir = tmp_path / "film"

    result = run_formal_pilot_mock_dry_run(
        config,
        "film_scalar_potential",
        out_dir,
        episodes=1,
        max_decision_steps_per_episode=1,
    )

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert result["status"] == "MOCK_DRY_RUN_DONE"
    assert metadata["pilot_runner_skeleton"] is True
    assert metadata["mock_env"] is True
    assert metadata["real_env_rollout"] is False
    assert metadata["real_ppo_update"] is False
    assert metadata["cityflow_optimizer_step"] is False
    assert metadata["pilot_execution"] is False
    assert metadata["performance_claim"] is False
    assert metadata["state_encoder_hash"] == "4d1c2b4e276043ac"
    assert metadata["objective_normalizer_hash"] == "b2c55e7d2c42856a"
    assert metadata["film_model_hash"] == "08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642"
    assert metadata["ppo_config_hash"] == config.ppo_config_hash()
    assert metadata["pilot_spec_hash"] == spec_hash
    assert metadata["pilot_spec_hash_verified"] is True
    assert "performance" not in metadata.get("result_mode", "")
    assert "ranking" not in metadata.get("result_mode", "")


def test_formal_pilot_runner_mock_only_rejects_missing_hash_lock(tmp_path: Path):
    config, _ = _config(tmp_path)
    config.pilot.pop("objective_normalizer_hash")

    try:
        run_formal_pilot_mock_dry_run(
            config,
            "film_scalar_potential",
            tmp_path / "blocked",
            episodes=1,
            max_decision_steps_per_episode=1,
        )
    except ValueError as exc:
        assert "objective_normalizer_hash" in str(exc)
    else:
        raise AssertionError("missing objective_normalizer_hash should fail")
