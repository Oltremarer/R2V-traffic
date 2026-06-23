from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_runner import (
    FORMAL_PILOT_WIRING_METHOD,
    _formal_pilot_wiring_context,
    _tiny_update_metadata,
    run_formal_pilot_wiring_dry_run,
    validate_formal_pilot_wiring_dry_run_limits,
)
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig


def _config(tmp_path: Path) -> FormalPPODryRunConfig:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"formal_representation_pass": True}), encoding="utf-8")
    spec = tmp_path / "pilot.md"
    spec.write_text("pilot spec\n", encoding="utf-8")
    packet = tmp_path / "representation_formal_gate_packet.json"
    packet.write_text(json.dumps({"formal_representation_pass": True}), encoding="utf-8")
    return FormalPPODryRunConfig.from_dict(
        {
            "pilot": {
                "scenario": "jinan",
                "traffic_file": "anon_3_4_jinan_real.json",
                "cityflow_seed": 0,
                "policy_seed": 0,
                "model_seed": 0,
                "methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
                "formal_gate_decision_path": str(gate),
                "pilot_spec_path": str(spec),
                "representation_gate_packet_path": str(packet),
                "representation_run_id": "v3_rev15_m02_iso3_c15_u03",
                "state_encoder_hash": "4d1c2b4e276043ac",
                "objective_normalizer_hash": "norm_hash",
                "vector_model_hash": "vector_hash",
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


def test_formal_pilot_wiring_limits_are_seed0_jinan_only(tmp_path: Path):
    config = _config(tmp_path)

    validate_formal_pilot_wiring_dry_run_limits(config, episodes=1, max_decision_steps_per_episode=3)

    with pytest.raises(ValueError, match="episodes must be exactly 1"):
        validate_formal_pilot_wiring_dry_run_limits(config, episodes=2, max_decision_steps_per_episode=3)
    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        validate_formal_pilot_wiring_dry_run_limits(config, episodes=1, max_decision_steps_per_episode=4)

    bad_seed = _config(tmp_path)
    bad_seed.pilot["cityflow_seed"] = 1
    with pytest.raises(ValueError, match="cityflow_seed=0"):
        validate_formal_pilot_wiring_dry_run_limits(bad_seed, episodes=1, max_decision_steps_per_episode=3)


def test_formal_pilot_wiring_context_has_no_formal_or_performance_flags(tmp_path: Path):
    config = _config(tmp_path)

    context = _formal_pilot_wiring_context(
        config,
        vector_model_dir=tmp_path / "vector",
        vector_model_hash="vector_hash",
    )

    assert context["approval_phrase"] == "PARETO PPO FORMAL-PILOT DRY-RUN GO"
    assert context["formal_pilot_wiring_dry_run"] is True
    assert context["formal_experiment"] is False
    assert context["performance_claim"] is False
    assert context["traffic_result_value_reading_executed"] is False
    assert context["method_ranking_executed"] is False
    assert context["paper_result_claim"] is False
    assert context["formal_experiment_requires_new_pro_approval"] is True
    assert context["representation_run_id"] == "v3_rev15_m02_iso3_c15_u03"


def test_formal_pilot_wiring_wrapper_uses_vector_method_and_suppresses_exploratory_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = _config(tmp_path)
    normalizer = tmp_path / "objective_norm.json"
    normalizer.write_text(json.dumps({"hash": "norm_hash"}), encoding="utf-8")
    vector_dir = tmp_path / "vector"
    vector_dir.mkdir()
    (vector_dir / "model.pt").write_bytes(b"vector checkpoint")
    captured: dict = {}

    def fake_bounded_runner(*args, **kwargs):
        captured["method"] = args[1]
        captured["kwargs"] = kwargs
        out = Path(args[2])
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(
            json.dumps(
                {
                    "formal_pilot_wiring_dry_run": True,
                    "vector_model_loaded": True,
                    "vector_model_hash_verified": True,
                    "reward_adapter": FORMAL_PILOT_WIRING_METHOD,
                    "formal_experiment": False,
                    "performance_claim": False,
                }
            ),
            encoding="utf-8",
        )
        (out / "status.json").write_text(json.dumps({"status": "FORMAL_PILOT_WIRING_DRY_RUN_DONE"}), encoding="utf-8")
        return {"status": "FORMAL_PILOT_WIRING_DRY_RUN_DONE", "formal_pilot_wiring_dry_run": True}

    monkeypatch.setattr("pareto.rl.formal_pilot_runner.run_bounded_jinan_pilot_dry_run", fake_bounded_runner)

    run_formal_pilot_wiring_dry_run(
        config,
        tmp_path / "out",
        episodes=1,
        max_decision_steps_per_episode=3,
        objective_normalizer=normalizer,
        objective_normalizer_hash="norm_hash",
        vector_model_dir=vector_dir,
        vector_model_hash="vector_hash",
    )

    assert captured["method"] == FORMAL_PILOT_WIRING_METHOD
    assert captured["kwargs"]["formal_execution_context"]["formal_pilot_wiring_dry_run"] is True
    assert captured["kwargs"]["require_final_method_list"] is False
    assert captured["kwargs"]["approved_methods"] == (FORMAL_PILOT_WIRING_METHOD,)
    assert not (tmp_path / "out" / "pilot_status.json").exists()
    assert not (tmp_path / "out" / "pilot_guard_report.md").exists()


def test_formal_pilot_wiring_metadata_preserves_actual_and_config_vector_hash(tmp_path: Path):
    config = _config(tmp_path)

    metadata = _tiny_update_metadata(
        config,
        FORMAL_PILOT_WIRING_METHOD,
        episodes=1,
        max_decision_steps_per_episode=3,
        max_policy_updates=1,
        objective_normalizer_path="objective_norm.json",
        objective_normalizer_hash="norm_hash",
        objective_normalizer_hash_expected="norm_hash",
        objective_normalizer_hash_verified=True,
        objective_normalizer_used_by_reward=False,
        film_model_dir=None,
        film_model_hash=None,
        film_model_hash_expected=None,
        film_model_hash_verified=False,
        film_model_source=None,
        film_model_loaded=False,
        vector_model_dir="vector",
        vector_model_hash="actual_vector_hash",
        vector_model_hash_expected="vector_hash",
        vector_model_hash_verified=True,
        vector_model_loaded=True,
        vector_model_config={"architecture": "residual_tower", "score_mode": "low_rank_interaction"},
        observed_feature_hash="4d1c2b4e276043ac",
        observed_obs_dim=193,
    )

    assert metadata["vector_model_required"] is True
    assert metadata["vector_model_hash"] == "actual_vector_hash"
    assert metadata["vector_model_hash_expected"] == "vector_hash"
    assert metadata["vector_model_hash_config"] == "vector_hash"
    assert metadata["vector_model_hash_verified"] is True
    assert metadata["vector_model_architecture"] == "residual_tower"
