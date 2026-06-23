from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pareto.rl.real_cityflow_preflight_smoke import (
    assert_no_learning_artifacts,
    build_real_cityflow_preflight_metadata,
    sha256_file,
    validate_hash,
    validate_real_cityflow_preflight_artifacts,
    validate_real_cityflow_preflight_limits,
)


ROOT = Path(__file__).resolve().parents[2]


def test_real_cityflow_preflight_requires_explicit_flag():
    with pytest.raises(ValueError, match="--real_env_preflight"):
        validate_real_cityflow_preflight_limits(
            real_env_preflight=False,
            episodes=1,
            max_decision_steps_per_episode=3,
        )


def test_real_cityflow_preflight_budget_is_tiny():
    with pytest.raises(ValueError, match="episodes"):
        validate_real_cityflow_preflight_limits(
            real_env_preflight=True,
            episodes=2,
            max_decision_steps_per_episode=3,
        )
    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        validate_real_cityflow_preflight_limits(
            real_env_preflight=True,
            episodes=1,
            max_decision_steps_per_episode=4,
        )


def test_real_cityflow_reward_readiness_budget_covers_five_templates():
    validate_real_cityflow_preflight_limits(
        real_env_preflight=True,
        episodes=5,
        max_decision_steps_per_episode=1,
        reward_readiness=True,
    )
    with pytest.raises(ValueError, match="episodes must be exactly 5"):
        validate_real_cityflow_preflight_limits(
            real_env_preflight=True,
            episodes=1,
            max_decision_steps_per_episode=1,
            reward_readiness=True,
        )
    with pytest.raises(ValueError, match="max_decision_steps_per_episode must be exactly 1"):
        validate_real_cityflow_preflight_limits(
            real_env_preflight=True,
            episodes=5,
            max_decision_steps_per_episode=2,
            reward_readiness=True,
        )


def test_real_cityflow_reward_readiness_requires_real_artifacts(tmp_path: Path):
    normalizer_path = tmp_path / "objective_norm.json"
    normalizer_path.write_text(json.dumps({"hash": "norm_hash"}), encoding="utf-8")
    film_dir = tmp_path / "film"
    film_dir.mkdir()
    model_path = film_dir / "model.pt"
    model_path.write_bytes(b"checkpoint")
    film_hash = sha256_file(model_path)

    validate_real_cityflow_preflight_artifacts(
        method="weighted_proxy",
        reward_readiness=True,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
        film_model_dir=None,
    )
    validate_real_cityflow_preflight_artifacts(
        method="film_scalar_potential",
        reward_readiness=True,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
        film_model_dir=film_dir,
        film_model_hash=film_hash,
    )

    with pytest.raises(ValueError, match="objective normalizer"):
        validate_real_cityflow_preflight_artifacts(
            method="weighted_proxy",
            reward_readiness=True,
            objective_normalizer=None,
            film_model_dir=None,
        )
    with pytest.raises(ValueError, match="trained FiLM checkpoint"):
        validate_real_cityflow_preflight_artifacts(
            method="film_scalar_potential",
            reward_readiness=True,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=None,
        )


def test_real_cityflow_reward_readiness_hash_validation_is_hard(tmp_path: Path):
    normalizer_path = tmp_path / "objective_norm.json"
    normalizer_path.write_text(json.dumps({"hash": "norm_hash"}), encoding="utf-8")
    film_dir = tmp_path / "film"
    film_dir.mkdir()
    model_path = film_dir / "model.pt"
    model_path.write_bytes(b"checkpoint")
    film_hash = sha256_file(model_path)

    validate_hash("unit_test_hash", "abc", "abc")
    with pytest.raises(ValueError, match="unit_test_hash mismatch"):
        validate_hash("unit_test_hash", "abc", "def")

    validate_real_cityflow_preflight_artifacts(
        method="film_scalar_potential",
        reward_readiness=True,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
        film_model_dir=film_dir,
        film_model_hash=film_hash,
    )
    with pytest.raises(ValueError, match="objective_normalizer_hash mismatch"):
        validate_real_cityflow_preflight_artifacts(
            method="film_scalar_potential",
            reward_readiness=True,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="wrong",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )
    with pytest.raises(ValueError, match="film_model_hash mismatch"):
        validate_real_cityflow_preflight_artifacts(
            method="film_scalar_potential",
            reward_readiness=True,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash="wrong",
        )


def test_real_cityflow_preflight_metadata_is_no_learning():
    metadata = build_real_cityflow_preflight_metadata(
        method="weighted_proxy",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        cityflow_seed=0,
        policy_seed=0,
        model_seed=0,
        episodes=1,
        max_decision_steps_per_episode=3,
        min_action_time=30,
        state_encoder_id="hybrid_v1",
        state_encoder_hash="4d1c2b4e276043ac",
        expected_feature_hash="4d1c2b4e276043ac",
        objective_normalizer_path=None,
        objective_normalizer_hash=None,
        objective_normalizer_hash_expected=None,
        objective_normalizer_hash_verified=False,
        objective_normalizer_file_sha256=None,
        objective_normalizer_loaded=False,
        film_model_dir=None,
        film_model_hash=None,
        film_model_hash_expected=None,
        film_model_hash_verified=False,
        film_model_source=None,
        film_model_loaded=False,
        reward_readiness=False,
        preference_name="efficiency",
        w=[1.0, 0.0, 0.0, 0.0],
        preference_coverage=["efficiency"],
    )

    assert metadata["real_env_preflight"] is True
    assert metadata["tiny_cityflow_preflight"] is True
    assert metadata["mock_env"] is False
    assert metadata["real_env_rollout"] is True
    assert metadata["cityflow_env_constructed"] is True
    assert metadata["cityflow_step_called"] is True
    assert metadata["pilot_execution"] is False
    assert metadata["formal_experiment"] is False
    assert metadata["performance_claim"] is False
    assert metadata["not_for_main_results"] is True
    assert metadata["real_reward_readiness"] is False
    assert metadata["reward_readiness_no_learning"] is False
    assert metadata["ppo_training"] is False
    assert metadata["policy_update"] is False
    assert metadata["optimizer_step"] is False
    assert metadata["sim_seconds_per_method"] == 90


def test_real_cityflow_reward_readiness_metadata_records_real_artifacts():
    metadata = build_real_cityflow_preflight_metadata(
        method="film_scalar_potential",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        cityflow_seed=0,
        policy_seed=0,
        model_seed=0,
        episodes=5,
        max_decision_steps_per_episode=1,
        min_action_time=30,
        state_encoder_id="hybrid_v1",
        state_encoder_hash="4d1c2b4e276043ac",
        expected_feature_hash="4d1c2b4e276043ac",
        objective_normalizer_path="data/normalizers/jinan/objective_norm_smoke3600.json",
        objective_normalizer_hash="b2c55e7d2c42856a",
        objective_normalizer_hash_expected="b2c55e7d2c42856a",
        objective_normalizer_hash_verified=True,
        objective_normalizer_file_sha256="normalizer_file_sha256",
        objective_normalizer_loaded=True,
        film_model_dir="model_weights/cond_scalar/jinan/preformal_final/film_rich_v2",
        film_model_hash="08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642",
        film_model_hash_expected="08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642",
        film_model_hash_verified=True,
        film_model_source="trained_film_checkpoint",
        film_model_loaded=True,
        reward_readiness=True,
        preference_name="episode_fixed_5_templates",
        w=[],
        preference_coverage=["efficiency", "safety", "fairness", "stability", "balanced"],
    )

    assert metadata["real_env_preflight"] is True
    assert metadata["real_reward_readiness"] is True
    assert metadata["reward_readiness_no_learning"] is True
    assert metadata["objective_normalizer_loaded"] is True
    assert metadata["objective_normalizer_hash"] == "b2c55e7d2c42856a"
    assert metadata["objective_normalizer_hash_expected"] == "b2c55e7d2c42856a"
    assert metadata["objective_normalizer_hash_verified"] is True
    assert metadata["objective_normalizer_file_sha256"] == "normalizer_file_sha256"
    assert metadata["film_model_loaded"] is True
    assert metadata["film_model_hash"] == "08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642"
    assert metadata["film_model_hash_expected"] == "08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642"
    assert metadata["film_model_hash_verified"] is True
    assert metadata["film_model_source"] == "trained_film_checkpoint"
    assert metadata["preference_coverage"] == ["efficiency", "safety", "fairness", "stability", "balanced"]
    assert metadata["sim_seconds_per_method"] == 150
    assert metadata["ppo_training"] is False
    assert metadata["policy_update"] is False
    assert metadata["optimizer_step"] is False


def test_real_cityflow_preflight_rejects_learning_artifacts(tmp_path: Path):
    (tmp_path / "loss_debug.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learning artifacts"):
        assert_no_learning_artifacts(tmp_path)

    (tmp_path / "loss_debug.jsonl").unlink()
    (tmp_path / "training_checkpoint_last.pt").write_bytes(b"not a real checkpoint")
    with pytest.raises(ValueError, match="learning artifacts"):
        assert_no_learning_artifacts(tmp_path)


def test_real_cityflow_preflight_cli_blocks_without_flag(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/real_cityflow_preflight_smoke.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "env_reward",
            "--out_dir",
            str(tmp_path / "blocked"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "requires --real_env_preflight" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked" / "metadata.json").exists()


def test_real_cityflow_preflight_allowed_artifacts_pass(tmp_path: Path):
    for name, payload in {
        "metadata.json": {"real_env_preflight": True},
        "status.json": {"status": "REAL_PREFLIGHT_DONE"},
        "reward_components.jsonl": "{}\n",
        "action_debug.jsonl": "{}\n",
        "preflight_metrics.jsonl": "{}\n",
        "stdout.log": "",
        "stderr.log": "",
        "REAL_PREFLIGHT_DONE": "done\n",
    }.items():
        path = tmp_path / name
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(payload, encoding="utf-8")

    assert_no_learning_artifacts(tmp_path)
