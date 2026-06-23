from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_runner import (
    _method_display_name,
    _summarize_tiny_env_reward_rows,
    run_tiny_real_ppo_update_preflight,
)
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig
from pareto.rl.real_cityflow_preflight_smoke import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path, film_hash: str = "film_hash") -> FormalPPODryRunConfig:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False}), encoding="utf-8")
    spec = tmp_path / "tiny_spec.md"
    spec.write_text("tiny ppo update preflight spec\n", encoding="utf-8")
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
                "objective_normalizer_hash": "norm_hash",
                "film_model_hash": film_hash,
            },
            "ppo": {
                "algorithm_label": "PPO",
                "requires_clipped_objective": True,
                "rollout_steps": 24,
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
            "model": {"obs_dim": 193, "preference_dim": 4, "action_dim": 4, "hidden_dim": 16},
        }
    )


def _normalizer(tmp_path: Path) -> Path:
    path = tmp_path / "objective_norm.json"
    path.write_text(json.dumps({"hash": "norm_hash"}), encoding="utf-8")
    return path


def _film_dir(tmp_path: Path) -> tuple[Path, str]:
    film_dir = tmp_path / "film"
    film_dir.mkdir()
    model_path = film_dir / "model.pt"
    model_path.write_bytes(b"trained-film")
    return film_dir, sha256_file(model_path)


def test_tiny_real_ppo_update_rejects_budget_above_gate(tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    with pytest.raises(ValueError, match="episodes must be exactly 1"):
        run_tiny_real_ppo_update_preflight(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_episodes",
            episodes=2,
            max_decision_steps_per_episode=2,
            max_policy_updates=1,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )

    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        run_tiny_real_ppo_update_preflight(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_steps",
            episodes=1,
            max_decision_steps_per_episode=3,
            max_policy_updates=1,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )

    with pytest.raises(ValueError, match="max_policy_updates must be exactly 1"):
        run_tiny_real_ppo_update_preflight(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_updates",
            episodes=1,
            max_decision_steps_per_episode=2,
            max_policy_updates=2,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )


def test_tiny_real_ppo_update_requires_hash_verified_artifacts(tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    with pytest.raises(ValueError, match="objective normalizer"):
        run_tiny_real_ppo_update_preflight(
            config,
            "weighted_proxy",
            tmp_path / "blocked_no_norm",
            episodes=1,
            max_decision_steps_per_episode=2,
            max_policy_updates=1,
            allow_nonfilm_tiny_preflight=True,
        )

    with pytest.raises(ValueError, match="objective_normalizer_hash mismatch"):
        run_tiny_real_ppo_update_preflight(
            config,
            "weighted_proxy",
            tmp_path / "blocked_bad_norm",
            episodes=1,
            max_decision_steps_per_episode=2,
            max_policy_updates=1,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="wrong",
            allow_nonfilm_tiny_preflight=True,
        )

    with pytest.raises(ValueError, match="film_model_hash mismatch"):
        run_tiny_real_ppo_update_preflight(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_bad_film",
            episodes=1,
            max_decision_steps_per_episode=2,
            max_policy_updates=1,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash="wrong",
        )


def test_tiny_real_ppo_update_cli_requires_explicit_flag(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "film_scalar_potential",
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
    assert "--mock_env" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked" / "metadata.json").exists()


def test_tiny_real_ppo_update_cli_requires_manual_confirmation(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "film_scalar_potential",
            "--tiny_real_ppo_update_preflight",
            "--episodes",
            "1",
            "--max_decision_steps_per_episode",
            "2",
            "--max_policy_updates",
            "1",
            "--out_dir",
            str(tmp_path / "blocked_manual"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing to run" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked_manual" / "metadata.json").exists()


def test_tiny_real_ppo_update_cli_manual_confirmation_reaches_budget_guard(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "film_scalar_potential",
            "--tiny_real_ppo_update_preflight",
            "--i_understand_this_runs_one_real_ppo_update",
            "--episodes",
            "2",
            "--max_decision_steps_per_episode",
            "2",
            "--max_policy_updates",
            "1",
            "--out_dir",
            str(tmp_path / "blocked_budget"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "episodes must be exactly 1" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked_budget" / "metadata.json").exists()


def test_tiny_real_ppo_update_first_execution_is_film_only(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    normalizer_path = _normalizer(tmp_path)

    def forbidden_core(*args, **kwargs):
        raise AssertionError("core should not run for a blocked first tiny preflight method")

    monkeypatch.setattr("pareto.rl.formal_pilot_runner._run_tiny_real_ppo_update_core", forbidden_core)

    with pytest.raises(ValueError, match="film_scalar_potential"):
        run_tiny_real_ppo_update_preflight(
            config,
            "weighted_proxy",
            tmp_path / "blocked_method",
            episodes=1,
            max_decision_steps_per_episode=2,
            max_policy_updates=1,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
        )


def test_tiny_real_ppo_update_cli_blocks_nonfilm_without_allow_flag(tmp_path: Path):
    for method in ("weighted_proxy", "env_reward"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "pareto/rl/formal_pilot_runner.py"),
                "--spec",
                str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
                "--method",
                method,
                "--tiny_real_ppo_update_preflight",
                "--i_understand_this_runs_one_real_ppo_update",
                "--episodes",
                "1",
                "--max_decision_steps_per_episode",
                "2",
                "--max_policy_updates",
                "1",
                "--out_dir",
                str(tmp_path / f"blocked_{method}"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode != 0
        assert "film_scalar_potential" in (result.stderr + result.stdout)
        assert not (tmp_path / f"blocked_{method}" / "metadata.json").exists()


def test_tiny_real_ppo_update_cli_nonfilm_allow_flag_reaches_budget_guard(tmp_path: Path):
    for method in ("weighted_proxy", "env_reward"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "pareto/rl/formal_pilot_runner.py"),
                "--spec",
                str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
                "--method",
                method,
                "--tiny_real_ppo_update_preflight",
                "--i_understand_this_runs_one_real_ppo_update",
                "--allow_nonfilm_tiny_preflight",
                "--episodes",
                "2",
                "--max_decision_steps_per_episode",
                "2",
                "--max_policy_updates",
                "1",
                "--out_dir",
                str(tmp_path / f"blocked_budget_{method}"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode != 0
        assert "episodes must be exactly 1" in (result.stderr + result.stdout)
        assert not (tmp_path / f"blocked_budget_{method}" / "metadata.json").exists()


def test_tiny_env_reward_metadata_uses_queue_penalty_display_name():
    assert _method_display_name("env_reward") == "EnvReward-QueuePenalty-PPO"
    assert _method_display_name("film_scalar_potential") == "FiLMScalar-PPO"


def test_tiny_env_reward_summary_requires_nonzero_signal():
    rows = [
        {"env_reward": -3.0, "env_reward_source": "cityflow_average_reward"},
        {"env_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
    ]

    summary = _summarize_tiny_env_reward_rows(rows)

    assert summary["finite"] is True
    assert summary["all_zero_reward"] is False
    assert summary["env_reward_nonzero_rate"] == 0.5
    assert summary["env_reward_sources"] == ["cityflow_average_reward"]


def test_tiny_env_reward_summary_marks_all_zero_as_blocker():
    rows = [
        {"env_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
        {"env_reward": 0.0, "env_reward_source": "cityflow_average_reward"},
    ]

    summary = _summarize_tiny_env_reward_rows(rows)

    assert summary["finite"] is True
    assert summary["all_zero_reward"] is True
    assert summary["env_reward_nonzero_rate"] == 0.0


def test_tiny_real_ppo_update_wraps_core_metadata(monkeypatch, tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    def fake_core(*args, **kwargs):
        out = Path(args[2])
        out.mkdir(parents=True, exist_ok=True)
        metadata = {
            "method": args[1],
            "real_env_rollout": True,
            "cityflow_env_constructed": True,
            "cityflow_step_called": True,
            "ppo_training": True,
            "real_ppo_update": True,
            "policy_update": True,
            "optimizer_step": True,
            "policy_update_count": 1,
            "max_policy_updates": 1,
            "pilot_execution": False,
            "formal_experiment": False,
            "performance_claim": False,
            "not_for_main_results": True,
            "objective_normalizer_hash_verified": True,
            "film_model_hash_verified": True,
            "film_model_loaded": True,
            "film_model_source": "trained_film_checkpoint",
            "state_encoder_hash": "4d1c2b4e276043ac",
        }
        (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (out / "status.json").write_text(json.dumps({"status": "TINY_PPO_PREFLIGHT_DONE", "steps": 2}), encoding="utf-8")
        (out / "loss_debug.jsonl").write_text(json.dumps({"total_loss": 0.1, "approx_kl": 0.0}) + "\n", encoding="utf-8")
        (out / "training_checkpoint_last.pt").write_bytes(b"checkpoint")
        (out / "reward_components.jsonl").write_text(json.dumps({"finite": True}) + "\n", encoding="utf-8")
        (out / "train_metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
        (out / "TINY_PPO_PREFLIGHT_DONE").write_text("done\n", encoding="utf-8")
        return {"status": "TINY_PPO_PREFLIGHT_DONE", "steps": 2, "policy_update_count": 1}

    monkeypatch.setattr("pareto.rl.formal_pilot_runner._run_tiny_real_ppo_update_core", fake_core)
    out_dir = tmp_path / "tiny"

    result = run_tiny_real_ppo_update_preflight(
        config,
        "film_scalar_potential",
        out_dir,
        episodes=1,
        max_decision_steps_per_episode=2,
        max_policy_updates=1,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
        film_model_dir=film_dir,
        film_model_hash=film_hash,
    )

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "TINY_PPO_PREFLIGHT_DONE"
    assert status["status"] == "TINY_PPO_PREFLIGHT_DONE"
    assert metadata["tiny_real_ppo_update_preflight"] is True
    assert metadata["tiny_preflight_not_pilot"] is True
    assert metadata["exclude_from_analysis"] is True
    assert metadata["checkpoint_use"] == "preflight_resume_test_only"
    assert metadata["pilot_execution"] is False
    assert metadata["formal_experiment"] is False
    assert metadata["performance_claim"] is False
    assert metadata["not_for_main_results"] is True
    assert metadata["real_env_rollout"] is True
    assert metadata["ppo_training"] is True
    assert metadata["real_ppo_update"] is True
    assert metadata["policy_update"] is True
    assert metadata["optimizer_step"] is True
    assert metadata["max_policy_updates"] == 1
    assert metadata["policy_update_count"] == 1
    assert metadata["episodes"] == 1
    assert metadata["max_decision_steps_per_episode"] == 2
    assert metadata["sim_seconds_per_method"] == 60
    assert metadata["objective_normalizer_hash_verified"] is True
    assert metadata["film_model_required"] is True
    assert metadata["film_model_loaded"] is True
    assert metadata["film_model_hash_verified"] is True
    assert (out_dir / "loss_debug.jsonl").exists()
    assert (out_dir / "training_checkpoint_last.pt").exists()
    assert (out_dir / "TINY_PPO_PREFLIGHT_DONE").exists()
