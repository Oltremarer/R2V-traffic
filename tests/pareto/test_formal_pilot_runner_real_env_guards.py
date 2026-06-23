from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_runner import (
    EXPLORATORY_PILOT_ALLOWED_OUTPUTS,
    FINAL_JINAN_PILOT_METHODS,
    _exploratory_pilot_gate_metadata,
    _finalize_exploratory_pilot_outputs,
    _maybe_write_exploratory_pilot_guard_artifacts,
    _summarize_action_distribution_guard,
    _update_from_buffer,
    _write_exploratory_pilot_guard_artifacts,
    run_formal_pilot_real_env_dry_run,
    validate_bounded_jinan_pilot_dry_run_limits,
)
from pareto.rl import formal_pilot_runner as formal_pilot_runner_module
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig
from pareto.rl.ppo_actor_critic import PreferenceConditionedActorCritic
from pareto.rl.ppo_buffer import PPORolloutBuffer
from pareto.rl.real_cityflow_preflight_smoke import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path, film_hash: str = "film_hash") -> FormalPPODryRunConfig:
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


def test_formal_pilot_runner_rejects_real_env_flag(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "film_scalar_potential",
            "--real_env",
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
    assert "real env pilot execution is not allowed" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked" / "metadata.json").exists()


def test_bounded_jinan_pilot_dry_run_limits_are_locked_to_pro_gate(tmp_path: Path):
    config = _config(tmp_path)
    config.pilot["model_seed"] = 0

    validate_bounded_jinan_pilot_dry_run_limits(config, "weighted_proxy", episodes=5, max_decision_steps_per_episode=120)

    with pytest.raises(ValueError, match="episodes must be in"):
        validate_bounded_jinan_pilot_dry_run_limits(config, "weighted_proxy", episodes=6, max_decision_steps_per_episode=120)
    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        validate_bounded_jinan_pilot_dry_run_limits(config, "weighted_proxy", episodes=5, max_decision_steps_per_episode=121)

    bad_seed = _config(tmp_path)
    bad_seed.pilot["cityflow_seed"] = 1
    with pytest.raises(ValueError, match="cityflow_seed=0"):
        validate_bounded_jinan_pilot_dry_run_limits(bad_seed, "weighted_proxy", episodes=5, max_decision_steps_per_episode=120)

    bad_methods = _config(tmp_path)
    bad_methods.pilot["model_seed"] = 0
    bad_methods.pilot["methods"] = list(reversed(FINAL_JINAN_PILOT_METHODS))
    with pytest.raises(ValueError, match="final Jinan method list"):
        validate_bounded_jinan_pilot_dry_run_limits(bad_methods, "weighted_proxy", episodes=5, max_decision_steps_per_episode=120)


def test_bounded_jinan_pilot_dry_run_cli_requires_explicit_ack(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "weighted_proxy",
            "--bounded_jinan_pilot_dry_run",
            "--episodes",
            "1",
            "--max_decision_steps_per_episode",
            "1",
            "--out_dir",
            str(tmp_path / "blocked_bounded_pilot"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "bounded Jinan pilot dry-run performs real CityFlow PPO updates" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked_bounded_pilot" / "metadata.json").exists()


def test_exploratory_pilot_gate_metadata_matches_pro_scope(tmp_path: Path):
    config = _config(tmp_path)
    config.pilot["model_seed"] = 0

    metadata = _exploratory_pilot_gate_metadata(config, "weighted_proxy")

    assert metadata["exploratory_pilot"] is True
    assert metadata["formal_experiment"] is False
    assert metadata["performance_claim"] is False
    assert metadata["not_for_main_results"] is True
    assert metadata["seed_expansion_allowed"] is False
    assert metadata["city_expansion_allowed"] is False
    assert metadata["method_ranking_allowed"] is False
    assert metadata["paper_result_allowed"] is False
    assert metadata["offline_representation_gate"] == "partial_pass"
    assert metadata["allowed_result_interpretation"] == "closed_loop_stability_only"
    assert metadata["feature_schema_version"] == "hybrid_v1"


def test_action_distribution_guard_rejects_global_single_action_collapse():
    with pytest.raises(ValueError, match="action collapse"):
        _summarize_action_distribution_guard(
            Counter({0: 96, 1: 4}),
            [Counter({0: 8, 1: 2}) for _ in range(10)],
            max_single_action_rate=0.95,
        )

    summary = _summarize_action_distribution_guard(
        Counter({0: 94, 1: 6}),
        [Counter({0: 7, 1: 3}) for _ in range(10)],
        max_single_action_rate=0.95,
    )
    assert summary["unique_actions_used"] == 2
    assert summary["global_single_action_rate"] == 0.94
    assert summary["action_entropy"] > 0.0


def test_exploratory_pilot_guard_artifacts_are_non_performance_outputs(tmp_path: Path):
    status = {
        "status": "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
        "loss_debug_finite": True,
        "checkpoint_saved": True,
        "checkpoint_load_verified": True,
        "policy_update_count": 1,
        "reward_row_count": 24,
    }
    action_guard = _summarize_action_distribution_guard(
        Counter({0: 10, 1: 8, 2: 6}),
        [Counter({0: 2, 1: 1}) for _ in range(8)],
    )

    _write_exploratory_pilot_guard_artifacts(tmp_path, status, action_guard)

    pilot_status = json.loads((tmp_path / "pilot_status.json").read_text(encoding="utf-8"))
    assert pilot_status["exploratory_pilot"] is True
    assert pilot_status["formal_experiment"] is False
    assert pilot_status["performance_claim"] is False
    assert pilot_status["checkpoint_valid"] is True
    assert pilot_status["checkpoint_load_verified"] is True
    assert pilot_status["action_distribution_non_degenerate"] is True
    assert (tmp_path / "pilot_guard_report.md").exists()
    assert not (tmp_path / "main_results.csv").exists()


def test_exploratory_pilot_finalize_removes_non_allowlisted_outputs(tmp_path: Path):
    for name in EXPLORATORY_PILOT_ALLOWED_OUTPUTS:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "action_debug.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ppo_config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "llmlight_work").mkdir()
    (tmp_path / "llmlight_work" / "cityflow.log").write_text("log\n", encoding="utf-8")

    removed = _finalize_exploratory_pilot_outputs(tmp_path)

    assert "action_debug.jsonl" in removed
    assert "ppo_config.json" in removed
    assert "llmlight_work" in removed
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(EXPLORATORY_PILOT_ALLOWED_OUTPUTS)


def test_exploratory_pilot_finalize_retries_directory_not_empty_cleanup(tmp_path: Path, monkeypatch):
    for name in EXPLORATORY_PILOT_ALLOWED_OUTPUTS:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    work_dir = tmp_path / "llmlight_work"
    work_dir.mkdir()
    (work_dir / "cityflow.log").write_text("log\n", encoding="utf-8")
    attempts = {"count": 0}
    original_rmtree = formal_pilot_runner_module.shutil.rmtree

    def flaky_rmtree(path: Path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError(39, "Directory not empty", str(path))
        return original_rmtree(path)

    monkeypatch.setattr(formal_pilot_runner_module.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(formal_pilot_runner_module.time, "sleep", lambda _seconds: None)

    removed = _finalize_exploratory_pilot_outputs(tmp_path)

    assert removed == ["llmlight_work"]
    assert attempts["count"] == 2
    assert not work_dir.exists()


def test_exploratory_pilot_finalize_quarantines_stubborn_work_dir(tmp_path: Path, monkeypatch):
    for name in EXPLORATORY_PILOT_ALLOWED_OUTPUTS:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    work_dir = tmp_path / "llmlight_work"
    work_dir.mkdir()
    (work_dir / "cityflow.log").write_text("log\n", encoding="utf-8")
    original_rmtree = formal_pilot_runner_module.shutil.rmtree

    def stubborn_rmtree(path: Path, *args, **kwargs):
        if Path(path) == work_dir:
            raise OSError(39, "Directory not empty", str(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(formal_pilot_runner_module.shutil, "rmtree", stubborn_rmtree)
    monkeypatch.setattr(formal_pilot_runner_module.time, "sleep", lambda _seconds: None)

    removed = _finalize_exploratory_pilot_outputs(tmp_path)

    assert removed == ["llmlight_work"]
    assert not work_dir.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(EXPLORATORY_PILOT_ALLOWED_OUTPUTS)


def test_exploratory_pilot_guard_refuses_unverified_checkpoint(tmp_path: Path):
    status = {
        "status": "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
        "loss_debug_finite": True,
        "checkpoint_saved": True,
        "checkpoint_load_verified": False,
        "policy_update_count": 1,
        "reward_row_count": 24,
    }
    action_guard = _summarize_action_distribution_guard(
        Counter({0: 10, 1: 8, 2: 6}),
        [Counter({0: 2, 1: 1}) for _ in range(8)],
    )

    with pytest.raises(ValueError, match="checkpoint load verification"):
        _write_exploratory_pilot_guard_artifacts(tmp_path, status, action_guard)

    assert not (tmp_path / "pilot_status.json").exists()


def test_formal_execution_context_does_not_write_exploratory_pilot_artifacts(tmp_path: Path):
    status = {
        "status": "FORMAL_JINAN_3SEED_RUN_DONE",
        "loss_debug_finite": True,
        "checkpoint_saved": True,
        "checkpoint_load_verified": True,
        "policy_update_count": 1,
        "reward_row_count": 24,
    }
    action_guard = _summarize_action_distribution_guard(
        Counter({0: 10, 1: 8, 2: 6}),
        [Counter({0: 2, 1: 1}) for _ in range(8)],
    )

    wrote = _maybe_write_exploratory_pilot_guard_artifacts(
        tmp_path,
        status,
        action_guard,
        formal_execution_context={"formal_jinan_3seed_execution": True},
    )

    assert wrote is False
    assert not (tmp_path / "pilot_status.json").exists()
    assert not (tmp_path / "pilot_guard_report.md").exists()


def test_update_from_buffer_moves_rollout_batch_to_model_device(tmp_path: Path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device mismatch regression needs CUDA")
    config = _config(tmp_path)
    model = PreferenceConditionedActorCritic(obs_dim=193, preference_dim=4, action_dim=4, hidden_dim=16).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    buffer = PPORolloutBuffer()
    for idx in range(24):
        buffer.add(
            obs=torch.ones(193) * (idx + 1),
            w=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            action=idx % 4,
            reward=float(idx % 3),
            done=False,
            value=0.0,
            log_prob=-1.386294,
        )

    update_count = _update_from_buffer(model, optimizer, buffer, config, tmp_path, 0)

    assert update_count > 0
    assert (tmp_path / "loss_debug.jsonl").exists()



def test_formal_pilot_real_env_dry_run_rejects_budget_above_gate(tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    with pytest.raises(ValueError, match="episodes must be exactly 1"):
        run_formal_pilot_real_env_dry_run(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_episodes",
            episodes=2,
            max_decision_steps_per_episode=3,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )

    with pytest.raises(ValueError, match="max_decision_steps_per_episode"):
        run_formal_pilot_real_env_dry_run(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_steps",
            episodes=1,
            max_decision_steps_per_episode=4,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash=film_hash,
        )


def test_formal_pilot_real_env_dry_run_requires_hash_verified_artifacts(tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    with pytest.raises(ValueError, match="objective normalizer"):
        run_formal_pilot_real_env_dry_run(
            config,
            "weighted_proxy",
            tmp_path / "blocked_no_norm",
            episodes=1,
            max_decision_steps_per_episode=3,
        )

    with pytest.raises(ValueError, match="objective_normalizer_hash mismatch"):
        run_formal_pilot_real_env_dry_run(
            config,
            "weighted_proxy",
            tmp_path / "blocked_bad_norm",
            episodes=1,
            max_decision_steps_per_episode=3,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="wrong",
        )

    with pytest.raises(ValueError, match="film_model_hash mismatch"):
        run_formal_pilot_real_env_dry_run(
            config,
            "film_scalar_potential",
            tmp_path / "blocked_bad_film",
            episodes=1,
            max_decision_steps_per_episode=3,
            objective_normalizer=normalizer_path,
            objective_normalizer_hash="norm_hash",
            film_model_dir=film_dir,
            film_model_hash="wrong",
        )


def test_formal_pilot_real_env_dry_run_wraps_no_learning_core(monkeypatch, tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    def fake_real_cityflow_core(*args, **kwargs):
        out = Path(args[2])
        out.mkdir(parents=True, exist_ok=True)
        metadata = {
            "method": kwargs["method"] if "method" in kwargs else args[1],
            "mock_env": False,
            "real_env_rollout": True,
            "cityflow_env_constructed": True,
            "cityflow_step_called": True,
            "policy_update": False,
            "optimizer_step": False,
            "ppo_training": False,
            "pilot_execution": False,
            "performance_claim": False,
            "objective_normalizer_hash": "norm_hash",
            "objective_normalizer_hash_expected": "norm_hash",
            "objective_normalizer_hash_verified": True,
            "film_model_hash": film_hash,
            "film_model_hash_expected": film_hash,
            "film_model_hash_verified": True,
            "film_model_source": "trained_film_checkpoint",
            "film_model_loaded": True,
            "state_encoder_hash": "4d1c2b4e276043ac",
            "obs_dim": 193,
        }
        (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (out / "status.json").write_text(json.dumps({"status": "REAL_PREFLIGHT_DONE", "steps": 3}), encoding="utf-8")
        (out / "preflight_metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
        (out / "reward_components.jsonl").write_text(json.dumps({"finite": True}) + "\n", encoding="utf-8")
        (out / "action_debug.jsonl").write_text(json.dumps({"action_list_len": 12}) + "\n", encoding="utf-8")
        (out / "REAL_PREFLIGHT_DONE").write_text("done\n", encoding="utf-8")
        return {"status": "REAL_PREFLIGHT_DONE", "steps": 3, "reward_row_count": 36, "obs_dim": 193}

    monkeypatch.setattr("pareto.rl.formal_pilot_runner.run_real_cityflow_preflight_smoke", fake_real_cityflow_core)
    out_dir = tmp_path / "film_real_env_dryrun"

    result = run_formal_pilot_real_env_dry_run(
        config,
        "film_scalar_potential",
        out_dir,
        episodes=1,
        max_decision_steps_per_episode=3,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
        film_model_dir=film_dir,
        film_model_hash=film_hash,
    )

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "REAL_ENV_DRY_RUN_DONE"
    assert status["status"] == "REAL_ENV_DRY_RUN_DONE"
    assert metadata["pilot_runner_skeleton"] is True
    assert metadata["real_env_dry_run"] is True
    assert metadata["mock_env"] is False
    assert metadata["real_env_rollout"] is True
    assert metadata["real_ppo_update"] is False
    assert metadata["cityflow_optimizer_step"] is False
    assert metadata["policy_update"] is False
    assert metadata["optimizer_step"] is False
    assert metadata["pilot_execution"] is False
    assert metadata["formal_experiment"] is False
    assert metadata["performance_claim"] is False
    assert metadata["not_for_main_results"] is True
    assert metadata["objective_normalizer_hash_verified"] is True
    assert metadata["film_model_hash_verified"] is True
    assert metadata["film_model_source"] == "trained_film_checkpoint"
    assert metadata["smoke_scalar_quality_model"] is False
    assert metadata["ppo_config_hash"] == config.ppo_config_hash()
    assert (out_dir / "runner_metrics.jsonl").exists()
    assert not (out_dir / "preflight_metrics.jsonl").exists()
    assert not (out_dir / "REAL_PREFLIGHT_DONE").exists()
    assert (out_dir / "REAL_ENV_DRY_RUN_DONE").exists()
    assert not (out_dir / "loss_debug.jsonl").exists()
    assert not (out_dir / "training_checkpoint_last.pt").exists()


def test_formal_pilot_real_env_dry_run_keeps_non_film_hash_semantics(monkeypatch, tmp_path: Path):
    film_dir, film_hash = _film_dir(tmp_path)
    del film_dir
    config = _config(tmp_path, film_hash=film_hash)
    normalizer_path = _normalizer(tmp_path)

    def fake_real_cityflow_core(*args, **kwargs):
        out = Path(args[2])
        out.mkdir(parents=True, exist_ok=True)
        metadata = {
            "method": args[1],
            "mock_env": False,
            "real_env_rollout": True,
            "cityflow_env_constructed": True,
            "cityflow_step_called": True,
            "policy_update": False,
            "optimizer_step": False,
            "ppo_training": False,
            "pilot_execution": False,
            "performance_claim": False,
            "objective_normalizer_hash": "norm_hash",
            "objective_normalizer_hash_expected": "norm_hash",
            "objective_normalizer_hash_verified": True,
            "film_model_hash": None,
            "film_model_hash_expected": None,
            "film_model_hash_verified": False,
            "film_model_source": None,
            "film_model_loaded": False,
            "state_encoder_hash": "4d1c2b4e276043ac",
            "obs_dim": 193,
        }
        (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (out / "status.json").write_text(json.dumps({"status": "REAL_PREFLIGHT_DONE", "steps": 3}), encoding="utf-8")
        (out / "preflight_metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
        (out / "REAL_PREFLIGHT_DONE").write_text("done\n", encoding="utf-8")
        return {"status": "REAL_PREFLIGHT_DONE", "steps": 3, "reward_row_count": 36, "obs_dim": 193}

    monkeypatch.setattr("pareto.rl.formal_pilot_runner.run_real_cityflow_preflight_smoke", fake_real_cityflow_core)
    out_dir = tmp_path / "weighted_real_env_dryrun"

    run_formal_pilot_real_env_dry_run(
        config,
        "weighted_proxy",
        out_dir,
        episodes=1,
        max_decision_steps_per_episode=3,
        objective_normalizer=normalizer_path,
        objective_normalizer_hash="norm_hash",
    )

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["film_model_hash"] is None
    assert metadata["film_model_hash_config"] == film_hash
    assert metadata["film_model_hash_verified"] is False
    assert metadata["smoke_scalar_quality_model"] is False
