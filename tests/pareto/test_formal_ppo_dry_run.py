from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> Path:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ppo_formal_allowed": False}), encoding="utf-8")
    spec = tmp_path / "pilot_spec.md"
    spec.write_text("pilot spec\n", encoding="utf-8")
    config = {
        "pilot": {
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "cityflow_seed": 0,
            "policy_seed": 0,
            "model_seed": 0,
            "methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
            "formal_gate_decision_path": str(gate),
            "pilot_spec_path": str(spec),
        },
        "ppo": {
            "algorithm_label": "PPO",
            "requires_clipped_objective": True,
            "rollout_steps": 12,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "update_epochs": 2,
            "minibatch_size": 6,
            "lr": 0.0003,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantages": True,
        },
        "model": {"obs_dim": 8, "preference_dim": 4, "action_dim": 4, "hidden_dim": 16},
    }
    path = tmp_path / "dryrun.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_formal_ppo_dry_run_uses_synthetic_batch_and_writes_guarded_metadata(tmp_path: Path):
    config_path = _config(tmp_path)
    out_dir = tmp_path / "film"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/run_formal_ppo_dry_run.py"),
            "--spec",
            str(config_path),
            "--method",
            "film_scalar_potential",
            "--dry_run",
            "--out_dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/run_formal_ppo_dry_run.py"),
            "--spec",
            str(config_path),
            "--method",
            "film_scalar_potential",
            "--dry_run",
            "--out_dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dry_run"] is True
    assert metadata["env_rollout"] is False
    assert metadata["env_training"] is False
    assert metadata["ppo_training"] is False
    assert metadata["ppo_training_on_env"] is False
    assert metadata["synthetic_ppo_update"] is True
    assert metadata["synthetic_update_only"] is True
    assert metadata["policy_update"] is True
    assert metadata["performance_claim"] is False
    assert metadata["pilot_execution"] is False
    assert metadata["formal_experiment"] is False
    assert metadata["formal_ppo_dryrun_spec_snapshot"] is True
    assert metadata["formal_experiment_spec_is_snapshot_only"] is True
    assert metadata["spec_snapshot_name"] == "formal_ppo_dryrun_spec.json"
    assert (out_dir / "checkpoint_last.pt").exists()
    assert (out_dir / "formal_ppo_dryrun_spec.json").exists()
    assert (out_dir / "DRY_RUN_DONE").exists()
    assert not (out_dir / "main_results.csv").exists()
    loss_rows = [json.loads(line) for line in (out_dir / "loss_debug.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(loss_rows) == 4
    assert "approx_kl" in loss_rows[0]
    assert "clip_fraction" in loss_rows[0]
