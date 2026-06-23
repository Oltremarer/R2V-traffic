from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from pareto.models.conditioned_scalar import build_conditioned_scalar_model


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_model(path: Path) -> str:
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
    path.mkdir(parents=True, exist_ok=True)
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
        path / "model.pt",
    )
    return hashlib.sha256((path / "model.pt").read_bytes()).hexdigest()


def _payload(tmp_path: Path, reward_adapter: str) -> dict:
    objective_norm = tmp_path / "objective_norm.json"
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(objective_norm, {"hash": "objective_hash"})
    _write_json(gate, {"ppo_formal_allowed": False})
    _write_json(selection, {"selected": "film"})
    model_dir = tmp_path / "film_model"
    model_hash = _write_model(model_dir)
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
        "film_model_dir": str(model_dir) if reward_adapter == "film_scalar_potential" else None,
        "film_model_hash": model_hash if reward_adapter == "film_scalar_potential" else None,
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


def test_formal_preflight_runner_checks_only_writes_non_rollout_report(tmp_path: Path):
    specs = []
    for reward_adapter in ("film_scalar_potential", "weighted_proxy", "env_reward"):
        path = tmp_path / f"{reward_adapter}.json"
        _write_json(path, _payload(tmp_path, reward_adapter))
        specs.append(str(path))
    out_dir = tmp_path / "checks"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_preflight_runner.py"),
            "--specs",
            *specs,
            "--checks_only",
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "passed" in completed.stdout
    report = json.loads((out_dir / "preflight_checks.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["checks_only"] is True
    assert report["env_rollout"] is False
    assert report["ppo_training"] is False
    assert report["policy_update"] is False
    assert not (out_dir / "metrics.csv").exists()
