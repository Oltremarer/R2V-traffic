from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.rl.formal_experiment_spec import FormalExperimentSpec
from pareto.rl.formal_preflight_checks import run_preflight_checks


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_film_model(tmp_path: Path) -> tuple[str, str]:
    torch.manual_seed(321)
    model_dir = tmp_path / "film"
    model = build_conditioned_scalar_model(
        "film",
        input_dim=6,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        film_layers=1,
        head_layers=1,
    )
    model_dir.mkdir(parents=True)
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
    return str(model_dir), hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()


def _spec(tmp_path: Path) -> FormalExperimentSpec:
    gate = tmp_path / "gate.json"
    objective_norm = tmp_path / "objective_norm.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, {"ppo_formal_allowed": False})
    _write_json(objective_norm, {"hash": "objective_hash"})
    _write_json(selection, {"selected": "film"})
    model_dir, model_hash = _write_film_model(tmp_path)
    payload = {
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
        "film_model_dir": model_dir,
        "film_model_hash": model_hash,
        "film_model_selection_report": str(selection),
        "film_training_commit": "8f0c9d4",
        "reward_adapter": "film_scalar_potential",
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
    return FormalExperimentSpec.from_dict(payload)


def test_real_record_sensitivity_uses_next_sample_links_and_reports_deltas(tmp_path: Path):
    records = []
    for idx in range(6):
        records.append(
            {
                "sample_id": f"s{idx}",
                "next_sample_id": f"s{idx + 1}" if idx < 5 else None,
                "obs_features": [float(idx + j) / 10.0 for j in range(6)],
                "objective_values_norm": {
                    "efficiency": 0.0,
                    "safety": 0.0,
                    "fairness": 0.0,
                    "stability": 0.0,
                },
                "objective_valid_mask": {
                    "efficiency": True,
                    "safety": True,
                    "fairness": True,
                    "stability": True,
                },
                "metadata": {"objective_normalizer_hash": "objective_hash"},
            }
        )
    records_path = tmp_path / "val.jsonl"
    _write_jsonl(records_path, records)

    report = run_preflight_checks(
        [_spec(tmp_path)],
        root=tmp_path,
        device="cpu",
        real_records_path=records_path,
        num_sensitivity_records=4,
        min_real_mean_delta=1e-7,
        min_real_max_delta=1e-7,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["real_record_film_reward_sensitivity"]["passed"] is True
    details = checks["real_record_film_reward_sensitivity"]["details"]
    assert details["num_records"] == 4
    assert details["mean_delta"] > 0.0
    assert details["max_delta"] > 0.0
    assert details["records_source"].endswith("val.jsonl")
