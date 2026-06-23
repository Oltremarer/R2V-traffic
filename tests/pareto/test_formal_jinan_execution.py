from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_jinan_execution import (
    DEPRECATED_FORMAL_JINAN_EXECUTION_MESSAGE,
    FORMAL_ROOT_ALLOWED_OUTPUTS,
    build_formal_execution_context,
    finalize_formal_jinan_seed_outputs,
    prepare_seed_bound_config,
    validate_formal_jinan_execution_request,
)
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig
from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
from pareto.rl.formal_run_plan import FORMAL_EXECUTION_APPROVAL_PHRASE


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
                "model_seed": 0,
                "methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
                "formal_gate_decision_path": str(gate),
                "pilot_spec_path": str(spec),
                "state_encoder_hash": "4d1c2b4e276043ac",
                "objective_normalizer_hash": "norm_hash",
                "film_model_hash": "film_hash",
            },
            "ppo": {
                "algorithm_label": "PPO",
                "requires_clipped_objective": True,
                "rollout_steps": 120,
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
            "model": {"obs_dim": 193, "preference_dim": 4, "action_dim": 4, "hidden_dim": 64},
        }
    )


def test_prepare_seed_bound_config_binds_all_seeds(tmp_path: Path):
    config = _config(tmp_path)

    bound = prepare_seed_bound_config(config, seed_id=2)

    assert bound.pilot["cityflow_seed"] == 2
    assert bound.pilot["policy_seed"] == 2
    assert bound.pilot["model_seed"] == 2
    assert config.pilot["cityflow_seed"] == 0


def test_validate_formal_jinan_execution_request_hard_fails_before_old_phrase_path():
    with pytest.raises(ValueError, match="formal_jinan_execution.py is deprecated"):
        validate_formal_jinan_execution_request(
            "configs/formal/formal_jinan_3seed_run_plan_2026-05-31.json",
            method="weighted_proxy",
            seed_id=1,
            approval_phrase="close but not exact",
        )


def test_formal_jinan_execution_wrapper_is_deprecated_and_uses_single_phrase_source():
    assert FORMAL_EXECUTION_APPROVAL_PHRASE == FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
    with pytest.raises(ValueError, match="formal_jinan_execution.py is deprecated"):
        validate_formal_jinan_execution_request(
            "configs/formal/formal_jinan_3seed_run_plan_2026-05-31.json",
            method="weighted_proxy",
            seed_id=1,
            approval_phrase=FORMAL_EXECUTION_APPROVAL_PHRASE,
        )
    assert "formal_pilot_runner.py --formal_jinan_3seed_execution" in DEPRECATED_FORMAL_JINAN_EXECUTION_MESSAGE


def test_build_formal_execution_context_marks_no_claim_execution():
    request = {
        "method": "env_reward",
        "seed_id": 2,
        "run_plan_path": "configs/formal/formal_jinan_3seed_run_plan_2026-05-31.json",
        "approval_phrase": FORMAL_EXECUTION_APPROVAL_PHRASE,
    }

    context = build_formal_execution_context(request, packet_commit="abc123")

    assert context["formal_jinan_3seed_execution"] is True
    assert context["formal_experiment"] is True
    assert context["performance_claim"] is False
    assert context["method_ranking_allowed"] is False
    assert context["performance_table_allowed"] is False
    assert context["pro_approval_phrase_verified"] is True
    assert context["status_label"] == "FORMAL_JINAN_3SEED_RUN_DONE"
    assert context["cityflow_seed"] == 2
    assert context["policy_seed"] == 2
    assert context["model_seed"] == 2


def test_finalize_formal_jinan_seed_outputs_removes_non_allowlisted_root_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in FORMAL_ROOT_ALLOWED_OUTPUTS:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (run_dir / "ppo_config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "pilot_spec.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "formal_gate_decision.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "action_debug.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "BOUNDED_JINAN_PILOT_DRY_RUN_DONE").write_text("done\n", encoding="utf-8")
    (run_dir / "llmlight_work").mkdir()
    (run_dir / "llmlight_work" / "cityflow.log").write_text("log\n", encoding="utf-8")

    finalize_formal_jinan_seed_outputs(run_dir)

    assert sorted(path.name for path in run_dir.iterdir()) == sorted(FORMAL_ROOT_ALLOWED_OUTPUTS)
