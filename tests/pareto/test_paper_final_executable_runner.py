from __future__ import annotations

from collections import Counter
import subprocess
import sys
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_runner import (
    PAPER_FINAL_EPISODES_PER_SEED,
    PAPER_FINAL_EXECUTION_APPROVAL_PHRASE,
    PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE,
    run_paper_final_execution,
    validate_paper_final_execution_limits,
)
from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config
from pareto.rl.paper_final_executable_runner import build_executable_runner_specs

ROOT = Path(__file__).resolve().parents[2]


def _preview_row(method: str = "Random", *, city: str = "jinan", preference_template: str = "balanced") -> dict:
    return {
        "city": city,
        "traffic_file": "anon_3_4_jinan_real.json",
        "method": method,
        "seed": 0,
        "preference_template": preference_template,
        "preference_weights": [0.25, 0.25, 0.25, 0.25],
        "out_dir": "records/paper_final/train_20260602_v1/jinan/Random/seed0",
        "command": "preview-only command",
        "approval_phrase_required": True,
        "executes_training_now": False,
        "ranking_generated": False,
        "paper_table_generated": False,
        "paper_result_text_generated": False,
    }


def _artifact_inventory() -> dict:
    rows = []
    for baseline, family in (
        ("Cond-Scalar-RL", "cond_scalar"),
        ("VectorQ-PPO", "pareto_quality"),
    ):
        for city in ("jinan", "hangzhou", "newyork_28x7"):
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "status": "implemented_guarded_preview",
                    "model_path": f"model_weights/{family}/{city}/paper_final/run1/model.pt",
                    "model_hash": "a" * 64,
                    "objective_normalizer_path": f"model_weights/{family}/{city}/paper_final/run1/objective_normalizer.json",
                    "objective_normalizer_hash": f"{baseline}_{city}_normalizer_hash",
                    "objective_normalizer_file_sha256": "b" * 64,
                    "executes_training_now": False,
                }
            )
    return {
        "packet_type": "paper_learned_artifact_inventory",
        "coverage_status": "complete",
        "rows": rows,
        "executes_training_now": False,
    }


def test_reference_preview_maps_to_guarded_adapter_cli_args():
    spec = build_executable_runner_specs([_preview_row()], python_bin="python3")
    row = spec["rows"][0]

    assert spec["status"] == "ready_request"
    assert row["status"] == "executable_preview"
    assert row["runner_family"] == "paper_final_reference_adapter"
    assert row["command_argv"][:2] == ["python3", "pareto/rl/paper_final_reference_runner.py"]
    assert "--method" in row["command_argv"]
    assert "Random" in row["command_argv"]
    assert "--legacy_script" in row["command_argv"]
    assert "run_random.py" in row["command_argv"]
    assert "--dataset" in row["command_argv"]
    assert "jinan" in row["command_argv"]
    assert "--traffic_file" in row["command_argv"]
    assert "--seed_id" in row["command_argv"]
    assert "0" in row["command_argv"]
    assert "--out_dir" in row["command_argv"]
    assert row["out_dir"] in row["command_argv"]
    assert str(row["out_dir"]).startswith("records/paper_final/")
    assert row["executes_now"] is False


def test_c2t_scope_exclusion_has_no_command_and_is_not_executable():
    row = _preview_row("C2T-scalar")
    row["command"] = None
    row["scope_limitation"] = "excluded_by_reviewer"
    row["paper_claim_limitation"] = "C2T-scalar is excluded."

    spec = build_executable_runner_specs([row])
    output = spec["rows"][0]

    assert spec["status"] == "ready_request"
    assert output["status"] == "excluded_by_scope_limitation"
    assert output["command_argv"] == []
    assert output["executes_now"] is False


def test_learned_ppo_rows_remain_blocked_until_artifact_inventory_is_supplied():
    spec = build_executable_runner_specs([_preview_row("VectorQ-PPO")])
    row = spec["rows"][0]

    assert spec["status"] == "missing_blocker"
    assert row["status"] == "missing_blocker"
    assert "artifact missing" in row["blocker"]


def test_learned_ppo_rows_map_to_guarded_paper_final_runner_with_inventory():
    spec = build_executable_runner_specs(
        [
            _preview_row("Cond-Scalar-RL"),
            _preview_row("Weighted-RL", preference_template="efficiency_focused"),
            _preview_row("VectorQ-PPO"),
        ],
        python_bin="python3",
        learned_artifact_inventory=_artifact_inventory(),
        device="cuda",
    )

    assert spec["status"] == "ready_request"
    rows = {row["method"]: row for row in spec["rows"]}
    assert Counter(row["status"] for row in spec["rows"]) == {"executable_preview": 3}
    for row in rows.values():
        argv = row["command_argv"]
        assert argv[:2] == ["python3", "pareto/rl/formal_pilot_runner.py"]
        assert "--paper_final_execution" in argv
        assert "--approval_phrase" in argv
        assert "${PPTS_PARETO_PPO_FINAL_EXECUTION_APPROVAL_PHRASE}" in argv
        assert "--rollout_steps" in argv
        assert "3600" in argv
        assert "--total_env_steps_per_seed" in argv
        assert "1000000" in argv
        assert "--fixed_preference_template" in argv
        assert "--episodes" in argv
        assert argv[argv.index("--episodes") + 1] == "278"
        assert "--max_decision_steps_per_episode" in argv
        assert argv[argv.index("--max_decision_steps_per_episode") + 1] == "120"
        assert row["executes_now"] is False
    assert "--film_model_dir" in rows["Cond-Scalar-RL"]["command_argv"]
    assert "--vector_model_dir" in rows["VectorQ-PPO"]["command_argv"]
    assert "--film_model_dir" not in rows["Weighted-RL"]["command_argv"]
    assert rows["Weighted-RL"]["method_id"] == "weighted_proxy"
    cond_argv = rows["Cond-Scalar-RL"]["command_argv"]
    norm_hash_index = cond_argv.index("--objective_normalizer_hash") + 1
    assert cond_argv[norm_hash_index] == "Cond-Scalar-RL_jinan_normalizer_hash"
    assert rows["Cond-Scalar-RL"]["objective_normalizer_file_sha256"] == "b" * 64


def test_paper_final_runner_limits_accept_all_city_seed_and_paper_scale_budget():
    config = load_formal_ppo_dryrun_config("configs/formal/paper_final_hangzhou_5seed_ppo.json")

    request = validate_paper_final_execution_limits(
        config,
        "weighted_proxy",
        seed_id=4,
        approval_phrase=PAPER_FINAL_EXECUTION_APPROVAL_PHRASE,
        rollout_steps=3600,
        total_env_steps_per_seed=1_000_000,
        episodes=278,
        max_decision_steps_per_episode=120,
        fixed_preference_template="safety_focused",
    )

    assert request["scenario"] == "hangzhou"
    assert request["seed_id"] == 4
    assert request["episodes"] == 278
    assert request["actual_sim_seconds_per_seed"] == 1_000_800
    assert request["fixed_preference_weights"] == [0.1, 0.7, 0.1, 0.1]


def test_paper_final_runner_limits_reject_short_run_shape():
    config = load_formal_ppo_dryrun_config("configs/formal/paper_final_jinan_5seed_ppo.json")

    with pytest.raises(ValueError, match="run shape"):
        validate_paper_final_execution_limits(
            config,
            "weighted_proxy",
            seed_id=0,
            approval_phrase=PAPER_FINAL_EXECUTION_APPROVAL_PHRASE,
            rollout_steps=3600,
            total_env_steps_per_seed=1_000_000,
            episodes=5,
            max_decision_steps_per_episode=120,
            fixed_preference_template="balanced",
        )


def test_paper_final_runner_limits_reject_missing_final_execution_phrase():
    config = load_formal_ppo_dryrun_config("configs/formal/paper_final_newyork_28x7_5seed_ppo.json")

    with pytest.raises(ValueError, match="exact external approval phrase"):
        validate_paper_final_execution_limits(
            config,
            "vector_quality_potential",
            seed_id=0,
            approval_phrase="not approved",
            rollout_steps=3600,
            total_env_steps_per_seed=1_000_000,
            episodes=278,
            max_decision_steps_per_episode=120,
            fixed_preference_template="balanced",
        )


def test_paper_final_execution_passes_paper_scale_limits_to_internal_runner(monkeypatch, tmp_path: Path):
    config = load_formal_ppo_dryrun_config("configs/formal/paper_final_jinan_5seed_ppo.json")
    captured = {}

    def fake_bounded_runner(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "FAKE_PAPER_FINAL_RUNNER"}

    monkeypatch.setattr("pareto.rl.formal_pilot_runner.run_bounded_jinan_pilot_dry_run", fake_bounded_runner)
    monkeypatch.setattr("pareto.rl.formal_pilot_runner._finalize_paper_final_execution_outputs", lambda out_dir: None)

    payload = run_paper_final_execution(
        config,
        "weighted_proxy",
        tmp_path / "paper_final_weighted_proxy",
        seed_id=0,
        approval_phrase=PAPER_FINAL_EXECUTION_APPROVAL_PHRASE,
        rollout_steps=3600,
        total_env_steps_per_seed=1_000_000,
        fixed_preference_template="balanced",
        episodes=PAPER_FINAL_EPISODES_PER_SEED,
        max_decision_steps_per_episode=PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE,
        objective_normalizer_hash="b" * 64,
        device="cpu",
    )

    assert payload["status"] == "FAKE_PAPER_FINAL_RUNNER"
    assert captured["max_episodes"] == PAPER_FINAL_EPISODES_PER_SEED
    assert captured["max_decision_steps"] == PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE
    assert captured["scope_label"] == "paper-final learned PPO execution"


def test_paper_final_cli_refuses_without_exact_final_execution_phrase(tmp_path: Path):
    out_dir = tmp_path / "blocked"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/paper_final_jinan_5seed_ppo.json"),
            "--method",
            "weighted_proxy",
            "--paper_final_execution",
            "--approval_phrase",
            "${PPTS_PARETO_PPO_FINAL_EXECUTION_APPROVAL_PHRASE}",
            "--seed_id",
            "0",
            "--rollout_steps",
            "3600",
            "--total_env_steps_per_seed",
            "1000000",
            "--fixed_preference_template",
            "balanced",
            "--objective_normalizer",
            "model_weights/cond_scalar/jinan/paper_final/run1/objective_normalizer.json",
            "--objective_normalizer_hash",
            "b" * 64,
            "--out_dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "exact external approval phrase" in (result.stderr + result.stdout)
    assert not out_dir.exists()
