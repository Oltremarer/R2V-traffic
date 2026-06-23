from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_jinan_audit import audit_formal_jinan_runs


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_run(root: Path, *, seed: int, method: str, steps: int = 2, reward_rows: int = 4) -> None:
    out = root / f"seed{seed}" / method
    out.mkdir(parents=True)
    metadata = {
        "formal_jinan_3seed_execution": True,
        "formal_experiment": True,
        "performance_claim": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "pro_approval_phrase_verified": True,
        "method": method,
        "method_display_name": {
            "film_scalar_potential": "FiLMScalar-PPO",
            "weighted_proxy": "WeightedProxy-PPO",
            "env_reward": "EnvReward-QueuePenalty-PPO",
        }[method],
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "cityflow_seed": seed,
        "policy_seed": seed,
        "model_seed": seed,
        "episodes": 1,
        "max_decision_steps_per_episode": steps,
        "min_action_time": 30,
        "reference_only_methods": ["MaxPressure", "AdvancedMaxPressure"],
        "reward_adapter_semantics": "queue_length_penalty_proxy" if method == "env_reward" else "not_env_reward",
    }
    if method == "env_reward":
        metadata["env_reward_summary"] = {
            "finite": True,
            "all_zero_reward": False,
            "nonzero_reward_count": reward_rows,
            "env_reward_sources": ["cityflow_average_reward"],
        }
        metadata["env_reward_info_source"] = "pareto_nonzero_queue_length_proxy"
        metadata["active_env_reward_info"] = {"queue_length": -1.0}
    _write_json(out / "metadata.json", metadata)
    _write_json(
        out / "status.json",
        {
            **metadata,
            "status": "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
            "steps": steps,
            "reward_row_count": reward_rows,
            "policy_update_count": 1,
            "loss_debug_finite": True,
        },
    )
    _write_jsonl(out / "train_metrics.jsonl", [{"step": idx, "reward_finite": True} for idx in range(steps)])
    _write_jsonl(out / "reward_components.jsonl", [{"total_reward": 1.0} for _ in range(reward_rows)])
    _write_jsonl(out / "loss_debug.jsonl", [{"total_loss": 0.5}])
    for name in (
        "checkpoint_last.pt",
        "command.txt",
        "formal_run_plan.json",
        "formal_run_plan.md",
        "stderr.txt",
        "stdout.txt",
        "training_checkpoint_last.pt",
    ):
        (out / name).write_text("x\n", encoding="utf-8")


def test_formal_jinan_audit_passes_minimal_guarded_tree(tmp_path: Path):
    for seed in (0, 1, 2):
        for method in ("film_scalar_potential", "weighted_proxy", "env_reward"):
            _make_run(tmp_path, seed=seed, method=method)

    report = audit_formal_jinan_runs(
        tmp_path,
        expected_steps=2,
        expected_reward_rows=4,
        check_checkpoints=False,
    )

    assert report["report_status"] == "FORMAL_JINAN_3SEED_GUARD_PASS"
    assert report["failures"] == []
    assert report["budget_consistent"] is True
    assert report["performance_claim"] is False
    assert report["ranking_generated"] is False


def test_formal_jinan_audit_rejects_forbidden_artifact(tmp_path: Path):
    for seed in (0, 1, 2):
        for method in ("film_scalar_potential", "weighted_proxy", "env_reward"):
            _make_run(tmp_path, seed=seed, method=method)
    (tmp_path / "seed0" / "weighted_proxy" / "leaderboard.csv").write_text("bad\n", encoding="utf-8")

    report = audit_formal_jinan_runs(
        tmp_path,
        expected_steps=2,
        expected_reward_rows=4,
        check_checkpoints=False,
    )

    assert report["report_status"] == "FORMAL_JINAN_3SEED_GUARD_FAIL"
    assert any("forbidden" in failure for failure in report["failures"])


def test_formal_jinan_audit_rejects_seed_mismatch(tmp_path: Path):
    for seed in (0, 1, 2):
        for method in ("film_scalar_potential", "weighted_proxy", "env_reward"):
            _make_run(tmp_path, seed=seed, method=method)
    path = tmp_path / "seed2" / "env_reward" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["policy_seed"] = 0
    _write_json(path, metadata)

    report = audit_formal_jinan_runs(
        tmp_path,
        expected_steps=2,
        expected_reward_rows=4,
        check_checkpoints=False,
    )

    assert report["report_status"] == "FORMAL_JINAN_3SEED_GUARD_FAIL"
    assert any("seed binding mismatch" in failure for failure in report["failures"])
