from __future__ import annotations

import json
from pathlib import Path

from pareto.rl import paper_final_learned_eval_backfill as backfill
from pareto.rl.paper_final_learned_eval_runner import PAPER_FINAL_LEARNED_EVAL_DONE


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _learned_row(out_dir: str, *, city: str = "jinan", seed: int = 0, preference: str = "balanced") -> dict:
    return {
        "row_index": 12,
        "status": "executable_preview",
        "runner_family": "formal_pilot_paper_final",
        "method": "VectorQ-PPO",
        "city": city,
        "traffic_file": "anon_3_4_jinan_real.json" if city == "jinan" else "anon_4_4_hangzhou_real.json",
        "seed": seed,
        "preference_template": preference,
        "spec_path": f"configs/formal/paper_final_{city}_5seed_ppo.json",
        "out_dir": out_dir,
    }


def test_learned_eval_backfill_plan_builds_eval_command_for_complete_training_row(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "ROOT", tmp_path)
    train_dir = Path("records/paper_final/train_20260602_v1/jinan/anon_3_4_jinan_real/VectorQ-PPO/seed0")
    _write_json(
        tmp_path / train_dir / "status.json",
        {"status": "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE", "fixed_preference_template": "balanced"},
    )
    (tmp_path / train_dir / "checkpoint_last.pt").write_text("checkpoint", encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, {"rows": [_learned_row(train_dir.as_posix())]})

    plan = backfill.build_learned_eval_backfill_plan(
        spec_json=spec_path,
        python_bin="python3",
        device="cpu",
        excluded_cities=(),
    )

    assert plan["counts"]["expected_eval_rows"] == 1
    assert plan["counts"]["rows_to_run"] == 1
    row = plan["rows"][0]
    assert row["status"] == "pending_eval"
    assert row["train_dir"] == train_dir.as_posix()
    assert row["eval_out_dir"].endswith("/VectorQ-PPO/seed0/balanced")
    assert row["command_argv"][:2] == ["python3", "pareto/rl/paper_final_learned_eval_runner.py"]
    assert "--execute" in row["command_argv"]
    assert "--device" in row["command_argv"]


def test_learned_eval_backfill_plan_skips_completed_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "ROOT", tmp_path)
    train_dir = Path("records/paper_final/train_20260602_v1/jinan/anon_3_4_jinan_real/VectorQ-PPO/seed0")
    eval_dir = Path("records/paper_final/eval_20260602_v1/jinan/anon_3_4_jinan_real/VectorQ-PPO/seed0/balanced")
    _write_json(
        tmp_path / train_dir / "status.json",
        {"status": "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE", "fixed_preference_template": "balanced"},
    )
    (tmp_path / train_dir / "checkpoint_last.pt").write_text("checkpoint", encoding="utf-8")
    _write_json(tmp_path / eval_dir / "paper_final_learned_eval_status.json", {"status": PAPER_FINAL_LEARNED_EVAL_DONE})
    _write_json(
        tmp_path / eval_dir / "paper_final_learned_eval_metrics.json",
        {
            "test_reward_over": -1.0,
            "test_avg_queue_len_over": 2.5,
            "test_queuing_vehicle_num_over": 30.0,
            "test_avg_waiting_time_over": 4.0,
            "test_avg_travel_time_over": 51.25,
        },
    )
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, {"rows": [_learned_row(train_dir.as_posix())]})

    plan = backfill.build_learned_eval_backfill_plan(
        spec_json=spec_path,
        python_bin="python3",
        device="cpu",
        excluded_cities=(),
    )

    assert plan["counts"]["expected_eval_rows"] == 1
    assert plan["counts"]["already_complete"] == 1
    assert plan["counts"]["rows_to_run"] == 0
    assert plan["rows"][0]["status"] == "already_complete"
