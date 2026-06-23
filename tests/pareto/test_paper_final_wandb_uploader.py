from __future__ import annotations

import json
from pathlib import Path

import pareto.rl.paper_final_wandb_uploader as uploader
from pareto.rl.paper_final_wandb_uploader import (
    PAPER_FINAL_WANDB_PROJECT,
    paper_final_wandb_env,
    upload_paper_final_learned_eval_to_wandb,
    upload_paper_final_run_to_wandb,
)


class FakeWandb:
    def __init__(self) -> None:
        self.inits: list[dict] = []
        self.logs: list[tuple[dict, int | None]] = []
        self.saved: list[dict] = []
        self.finished = 0

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return object()

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logs.append((payload, step))

    def save(self, path: str, **kwargs) -> None:
        self.saved.append({"path": path, **kwargs})

    def finish(self) -> None:
        self.finished += 1


class FakeWandbWithSettings(FakeWandb):
    class Settings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs


class FlakyInitWandb(FakeWandbWithSettings):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def init(self, **kwargs):
        self.inits.append(kwargs)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary wandb init failure")
        return object()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sample_out_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "records" / "paper_final" / "jinan" / "VectorQ-PPO" / "seed0" / "balanced"
    out_dir.mkdir(parents=True)
    _write_json(
        out_dir / "metadata.json",
        {
            "paper_final_execution": True,
            "method": "vector_quality_potential",
            "episodes": 2,
        },
    )
    _write_json(
        out_dir / "status.json",
        {
            "status": "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE",
            "episodes": 2,
            "total_env_steps_per_seed": 1000,
        },
    )
    _write_jsonl(
        out_dir / "train_metrics.jsonl",
        [
            {
                "episode": 0,
                "step": 1,
                "sim_time": 30,
                "preference_name": "balanced",
                "reward_mean": 1.5,
            },
            {
                "episode": 0,
                "step": 2,
                "sim_time": 60,
                "preference_name": "balanced",
                "reward_mean": 1.75,
            },
        ],
    )
    return out_dir


def _sample_row(out_dir: Path) -> dict:
    return {
        "row_index": 7,
        "method": "VectorQ-PPO",
        "city": "jinan",
        "seed": 0,
        "preference_template": "balanced",
        "runner_family": "formal_pilot_paper_final",
        "out_dir": out_dir.as_posix(),
    }


def _sample_result() -> dict:
    return {
        "row_index": 7,
        "returncode": 0,
        "stdout_path": "logs/007.stdout.txt",
        "stderr_path": "logs/007.stderr.txt",
    }


def test_paper_final_wandb_env_defaults_to_online_upload():
    env = paper_final_wandb_env(base_env={})

    assert env["WANDB_MODE"] == "online"
    assert env["WANDB_PROJECT"] == PAPER_FINAL_WANDB_PROJECT


def test_upload_paper_final_run_logs_train_metrics_and_status(tmp_path):
    out_dir = _sample_out_dir(tmp_path)
    fake = FakeWandb()

    summary = upload_paper_final_run_to_wandb(
        _sample_row(out_dir),
        _sample_result(),
        wandb_module=fake,
        env={},
    )

    assert summary["status"] == "uploaded"
    assert summary["project"] == PAPER_FINAL_WANDB_PROJECT
    assert summary["metric_rows_logged"] == 2
    assert fake.inits[0]["project"] == PAPER_FINAL_WANDB_PROJECT
    assert fake.inits[0]["name"] == "learned__jinan__VectorQ-PPO__seed0__balanced__row007"
    assert fake.inits[0]["group"] == "paper_final/no_newyork/learned/jinan/VectorQ-PPO/balanced"
    assert fake.inits[0]["tags"] == [
        "paper_final",
        "no_newyork",
        "learned",
        "jinan",
        "VectorQ-PPO",
        "seed_0",
        "balanced",
    ]
    assert fake.logs[0][0]["train/reward_mean"] == 1.5
    assert fake.logs[0][1] == 0
    assert fake.logs[1][0]["train/reward_mean"] == 1.75
    assert fake.logs[2][0]["status/episodes"] == 2
    assert fake.saved[0]["path"].endswith("metadata.json")
    assert fake.saved[1]["path"].endswith("status.json")
    assert fake.saved[2]["path"].endswith("train_metrics.jsonl")
    assert fake.finished == 1


def test_upload_paper_final_run_passes_wandb_timeout_settings(tmp_path):
    out_dir = _sample_out_dir(tmp_path)
    fake = FakeWandbWithSettings()

    summary = upload_paper_final_run_to_wandb(
        _sample_row(out_dir),
        _sample_result(),
        wandb_module=fake,
        env={"WANDB_INIT_TIMEOUT": "180", "WANDB__SERVICE_WAIT": "180"},
    )

    assert summary["status"] == "uploaded"
    settings = fake.inits[0]["settings"]
    assert settings.kwargs == {"init_timeout": 180, "_service_wait": 180}


def test_upload_paper_final_run_retries_transient_init_failure(tmp_path, monkeypatch):
    out_dir = _sample_out_dir(tmp_path)
    fake = FlakyInitWandb()
    sleeps: list[int] = []
    monkeypatch.setattr(uploader.time, "sleep", sleeps.append)

    summary = upload_paper_final_run_to_wandb(
        _sample_row(out_dir),
        _sample_result(),
        wandb_module=fake,
        env={"WANDB_UPLOAD_RETRIES": "2", "WANDB_UPLOAD_RETRY_DELAY_SECONDS": "1"},
    )

    assert summary["status"] == "uploaded"
    assert summary["attempt"] == 2
    assert len(fake.inits) == 2
    assert fake.finished == 2
    assert sleeps == [1]


def test_upload_learned_eval_logs_reference_schema_metrics(tmp_path):
    eval_dir = tmp_path / "records" / "paper_final" / "eval_20260602_v1" / "jinan" / "anon_3_4_jinan_real" / "VectorQ-PPO" / "seed0" / "balanced"
    eval_dir.mkdir(parents=True)
    _write_json(
        eval_dir / "paper_final_learned_eval_metadata.json",
        {
            "method": "VectorQ-PPO",
            "city": "jinan",
            "seed_id": 0,
            "fixed_preference_template": "balanced",
        },
    )
    _write_json(
        eval_dir / "paper_final_learned_eval_status.json",
        {
            "status": "PAPER_FINAL_LEARNED_EVAL_DONE",
            "decision_steps": 120,
        },
    )
    _write_json(
        eval_dir / "paper_final_learned_eval_metrics.json",
        {
            "test_reward_over": -1.0,
            "test_avg_queue_len_over": 2.5,
            "test_queuing_vehicle_num_over": 30.0,
            "test_avg_waiting_time_over": 4.0,
            "test_avg_travel_time_over": 51.25,
        },
    )
    fake = FakeWandb()

    summary = upload_paper_final_learned_eval_to_wandb(
        {
            "row_index": 7,
            "method": "VectorQ-PPO",
            "city": "jinan",
            "seed": 0,
            "preference_template": "balanced",
            "eval_out_dir": eval_dir.as_posix(),
        },
        wandb_module=fake,
        env={},
    )

    assert summary["status"] == "uploaded"
    assert summary["metric_keys"] == [
        "test_reward_over",
        "test_avg_queue_len_over",
        "test_queuing_vehicle_num_over",
        "test_avg_waiting_time_over",
        "test_avg_travel_time_over",
    ]
    assert fake.inits[0]["name"] == "learned_eval__jinan__VectorQ-PPO__seed0__balanced__row007"
    assert fake.inits[0]["group"] == "paper_final/no_newyork/learned_eval/jinan/VectorQ-PPO/balanced"
    assert fake.logs[0][0]["test_avg_travel_time_over"] == 51.25
    assert fake.logs[0][1] == 0
    assert any(item["path"].endswith("paper_final_learned_eval_metrics.json") for item in fake.saved)
