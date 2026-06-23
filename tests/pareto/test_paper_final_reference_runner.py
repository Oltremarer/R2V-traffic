from __future__ import annotations

import pytest

from pareto.rl.paper_final_reference_runner import (
    PAPER_FINAL_REFERENCE_EXECUTION_APPROVAL_PHRASE,
    build_legacy_reference_command,
    extract_reference_metrics_from_stdout,
    paper_final_reference_env,
    prepare_reference_paths_for_execution,
    reference_model_dir,
    upload_reference_metrics_to_wandb,
    validate_reference_request,
)


def test_reference_request_binds_seed_output_root_and_legacy_command():
    request = validate_reference_request(
        method="Random",
        dataset="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed_id=0,
        out_dir="records/paper_final/train_20260602_v1/jinan/Random/seed0",
        legacy_script="run_random.py",
        execute=False,
        approval_phrase=None,
    )

    command = build_legacy_reference_command(request, python_bin="python3")
    env = paper_final_reference_env(request, base_env={})

    assert command[:2] == ["python3", "run_random.py"]
    assert "--dataset" in command
    assert "jinan" in command
    assert "--traffic_file" in command
    assert "anon_3_4_jinan_real.json" in command
    assert env["PAPER_FINAL_REFERENCE_RUN"] == "1"
    assert env["PAPER_FINAL_REFERENCE_SEED_ID"] == "0"
    assert env["PAPER_FINAL_REFERENCE_OUT_DIR"] == "records/paper_final/train_20260602_v1/jinan/Random/seed0"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["WANDB_MODE"] == "disabled"
    assert env["PAPER_FINAL_REFERENCE_MODEL_DIR"] == "model/paper_final/train_20260602_v1/jinan/Random/seed0"


def test_reference_request_rejects_non_paper_final_output_root():
    with pytest.raises(ValueError, match="records/paper_final"):
        validate_reference_request(
            method="Random",
            dataset="jinan",
            traffic_file="anon_3_4_jinan_real.json",
            seed_id=0,
            out_dir="records/default/jinan/Random/seed0",
            legacy_script="run_random.py",
            execute=False,
            approval_phrase=None,
        )


def test_reference_request_rejects_method_script_mismatch():
    with pytest.raises(ValueError, match="legacy_script mismatch"):
        validate_reference_request(
            method="Random",
            dataset="jinan",
            traffic_file="anon_3_4_jinan_real.json",
            seed_id=0,
            out_dir="records/paper_final/train_20260602_v1/jinan/Random/seed0",
            legacy_script="run_presslight.py",
            execute=False,
            approval_phrase=None,
        )


def test_reference_request_execution_requires_exact_phrase():
    with pytest.raises(ValueError, match="exact external approval phrase"):
        validate_reference_request(
            method="MaxPressure",
            dataset="hangzhou",
            traffic_file="anon_4_4_hangzhou_real.json",
            seed_id=4,
            out_dir="records/paper_final/train_20260602_v1/hangzhou/MaxPressure/seed4",
            legacy_script="run_maxpressure.py",
            execute=True,
            approval_phrase="wrong phrase",
        )

    request = validate_reference_request(
        method="MaxPressure",
        dataset="hangzhou",
        traffic_file="anon_4_4_hangzhou_real.json",
        seed_id=4,
        out_dir="records/paper_final/train_20260602_v1/hangzhou/MaxPressure/seed4",
        legacy_script="run_maxpressure.py",
        execute=True,
        approval_phrase=PAPER_FINAL_REFERENCE_EXECUTION_APPROVAL_PHRASE,
    )
    assert request["execute"] is True


def test_reference_path_preparation_removes_incomplete_output_and_model_dir(tmp_path, monkeypatch):
    request = validate_reference_request(
        method="PressLight",
        dataset="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed_id=0,
        out_dir="records/paper_final/train_20260602_v1/jinan/PressLight/seed0",
        legacy_script="run_presslight.py",
        execute=False,
        approval_phrase=None,
    )
    monkeypatch.setattr("pareto.rl.paper_final_reference_runner.ROOT", tmp_path)
    out_dir = tmp_path / request["out_dir"]
    model_dir = tmp_path / reference_model_dir(request)
    out_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (out_dir / "partial.txt").write_text("partial", encoding="utf-8")
    (model_dir / "partial.txt").write_text("partial", encoding="utf-8")

    prepare_reference_paths_for_execution(request)

    assert not out_dir.exists()
    assert not model_dir.exists()


def test_reference_metrics_are_extracted_from_legacy_stdout():
    stdout = """
    round 0 starts
    {'test_reward_over': -1.0, 'test_avg_queue_len_over': 2.5, 'test_queuing_vehicle_num_over': 30, 'test_avg_waiting_time_over': 4.0, 'test_avg_travel_time_over': 51.25}
    pipeline_wrapper end
    """

    metrics = extract_reference_metrics_from_stdout(stdout)

    assert metrics == {
        "test_reward_over": -1.0,
        "test_avg_queue_len_over": 2.5,
        "test_queuing_vehicle_num_over": 30.0,
        "test_avg_waiting_time_over": 4.0,
        "test_avg_travel_time_over": 51.25,
    }


def test_reference_metrics_accept_numpy_scalar_repr():
    stdout = (
        "{'test_reward_over': np.float64(-1.0), 'test_avg_queue_len_over': np.float64(2.5), "
        "'test_queuing_vehicle_num_over': np.int64(30), 'test_avg_waiting_time_over': np.float64(4.0), "
        "'test_avg_travel_time_over': np.float64(51.25)}"
    )

    assert extract_reference_metrics_from_stdout(stdout)["test_queuing_vehicle_num_over"] == 30.0


class _FakeWandb:
    def __init__(self):
        self.inits = []
        self.logs = []
        self.saved = []
        self.finished = 0

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return self

    def log(self, payload, step=None):
        self.logs.append((payload, step))

    def save(self, path, base_path=None, policy=None):
        self.saved.append((path, base_path, policy))

    def finish(self):
        self.finished += 1


def test_reference_upload_uses_clean_identity_and_top_level_metrics():
    request = validate_reference_request(
        method="Random",
        dataset="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed_id=0,
        out_dir="records/paper_final/train_20260602_v1/jinan/Random/seed0",
        legacy_script="run_random.py",
        execute=False,
        approval_phrase=None,
    )
    fake = _FakeWandb()

    summary = upload_reference_metrics_to_wandb(
        request,
        {
            "test_reward_over": -1.0,
            "test_avg_queue_len_over": 2.5,
            "test_queuing_vehicle_num_over": 30.0,
            "test_avg_waiting_time_over": 4.0,
            "test_avg_travel_time_over": 51.25,
        },
        wandb_module=fake,
        env={"WANDB_PROJECT": "paper_final_scope_limited"},
    )

    assert summary["status"] == "uploaded"
    assert summary["project"] == "paper_final_scope_limited"
    assert fake.inits[0]["name"] == "reference__jinan__Random__seed00"
    assert fake.inits[0]["group"] == "paper_final/no_newyork/reference/jinan/Random"
    assert fake.inits[0]["tags"] == ["paper_final", "no_newyork", "reference", "jinan", "Random", "seed_0"]
    assert fake.logs[0][0]["test_reward_over"] == -1.0
