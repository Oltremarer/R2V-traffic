from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pareto.r2v.result_aggregation import aggregate_r2v_results


def test_aggregate_r2v_results_separates_performance_from_integrity(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "method": "baseline",
                    "seed": 0,
                    "average_travel_time": 100.0,
                    "queue_length": 10.0,
                    "delay": 3.0,
                    "throughput": 50.0,
                    "reward": -20.0,
                    "status": "DONE",
                },
                {
                    "method": "r2v",
                    "seed": 0,
                    "average_travel_time": 90.0,
                    "queue_length": 8.0,
                    "delay": 2.5,
                    "throughput": 55.0,
                    "reward": -15.0,
                    "status": "DONE",
                },
            ]
        ),
        encoding="utf-8",
    )
    integrity = tmp_path / "r2v_summary.json"
    integrity.write_text(
        json.dumps(
            {
                "candidate_count": 10,
                "admitted_count": 4,
                "gate_counts": {"rare": 5, "value": 4, "support": 8, "dynamics": 6},
                "status": "COMPLETED",
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_r2v_results(
        performance_paths=[perf],
        integrity_paths=[integrity],
    )

    assert result["schema_version"] == "r2v-traffic-result-aggregation-v1"
    assert set(result["performance"]["metrics"]) == {
        "average_travel_time",
        "queue_length",
        "delay",
        "throughput",
        "reward",
    }
    assert "status" not in result["performance"]["metrics"]
    assert result["performance"]["by_method"]["r2v"]["average_travel_time"]["mean"] == 90.0
    assert result["integrity"]["total_candidate_count"] == 10
    assert result["integrity"]["total_admitted_count"] == 4


def test_aggregate_r2v_results_reports_row_count_separately_from_metric_values(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "method": "baseline",
                    "seed": 0,
                    "average_travel_time": 100.0,
                    "queue_length": 10.0,
                    "delay": 3.0,
                    "throughput": 50.0,
                    "reward": -20.0,
                },
                {
                    "method": "r2v",
                    "seed": 0,
                    "average_travel_time": 90.0,
                    "queue_length": 8.0,
                    "delay": 2.5,
                    "throughput": 55.0,
                    "reward": -15.0,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = aggregate_r2v_results(performance_paths=[perf], integrity_paths=[])

    assert result["performance"]["row_count"] == 2
    assert result["performance"]["metric_value_count"] == 10
    assert result["performance"]["by_method_row_count"] == {"baseline": 1, "r2v": 1}


def test_aggregate_r2v_results_records_input_artifact_hashes(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "r2v",
                "seed": 0,
                "average_travel_time": 90.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    integrity = tmp_path / "r2v_summary.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": "r2v-tsc-candidate-summary-v1",
                "candidate_count": 4,
                "admitted_count": 2,
                "gate_counts": {"rare": 4, "value": 3, "support": 4, "safety": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = aggregate_r2v_results(performance_paths=[perf], integrity_paths=[integrity])

    performance_input = result["input_artifacts"]["performance"][0]
    integrity_input = result["input_artifacts"]["integrity"][0]
    assert performance_input["path"] == str(perf)
    assert performance_input["sha256"] == hashlib.sha256(perf.read_bytes()).hexdigest()
    assert performance_input["size_bytes"] == perf.stat().st_size
    assert performance_input["line_count"] == 1
    assert integrity_input["path"] == str(integrity)
    assert integrity_input["sha256"] == hashlib.sha256(integrity.read_bytes()).hexdigest()
    assert integrity_input["size_bytes"] == integrity.stat().st_size
    assert integrity_input["line_count"] == 1


def test_aggregate_r2v_results_rejects_status_only_performance_rows(tmp_path: Path):
    status_only = tmp_path / "status.jsonl"
    status_only.write_text(json.dumps({"method": "r2v", "seed": 0, "status": "DONE"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no performance metrics"):
        aggregate_r2v_results(performance_paths=[status_only], integrity_paths=[])


def test_aggregate_r2v_results_rejects_nonfinite_metric(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "r2v",
                "seed": 0,
                "average_travel_time": "nan",
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite performance metric"):
        aggregate_r2v_results(performance_paths=[perf], integrity_paths=[])


def test_aggregate_r2v_results_accepts_legacy_eval_metric_aliases(tmp_path: Path):
    perf = tmp_path / "legacy_eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "baseline",
                "seed": 0,
                "test_avg_travel_time_over": 100.0,
                "test_avg_queue_len_over": 10.0,
                "test_avg_waiting_time_over": 3.0,
                "throughput": 40.0,
                "test_reward_over": -20.0,
                "test_queuing_vehicle_num_over": 999.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = aggregate_r2v_results(performance_paths=[perf], integrity_paths=[])

    metrics = result["performance"]["by_method"]["baseline"]
    assert metrics["average_travel_time"]["mean"] == 100.0
    assert metrics["queue_length"]["mean"] == 10.0
    assert metrics["delay"]["mean"] == 3.0
    assert metrics["throughput"]["mean"] == 40.0
    assert "test_queuing_vehicle_num_over" not in result["performance"]["metrics"]


def test_result_aggregation_cli_writes_output_json(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "r2v",
                "seed": 0,
                "average_travel_time": 90.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    integrity = tmp_path / "r2v_summary.json"
    integrity.write_text(
        json.dumps({"candidate_count": 4, "admitted_count": 2, "status": "COMPLETED"}),
        encoding="utf-8",
    )
    output = tmp_path / "aggregation.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.result_aggregation",
            "--performance_path",
            str(perf),
            "--integrity_path",
            str(integrity),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["performance"]["row_count"] == 1
    assert payload["integrity"]["total_admitted_count"] == 2
