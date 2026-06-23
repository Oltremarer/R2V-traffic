from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pareto.rl.formal_result_analysis_plan_validator import load_formal_result_analysis_plan


def _valid_plan() -> dict:
    return {
        "plan_type": "formal_jinan_result_analysis_plan_packet",
        "approval": {
            "plan_creation_exact_phrase": "FORMAL JINAN RESULT-ANALYSIS PLAN GO",
            "result_analysis_allowed_now": False,
            "received_future_result_analysis_phrase": False,
        },
        "scope": {
            "stage": "result_analysis_plan_only",
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "consumes_run_logs_now": False,
            "generates_formal_result_values_now": False,
            "generates_method_ordering_now": False,
            "generates_result_table_now": False,
        },
        "allowed_inputs_now": [
            "docs/pro_reviews/formal_jinan_no_ranking_analysis_2026-05-31/guard_audit_summary.json",
            "docs/pro_reviews/formal_jinan_no_ranking_analysis_2026-05-31/training_stability_sanity.json",
            "docs/pro_reviews/formal_jinan_no_ranking_analysis_2026-05-31/formal_analysis_packet.md",
            "docs/pro_reviews/formal_jinan_3seed_guard_audit_2026-05-31.json",
            "docs/pro_reviews/formal_jinan_3seed_run_plan_2026-05-31.md",
            "configs/formal/formal_jinan_analysis_plan_2026-05-31.json",
            "pareto/common/artifact_guard.py",
            "pareto/rl/formal_result_analysis_plan_validator.py",
            "tests/pareto/test_formal_result_analysis_plan_validator.py",
        ],
        "allowed_outputs_now": [
            "formal_jinan_result_analysis_plan.md",
            "formal_jinan_result_analysis_plan.json",
            "formal_result_analysis_plan_validator.py",
            "test_formal_result_analysis_plan_validator.py",
        ],
        "permissions": {
            "read_run_logs_for_result_analysis_allowed": False,
            "result_table_allowed": False,
            "method_ordering_allowed": False,
            "method_comparison_claim_allowed": False,
            "best_method_claim_allowed": False,
            "traffic_control_improvement_claim_allowed": False,
            "city_expansion_allowed": False,
            "seed_expansion_allowed": False,
            "extra_methods_allowed": False,
        },
        "future_analysis_policy": {
            "requires_new_pro_phrase_before_reading_run_logs": True,
            "may_define_future_log_fields": True,
            "actual_log_value_extraction_now": False,
            "formal_result_table_now": False,
        },
        "forbidden_metrics": [
            "travel_time",
            "queue",
            "delay",
            "throughput",
            "waiting_time",
            "traffic metrics",
            "reward total as performance",
            "score",
            "mean/std performance",
            "improvement rate",
            "win/loss count",
        ],
        "forbidden_outputs": [
            "best_method.json",
            "best_method.txt",
            "leaderboard.csv",
            "main_results.csv",
            "method_ranking.csv",
            "paper_results.csv",
            "performance_table.csv",
            "performance_table.json",
            "performance_table.md",
            "performance_table.tex",
            "preference_response_plot.pdf",
            "preference_response_plot.png",
            "preference_sweep.csv",
            "ranking.csv",
            "traffic_metrics.csv",
        ],
        "forbidden_wording": [
            "best method",
            "beats",
            "outperforms",
            "ranked",
            "leaderboard",
            "traffic improvement",
            "state-of-the-art",
            "better than",
            "wins",
            "performance gain",
            "main result",
            "paper result",
        ],
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "result_analysis_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_result_analysis_plan_accepts_plan_only_packet(tmp_path: Path):
    plan = load_formal_result_analysis_plan(_write(tmp_path, _valid_plan()))

    assert plan.to_dict()["plan_type"] == "formal_jinan_result_analysis_plan_packet"


def test_result_analysis_plan_rejects_run_log_reading_now(tmp_path: Path):
    payload = _valid_plan()
    payload["scope"]["consumes_run_logs_now"] = True

    with pytest.raises(ValueError, match="consumes_run_logs_now"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))


def test_result_analysis_plan_rejects_run_log_input_path(tmp_path: Path):
    payload = _valid_plan()
    payload["allowed_inputs_now"].append("records/formal_jinan_3seed_guarded_20260531/seed0/env_reward/train_metrics.jsonl")

    with pytest.raises(ValueError, match="run log input"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))


def test_result_analysis_plan_rejects_result_output_now(tmp_path: Path):
    payload = _valid_plan()
    payload["allowed_outputs_now"].append("main_results.csv")

    with pytest.raises(ValueError, match="forbidden output"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))


def test_result_analysis_plan_rejects_missing_forbidden_metric(tmp_path: Path):
    payload = _valid_plan()
    payload["forbidden_metrics"].remove("queue")

    with pytest.raises(ValueError, match="forbidden_metrics"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))


def test_result_analysis_plan_rejects_missing_forbidden_wording(tmp_path: Path):
    payload = _valid_plan()
    payload["forbidden_wording"].remove("better than")

    with pytest.raises(ValueError, match="forbidden_wording"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))


def test_result_analysis_plan_rejects_result_permission(tmp_path: Path):
    payload = _valid_plan()
    payload["permissions"] = copy.copy(payload["permissions"])
    payload["permissions"]["result_table_allowed"] = True

    with pytest.raises(ValueError, match="result_table_allowed"):
        load_formal_result_analysis_plan(_write(tmp_path, payload))
