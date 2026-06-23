from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pareto.rl.formal_analysis_plan import load_formal_analysis_plan


def _valid_plan() -> dict:
    return {
        "plan_type": "formal_jinan_postrun_analysis_plan",
        "approval": {
            "analysis_allowed_now": False,
            "required_exact_phrase": "FORMAL JINAN NO-RANKING ANALYSIS GO",
            "received_exact_phrase": False,
        },
        "scope": {
            "stage": "analysis_plan_only",
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "consumes_run_outputs": False,
            "generates_analysis_outputs": False,
            "generates_ranking_or_performance_table": False,
        },
        "inputs": {
            "guard_audit_json": "docs/pro_reviews/formal_jinan_3seed_guard_audit_2026-05-31.json",
            "allowed_future_input_roots": ["records/formal_jinan_3seed_guarded_20260531"],
        },
        "permissions": {
            "ranking_allowed": False,
            "performance_table_allowed": False,
            "best_method_claim_allowed": False,
            "traffic_control_improvement_claim_allowed": False,
            "city_expansion_allowed": False,
            "seed_expansion_allowed": False,
            "extra_methods_allowed": False,
        },
        "allowed_future_outputs": [
            "guard_audit_summary.json",
            "training_stability_sanity.json",
            "formal_analysis_packet.md",
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
            "traffic improvement",
            "leaderboard",
            "ranked",
        ],
        "allowed_future_metrics": [
            "guard_pass_fail",
            "budget_consistency",
            "finite_training_logs",
            "checkpoint_load_status",
            "env_reward_source_nonzero_status",
        ],
        "statistical_policy": {
            "ranking": "forbidden",
            "mean_std_performance_table": "forbidden",
            "method_comparison_claim": "forbidden",
            "allowed_summary": "guard_and_training_stability_only",
        },
        "method_policy": {
            "ppo_methods": ["film_scalar_potential", "weighted_proxy", "env_reward"],
            "reference_only_methods": ["MaxPressure", "AdvancedMaxPressure"],
            "env_reward_role": "diagnostic_ablation_only",
        },
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "analysis_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_analysis_plan_accepts_plan_only_packet(tmp_path: Path):
    plan = load_formal_analysis_plan(_write(tmp_path, _valid_plan()))

    assert plan.to_dict()["plan_type"] == "formal_jinan_postrun_analysis_plan"


def test_formal_analysis_plan_rejects_analysis_execution_permission(tmp_path: Path):
    payload = _valid_plan()
    payload["approval"]["analysis_allowed_now"] = True

    with pytest.raises(ValueError, match="analysis_allowed_now"):
        load_formal_analysis_plan(_write(tmp_path, payload))


def test_formal_analysis_plan_rejects_ranking_permission(tmp_path: Path):
    payload = _valid_plan()
    payload["permissions"]["ranking_allowed"] = True

    with pytest.raises(ValueError, match="ranking_allowed"):
        load_formal_analysis_plan(_write(tmp_path, payload))


def test_formal_analysis_plan_rejects_forbidden_output_gap(tmp_path: Path):
    payload = _valid_plan()
    payload["forbidden_outputs"].remove("leaderboard.csv")

    with pytest.raises(ValueError, match="forbidden_outputs"):
        load_formal_analysis_plan(_write(tmp_path, payload))


def test_formal_analysis_plan_rejects_forbidden_wording_gap(tmp_path: Path):
    payload = _valid_plan()
    payload["forbidden_wording"].remove("beats")

    with pytest.raises(ValueError, match="forbidden_wording"):
        load_formal_analysis_plan(_write(tmp_path, payload))


def test_formal_analysis_plan_rejects_performance_metric(tmp_path: Path):
    payload = _valid_plan()
    payload["allowed_future_metrics"] = copy.copy(payload["allowed_future_metrics"]) + ["average_travel_time"]

    with pytest.raises(ValueError, match="performance-like metric"):
        load_formal_analysis_plan(_write(tmp_path, payload))
