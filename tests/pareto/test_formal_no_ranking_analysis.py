from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_analysis_plan import FORMAL_ANALYSIS_APPROVAL_PHRASE
from pareto.rl.formal_no_ranking_analysis import run_no_ranking_analysis


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _minimal_plan(tmp_path: Path) -> Path:
    plan = {
        "plan_type": "formal_jinan_postrun_analysis_plan",
        "approval": {
            "analysis_allowed_now": False,
            "required_exact_phrase": FORMAL_ANALYSIS_APPROVAL_PHRASE,
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
            "guard_audit_json": "audit.json",
            "allowed_future_input_roots": ["runs"],
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
            "leaderboard",
            "outperforms",
            "ranked",
            "traffic improvement",
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
    path = tmp_path / "analysis_plan.json"
    _write_json(path, plan)
    return path


def _minimal_run_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "runs"
    audit = {
        "report_status": "FORMAL_JINAN_3SEED_GUARD_PASS",
        "failures": [],
        "budget_consistent": True,
        "runs": [],
    }
    for seed in (0, 1, 2):
        for method in ("film_scalar_potential", "weighted_proxy", "env_reward"):
            run_dir = root / f"seed{seed}" / method
            run_dir.mkdir(parents=True)
            _write_json(
                run_dir / "metadata.json",
                {
                    "method": method,
                    "cityflow_seed": seed,
                    "policy_seed": seed,
                    "model_seed": seed,
                    "performance_claim": False,
                    "method_ranking_allowed": False,
                    "performance_table_allowed": False,
                    "env_reward_summary": {
                        "finite": True,
                        "all_zero_reward": False,
                        "env_reward_sources": ["cityflow_average_reward"],
                    }
                    if method == "env_reward"
                    else None,
                },
            )
            _write_json(run_dir / "status.json", {"policy_update_count": 1})
            _write_jsonl(run_dir / "loss_debug.jsonl", [{"total_loss": 0.1}])
            _write_jsonl(run_dir / "reward_components.jsonl", [{"total_reward": 0.1}])
            audit["runs"].append(
                {
                    "seed": seed,
                    "method": method,
                    "checkpoint_loads": True,
                    "policy_update_count": 1,
                    "loss_debug_rows": 1,
                }
            )
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)
    return root, audit_path


def test_no_ranking_analysis_writes_only_allowed_outputs(tmp_path: Path):
    plan = _minimal_plan(tmp_path)
    root, audit = _minimal_run_tree(tmp_path)
    out = tmp_path / "out"

    report = run_no_ranking_analysis(
        root=root,
        guard_audit_json=audit,
        analysis_plan=plan,
        out_dir=out,
        approval_phrase=FORMAL_ANALYSIS_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_NO_RANKING_ANALYSIS_PASS"
    assert sorted(path.name for path in out.iterdir()) == [
        "formal_analysis_packet.md",
        "guard_audit_summary.json",
        "training_stability_sanity.json",
    ]
    packet = (out / "formal_analysis_packet.md").read_text(encoding="utf-8")
    assert "leaderboard" not in packet.lower()
    assert "beats" not in packet.lower()


def test_no_ranking_analysis_rejects_wrong_phrase(tmp_path: Path):
    plan = _minimal_plan(tmp_path)
    root, audit = _minimal_run_tree(tmp_path)

    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        run_no_ranking_analysis(
            root=root,
            guard_audit_json=audit,
            analysis_plan=plan,
            out_dir=tmp_path / "out",
            approval_phrase="wrong",
        )
