from __future__ import annotations

import json
from pathlib import Path

from pareto.eval.paper_representation_artifact_sources import (
    PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID,
    PAPER_FINAL_SCALAR_EVIDENCE_ID,
    PAPER_FINAL_VECTOR_EVIDENCE_ID,
)
from pareto.eval.paper_representation_formal_pass_guard import PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT
from pareto.eval.paper_representation_remediation_diagnostics import (
    build_representation_remediation_diagnostics,
    remediation_diagnostic_blockers,
    write_representation_remediation_diagnostics,
)


def _packet(city: str, *, rev_acc: float = 0.42, rev_threshold: float = 0.6) -> dict:
    return {
        "packet_version": "representation-formal-gate-packet-v1",
        "pro_approval_phrase": "PARETO PPO OFFLINE REPRESENTATION FORMAL-GATE REMEDIATION GO",
        "formal_experiment_requires_new_pro_approval": True,
        "formal_representation_pass": False,
        "formal_gate_decision": {
            "representation_gate_pass": False,
            "claim_mode": "diagnostics_only",
            "failed_reasons": ["rev_acc below formal threshold"],
        },
        "scope": {
            "city": f"{city} paper-final offline records only",
            "formal_experiment_executed": False,
            "new_cityflow_ppo_run_executed": False,
            "multi_seed_executed": False,
            "city_expansion_executed": False,
            "traffic_result_value_reading_executed": False,
            "method_ranking_executed": False,
            "paper_result_claim": False,
        },
        "selected_vector_run": {
            "model_dir": f"model_weights/pareto_quality/{city}/paper_final/{PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID}",
            "run_id": PAPER_FINAL_VECTOR_EVIDENCE_ID,
        },
        "dangerous_scalar_baseline": {
            "model_dir": f"model_weights/cond_scalar/{city}/paper_final/{PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID}",
            "run_id": PAPER_FINAL_SCALAR_EVIDENCE_ID,
        },
        "threshold_checks": {
            "rev_acc": {"value": rev_acc, "threshold": rev_threshold, "pass": False, "bootstrap": {"low": 0.31, "high": 0.5, "n": 40}},
            "pref_acc": {"value": 0.7, "threshold": 0.6, "pass": True, "bootstrap": {"low": 0.62, "high": 0.8, "n": 40}},
            "obj_acc_mean": {"value": 0.8, "threshold": 0.6, "pass": True, "bootstrap": {"low": 0.7, "high": 0.9, "n": 40}},
            "dpr_head": {"value": 0.8, "threshold": 0.6, "pass": True, "bootstrap": {"low": 0.7, "high": 0.9, "n": 40}},
            "dpr_utility": {"value": 0.8, "threshold": 0.6, "pass": True, "bootstrap": {"low": 0.7, "high": 0.9, "n": 40}},
            "head_leakage_diag_offdiag_gap": {"value": 0.3, "threshold": 0.1, "pass": True, "bootstrap": {"low": 0.2, "high": 0.4, "n": 40}},
        },
        "bootstrap_lower_bound_checks": {
            "rev_acc": {"low": 0.31, "threshold": rev_threshold, "pass": False},
            "pref_acc": {"low": 0.62, "threshold": 0.6, "pass": True},
        },
        "pair_coverage_check": {
            "pass": True,
            "splits": {
                "test": {
                    "counts": {"preference_pairs": 100, "dominance_pairs": 50, "reversal_pairs": 40, "objective_pairs": 80},
                    "sampling_strategy_counts": {"efficiency_safety_conflict": 20},
                }
            },
        },
        "split_leakage_check": {"pass": True},
        "train_only_normalizer_check": {"pass": True},
        "bootstrap_pair_count_consistency_check": {"pass": True},
        "objective_sanity_status": {"pass": True},
        "dominance_audit_summary": {"DPR_head_by_objective": {"efficiency": 0.8}},
        "vector_metrics": {"test": {"rev_acc": rev_acc}, "test_bootstrap": {"rev_acc": {"low": 0.31, "high": 0.5}}},
        "condscalar_metrics": {"test_bootstrap": {}},
    }


def _write_packet(root: Path, city: str) -> None:
    packet_dir = root / PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT / city / "vectorq_ppo"
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "representation_formal_gate_packet.json").write_text(
        json.dumps(_packet(city), sort_keys=True),
        encoding="utf-8",
    )


def test_remediation_diagnostics_reports_threshold_margins_without_execution(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    diagnostics = build_representation_remediation_diagnostics(tmp_path)

    assert diagnostics["status"] == "diagnostic_only"
    assert diagnostics["executes_training_now"] is False
    assert diagnostics["regenerates_evidence"] is False
    assert diagnostics["writes_records_paper_final"] is False
    assert remediation_diagnostic_blockers(diagnostics) == []
    jinan = next(row for row in diagnostics["rows"] if row["city"] == "jinan")
    assert jinan["formal_representation_pass"] is False
    assert jinan["threshold_margins"]["rev_acc"]["point_margin"] == -0.18
    assert jinan["threshold_margins"]["rev_acc"]["bootstrap_low_margin"] == -0.29
    assert jinan["pair_counts"]["reversal_pairs"] == 40


def test_remediation_diagnostics_blocks_missing_city_packet(tmp_path: Path):
    _write_packet(tmp_path, "jinan")

    diagnostics = build_representation_remediation_diagnostics(tmp_path)

    assert diagnostics["status"] == "missing_blocker"
    assert any("hangzhou" in blocker for blocker in remediation_diagnostic_blockers(diagnostics))


def test_remediation_diagnostics_derives_reversal_pairs_from_template_counts(tmp_path: Path):
    packet_dir = tmp_path / PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT / "jinan" / "vectorq_ppo"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet = _packet("jinan")
    packet["pair_coverage_check"]["splits"]["test"]["counts"] = {}
    packet["pair_coverage_check"]["splits"]["test"]["reversal_by_template_pair"] = {
        "efficiency__safety": 2,
        "efficiency__stability": 3,
    }
    (packet_dir / "representation_formal_gate_packet.json").write_text(json.dumps(packet), encoding="utf-8")
    for city in ("hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    diagnostics = build_representation_remediation_diagnostics(tmp_path)
    jinan = next(row for row in diagnostics["rows"] if row["city"] == "jinan")

    assert jinan["pair_counts"]["reversal_pairs"] == 5


def test_write_remediation_diagnostics_stays_out_of_records_paper_final(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)
    diagnostics = build_representation_remediation_diagnostics(tmp_path)

    out_dir = tmp_path / "docs" / "pro_reviews" / "diag"
    write_representation_remediation_diagnostics(diagnostics, out_dir)

    assert (out_dir / "representation_remediation_diagnostics.json").is_file()
    assert (out_dir / "representation_remediation_diagnostics.md").is_file()
    assert not (tmp_path / "records" / "paper_final").exists()
