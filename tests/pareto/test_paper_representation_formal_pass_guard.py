from __future__ import annotations

import json
from pathlib import Path

from pareto.eval.paper_representation_artifact_sources import (
    PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID,
    PAPER_FINAL_SCALAR_EVIDENCE_ID,
    PAPER_FINAL_VECTOR_EVIDENCE_ID,
)
from pareto.eval.paper_representation_formal_pass_guard import (
    PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT,
    REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL,
    build_representation_formal_pass_audit,
    build_representation_scope_limitation,
    representation_formal_pass_blockers,
    validate_representation_scope_limitation,
)


REMEDIATION_OUTPUT_ROOT = "docs/pro_reviews/pareto_ppo_final_representation_remediation_execution_2026-06-03"


def _packet(city: str, *, passes: bool = True, claim_mode: str = "formal_pass") -> dict:
    failed_reasons = [] if passes else ["rev_acc below formal threshold"]
    return {
        "packet_version": "representation-formal-gate-packet-v1",
        "pro_approval_phrase": "PARETO PPO OFFLINE REPRESENTATION FORMAL-GATE REMEDIATION GO",
        "formal_experiment_requires_new_pro_approval": True,
        "formal_representation_pass": passes,
        "formal_gate_decision": {
            "representation_gate_pass": passes,
            "claim_mode": claim_mode,
            "failed_reasons": failed_reasons,
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
        "vector_metrics": {},
        "condscalar_metrics": {},
        "threshold_checks": {
            "obj_acc_mean": {"pass": passes},
            "pref_acc": {"pass": passes},
            "rev_acc": {"pass": passes},
            "dpr_head": {"pass": passes},
            "dpr_utility": {"pass": passes},
        },
        "bootstrap_lower_bound_checks": {
            "obj_acc_mean": {"pass": passes},
            "pref_acc": {"pass": passes},
            "rev_acc": {"pass": passes},
            "dpr_head": {"pass": passes},
            "dpr_utility": {"pass": passes},
        },
    }


def _write_packet(
    root: Path,
    city: str,
    *,
    passes: bool = True,
    claim_mode: str = "formal_pass",
    suffix: str = "vectorq_ppo",
    output_root: str = PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT,
) -> None:
    packet_dir = root / output_root / city / suffix
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "representation_formal_gate_packet.json").write_text(
        json.dumps(_packet(city, passes=passes, claim_mode=claim_mode), sort_keys=True),
        encoding="utf-8",
    )


def test_formal_pass_guard_accepts_all_city_passing_packets(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    audit = build_representation_formal_pass_audit(tmp_path)

    assert audit["status"] == "pass"
    assert representation_formal_pass_blockers(audit) == []


def test_formal_pass_guard_accepts_custom_remediation_output_root(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city, output_root=REMEDIATION_OUTPUT_ROOT)

    audit = build_representation_formal_pass_audit(tmp_path, packet_output_root=REMEDIATION_OUTPUT_ROOT)

    assert audit["status"] == "pass"
    assert representation_formal_pass_blockers(audit) == []


def test_formal_pass_guard_can_ignore_old_packets_for_remediation_root(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city, passes=False, claim_mode="diagnostics_only")
        _write_packet(tmp_path, city, output_root=REMEDIATION_OUTPUT_ROOT)

    audit = build_representation_formal_pass_audit(
        tmp_path,
        packet_output_root=REMEDIATION_OUTPUT_ROOT,
        ignore_packets_outside_output_root=True,
    )

    assert audit["status"] == "pass"
    assert representation_formal_pass_blockers(audit) == []
    assert any("outside selected representation packet output root" in row["reason"] for row in audit["ignored_packets"])


def test_formal_pass_guard_blocks_diagnostics_only_packets(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city, passes=False, claim_mode="diagnostics_only")

    audit = build_representation_formal_pass_audit(tmp_path)
    blockers = representation_formal_pass_blockers(audit)

    assert audit["status"] == "blocked"
    assert any("jinan" in blocker and "diagnostics_only" in blocker for blocker in blockers)
    assert any("newyork_28x7" in blocker and "rev_acc below formal threshold" in blocker for blocker in blockers)


def test_formal_pass_guard_blocks_result_value_reading_flag(tmp_path: Path):
    packet = _packet("jinan")
    packet["scope"]["traffic_result_value_reading_executed"] = True
    packet_dir = tmp_path / PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT / "jinan" / "vectorq_ppo"
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "representation_formal_gate_packet.json").write_text(json.dumps(packet), encoding="utf-8")
    for city in ("hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    blockers = representation_formal_pass_blockers(build_representation_formal_pass_audit(tmp_path))

    assert any("jinan" in blocker and "traffic_result_value_reading_executed" in blocker for blocker in blockers)


def test_formal_pass_guard_blocks_multiple_hashes_for_one_city(tmp_path: Path):
    _write_packet(tmp_path, "jinan", suffix="vectorq_ppo")
    _write_packet(tmp_path, "jinan", passes=False, claim_mode="diagnostics_only", suffix="cond_scalar_rl")
    for city in ("hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    blockers = representation_formal_pass_blockers(build_representation_formal_pass_audit(tmp_path))

    assert any("jinan" in blocker and "multiple paper-final packet hashes" in blocker for blocker in blockers)


def test_formal_pass_guard_blocks_paper_final_packet_outside_output_root(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city, output_root="docs/pro_reviews/paper_final_packets")

    blockers = representation_formal_pass_blockers(build_representation_formal_pass_audit(tmp_path))

    assert any("jinan" in blocker and "outside paper-final representation packet output root" in blocker for blocker in blockers)


def test_representation_scope_limitation_requires_exact_phrase_and_claim_limitation():
    blocked = build_representation_scope_limitation(approval_phrase="close but wrong")

    assert blocked["status"] == "missing_blocker"
    assert "exact reviewer approval" in blocked["blocker"]

    row = build_representation_scope_limitation(
        approval_phrase=REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL,
        paper_claim_limitation="Representation formal gate failed and is reported only as a diagnostic limitation.",
    )

    assert row["status"] == "diagnostic_limitation_by_reviewer"
    assert row["executes_now"] is False
    validate_representation_scope_limitation(row)
