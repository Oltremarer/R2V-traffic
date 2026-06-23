from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.eval.paper_representation_artifact_sources import (
    PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID,
    PAPER_FINAL_SCALAR_EVIDENCE_ID,
    PAPER_FINAL_VECTOR_EVIDENCE_ID,
    inventory_representation_artifact_sources,
    representation_artifact_blockers,
    validate_representation_artifact_row,
    validate_representation_packet,
)


REMEDIATION_RUN_ID = "paper_final_rep_remediate_20260603_v1"
REMEDIATION_VECTOR_ID = "pareto_quality_paper_final_rep_remediate_20260603_v1"
REMEDIATION_SCALAR_ID = "cond_scalar_paper_final_rep_remediate_20260603_v1"


def _packet(
    city: str = "jinan",
    *,
    paper_final: bool = True,
    learned_artifact_run_id: str = PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID,
    vector_evidence_id: str = PAPER_FINAL_VECTOR_EVIDENCE_ID,
    scalar_evidence_id: str = PAPER_FINAL_SCALAR_EVIDENCE_ID,
) -> dict:
    vector_model_dir = (
        f"model_weights/pareto_quality/{city}/paper_final/{learned_artifact_run_id}"
        if paper_final
        else "model_weights/pareto_quality/jinan/dpr_margin_remediation/dpr_e40_m03_iso4_c2_u03"
    )
    scalar_model_dir = (
        f"model_weights/cond_scalar/{city}/paper_final/{learned_artifact_run_id}"
        if paper_final
        else "model_weights/cond_scalar/jinan/preformal_final/film_rich_v2"
    )
    return {
        "packet_version": "representation-formal-gate-packet-v1",
        "pro_approval_phrase": "PARETO PPO OFFLINE REPRESENTATION EVAL-CONSISTENCY REV-ACC REMEDIATION GO",
        "formal_experiment_requires_new_pro_approval": True,
        "scope": {
            "city": f"{city} paper-final offline records only" if paper_final else "Jinan offline records only",
            "traffic_result_value_reading_executed": False,
            "paper_result_claim": False,
        },
        "selected_vector_run": {
            "model_dir": vector_model_dir,
            "run_id": vector_evidence_id if paper_final else "dpr_e40_m03_iso4_c2_u03",
        },
        "dangerous_scalar_baseline": {
            "model_dir": scalar_model_dir,
            "run_id": scalar_evidence_id if paper_final else "film_rich_v2",
        },
        "vector_metrics": {},
        "condscalar_metrics": {},
        "threshold_checks": {
            "obj_acc_mean": {},
            "pref_acc": {},
            "rev_acc": {},
            "dpr_head": {},
            "dpr_utility": {},
        },
    }


def _write_packet(root: Path, city: str, *, paper_final: bool = True, name: str | None = None) -> None:
    packet_dir = root / "docs" / "pro_reviews" / (name or f"representation_{city}")
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "representation_formal_gate_packet.json").write_text(
        json.dumps(_packet(city, paper_final=paper_final)),
        encoding="utf-8",
    )


def test_representation_packet_validates_required_metric_keys():
    source = validate_representation_packet(_packet())

    assert source["metrics"] == ["obj_acc", "pref_acc", "rev_acc", "dpr"]
    assert source["packet_keys"]["dpr"] == ["dpr_head", "dpr_utility"]


def test_representation_packet_accepts_remediation_run_ids():
    source = validate_representation_packet(
        _packet(
            "jinan",
            learned_artifact_run_id=REMEDIATION_RUN_ID,
            vector_evidence_id=REMEDIATION_VECTOR_ID,
            scalar_evidence_id=REMEDIATION_SCALAR_ID,
        )
    )

    assert source["city"] == "jinan"
    assert set(source["model_families"]) == {"VectorQ-PPO", "Cond-Scalar-RL"}


def test_representation_packet_rejects_missing_dpr_key():
    packet = _packet()
    del packet["threshold_checks"]["dpr_utility"]

    with pytest.raises(ValueError, match="missing representation packet key"):
        validate_representation_packet(packet)


def test_representation_packet_rejects_legacy_non_paper_final_scope():
    with pytest.raises(ValueError, match="paper-final"):
        validate_representation_packet(_packet("jinan", paper_final=False))


def test_representation_inventory_ignores_legacy_jinan_packets(tmp_path: Path):
    _write_packet(tmp_path, "jinan", paper_final=False, name="legacy_jinan")

    audit = inventory_representation_artifact_sources(tmp_path)
    blockers = representation_artifact_blockers(audit)

    assert any("jinan" in blocker for blocker in blockers)
    assert audit["ignored_packets"] == [
        {
            "packet_path": "docs/pro_reviews/legacy_jinan/representation_formal_gate_packet.json",
            "reason": "representation packet is not paper-final scope",
        }
    ]


def test_representation_inventory_keeps_missing_city_coverage_blocked(tmp_path: Path):
    _write_packet(tmp_path, "jinan")

    audit = inventory_representation_artifact_sources(tmp_path)
    blockers = representation_artifact_blockers(audit)

    assert any("hangzhou" in blocker for blocker in blockers)
    assert any("newyork_28x7" in blocker for blocker in blockers)


def test_representation_inventory_accepts_all_city_packet_coverage(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city)

    audit = inventory_representation_artifact_sources(tmp_path)

    assert representation_artifact_blockers(audit) == []
    for row in audit["rows"]:
        assert row["status"] == "implemented_guarded_preview"
        assert row["executes_training_now"] is False
        validate_representation_artifact_row(row)


def test_representation_inventory_prefers_family_specific_packet_paths(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _write_packet(tmp_path, city, name=f"representation_{city}/cond_scalar_rl")
        _write_packet(tmp_path, city, name=f"representation_{city}/vectorq_ppo")

    audit = inventory_representation_artifact_sources(tmp_path)

    assert representation_artifact_blockers(audit) == []
    for row in audit["rows"]:
        if row["model_family"] == "VectorQ-PPO":
            assert "/vectorq_ppo/" in row["packet_path"]
        if row["model_family"] == "Cond-Scalar-RL":
            assert "/cond_scalar_rl/" in row["packet_path"]
