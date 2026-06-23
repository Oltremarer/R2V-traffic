from __future__ import annotations

import pytest

from pareto.eval.paper_representation_command_preview import (
    build_representation_command_preview,
    representation_command_blockers,
    validate_representation_command_preview,
)


def test_representation_command_preview_blocks_until_learned_artifacts_ready():
    preview = build_representation_command_preview(learned_artifacts_ready=False, request_id="paper_final_rep_v1")

    assert preview["status"] == "missing_blocker"
    assert representation_command_blockers(preview) == ["representation: learned artifacts or evidence dirs incomplete"]


def test_representation_command_preview_builds_non_executing_all_city_commands():
    preview = build_representation_command_preview(learned_artifacts_ready=True, request_id="paper_final_rep_v1")

    assert preview["status"] == "ready_request"
    assert len(preview["rows"]) == 6
    assert all(row["executes_generation_now"] is False for row in preview["rows"])
    assert all(row["reads_final_traffic_result_values"] is False for row in preview["rows"])
    assert all("python -m pareto.eval.representation_formal_gate_packet" in row["command_preview"] for row in preview["rows"])
    validate_representation_command_preview(preview)


def test_representation_command_preview_default_does_not_point_to_old_artifact_root():
    preview = build_representation_command_preview(learned_artifacts_ready=True, request_id="paper_final_rep_v1")

    commands = "\n".join(row["command_preview"] for row in preview["rows"])
    assert "paper_final_20260602_v1" not in commands
    assert "paper_final_20260603_v1" in commands


def test_representation_command_preview_uses_distinct_evidence_ids_and_requested_learned_artifact_run_id():
    preview = build_representation_command_preview(
        learned_artifacts_ready=True,
        request_id="paper_final_rep_v1",
        learned_artifact_run_id="paper_final_20260603_v1",
        vector_evidence_id="pareto_quality_paper_final_20260603_v1",
        scalar_evidence_id="cond_scalar_paper_final_20260603_v1",
    )

    for row in preview["rows"]:
        assert "--vector_run_id pareto_quality_paper_final_20260603_v1" in row["command_preview"]
        assert "--scalar_run_id cond_scalar_paper_final_20260603_v1" in row["command_preview"]
        assert "--normalizer_filename objective_norm_paper_final.json" in row["command_preview"]
        assert f"--city {row['city']}" in row["command_preview"]
        assert f"model_weights/pareto_quality/{row['city']}/paper_final/paper_final_20260603_v1" in row["command_preview"]
        assert f"model_weights/cond_scalar/{row['city']}/paper_final/paper_final_20260603_v1" in row["command_preview"]


def test_representation_command_preview_accepts_remediation_evidence_dir_suffix():
    preview = build_representation_command_preview(
        learned_artifacts_ready=True,
        request_id="paper_final_rep_remediation_v1",
        learned_artifact_run_id="paper_final_rep_remediate_20260603_v1",
        vector_evidence_id="pareto_quality_paper_final_rep_remediate_20260603_v1",
        scalar_evidence_id="cond_scalar_paper_final_rep_remediate_20260603_v1",
        evidence_dir_suffix="paper_final_remediation_evidence",
    )

    assert preview["evidence_dir_suffix"] == "paper_final_remediation_evidence"
    for row in preview["rows"]:
        assert "paper_final_remediation_evidence" in row["evidence_dir"]
        assert "--evidence_dir" in row["command_preview"]
        assert "paper_final_remediation_evidence" in row["command_preview"]


def test_representation_command_preview_rejects_final_result_reading():
    preview = build_representation_command_preview(learned_artifacts_ready=True, request_id="paper_final_rep_v1")
    preview["rows"][0] = dict(preview["rows"][0], reads_final_traffic_result_values=True)

    with pytest.raises(ValueError, match="final traffic result"):
        validate_representation_command_preview(preview)
