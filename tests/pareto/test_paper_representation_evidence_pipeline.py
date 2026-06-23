from __future__ import annotations

import json
from pathlib import Path

from pareto.eval.paper_representation_evidence_pipeline import (
    SCALAR_EVIDENCE_ID,
    VECTOR_EVIDENCE_ID,
    build_representation_evidence_plan,
    expected_evidence_filenames,
    main,
    materialize_city_evidence,
)


RUN_ID = "paper_final_20260603_v1"
REMEDIATION_RUN_ID = "paper_final_rep_remediate_20260603_v1"
REMEDIATION_VECTOR_ID = "pareto_quality_paper_final_rep_remediate_20260603_v1"
REMEDIATION_SCALAR_ID = "cond_scalar_paper_final_rep_remediate_20260603_v1"


def _write(path: Path, payload: dict | str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload) if isinstance(payload, dict) else payload
    path.write_text(text + "\n", encoding="utf-8")


def _populate_city(root: Path, city: str = "jinan", run_id: str = RUN_ID) -> None:
    raw = root / "data" / "pareto_records_split" / city / "paper_final"
    norm = root / "data" / "pareto_records_split_norm" / city / "paper_final"
    pairs = root / "data" / "pareto_pairs" / city / "paper_final"
    for split in ("train", "val", "test"):
        _write(raw / f"{split}_raw.jsonl", "{\"sample_id\":\"s1\"}")
        _write(norm / f"{split}_raw.jsonl", "{\"sample_id\":\"s1\"}")
        split_dir = pairs / split
        for name in ("objective_pairs", "preference_pairs", "dominance_pairs", "reversal_pairs"):
            _write(split_dir / f"{name}.jsonl", "")
        _write(split_dir / "pair_report.json", {"split": split, "counts": {}})
    _write(raw / "split_records_report.json", {"city": city})
    _write(root / "data" / "normalizers" / city / "objective_norm_paper_final.json", {"hash": city})
    for family, marker in (("pareto_quality", "vector"), ("cond_scalar", "scalar")):
        artifact = root / "model_weights" / family / city / "paper_final" / run_id
        _write(artifact / "model.pt", marker)
        _write(artifact / "metadata.json", {"family": family, "param_count": 10})
        _write(artifact / "diagnostics_val.json", {"family": family, "split": "val"})
        _write(artifact / "diagnostics_test.json", {"family": family, "split": "test"})
        _write(artifact / "objective_normalizer.json", {"hash": city})


def _fake_runner(command: list[str]) -> None:
    out = Path(command[command.index("--out") + 1])
    if "objective_sanity.py" in " ".join(command):
        _write(out, {"strict_failures": [], "warnings": [], "safety_valid_rate": 1.0, "objective_correlations": {}})
    elif "offline_pair_bootstrap.py" in " ".join(command):
        _write(out, {"vector": {"metrics": {}}, "cond_scalar": {"metrics": {}}})
    elif "dominance_error_audit.py" in " ".join(command):
        _write(out, {"audit": {}})
    else:
        raise AssertionError(f"unexpected command: {command}")


def test_evidence_plan_reports_ready_rows_for_complete_fixture(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _populate_city(tmp_path, city)

    plan = build_representation_evidence_plan(tmp_path, learned_artifact_run_id=RUN_ID)

    assert plan["status"] == "ready_request"
    assert len(plan["rows"]) == 3
    assert all(row["status"] == "ready" for row in plan["rows"])
    assert plan["vector_evidence_id"] == VECTOR_EVIDENCE_ID
    assert plan["scalar_evidence_id"] == SCALAR_EVIDENCE_ID
    assert plan["executes_generation_now"] is False
    assert plan["reads_final_traffic_result_values"] is False


def test_evidence_plan_fails_closed_when_scalar_diagnostics_missing(tmp_path: Path):
    _populate_city(tmp_path, "jinan")
    missing = tmp_path / "model_weights" / "cond_scalar" / "jinan" / "paper_final" / RUN_ID / "diagnostics_test.json"
    missing.unlink()

    plan = build_representation_evidence_plan(tmp_path, learned_artifact_run_id=RUN_ID, cities=("jinan",))

    row = plan["rows"][0]
    assert row["status"] == "missing_blocker"
    assert any("cond_scalar" in path and "diagnostics_test.json" in path for path in row["missing_files"])


def test_evidence_plan_fails_closed_when_records_paper_final_not_empty(tmp_path: Path):
    _populate_city(tmp_path, "jinan")
    _write(tmp_path / "records" / "paper_final" / "unexpected.json", {})

    plan = build_representation_evidence_plan(tmp_path, learned_artifact_run_id=RUN_ID, cities=("jinan",))

    row = plan["rows"][0]
    assert row["status"] == "missing_blocker"
    assert any("records/paper_final" in path for path in row["missing_files"])


def test_materialize_city_evidence_uses_distinct_vector_and_scalar_filenames(tmp_path: Path):
    _populate_city(tmp_path, "jinan")

    result = materialize_city_evidence(tmp_path, "jinan", learned_artifact_run_id=RUN_ID, command_runner=_fake_runner)

    evidence = Path(result["evidence_dir"])
    expected = set(expected_evidence_filenames())
    assert expected <= {path.name for path in evidence.iterdir()}
    vector_test = json.loads((evidence / f"{VECTOR_EVIDENCE_ID}_diagnostics_test.json").read_text())
    scalar_test = json.loads((evidence / f"{SCALAR_EVIDENCE_ID}_diagnostics_test.json").read_text())
    assert vector_test["family"] == "pareto_quality"
    assert scalar_test["family"] == "cond_scalar"
    assert (evidence / f"{VECTOR_EVIDENCE_ID}_metadata.json").read_text() != (
        evidence / f"{SCALAR_EVIDENCE_ID}_metadata.json"
    ).read_text()


def test_materialize_city_evidence_uses_configured_python_executable(tmp_path: Path):
    _populate_city(tmp_path, "jinan")
    commands: list[list[str]] = []

    def runner(command: list[str]) -> None:
        commands.append(command)
        _fake_runner(command)

    materialize_city_evidence(
        tmp_path,
        "jinan",
        learned_artifact_run_id=RUN_ID,
        command_runner=runner,
        python_executable="/remote/c2t/python",
    )

    assert commands
    assert all(command[0] == "/remote/c2t/python" for command in commands)


def test_evidence_pipeline_accepts_custom_remediation_ids_and_evidence_dir(tmp_path: Path):
    _populate_city(tmp_path, "jinan", run_id=REMEDIATION_RUN_ID)
    commands: list[list[str]] = []

    def runner(command: list[str]) -> None:
        commands.append(command)
        _fake_runner(command)

    plan = build_representation_evidence_plan(
        tmp_path,
        learned_artifact_run_id=REMEDIATION_RUN_ID,
        vector_evidence_id=REMEDIATION_VECTOR_ID,
        scalar_evidence_id=REMEDIATION_SCALAR_ID,
        evidence_dir_suffix="paper_final_remediation_evidence",
        cities=("jinan",),
    )
    result = materialize_city_evidence(
        tmp_path,
        "jinan",
        learned_artifact_run_id=REMEDIATION_RUN_ID,
        vector_evidence_id=REMEDIATION_VECTOR_ID,
        scalar_evidence_id=REMEDIATION_SCALAR_ID,
        evidence_dir_suffix="paper_final_remediation_evidence",
        command_runner=runner,
    )

    evidence = Path(result["evidence_dir"])
    assert plan["vector_evidence_id"] == REMEDIATION_VECTOR_ID
    assert plan["scalar_evidence_id"] == REMEDIATION_SCALAR_ID
    assert "paper_final_remediation_evidence" in plan["rows"][0]["evidence_dir"]
    assert evidence.name == "pareto_offline_representation_jinan_paper_final_remediation_evidence"
    assert (evidence / f"{REMEDIATION_VECTOR_ID}_diagnostics_test.json").is_file()
    assert (evidence / f"{REMEDIATION_SCALAR_ID}_diagnostics_test.json").is_file()
    assert (evidence / f"{REMEDIATION_VECTOR_ID}_formal_gate_decision.json").is_file()
    assert all(REMEDIATION_RUN_ID in " ".join(command) for command in commands if "--model_dir" in command or "--vector_model_dir" in command)


def test_cli_preview_mode_does_not_create_evidence_dir(tmp_path: Path):
    _populate_city(tmp_path, "jinan")

    main(["--root", str(tmp_path), "--city", "jinan"])

    assert not (tmp_path / "docs" / "pro_reviews" / "pareto_offline_representation_jinan_paper_final_evidence").exists()
