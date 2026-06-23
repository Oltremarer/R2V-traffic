from __future__ import annotations

from pathlib import Path

import pytest

from pareto.data.paper_final_data_roots import (
    build_paper_final_data_root_audit,
    data_root_blockers,
    validate_paper_final_data_root_audit,
)


def _populate_city(root: Path, city: str) -> None:
    records = root / "data" / "pareto_records_split_norm" / city / "paper_final"
    pairs = root / "data" / "pareto_pairs" / city / "paper_final"
    records.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (records / f"{split}_raw.jsonl").write_text("", encoding="utf-8")
        split_dir = pairs / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in ("objective_pairs", "preference_pairs", "dominance_pairs", "reversal_pairs"):
            (split_dir / f"{name}.jsonl").write_text("", encoding="utf-8")


def test_data_root_audit_reports_missing_all_city_roots(tmp_path: Path):
    audit = build_paper_final_data_root_audit(tmp_path)
    blockers = data_root_blockers(audit)

    assert audit["status"] == "missing_blocker"
    assert any("jinan" in blocker for blocker in blockers)
    assert any("hangzhou" in blocker for blocker in blockers)
    assert any("newyork_28x7" in blocker for blocker in blockers)


def test_data_root_audit_passes_when_all_required_split_files_exist(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _populate_city(tmp_path, city)

    audit = build_paper_final_data_root_audit(tmp_path)

    assert audit["status"] == "complete"
    assert data_root_blockers(audit) == []
    validate_paper_final_data_root_audit(audit)


def test_data_root_audit_rejects_roots_outside_expected_prefix(tmp_path: Path):
    audit = build_paper_final_data_root_audit(tmp_path)
    audit["rows"][0] = dict(audit["rows"][0], records_root="records/paper_final/bad")

    with pytest.raises(ValueError, match="pareto_records_split_norm"):
        validate_paper_final_data_root_audit(audit)
