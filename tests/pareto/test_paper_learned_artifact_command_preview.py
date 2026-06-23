from __future__ import annotations

from pathlib import Path

import pytest

from pareto.data.paper_final_data_roots import build_paper_final_data_root_audit
from pareto.rl.paper_learned_artifact_command_preview import (
    build_learned_artifact_command_preview,
    learned_artifact_command_blockers,
    validate_learned_artifact_command_preview,
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


def test_learned_artifact_command_preview_blocks_when_data_roots_missing(tmp_path: Path):
    data_audit = build_paper_final_data_root_audit(tmp_path)
    preview = build_learned_artifact_command_preview(data_audit=data_audit, run_id="paper_final_20260602_v1")

    assert preview["status"] == "missing_blocker"
    assert any("data roots incomplete" in blocker for blocker in learned_artifact_command_blockers(preview))


def test_learned_artifact_command_preview_builds_non_executing_commands_when_data_roots_complete(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _populate_city(tmp_path, city)
    data_audit = build_paper_final_data_root_audit(tmp_path)

    preview = build_learned_artifact_command_preview(data_audit=data_audit, run_id="paper_final_20260602_v1")

    assert preview["status"] == "ready_request"
    assert len(preview["rows"]) == 6
    assert all(row["executes_training_now"] is False for row in preview["rows"])
    assert all(row["model_hash"] is None for row in preview["rows"])
    assert any("train_conditioned_scalar.py" in row["command_preview"] for row in preview["rows"])
    assert any("train_vector_quality.py" in row["command_preview"] for row in preview["rows"])
    validate_learned_artifact_command_preview(preview)


def test_learned_artifact_command_preview_uses_paper_scale_hidden_dim_and_normalizer_copy(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _populate_city(tmp_path, city)
    preview = build_learned_artifact_command_preview(
        data_audit=build_paper_final_data_root_audit(tmp_path),
        run_id="paper_final_20260602_v1",
    )

    for row in preview["rows"]:
        command = row["command_preview"]
        normalizer_source = f"data/normalizers/{row['city']}/objective_norm_paper_final.json"
        normalizer_target = f"{row['output_dir']}/objective_normalizer.json"
        assert "--hidden_dim 256" in command
        assert row["objective_normalizer_source_path"] == normalizer_source
        assert f"cp {normalizer_source} {normalizer_target}" in command


def test_learned_artifact_command_preview_rejects_hash_claims(tmp_path: Path):
    for city in ("jinan", "hangzhou", "newyork_28x7"):
        _populate_city(tmp_path, city)
    preview = build_learned_artifact_command_preview(
        data_audit=build_paper_final_data_root_audit(tmp_path),
        run_id="paper_final_20260602_v1",
    )
    preview["rows"][0] = dict(preview["rows"][0], model_hash="a" * 64)

    with pytest.raises(ValueError, match="must not claim model_hash"):
        validate_learned_artifact_command_preview(preview)
