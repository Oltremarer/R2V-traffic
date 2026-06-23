from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pareto.rl.paper_learned_artifact_inventory import (
    REQUIRED_LEARNED_ARTIFACT_BASELINES,
    inventory_learned_artifacts,
    learned_artifact_blockers,
    validate_learned_artifact_row,
)


def _write(path: Path, text: str = "artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_normalizer(path: Path, normalizer_hash: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hash": normalizer_hash, "version": "test_normalizer"}
    text = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _populate_all_city_artifacts(root: Path) -> None:
    for family in ("cond_scalar", "pareto_quality"):
        for city in ("jinan", "hangzhou", "newyork_28x7"):
            run_dir = root / "model_weights" / family / city / "paper_final" / "run_a"
            _write(run_dir / "model.pt", f"{family}-{city}-model")
            _write_normalizer(run_dir / "objective_normalizer.json", f"{family}_{city}_normalizer_hash")


def test_inventory_reports_missing_hangzhou_newyork_as_blockers(tmp_path: Path):
    run_dir = tmp_path / "model_weights" / "cond_scalar" / "jinan" / "paper_final" / "run_a"
    _write(run_dir / "model.pt", "jinan cond scalar")
    _write_normalizer(run_dir / "objective_normalizer.json", "jinan_normalizer_hash")

    audit = inventory_learned_artifacts(tmp_path)
    blockers = learned_artifact_blockers(audit)

    assert set(audit["required_baselines"]) == set(REQUIRED_LEARNED_ARTIFACT_BASELINES)
    assert any("Cond-Scalar-RL hangzhou" in blocker for blocker in blockers)
    assert any("Cond-Scalar-RL newyork_28x7" in blocker for blocker in blockers)
    assert any("VectorQ-PPO jinan" in blocker for blocker in blockers)


def test_inventory_requires_objective_normalizer_hash(tmp_path: Path):
    run_dir = tmp_path / "model_weights" / "pareto_quality" / "jinan" / "paper_final" / "run_a"
    _write(run_dir / "model.pt", "vector model")

    audit = inventory_learned_artifacts(tmp_path)
    blockers = learned_artifact_blockers(audit)

    assert any("VectorQ-PPO jinan" in blocker and "objective normalizer hash missing" in blocker for blocker in blockers)


def test_inventory_hashes_all_city_model_and_normalizer_artifacts(tmp_path: Path):
    _populate_all_city_artifacts(tmp_path)

    audit = inventory_learned_artifacts(tmp_path)

    assert learned_artifact_blockers(audit) == []
    for row in audit["rows"]:
        assert row["status"] == "implemented_guarded_preview"
        assert len(row["model_hash"]) == 64
        assert row["objective_normalizer_hash"].endswith("_normalizer_hash")
        assert len(row["objective_normalizer_file_sha256"]) == 64
        assert row["executes_training_now"] is False
        validate_learned_artifact_row(row)


def test_inventory_distinguishes_normalizer_internal_hash_from_file_sha256(tmp_path: Path):
    run_dir = tmp_path / "model_weights" / "cond_scalar" / "jinan" / "paper_final" / "run_a"
    _write(run_dir / "model.pt", "jinan cond scalar")
    file_sha256 = _write_normalizer(run_dir / "objective_normalizer.json", "internal_norm_hash")

    audit = inventory_learned_artifacts(tmp_path)
    row = next(item for item in audit["rows"] if item["baseline"] == "Cond-Scalar-RL" and item["city"] == "jinan")

    assert row["objective_normalizer_hash"] == "internal_norm_hash"
    assert row["objective_normalizer_file_sha256"] == file_sha256
    assert row["objective_normalizer_hash"] != row["objective_normalizer_file_sha256"]


def test_inventory_rejects_paths_outside_paper_final_root():
    row = {
        "baseline": "VectorQ-PPO",
        "city": "jinan",
        "status": "implemented_guarded_preview",
        "model_path": "model_weights/pareto_quality/jinan/preformal_final/run_a/model.pt",
        "model_hash": "a" * 64,
        "objective_normalizer_path": "model_weights/pareto_quality/jinan/preformal_final/run_a/objective_normalizer.json",
        "objective_normalizer_hash": "normalizer_hash",
        "objective_normalizer_file_sha256": "b" * 64,
        "executes_training_now": False,
    }

    with pytest.raises(ValueError, match="paper_final"):
        validate_learned_artifact_row(row)
