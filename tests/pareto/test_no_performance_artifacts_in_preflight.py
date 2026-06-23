from __future__ import annotations

from pathlib import Path

import pytest

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts


def test_preflight_artifact_guard_allows_debug_logs(tmp_path: Path):
    (tmp_path / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reward_components.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "loss_debug.jsonl").write_text("{}\n", encoding="utf-8")

    assert_no_forbidden_performance_artifacts(tmp_path)


def test_preflight_artifact_guard_rejects_performance_tables(tmp_path: Path):
    (tmp_path / "main_results.csv").write_text("method,reward\n", encoding="utf-8")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "performance_table.tex").write_text("\\begin{tabular}{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden preflight performance artifacts"):
        assert_no_forbidden_performance_artifacts(tmp_path)
