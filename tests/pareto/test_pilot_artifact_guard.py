from __future__ import annotations

from pathlib import Path

import pytest

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts


def test_pilot_artifact_guard_rejects_best_method_and_plots(tmp_path: Path):
    (tmp_path / "best_method.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preference_response_plot.png").write_text("png", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden preflight performance artifacts"):
        assert_no_forbidden_performance_artifacts(tmp_path)


def test_pilot_artifact_guard_rejects_traffic_metrics_and_preference_sweep(tmp_path: Path):
    (tmp_path / "traffic_metrics.csv").write_text("step,reward\n", encoding="utf-8")
    (tmp_path / "preference_sweep.csv").write_text("method,utility\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden preflight performance artifacts"):
        assert_no_forbidden_performance_artifacts(tmp_path)
