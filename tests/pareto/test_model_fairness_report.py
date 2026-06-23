import json
from pathlib import Path

from pareto.eval.model_fairness_report import build_model_fairness_report, run as run_fairness_report


def test_model_fairness_report_marks_large_param_gap():
    report = build_model_fairness_report(
        {"param_count": 100, "epochs": 20, "batch_size": 128, "lr": 1e-3},
        {"param_count": 200, "epochs": 20, "batch_size": 128, "lr": 1e-3},
        max_param_gap=0.30,
    )

    assert report["param_gap_status"] == "warn"
    assert report["budget_comparison"]["epochs"]["match"] is True


def test_model_fairness_report_cli_reads_metadata(tmp_path: Path):
    vector_dir = tmp_path / "vector"
    scalar_dir = tmp_path / "scalar"
    vector_dir.mkdir()
    scalar_dir.mkdir()
    (vector_dir / "metadata.json").write_text(json.dumps({"param_count": 100, "epochs": 20}), encoding="utf-8")
    (scalar_dir / "metadata.json").write_text(json.dumps({"param_count": 110, "epochs": 20}), encoding="utf-8")
    out = tmp_path / "fairness.json"

    report = run_fairness_report(vector_dir, scalar_dir, out)

    assert out.exists()
    assert report["param_gap_status"] == "pass"
