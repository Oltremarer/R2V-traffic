import json
from pathlib import Path

from pareto.eval.offline_model_selection import composite_val_score, select_offline_model


def _write_run(root: Path, name: str, val: dict, test: dict) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": name}), encoding="utf-8")
    (run_dir / "diagnostics_val.json").write_text(json.dumps(val), encoding="utf-8")
    (run_dir / "diagnostics_test.json").write_text(json.dumps(test), encoding="utf-8")
    return run_dir


def test_selects_by_composite_val_score_not_test(tmp_path: Path):
    run_a = _write_run(
        tmp_path,
        "run_a",
        val={"rev_acc": 0.20, "dpr_head": 0.20, "dpr_utility": 0.80, "pref_acc": 0.60},
        test={"rev_acc": 0.95, "dpr_head": 0.95, "dpr_utility": 0.95, "pref_acc": 0.95},
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        val={"rev_acc": 0.60, "dpr_head": 0.70, "dpr_utility": 0.90, "pref_acc": 0.72},
        test={"rev_acc": 0.40, "dpr_head": 0.40, "dpr_utility": 0.40, "pref_acc": 0.40},
    )

    out_path = tmp_path / "selection.json"
    report = select_offline_model([run_a, run_b], out_path)

    assert report["selected_run_id"] == "run_b"
    assert report["selected_by"] == "composite_val_score"
    assert out_path.exists()


def test_composite_score_exposes_components():
    metrics = {
        "rev_acc": 0.60,
        "dpr_head": 0.70,
        "dpr_utility": 0.90,
        "pref_acc": 0.72,
        "obj_acc_mean": 0.68,
        "head_leakage_diag_offdiag_gap": 0.18,
    }
    score, components = composite_val_score(metrics, return_components=True)

    assert 0.0 < score < 1.0
    assert components["rev_acc"]["weight"] == 0.30
    assert components["head_leakage_diag_offdiag_gap"]["value"] == 0.18
