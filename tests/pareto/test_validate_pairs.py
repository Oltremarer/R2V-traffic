import json
from pathlib import Path

from pareto.data.validate_pairs import main as validate_pairs_main


def test_validate_pairs_strict_gate_exits_on_low_counts(tmp_path: Path, monkeypatch):
    (tmp_path / "objective_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "preference_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dominance_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "reversal_pairs.jsonl").write_text("", encoding="utf-8")
    report = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_pairs.py",
            "--pairs_dir",
            str(tmp_path),
            "--report",
            str(report),
            "--strict",
            "--min_preference_pairs",
            "1",
        ],
    )

    try:
        validate_pairs_main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("strict validation should fail")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["error_count"] == 1
    assert "preference_pairs count" in payload["errors"][0]["error"]


def test_validate_pairs_strict_gate_checks_contrast_split_and_reversal_templates(tmp_path: Path, monkeypatch):
    objective = {
        "pair_id": "obj0",
        "split": "train",
        "scenario": "jinan",
        "a_id": "a",
        "b_id": "b",
        "objective": "fairness",
        "label": 1,
        "is_tie": False,
        "margin_raw": 1.0,
        "margin_norm": 1.0,
        "source": "rule",
        "rule_version": "v1",
        "sampling_strategy": "eff_controlled_fairness",
    }
    preference = {
        "pair_id": "pref0",
        "split": "train",
        "scenario": "jinan",
        "a_id": "a",
        "b_id": "b",
        "w": [0.25, 0.25, 0.25, 0.25],
        "label": 1,
        "is_tie": False,
        "source": "rule",
        "rule_utility_a": 1.0,
        "rule_utility_b": 0.0,
        "rule_margin": 1.0,
        "sampling_strategy": "preference_balanced",
    }
    reversal = {
        "pair_id": "rev0",
        "split": "train",
        "scenario": "jinan",
        "a_id": "a",
        "b_id": "b",
        "w_1_name": "efficiency",
        "w_1": [0.7, 0.1, 0.1, 0.1],
        "label_1": 1,
        "margin_1": 1.0,
        "w_2_name": "fairness",
        "w_2": [0.1, 0.1, 0.7, 0.1],
        "label_2": 0,
        "margin_2": -1.0,
        "sampling_strategy": "reversal",
    }
    (tmp_path / "objective_pairs.jsonl").write_text(json.dumps(objective) + "\n", encoding="utf-8")
    (tmp_path / "preference_pairs.jsonl").write_text(json.dumps(preference) + "\n", encoding="utf-8")
    (tmp_path / "dominance_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "reversal_pairs.jsonl").write_text(json.dumps(reversal) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_pairs.py",
            "--pairs_dir",
            str(tmp_path),
            "--report",
            str(report),
            "--strict",
            "--min_eff_controlled_fairness",
            "2",
            "--min_split_pairs",
            "val:1",
            "--min_reversal_template_pair",
            "efficiency__fairness:2",
        ],
    )

    try:
        validate_pairs_main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("strict validation should fail")

    payload = json.loads(report.read_text(encoding="utf-8"))
    errors = " ".join(error["error"] for error in payload["errors"])
    assert "eff_controlled_fairness" in errors
    assert "val split pair count" in errors
    assert "efficiency__fairness reversal count" in errors


def test_validate_pairs_reports_positive_ratios_by_objective_and_strategy(tmp_path: Path, monkeypatch):
    rows = [
        {
            "pair_id": "obj0",
            "split": "train",
            "scenario": "jinan",
            "a_id": "a",
            "b_id": "b",
            "objective": "safety",
            "label": 1,
            "is_tie": False,
            "margin_raw": 1.0,
            "margin_norm": 1.0,
            "source": "rule",
            "rule_version": "v1",
            "sampling_strategy": "efficiency_safety_conflict",
        },
        {
            "pair_id": "obj1",
            "split": "train",
            "scenario": "jinan",
            "a_id": "c",
            "b_id": "d",
            "objective": "safety",
            "label": 1,
            "is_tie": False,
            "margin_raw": 1.0,
            "margin_norm": 1.0,
            "source": "rule",
            "rule_version": "v1",
            "sampling_strategy": "efficiency_safety_conflict",
        },
    ]
    (tmp_path / "objective_pairs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "preference_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dominance_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "reversal_pairs.jsonl").write_text("", encoding="utf-8")
    report = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_pairs.py",
            "--pairs_dir",
            str(tmp_path),
            "--report",
            str(report),
            "--strict",
            "--positive_ratio_by_objective_low",
            "0.3",
            "--positive_ratio_by_objective_high",
            "0.7",
        ],
    )

    try:
        validate_pairs_main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("strict validation should fail")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["positive_ratio_by_objective"]["safety"] == 1.0
    assert payload["positive_ratio_by_strategy"]["efficiency_safety_conflict"] == 1.0
    assert "safety positive ratio outside bounds" in payload["errors"][0]["error"]


def test_validate_pairs_gates_efficiency_stability_conflict(tmp_path: Path, monkeypatch):
    objective = {
        "pair_id": "obj0",
        "split": "train",
        "scenario": "jinan",
        "a_id": "a",
        "b_id": "b",
        "objective": "stability",
        "label": 1,
        "is_tie": False,
        "margin_raw": 1.0,
        "margin_norm": 1.0,
        "source": "rule",
        "rule_version": "v1",
        "sampling_strategy": "efficiency_stability_conflict",
    }
    (tmp_path / "objective_pairs.jsonl").write_text(json.dumps(objective) + "\n", encoding="utf-8")
    (tmp_path / "preference_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dominance_pairs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "reversal_pairs.jsonl").write_text("", encoding="utf-8")
    report = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_pairs.py",
            "--pairs_dir",
            str(tmp_path),
            "--report",
            str(report),
            "--strict",
            "--min_efficiency_stability_conflict",
            "2",
        ],
    )

    try:
        validate_pairs_main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("strict validation should fail")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["sampling_strategy_counts"]["efficiency_stability_conflict"] == 1
    assert "efficiency_stability_conflict count below minimum" in payload["errors"][0]["error"]
