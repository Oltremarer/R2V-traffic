from pathlib import Path

from pareto.data.normalization import RobustObjectiveNormalizer


def test_robust_objective_normalizer_round_trips(tmp_path: Path):
    records = [
        {"objective_values_raw": {"efficiency": -10.0, "safety": -1.0, "fairness": -0.5, "stability": -0.2}},
        {"objective_values_raw": {"efficiency": -20.0, "safety": -0.5, "fairness": -0.1, "stability": -0.1}},
        {"objective_values_raw": {"efficiency": -30.0, "safety": 0.0, "fairness": 0.0, "stability": 0.0}},
    ]
    normalizer = RobustObjectiveNormalizer.fit(records)
    transformed = normalizer.transform(records[0]["objective_values_raw"])

    assert set(transformed) == {"efficiency", "safety", "fairness", "stability"}
    assert transformed["efficiency"] > transformed["efficiency"] - 1e-9

    out = tmp_path / "norm.json"
    normalizer.save(out)
    restored = RobustObjectiveNormalizer.load(out)
    assert restored.transform(records[0]["objective_values_raw"]) == transformed


def test_robust_objective_normalizer_reports_fit_statistics():
    records = [
        {
            "objective_values_raw": {"efficiency": -10.0, "safety": 0.0, "fairness": -0.5, "stability": -0.2},
            "objective_valid_mask": {"efficiency": True, "safety": True, "fairness": True, "stability": True},
        },
        {
            "objective_values_raw": {"efficiency": -20.0, "safety": 0.0, "fairness": -0.1, "stability": -0.1},
            "objective_valid_mask": {"efficiency": True, "safety": True, "fairness": True, "stability": True},
        },
    ]

    normalizer = RobustObjectiveNormalizer.fit(records, fit_input_files=["buffer.jsonl"])
    payload = normalizer.to_dict()

    assert payload["valid_count"]["efficiency"] == 2
    assert payload["raw_q25"]["efficiency"] < payload["raw_q75"]["efficiency"]
    assert payload["raw_q50"] == payload["median"]
    assert "safety" in payload["zero_iqr_objectives"]
    assert payload["fit_input_files"] == ["buffer.jsonl"]
