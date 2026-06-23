from pareto.data.contrast_mining import (
    mine_efficiency_controlled_pairs,
    mine_reversal_pairs,
)


def test_efficiency_controlled_fairness_pair_keeps_efficiency_close():
    records = [
        {"sample_id": "a", "objective_values_norm": {"efficiency": 1.0, "fairness": 1.0, "safety": 0.0, "stability": 0.0}},
        {"sample_id": "b", "objective_values_norm": {"efficiency": 1.02, "fairness": -1.0, "safety": 0.0, "stability": 0.0}},
    ]
    pairs = mine_efficiency_controlled_pairs(
        records,
        target_objective="fairness",
        n=1,
        eps_efficiency=0.05,
        margin_target=1.0,
    )
    assert len(pairs) == 1
    assert pairs[0]["label"] == 1
    assert pairs[0]["a_id"] == "a"


def test_efficiency_controlled_pair_preserves_negative_labels():
    records = [
        {"sample_id": "a", "objective_values_norm": {"efficiency": 1.0, "fairness": -1.0, "safety": 0.0, "stability": 0.0}},
        {"sample_id": "b", "objective_values_norm": {"efficiency": 1.02, "fairness": 1.0, "safety": 0.0, "stability": 0.0}},
    ]
    pairs = mine_efficiency_controlled_pairs(
        records,
        target_objective="fairness",
        n=1,
        eps_efficiency=0.05,
        margin_target=1.0,
    )

    assert len(pairs) == 1
    assert pairs[0]["a_id"] == "a"
    assert pairs[0]["b_id"] == "b"
    assert pairs[0]["label"] == 0


def test_reversal_mining_finds_opposite_preferences():
    records = [
        {"sample_id": "fast", "objective_values_norm": {"efficiency": 2.0, "safety": -1.0, "fairness": 0.0, "stability": 0.0}},
        {"sample_id": "safe", "objective_values_norm": {"efficiency": -1.0, "safety": 2.0, "fairness": 0.0, "stability": 0.0}},
    ]
    prefs = {
        "efficiency": [0.7, 0.1, 0.1, 0.1],
        "safety": [0.1, 0.7, 0.1, 0.1],
    }
    pairs = mine_reversal_pairs(records, prefs, n=1, min_margin=0.5)
    assert len(pairs) == 1
    assert pairs[0]["label_1"] != pairs[0]["label_2"]
