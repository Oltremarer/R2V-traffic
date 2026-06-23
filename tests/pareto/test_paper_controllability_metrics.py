from __future__ import annotations

import pytest

from pareto.eval.paper_controllability_metrics import (
    compute_calibration_error,
    compute_heldout_preference_utility,
    compute_monotonicity,
    compute_shifted_traffic_att,
    compute_smoothness,
)


def test_controllability_metrics_compute_from_preference_response_trace():
    weights = [0.0, 0.5, 1.0]
    responses = [1.0, 1.5, 2.0]

    assert compute_monotonicity(weights, responses) == 1.0
    assert compute_smoothness(responses) == 0.0
    assert compute_calibration_error([0.25, 0.75], [0.20, 0.70]) == pytest.approx(0.05)


def test_controllability_metrics_reject_short_sweep():
    with pytest.raises(ValueError, match="three sweep points"):
        compute_monotonicity([0.0, 1.0], [0.0, 1.0])


def test_generalization_metrics_require_declared_sources():
    assert compute_heldout_preference_utility({"heldout_preference_utility": 0.4}) == 0.4
    assert compute_shifted_traffic_att({"shifted_traffic_att": 22.0}) == 22.0

    with pytest.raises(ValueError, match="heldout_preference_utility"):
        compute_heldout_preference_utility({})
