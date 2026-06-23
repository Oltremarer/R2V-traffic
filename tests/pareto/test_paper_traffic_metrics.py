from __future__ import annotations

import pytest

from pareto.eval.paper_traffic_metrics import (
    compute_harsh_brake_rate,
    compute_paper_traffic_metrics,
    compute_ttc_metrics,
)


def _trace() -> dict:
    return {
        "completed_travel_times": [10.0, 20.0, 30.0],
        "waiting_times": [1.0, 2.0, 3.0, 4.0],
        "queue_lengths": [2.0, 4.0, 6.0],
        "ttc_values": [1.0, 2.0, 4.0, 8.0],
        "speed_history": [[10.0, 7.0, 6.0], [5.0, 5.0, 1.0]],
        "phase_sequence": [0, 0, 1, 1, 2, 1],
        "queue_volatility": [1.0, 3.0, 2.0, 6.0],
    }


def test_paper_traffic_metrics_compute_required_fields():
    metrics = compute_paper_traffic_metrics(_trace())

    assert metrics["average_travel_time"] == 20.0
    assert metrics["average_waiting_time"] == 2.5
    assert metrics["average_queue_length"] == 4.0
    assert metrics["ttc_violation_rate"] == 0.5
    assert metrics["phase_switch_rate"] > 0.0
    assert metrics["oscillation_index"] > 0.0


def test_ttc_metrics_reject_empty_pairs():
    with pytest.raises(ValueError, match="TTC values"):
        compute_ttc_metrics([])


def test_harsh_brake_rate_uses_speed_deltas():
    assert compute_harsh_brake_rate([[10.0, 7.0], [5.0, 4.5]], threshold=2.0) == 0.5


def test_paper_traffic_metrics_reject_missing_waiting_times():
    trace = _trace()
    trace["waiting_times"] = []

    with pytest.raises(ValueError, match="waiting-time observations"):
        compute_paper_traffic_metrics(trace)
