from __future__ import annotations

import pytest

from pareto.eval.paper_pareto_metrics import (
    compute_alignment,
    compute_coverage,
    compute_dominance_violation,
    compute_hypervolume_2d,
    compute_utility,
)


def test_pareto_metrics_compute_source_guarded_values():
    points = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

    assert compute_hypervolume_2d(points, reference=(0.0, 0.0)) > 0.0
    assert compute_coverage(points, reference_points=[(0.2, 0.8), (0.8, 0.2)]) == 1.0
    assert compute_dominance_violation([(1.0, 1.0), (0.5, 0.5)]) == 1
    assert compute_utility((0.5, 0.5), (0.25, 0.75)) == 0.5
    assert compute_alignment((0.25, 0.75), (0.25, 0.75)) == 1.0


def test_pareto_metrics_reject_empty_point_set():
    with pytest.raises(ValueError, match="point set"):
        compute_hypervolume_2d([], reference=(0.0, 0.0))
