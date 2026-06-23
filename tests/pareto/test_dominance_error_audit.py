from __future__ import annotations

import torch

from pareto.eval.dominance_error_audit import dominance_error_audit


class FeatureAsQ(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :4]


def test_dominance_error_audit_reports_head_and_margin_failures():
    records = {
        "dom": {"obs_features": [10.0, -1.0, 2.0, 2.0]},
        "sub": {"obs_features": [1.0, 0.0, 1.0, 1.0]},
    }
    pairs = [
        {
            "a_id": "dom",
            "b_id": "sub",
            "dominates": "a",
            "objective_margins_norm": {
                "efficiency": 1.0,
                "safety": 0.5,
                "fairness": 1.0,
                "stability": 1.0,
            },
        }
    ]

    report = dominance_error_audit(records, pairs, FeatureAsQ())

    assert report["dominance_pairs"] == 1
    assert report["DPR_head"] == 0.0
    assert report["DPR_head_by_objective"]["efficiency"] == 1.0
    assert report["DPR_head_by_objective"]["safety"] == 0.0
    assert report["violation_rate_by_objective"]["safety"] == 1.0
    assert report["utility_pass_head_fail_rate"] == 1.0
    assert report["violation_by_margin_bin"]["0.50-inf"]["head_fail_rate"] == 1.0
