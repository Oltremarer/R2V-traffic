from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pareto.eval.pair_margin_diagnostics import diagnose_pair_margins


class IdentityFourHeadModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :4]


def _record(sample_id: str, features: list[float], objectives: dict[str, float]) -> dict:
    return {
        "sample_id": sample_id,
        "obs_features": features,
        "objective_values_norm": objectives,
        "objective_valid_mask": {key: True for key in objectives},
    }


def test_pair_margin_diagnostics_reports_reversal_templates_and_margin_bins():
    records = {
        "a": _record("a", [1.0, 0.0, 0.6, 0.2], {
            "efficiency": 1.0,
            "safety": 0.0,
            "fairness": 0.6,
            "stability": 0.2,
        }),
        "b": _record("b", [0.0, 1.0, 0.1, 0.5], {
            "efficiency": 0.0,
            "safety": 1.0,
            "fairness": 0.1,
            "stability": 0.5,
        }),
    }
    pairs = {
        "objective": [
            {
                "a_id": "a",
                "b_id": "b",
                "objective": "efficiency",
                "label": 1,
                "margin_norm": 1.0,
                "sampling_strategy": "objective_contrast",
            },
            {
                "a_id": "a",
                "b_id": "b",
                "objective": "safety",
                "label": 0,
                "margin_norm": -1.0,
                "sampling_strategy": "efficiency_safety_conflict",
            },
        ],
        "reversal": [
            {
                "a_id": "a",
                "b_id": "b",
                "w_1_name": "efficiency",
                "w_1": [0.7, 0.1, 0.1, 0.1],
                "label_1": 1,
                "margin_1": 0.5,
                "w_2_name": "safety",
                "w_2": [0.1, 0.7, 0.1, 0.1],
                "label_2": 0,
                "margin_2": -0.4,
                "sampling_strategy": "reversal",
            }
        ],
        "dominance": [
            {
                "a_id": "a",
                "b_id": "b",
                "dominates": "a",
                "objective_margins_norm": {
                    "efficiency": 1.0,
                    "safety": 0.1,
                    "fairness": 0.5,
                    "stability": 0.2,
                },
            }
        ],
    }

    report = diagnose_pair_margins(records, pairs, model=IdentityFourHeadModel(), device="cpu")

    assert report["objective_margin_stats"]["efficiency"]["count"] == 1
    assert report["objective_accuracy_by_objective"]["safety"] == 1.0
    assert report["strategy_accuracy"]["efficiency_safety_conflict"] == 1.0
    template = report["reversal_by_template_pair"]["efficiency__safety"]
    assert template["count"] == 1
    assert template["accuracy"] == 1.0
    assert template["correct_both"] == 1
    assert template["same_sign_rate"] == 0.0
    assert report["dominance_violations_by_head"]["safety"] == 1
