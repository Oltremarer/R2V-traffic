from __future__ import annotations

import pytest

from pareto.eval.paper_representation_diagnostics import (
    REPRESENTATION_ABLATIONS,
    REQUIRED_REPRESENTATION_METRICS,
    representation_diagnostic_blockers,
    validate_representation_diagnostic_plan,
)


def _plan() -> dict:
    return {
        "metrics": {metric: {"status": "implemented", "source": "diagnostic_raw"} for metric in REQUIRED_REPRESENTATION_METRICS},
        "ablations": {
            ablation: {
                "status": "implemented",
                "checkpoint_source": f"model_weights/{ablation}/model.pt",
                "training_command": f"train_{ablation}",
            }
            for ablation in REPRESENTATION_ABLATIONS
        },
        "outputs": {
            "raw_only": True,
            "paper_table_generated": False,
            "ranking_generated": False,
        },
    }


def test_representation_plan_accepts_complete_raw_only_diagnostics():
    plan = validate_representation_diagnostic_plan(_plan())

    assert representation_diagnostic_blockers(plan) == []
    assert set(plan["metrics"]) == set(REQUIRED_REPRESENTATION_METRICS)


def test_representation_plan_reports_missing_ablation_blockers():
    plan = _plan()
    plan["ablations"]["without_l_dom"] = {
        "status": "missing_blocker",
        "blocker": "no ablation checkpoint",
    }
    validated = validate_representation_diagnostic_plan(plan)

    assert any("without_l_dom" in item for item in representation_diagnostic_blockers(validated))


def test_representation_plan_rejects_paper_table_generation():
    plan = _plan()
    plan["outputs"]["paper_table_generated"] = True

    with pytest.raises(ValueError, match="paper table"):
        validate_representation_diagnostic_plan(plan)
