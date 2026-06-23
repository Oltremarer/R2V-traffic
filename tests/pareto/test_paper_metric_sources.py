from __future__ import annotations

import pytest

from pareto.eval.paper_metric_sources import (
    METRIC_SOURCE_POLICY,
    build_metric_source_policy,
    metric_source_blockers,
    validate_metric_source,
    validate_metric_source_policy,
)
from pareto.eval.paper_representation_artifact_sources import (
    REQUIRED_REPRESENTATION_ARTIFACT_CITIES,
    REQUIRED_REPRESENTATION_METRIC_KEYS,
    REQUIRED_REPRESENTATION_MODEL_FAMILIES,
)


def test_metric_source_policy_reports_remaining_representation_blockers():
    policy = validate_metric_source_policy(METRIC_SOURCE_POLICY)
    blockers = metric_source_blockers(policy)

    assert any("obj_acc" in item for item in blockers)
    assert any("dpr" in item for item in blockers)


def test_metric_source_rejects_train_metrics_for_eval_metric():
    with pytest.raises(ValueError, match="not an approved source"):
        validate_metric_source("average_travel_time", "train_metrics.jsonl", 12.0)


def test_metric_source_accepts_common_eval_metric_from_eval_metrics_json():
    assert validate_metric_source("average_travel_time", "eval_metrics.json", 12.0) == 12.0


def test_metric_source_accepts_ttc_metric_from_eval_metrics_json():
    assert validate_metric_source("ttc_p10", "eval_metrics.json", 3.5) == 3.5


def test_metric_source_rejects_representation_blocker_metric_even_with_numeric_value():
    with pytest.raises(ValueError, match="source is not implemented"):
        validate_metric_source("obj_acc", "representation_diagnostics.json", 0.9)


def test_metric_source_accepts_representation_packet_only_after_complete_artifact_audit():
    audit = {
        "rows": [
            {
                "city": city,
                "model_family": model_family,
                "status": "implemented_guarded_preview",
                "packet_path": f"docs/pro_reviews/{city}/{model_family}/representation_formal_gate_packet.json",
                "packet_hash": "a" * 64,
                "metrics": list(REQUIRED_REPRESENTATION_METRIC_KEYS),
                "packet_keys": {
                    "obj_acc": ["obj_acc_mean"],
                    "pref_acc": ["pref_acc"],
                    "rev_acc": ["rev_acc"],
                    "dpr": ["dpr_head", "dpr_utility"],
                },
                "executes_training_now": False,
                "reads_final_traffic_result_values": False,
                "paper_result_claim": False,
            }
            for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES
            for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES
        ]
    }
    policy = build_metric_source_policy(representation_artifact_audit=audit)

    assert validate_metric_source("obj_acc", "representation_formal_gate_packet.json", 0.9, policy=policy) == 0.9
    with pytest.raises(ValueError, match="not an approved source"):
        validate_metric_source("obj_acc", "eval_metrics.json", 0.9, policy=policy)
