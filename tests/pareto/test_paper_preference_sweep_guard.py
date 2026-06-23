from __future__ import annotations

import pytest

from pareto.eval.preference_sweep import (
    PAPER_FINAL_PREFERENCE_SWEEP_APPROVAL_PHRASE,
    validate_paper_final_preference_sweep_request,
)


def _request() -> dict:
    return {
        "paper_final_manifest": "configs/formal/paper_final_experiment_manifest_2026-06-02.json",
        "city": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "method": "VectorQ-PPO",
        "seed": 0,
        "preference_id": "balanced",
        "output_root_empty": True,
        "metric_source_policy": "paper_metric_sources_v1",
        "deterministic_policy_loading": True,
        "action_diagnostics_enabled": True,
    }


def test_paper_preference_sweep_rejects_unapproved_non_dry_run():
    with pytest.raises(ValueError, match="requires exact reviewer approval"):
        validate_paper_final_preference_sweep_request(_request(), approval_phrase=None)


def test_paper_preference_sweep_accepts_guarded_request_with_exact_phrase():
    validated = validate_paper_final_preference_sweep_request(
        _request(),
        approval_phrase=PAPER_FINAL_PREFERENCE_SWEEP_APPROVAL_PHRASE,
    )

    assert validated["dry_run"] is False
    assert validated["city"] == "jinan"
    assert validated["preference_id"] == "balanced"


def test_paper_preference_sweep_rejects_missing_action_diagnostics():
    request = _request()
    request["action_diagnostics_enabled"] = False

    with pytest.raises(ValueError, match="action diagnostics"):
        validate_paper_final_preference_sweep_request(
            request,
            approval_phrase=PAPER_FINAL_PREFERENCE_SWEEP_APPROVAL_PHRASE,
        )
