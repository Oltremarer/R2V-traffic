from __future__ import annotations

import pytest

from pareto.rl.paper_c2t_scalar_guard import (
    REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL,
    build_c2t_scalar_readiness,
    validate_c2t_scalar_readiness,
)


def test_c2t_scalar_stays_blocked_without_command_or_exclusion():
    row = build_c2t_scalar_readiness()

    assert row["status"] == "missing_blocker"
    assert "command or exclusion approval missing" in row["blocker"]


def test_c2t_scalar_accepts_non_executing_all_city_command_preview():
    row = build_c2t_scalar_readiness(
        command_preview={
            "command_family": "scalar_reward_training",
            "city_support": ["jinan", "hangzhou", "newyork_28x7"],
            "executes_training_now": False,
            "output_root": "records/paper_final/train_20260602_v1",
        }
    )

    assert row["status"] == "implemented_guarded_preview"
    assert row["executes_training_now"] is False
    validate_c2t_scalar_readiness(row)


def test_c2t_scalar_command_preview_cannot_execute_now():
    with pytest.raises(ValueError, match="must be non-executing"):
        build_c2t_scalar_readiness(
            command_preview={
                "command_family": "scalar_reward_training",
                "city_support": ["jinan", "hangzhou", "newyork_28x7"],
                "executes_training_now": True,
                "output_root": "records/paper_final/train_20260602_v1",
            }
        )


def test_c2t_scalar_reviewer_exclusion_requires_claim_limitation():
    row = build_c2t_scalar_readiness(
        exclusion_approval_phrase=REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL,
        paper_claim_limitation="Paper claims exclude C2T-scalar from final baseline comparisons.",
    )

    assert row["status"] == "excluded_by_reviewer"
    assert row["executes_training_now"] is False
    validate_c2t_scalar_readiness(row)


def test_c2t_scalar_rejects_exclusion_without_exact_phrase():
    with pytest.raises(ValueError, match="exact reviewer exclusion approval"):
        validate_c2t_scalar_readiness(
            {
                "status": "excluded_by_reviewer",
                "approval_phrase": "not enough",
                "paper_claim_limitation": "Claims exclude C2T-scalar.",
                "executes_training_now": False,
            }
        )
