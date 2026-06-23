from __future__ import annotations

import math

import pytest

from pareto.r2v.artifact_validation import (
    build_r2v_weight_map,
    validate_unique_transition_ids,
    validate_weighted_transition_rows,
)


def _row(sample_id: str, transition_id: str, weight: float = 1.0) -> dict:
    return {
        "sample_id": sample_id,
        "transition_id": transition_id,
        "metadata": {
            "r2v_schema_version": "r2v-tsc-weighted-transition-v1",
            "r2v_sample_weight": weight,
            "r2v_admitted": weight > 1.0,
            "r2v_gates": {
                "rare": weight > 1.0,
                "value": weight > 1.0,
                "support": True,
                "safety": True,
            },
        },
    }


def test_validate_unique_transition_ids_rejects_missing_transition_id():
    rows = [_row("s0", "t0"), {"sample_id": "s1", "metadata": {}}]

    with pytest.raises(ValueError, match="missing transition_id"):
        validate_unique_transition_ids(rows)


def test_validate_unique_transition_ids_rejects_duplicate_transition_id():
    rows = [_row("s0", "dup"), _row("s1", "dup")]

    with pytest.raises(ValueError, match="duplicate transition_id"):
        validate_unique_transition_ids(rows)


def test_validate_weighted_transition_rows_rejects_missing_join_key():
    rows = [{"transition_id": "t0", "metadata": {"r2v_sample_weight": 1.0}}]

    with pytest.raises(ValueError, match="missing join key sample_id"):
        validate_weighted_transition_rows(rows, join_key="sample_id")


def test_validate_weighted_transition_rows_rejects_duplicate_join_key():
    rows = [_row("same", "t0", 1.0), _row("same", "t1", 2.0)]

    with pytest.raises(ValueError, match="duplicate join key sample_id"):
        validate_weighted_transition_rows(rows, join_key="sample_id")


def test_validate_weighted_transition_rows_rejects_missing_metadata():
    rows = [{"sample_id": "s0", "transition_id": "t0"}]

    with pytest.raises(ValueError, match="missing metadata"):
        validate_weighted_transition_rows(rows)


def test_validate_weighted_transition_rows_rejects_wrong_schema_version():
    rows = [_row("s0", "t0", 1.0)]
    rows[0]["metadata"]["r2v_schema_version"] = "wrong-schema"

    with pytest.raises(ValueError, match="unsupported r2v_schema_version"):
        validate_weighted_transition_rows(rows)


def test_validate_weighted_transition_rows_rejects_missing_admitted_flag():
    rows = [_row("s0", "t0", 1.0)]
    rows[0]["metadata"].pop("r2v_admitted")

    with pytest.raises(ValueError, match="missing r2v_admitted"):
        validate_weighted_transition_rows(rows)


def test_validate_weighted_transition_rows_rejects_missing_gate_metadata():
    rows = [_row("s0", "t0", 1.0)]
    rows[0]["metadata"].pop("r2v_gates", None)

    with pytest.raises(ValueError, match="missing r2v_gates"):
        validate_weighted_transition_rows(rows)


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, math.inf, math.nan])
def test_validate_weighted_transition_rows_rejects_invalid_weights(bad_weight: float):
    rows = [_row("s0", "t0", bad_weight)]

    with pytest.raises(ValueError, match="invalid r2v weight"):
        validate_weighted_transition_rows(rows)


def test_build_r2v_weight_map_uses_join_key_and_reports_summary():
    rows = [_row("s0", "t0", 1.0), _row("s1", "t1", 3.5)]

    weight_map, summary = build_r2v_weight_map(rows, join_key="sample_id")

    assert weight_map == {"s0": 1.0, "s1": 3.5}
    assert summary["weighted_row_count"] == 2
    assert summary["admitted_count"] == 1
    assert summary["weight_min"] == 1.0
    assert summary["weight_max"] == 3.5
    assert summary["weight_mean"] == 2.25
