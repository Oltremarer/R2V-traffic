from __future__ import annotations

import math

import pytest

from pareto.eval.paper_metric_schema import (
    REQUIRED_METRIC_FAMILIES,
    REQUIRED_METRIC_KEYS,
    validate_metric_family_schema,
    validate_metric_value,
)


def test_metric_schema_contains_paper_required_families_and_keys():
    schema = validate_metric_family_schema(REQUIRED_METRIC_FAMILIES)

    assert "efficiency" in schema
    assert "safety" in schema
    assert "representation" in schema
    assert "hypervolume" in REQUIRED_METRIC_KEYS
    assert "calibration_error" in REQUIRED_METRIC_KEYS


def test_metric_value_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite metric"):
        validate_metric_value("average_travel_time", math.inf)


def test_metric_schema_rejects_missing_required_family():
    bad = dict(REQUIRED_METRIC_FAMILIES)
    bad.pop("safety")

    with pytest.raises(ValueError, match="missing metric family"):
        validate_metric_family_schema(bad)
