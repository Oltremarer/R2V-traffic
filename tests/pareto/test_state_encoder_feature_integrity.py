from __future__ import annotations

import pytest

from pareto.rl.state_encoder import (
    feature_values_hash,
    validate_feature_integrity_sequence,
)


def test_feature_values_hash_is_stable_for_equal_values():
    assert feature_values_hash([1.0, 2.0, 3.0]) == feature_values_hash([1.0, 2.0, 3.0])


def test_feature_integrity_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty observation feature sequence"):
        validate_feature_integrity_sequence([])


def test_feature_integrity_rejects_empty_features():
    with pytest.raises(ValueError, match="empty observation feature vector"):
        validate_feature_integrity_sequence([[]])


def test_feature_integrity_rejects_nonfinite_features():
    with pytest.raises(ValueError, match="non-finite observation feature"):
        validate_feature_integrity_sequence([[1.0, float("nan")]])


def test_feature_integrity_rejects_length_drift():
    with pytest.raises(ValueError, match="feature length drift"):
        validate_feature_integrity_sequence([[1.0, 2.0], [1.0, 2.0, 3.0]])


def test_feature_integrity_rejects_constant_sequence_by_default():
    with pytest.raises(ValueError, match="constant observation feature sequence"):
        validate_feature_integrity_sequence([[1.0, 2.0], [1.0, 2.0]])


def test_feature_integrity_accepts_changing_sequence():
    summary = validate_feature_integrity_sequence([[1.0, 2.0], [1.0, 3.0]])
    assert summary["feature_length"] == 2
    assert summary["first_obs_feature_sha256"] != summary["final_obs_feature_sha256"]
    assert summary["feature_integrity_pass"] is True
