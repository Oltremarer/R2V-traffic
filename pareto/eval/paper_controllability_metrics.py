from __future__ import annotations

from typing import Sequence


def compute_monotonicity(weights: Sequence[float], responses: Sequence[float]) -> float:
    if len(weights) < 3 or len(responses) < 3:
        raise ValueError("monotonicity requires at least three sweep points")
    paired = sorted(zip([float(w) for w in weights], [float(r) for r in responses]), key=lambda item: item[0])
    good = sum(int(b[1] >= a[1]) for a, b in zip(paired, paired[1:]))
    return float(good / (len(paired) - 1))


def compute_smoothness(responses: Sequence[float]) -> float:
    values = [float(value) for value in responses]
    if len(values) < 3:
        raise ValueError("smoothness requires at least three sweep points")
    second_diffs = [abs(c - 2.0 * b + a) for a, b, c in zip(values, values[1:], values[2:])]
    return float(sum(second_diffs) / len(second_diffs))


def compute_calibration_error(target: Sequence[float], realized: Sequence[float]) -> float:
    target_values = [float(value) for value in target]
    realized_values = [float(value) for value in realized]
    if len(target_values) != len(realized_values) or not target_values:
        raise ValueError("target and realized preference vectors must have the same nonzero length")
    return float(sum(abs(a - b) for a, b in zip(target_values, realized_values)) / len(target_values))


def compute_heldout_preference_utility(row: dict) -> float:
    if "heldout_preference_utility" not in row:
        raise ValueError("heldout_preference_utility source is required")
    return float(row["heldout_preference_utility"])


def compute_shifted_traffic_att(row: dict) -> float:
    if "shifted_traffic_att" not in row:
        raise ValueError("shifted_traffic_att source is required")
    return float(row["shifted_traffic_att"])
