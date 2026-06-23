from __future__ import annotations

import math
from typing import Any


REQUIRED_METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "efficiency": ("average_travel_time", "average_waiting_time", "average_queue_length"),
    "safety": ("ttc_p10", "ttc_p50", "ttc_violation_rate", "harsh_brake_rate"),
    "fairness": ("waiting_time_imbalance",),
    "stability": ("phase_switch_rate", "oscillation_index"),
    "representation": ("obj_acc", "pref_acc", "rev_acc", "dpr"),
    "pareto": ("hypervolume", "coverage", "dominance_violation", "utility", "alignment"),
    "controllability": ("monotonicity", "smoothness", "calibration_error"),
    "generalization": ("heldout_preference_utility", "shifted_traffic_att"),
}
REQUIRED_METRIC_KEYS = tuple(metric for metrics in REQUIRED_METRIC_FAMILIES.values() for metric in metrics)


def validate_metric_value(metric_key: str, value: float | int) -> float:
    if metric_key not in REQUIRED_METRIC_KEYS:
        raise ValueError(f"unknown paper metric: {metric_key}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"non-finite metric {metric_key}: {value}")
    return normalized


def validate_metric_family_schema(schema: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    for family, required_metrics in REQUIRED_METRIC_FAMILIES.items():
        if family not in schema:
            raise ValueError(f"missing metric family: {family}")
        observed = tuple(schema[family])
        missing = sorted(set(required_metrics) - set(observed))
        if missing:
            raise ValueError(f"metric family {family} missing metrics: {missing}")
    return {family: tuple(metrics) for family, metrics in schema.items()}
