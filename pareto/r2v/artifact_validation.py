from __future__ import annotations

import math
from typing import Any, Iterable


_MISSING = object()
EXPECTED_WEIGHTED_SCHEMA_VERSION = "r2v-tsc-weighted-transition-v1"
EXPECTED_GATE_KEYS = ("rare", "value", "support", "safety")


def validate_unique_transition_ids(
    rows: Iterable[dict[str, Any]],
    *,
    required: bool = True,
    id_key: str = "transition_id",
) -> dict[str, Any]:
    seen: dict[str, int] = {}
    row_count = 0
    for idx, row in enumerate(rows):
        row_count += 1
        value = row.get(id_key, _MISSING)
        if value is _MISSING or value is None or str(value) == "":
            if required:
                raise ValueError(f"missing {id_key} at row {idx}")
            continue
        key = str(value)
        if key in seen:
            raise ValueError(f"duplicate {id_key} {key!r} at rows {seen[key]} and {idx}")
        seen[key] = idx
    return {
        "row_count": row_count,
        "unique_transition_id_count": len(seen),
    }


def validate_weighted_transition_rows(
    rows: Iterable[dict[str, Any]],
    *,
    join_key: str = "sample_id",
    weight_key: str = "metadata.r2v_sample_weight",
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    validate_unique_transition_ids(materialized)

    seen_join_keys: dict[str, int] = {}
    weights: list[float] = []
    admitted_count = 0
    schema_versions: set[str] = set()
    for idx, row in enumerate(materialized):
        join_value = _get_path(row, join_key)
        if join_value is _MISSING or join_value is None or str(join_value) == "":
            raise ValueError(f"missing join key {join_key} at row {idx}")
        join_id = str(join_value)
        if join_id in seen_join_keys:
            raise ValueError(f"duplicate join key {join_key}={join_id!r} at rows {seen_join_keys[join_id]} and {idx}")
        seen_join_keys[join_id] = idx

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing metadata for weighted R2V row {join_id!r}")
        schema_version = metadata.get("r2v_schema_version")
        if schema_version is None or str(schema_version) == "":
            raise ValueError(f"missing r2v_schema_version for weighted R2V row {join_id!r}")
        if str(schema_version) != EXPECTED_WEIGHTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported r2v_schema_version for weighted R2V row {join_id!r}: {schema_version!r}"
            )
        schema_versions.add(str(schema_version))
        if "r2v_admitted" not in metadata:
            raise ValueError(f"missing r2v_admitted for weighted R2V row {join_id!r}")
        gates = metadata.get("r2v_gates")
        if not isinstance(gates, dict):
            raise ValueError(f"missing r2v_gates for weighted R2V row {join_id!r}")
        missing_gates = [name for name in EXPECTED_GATE_KEYS if name not in gates]
        if missing_gates:
            raise ValueError(f"missing r2v_gates keys for weighted R2V row {join_id!r}: {missing_gates}")

        raw_weight = _get_path(row, weight_key)
        weight = _coerce_positive_finite_weight(raw_weight, join_key=join_key, join_value=join_id)
        weights.append(weight)
        if bool(metadata.get("r2v_admitted", False)):
            admitted_count += 1

    return _weight_summary(
        weights,
        admitted_count=admitted_count,
        join_key=join_key,
        weight_key=weight_key,
        schema_versions=sorted(schema_versions),
    )


def build_r2v_weight_map(
    rows: Iterable[dict[str, Any]],
    *,
    join_key: str = "sample_id",
    weight_key: str = "metadata.r2v_sample_weight",
) -> tuple[dict[str, float], dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    summary = validate_weighted_transition_rows(materialized, join_key=join_key, weight_key=weight_key)
    weight_map: dict[str, float] = {}
    for row in materialized:
        join_id = str(_get_path(row, join_key))
        weight_map[join_id] = _coerce_positive_finite_weight(
            _get_path(row, weight_key),
            join_key=join_key,
            join_value=join_id,
        )
    return weight_map, summary


def get_path_value(row: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value = _get_path(row, dotted_key)
    return default if value is _MISSING else value


def _get_path(row: dict[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _coerce_positive_finite_weight(raw_value: Any, *, join_key: str, join_value: str) -> float:
    try:
        weight = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid r2v weight for {join_key}={join_value!r}: {raw_value!r}") from None
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"invalid r2v weight for {join_key}={join_value!r}: {raw_value!r}")
    return weight


def _weight_summary(
    weights: list[float],
    *,
    admitted_count: int,
    join_key: str,
    weight_key: str,
    schema_versions: list[str],
) -> dict[str, Any]:
    if not weights:
        return {
            "weighted_row_count": 0,
            "admitted_count": 0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "weight_mean": 0.0,
            "join_key": join_key,
            "weight_key": weight_key,
            "schema_versions": schema_versions,
        }
    return {
        "weighted_row_count": len(weights),
        "admitted_count": int(admitted_count),
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "weight_mean": float(sum(weights) / len(weights)),
        "join_key": join_key,
        "weight_key": weight_key,
        "schema_versions": schema_versions,
    }
