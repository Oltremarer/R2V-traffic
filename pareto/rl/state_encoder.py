from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Tuple

import numpy as np


LLMLIGHT_FEATURE_SCHEMA_VERSION = "llmlight-feature-schema-v2"
LLM_MOVEMENT_KEYS = ("ET", "WT", "NT", "ST", "EL", "WL", "NL", "SL")
LLM_INCOMING_KEYS = ("E", "W", "N", "S")
LLM_CELL_COUNT = 4

LLMLIGHT_FEATURE_SPECS: tuple[tuple[str, int], ...] = (
    ("cur_phase", 1),
    ("time_this_phase", 1),
    ("lane_num_vehicle", 12),
    ("lane_num_vehicle_downstream", 12),
    ("lane_num_waiting_vehicle_in", 12),
    ("lane_num_waiting_vehicle_out", 12),
    ("pressure", 24),
    ("traffic_movement_pressure_queue", 12),
    ("traffic_movement_pressure_queue_efficient", 12),
    ("traffic_movement_pressure_num", 12),
    ("lane_enter_running_part", 12),
)


def _flatten_value(prefix: str, value: Any, expected_len: int | None = None) -> Tuple[list[float], list[str], Dict[str, Any]]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
    else:
        values = [float(value)]
    original_len = len(values)
    if expected_len is not None:
        if len(values) < expected_len:
            values = values + [0.0] * (expected_len - len(values))
        elif len(values) > expected_len:
            values = values[:expected_len]
    names = [f"{prefix}_{idx}" for idx in range(len(values))]
    return values, names, {
        "original_len": original_len,
        "encoded_len": len(values),
        "padded": expected_len is not None and original_len < expected_len,
        "truncated": expected_len is not None and original_len > expected_len,
    }


def _feature_hash(names: Iterable[str]) -> str:
    joined = "\n".join(names).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def feature_values_hash(values: Iterable[float]) -> str:
    payload = json.dumps([float(value) for value in values], separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_feature_integrity_sequence(
    feature_rows: Iterable[Iterable[float]],
    *,
    allow_constant: bool = False,
) -> dict[str, Any]:
    rows = [[float(value) for value in row] for row in feature_rows]
    if not rows:
        raise ValueError("empty observation feature sequence")
    if any(not row for row in rows):
        raise ValueError("empty observation feature vector")
    feature_length = len(rows[0])
    hashes: list[str] = []
    for idx, row in enumerate(rows):
        if len(row) != feature_length:
            raise ValueError(f"feature length drift at row {idx}")
        if any(not math.isfinite(value) for value in row):
            raise ValueError(f"non-finite observation feature at row {idx}")
        hashes.append(feature_values_hash(row))
    unique_hashes = sorted(set(hashes))
    if len(rows) > 1 and len(unique_hashes) == 1 and not allow_constant:
        raise ValueError("constant observation feature sequence")
    return {
        "feature_integrity_pass": True,
        "feature_row_count": len(rows),
        "feature_length": feature_length,
        "first_obs_feature_sha256": hashes[0],
        "final_obs_feature_sha256": hashes[-1],
        "unique_feature_hash_count": len(unique_hashes),
    }


class ParetoStateEncoder:
    def __init__(self, encoder_id: str = "hybrid_v1"):
        if encoder_id not in {"llm_abstraction", "llmlight_feature", "hybrid_v1"}:
            raise ValueError(f"unknown encoder_id: {encoder_id}")
        self.encoder_id = encoder_id

    def encode_snapshot(self, snapshot) -> tuple[np.ndarray, list[str], Dict[str, Any]]:
        values: list[float] = []
        names: list[str] = []
        self._last_missing_feature_keys = []
        self._last_padded_feature_keys = []
        self._last_truncated_feature_keys = []

        if self.encoder_id in {"llm_abstraction", "hybrid_v1"}:
            llm_values, llm_names = self._encode_llm_abstraction(snapshot)
            values.extend(llm_values)
            names.extend(llm_names)

        if self.encoder_id in {"llmlight_feature", "hybrid_v1"}:
            light_values, light_names = self._encode_llmlight_feature(snapshot)
            values.extend(light_values)
            names.extend(light_names)

        features = np.asarray(values, dtype=np.float32)
        feature_names_hash = _feature_hash(names)
        return features, names, {
            "encoder_id": self.encoder_id,
            "encoder_schema_version": LLMLIGHT_FEATURE_SCHEMA_VERSION,
            "feature_count": len(names),
            "feature_names_hash": feature_names_hash,
            "missing_feature_keys": getattr(self, "_last_missing_feature_keys", []),
            "padded_feature_keys": getattr(self, "_last_padded_feature_keys", []),
            "truncated_feature_keys": getattr(self, "_last_truncated_feature_keys", []),
        }

    def _encode_llm_abstraction(self, snapshot) -> tuple[list[float], list[str]]:
        values = [
            float(snapshot.current_phase),
            float(snapshot.time_this_phase),
            float(snapshot.mean_speed),
        ]
        names = ["llm_current_phase", "llm_time_this_phase", "llm_mean_speed"]

        for lane in LLM_MOVEMENT_KEYS:
            item = snapshot.state_detail.get(lane, {})
            values.append(float(item.get("queue_len", 0.0)))
            names.append(f"llm_{lane}_queue_len")
            values.append(float(item.get("avg_wait_time", 0.0)))
            names.append(f"llm_{lane}_avg_wait_time")
            cells = list(item.get("cells", []))[:LLM_CELL_COUNT]
            cells += [0.0] * (LLM_CELL_COUNT - len(cells))
            for idx, cell in enumerate(cells):
                values.append(float(cell))
                names.append(f"llm_{lane}_cell_{idx}")

        for lane in LLM_INCOMING_KEYS:
            item = snapshot.state_incoming.get(lane, {})
            values.append(float(item.get("queue_len", 0.0)))
            names.append(f"llm_in_{lane}_queue_len")
            cells = list(item.get("cells", []))[:LLM_CELL_COUNT]
            cells += [0.0] * (LLM_CELL_COUNT - len(cells))
            for idx, cell in enumerate(cells):
                values.append(float(cell))
                names.append(f"llm_in_{lane}_cell_{idx}")
        return values, names

    def _encode_llmlight_feature(self, snapshot) -> tuple[list[float], list[str]]:
        values: list[float] = []
        names: list[str] = []
        missing: list[str] = []
        padded: list[str] = []
        truncated: list[str] = []
        for key, expected_len in LLMLIGHT_FEATURE_SPECS:
            if key not in snapshot.dic_feature:
                missing.append(key)
                flat_values = [0.0] * expected_len
                flat_names = [f"llmlight_{key}_{idx}" for idx in range(expected_len)]
            else:
                flat_values, flat_names, info = _flatten_value(
                    f"llmlight_{key}",
                    snapshot.dic_feature[key],
                    expected_len=expected_len,
                )
                if info["padded"]:
                    padded.append(key)
                if info["truncated"]:
                    truncated.append(key)
            values.extend(flat_values)
            names.extend(flat_names)
        self._last_missing_feature_keys = missing
        self._last_padded_feature_keys = padded
        self._last_truncated_feature_keys = truncated
        return values, names
