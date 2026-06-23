#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_jsonl, write_json
from pareto.constants import OBJECTIVE_NAMES
from pareto.data.schema import DominancePairLabel, ObjectivePairLabel, PreferencePairLabel
from pareto.r2v.artifact_validation import build_r2v_weight_map, get_path_value, validate_weighted_transition_rows
from pareto.r2v.traffic_artifact_schema import validate_r2v_traffic_artifact


PREFERENCE_TEMPLATES = {
    "efficiency": [0.7, 0.1, 0.1, 0.1],
    "safety": [0.1, 0.7, 0.1, 0.1],
    "fairness": [0.1, 0.1, 0.7, 0.1],
    "stability": [0.1, 0.1, 0.1, 0.7],
    "balanced": [0.25, 0.25, 0.25, 0.25],
}

MAX_ENUMERATED_CANDIDATES = 10_000
DEFAULT_CANDIDATE_FLOOR = 10_000
DEFAULT_CANDIDATE_CEILING = 750_000
SUPPORTED_R2V_SAMPLING_MODES = {
    "off",
    "full_r2v",
    "admitted_only",
    "rare_only",
    "value_only",
    "random_same_count",
    "same_candidates_random_weights",
    "shuffled_value",
    "inverted_rarity",
}
SUPPORTED_ER_BASELINE_MODES = {
    "uniform",
    "recent",
    "pressure_priority",
    "reward_priority",
    "td_error_priority",
    "diversity",
    "intersection_balanced",
    "phase_action_balanced",
}
SUPPORTED_ER_R2V_COMBINE_MODES = {"multiply", "replace"}


def load_records(paths: Iterable[str | Path]) -> List[Dict]:
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _obj(record: Dict, objective: str) -> float:
    return float(record["objective_values_norm"][objective])


def _valid(record: Dict, objective: str) -> bool:
    return bool(record.get("objective_valid_mask", {}).get(objective, False))


def _utility(record: Dict, w: List[float]) -> float:
    return sum(weight * _obj(record, name) for weight, name in zip(w, OBJECTIVE_NAMES))


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    if path.exists():
        path.unlink()
    if rows:
        append_jsonl(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _maybe_flip(a: Dict, b: Dict, label: int, rng: random.Random):
    if rng.random() < 0.5:
        return b, a, 1 - label
    return a, b, label


def _total_candidate_pairs(record_count: int) -> int:
    if record_count < 2:
        return 0
    return record_count * (record_count - 1) // 2


def _candidate_budget(
    record_count: int,
    quota: int,
    *,
    multiplier: int = 1000,
    floor: int = DEFAULT_CANDIDATE_FLOOR,
    ceiling: int = DEFAULT_CANDIDATE_CEILING,
) -> int:
    total = _total_candidate_pairs(record_count)
    if total == 0 or quota <= 0:
        return 0
    desired = max(int(floor), int(quota) * int(multiplier))
    return min(total, desired, int(ceiling))


def _iter_candidate_pairs(
    records: List[Dict],
    rng: random.Random,
    max_checks: int,
) -> Iterator[tuple[Dict, Dict]]:
    record_count = len(records)
    total = _total_candidate_pairs(record_count)
    if total == 0 or max_checks <= 0:
        return
    if total <= min(max_checks, MAX_ENUMERATED_CANDIDATES):
        indices = [
            (left, right)
            for left in range(record_count - 1)
            for right in range(left + 1, record_count)
        ]
        rng.shuffle(indices)
        for left, right in indices:
            yield records[left], records[right]
        return

    seen: set[int] = set()
    while len(seen) < max_checks:
        left = rng.randrange(record_count)
        right = rng.randrange(record_count - 1)
        if right >= left:
            right += 1
        if left > right:
            left, right = right, left
        key = left * record_count + right
        if key in seen:
            continue
        seen.add(key)
        yield records[left], records[right]


def _has_r2v_sampling_weights(records: List[Dict]) -> bool:
    return any("_r2v_sampling_weight" in record for record in records)


def _strip_internal_sampling_fields(record: Dict) -> Dict:
    copied = dict(record)
    copied.pop("_er_sampling_weight", None)
    copied.pop("_r2v_sampling_weight", None)
    copied.pop("_pair_sampling_weight", None)
    return copied


def _weighted_candidate_budget(
    quota: int,
    *,
    multiplier: int,
    floor: int,
    ceiling: int,
) -> int:
    if quota <= 0:
        return 0
    desired = max(int(floor), int(quota) * int(multiplier))
    return min(desired, int(ceiling))


def _iter_weighted_candidate_pairs(
    records: List[Dict],
    rng: random.Random,
    max_checks: int,
) -> Iterator[tuple[Dict, Dict]]:
    if len(records) < 2 or max_checks <= 0:
        return
    weights = [max(0.0, float(record.get("_r2v_sampling_weight", 1.0))) for record in records]
    positive_indices = [idx for idx, weight in enumerate(weights) if weight > 0.0]
    if len(positive_indices) < 2:
        raise ValueError("R2V weighted sampling requires at least two positive-weight records")
    positive_weights = [weights[idx] for idx in positive_indices]
    cumulative_weights = _cumulative_weights(positive_weights)
    for _ in range(max_checks):
        left_pos = _weighted_choice_index(cumulative_weights, rng)
        right_pos = _weighted_choice_index(cumulative_weights, rng)
        retry_count = 0
        while right_pos == left_pos and retry_count < 12:
            right_pos = _weighted_choice_index(cumulative_weights, rng)
            retry_count += 1
        if right_pos == left_pos:
            right_pos = (left_pos + 1) % len(positive_indices)
        left = positive_indices[left_pos]
        right = positive_indices[right_pos]
        yield records[left], records[right]


def _cumulative_weights(weights: List[float]) -> List[float]:
    cumulative: List[float] = []
    total = 0.0
    for weight in weights:
        total += float(weight)
        cumulative.append(total)
    return cumulative


def _weighted_choice_index(cumulative_weights: List[float], rng: random.Random) -> int:
    total = cumulative_weights[-1] if cumulative_weights else 0.0
    if total <= 0.0:
        raise ValueError("R2V weighted sampling requires positive total weight")
    target = rng.random() * total
    return min(bisect.bisect_left(cumulative_weights, target), len(cumulative_weights) - 1)


def _candidate_pass(
    records: List[Dict],
    seed: int,
    label: str,
    quota: int,
    *,
    multiplier: int = 1000,
    floor: int = DEFAULT_CANDIDATE_FLOOR,
    ceiling: int = DEFAULT_CANDIDATE_CEILING,
) -> Iterator[tuple[Dict, Dict]]:
    if _has_r2v_sampling_weights(records):
        budget = _weighted_candidate_budget(
            quota,
            multiplier=multiplier,
            floor=floor,
            ceiling=ceiling,
        )
        return _iter_weighted_candidate_pairs(records, random.Random(f"{seed}:{label}:r2v"), budget)
    budget = _candidate_budget(
        len(records),
        quota,
        multiplier=multiplier,
        floor=floor,
        ceiling=ceiling,
    )
    return _iter_candidate_pairs(records, random.Random(f"{seed}:{label}"), budget)


def _prepare_sampling_records(
    records: List[Dict],
    *,
    er_baseline_mode: str,
    er_priority_key: str,
    er_r2v_combine: str,
    r2v_weighted_transitions: str | Path | None,
    r2v_sampling_mode: str,
    r2v_weight_key: str,
    r2v_join_key: str,
    seed: int,
) -> tuple[List[Dict], Dict[str, Any]]:
    if er_baseline_mode not in SUPPORTED_ER_BASELINE_MODES:
        raise ValueError(f"er_baseline_mode must be one of {sorted(SUPPORTED_ER_BASELINE_MODES)}")
    if er_r2v_combine not in SUPPORTED_ER_R2V_COMBINE_MODES:
        raise ValueError(f"er_r2v_combine must be one of {sorted(SUPPORTED_ER_R2V_COMBINE_MODES)}")

    cleaned_records = [_strip_internal_sampling_fields(record) for record in records]
    er_weights = _er_sampling_weights_for_mode(
        cleaned_records,
        mode=er_baseline_mode,
        priority_key=er_priority_key,
    )
    er_enabled = er_baseline_mode != "uniform"
    base_report = {
        "strategy": "er_weighted_index_pairs" if er_enabled else "bounded_random_index_pairs",
        "er_baseline_mode": er_baseline_mode,
        "er_baseline_enabled": er_enabled,
        "er_priority_key": er_priority_key,
        "er_weight_summary": _sampling_weight_summary(er_weights),
        "er_positive_sampling_record_count": sum(1 for weight in er_weights if weight > 0.0),
        "er_r2v_combine": er_r2v_combine,
        "r2v_sampling_enabled": False,
        "r2v_sampling_mode": r2v_sampling_mode,
    }

    if r2v_sampling_mode == "off":
        if r2v_weighted_transitions is not None:
            raise ValueError("r2v_weighted_transitions requires r2v_sampling_mode != 'off'")
        if not er_enabled:
            return cleaned_records, base_report
        weighted_records = [dict(record) for record in cleaned_records]
        for record, weight in zip(weighted_records, er_weights):
            record["_er_sampling_weight"] = weight
            record["_pair_sampling_weight"] = weight
            record["_r2v_sampling_weight"] = weight
        return weighted_records, base_report

    r2v_records, r2v_report = _prepare_r2v_sampling_records(
        cleaned_records,
        r2v_weighted_transitions=r2v_weighted_transitions,
        r2v_sampling_mode=r2v_sampling_mode,
        r2v_weight_key=r2v_weight_key,
        r2v_join_key=r2v_join_key,
        seed=seed,
    )
    r2v_weights = [float(record.get("_r2v_sampling_weight", 0.0)) for record in r2v_records]
    combined_weights = _compose_er_r2v_weights(er_weights, r2v_weights, mode=er_r2v_combine)
    for record, er_weight, combined_weight in zip(r2v_records, er_weights, combined_weights):
        record["_er_sampling_weight"] = er_weight
        record["_pair_sampling_weight"] = combined_weight
        record["_r2v_sampling_weight"] = combined_weight

    combined_positive_count = sum(1 for weight in combined_weights if weight > 0.0)
    if combined_positive_count < 2:
        raise ValueError(
            f"er_baseline_mode={er_baseline_mode!r} and r2v_sampling_mode={r2v_sampling_mode!r} "
            "produced fewer than two positive-weight records"
        )
    r2v_report.update({
        "strategy": "r2v_er_weighted_index_pairs" if er_enabled else "r2v_weighted_index_pairs",
        "er_baseline_mode": er_baseline_mode,
        "er_baseline_enabled": er_enabled,
        "er_priority_key": er_priority_key,
        "er_weight_summary": _sampling_weight_summary(er_weights),
        "er_positive_sampling_record_count": sum(1 for weight in er_weights if weight > 0.0),
        "er_r2v_combine": er_r2v_combine,
        "combined_weight_summary": _sampling_weight_summary(combined_weights),
        "combined_positive_sampling_record_count": combined_positive_count,
    })
    return r2v_records, r2v_report


def _er_sampling_weights_for_mode(
    records: List[Dict],
    *,
    mode: str,
    priority_key: str,
) -> List[float]:
    if mode == "uniform":
        return [1.0 for _ in records]
    if mode == "recent":
        return _normalize_er_weights(_recent_scores(records), mode)
    if mode == "pressure_priority":
        return _normalize_er_weights(_pressure_priority_scores(records), mode)
    if mode == "reward_priority":
        return _normalize_er_weights(_reward_priority_scores(records), mode)
    if mode == "td_error_priority":
        return _normalize_er_weights(_required_priority_scores(records, priority_key), mode)
    if mode == "diversity":
        return _normalize_er_weights(_diversity_scores(records), mode)
    if mode == "intersection_balanced":
        return _normalize_er_weights(_inverse_bucket_frequency_scores(records, _intersection_bucket), mode)
    if mode == "phase_action_balanced":
        return _normalize_er_weights(_inverse_bucket_frequency_scores(records, _phase_action_bucket), mode)
    raise ValueError(f"unsupported er_baseline_mode {mode!r}")


def _recent_scores(records: List[Dict]) -> List[float]:
    ordered = sorted(
        range(len(records)),
        key=lambda idx: (
            _finite_float_or_default(get_path_value(records[idx], "episode", 0.0), 0.0),
            _finite_float_or_default(get_path_value(records[idx], "step", idx), float(idx)),
            idx,
        ),
    )
    scores = [0.0 for _ in records]
    for rank, idx in enumerate(ordered, start=1):
        scores[idx] = float(rank)
    return scores


def _pressure_priority_scores(records: List[Dict]) -> List[float]:
    paths = (
        "metadata.pressure",
        "metadata.max_pressure",
        "metadata.queue_pressure",
        "metadata.queue_length",
        "pressure",
        "max_pressure",
        "queue_pressure",
        "queue_length",
    )
    explicit = [_first_finite_path(record, paths) for record in records]
    if any(value is not None for value in explicit):
        return [abs(value) if value is not None else 0.0 for value in explicit]

    efficiencies = [_objective_score(record, "efficiency") for record in records]
    if any(value is not None for value in efficiencies):
        finite_efficiencies = [value for value in efficiencies if value is not None]
        worst = max(finite_efficiencies)
        return [abs(worst - value) if value is not None else 0.0 for value in efficiencies]
    return [1.0 for _ in records]


def _reward_priority_scores(records: List[Dict]) -> List[float]:
    explicit = [_first_finite_path(record, ("env_reward", "metadata.env_reward", "reward", "metadata.reward")) for record in records]
    if any(value is not None for value in explicit):
        return [abs(value) if value is not None else 0.0 for value in explicit]
    return [abs(_balanced_objective_utility(record)) for record in records]


def _required_priority_scores(records: List[Dict], priority_key: str) -> List[float]:
    scores: List[float] = []
    missing: List[str] = []
    for idx, record in enumerate(records):
        raw_value = get_path_value(record, priority_key, None)
        value = _finite_float_or_default(raw_value, None)
        if value is None:
            missing.append(str(record.get("sample_id", idx)))
            scores.append(0.0)
        else:
            scores.append(abs(value))
    if missing:
        raise ValueError(
            f"er_baseline_mode='td_error_priority' requires finite er_priority_key {priority_key!r}; "
            f"missing/invalid rows: {missing[:5]}"
        )
    return scores


def _diversity_scores(records: List[Dict]) -> List[float]:
    vectors = [_record_feature_vector(record) for record in records]
    if not vectors:
        return []
    width = max((len(vector) for vector in vectors), default=0)
    if width == 0:
        return [1.0 for _ in records]
    padded = [vector + [0.0] * (width - len(vector)) for vector in vectors]
    centroid = [sum(vector[idx] for vector in padded) / len(padded) for idx in range(width)]
    return [
        math.sqrt(sum((value - centroid[idx]) ** 2 for idx, value in enumerate(vector)))
        for vector in padded
    ]


def _inverse_bucket_frequency_scores(records: List[Dict], bucket_fn) -> List[float]:
    buckets = [bucket_fn(record) for record in records]
    counts: Dict[str, int] = {}
    for bucket in buckets:
        counts[bucket] = counts.get(bucket, 0) + 1
    if len(counts) <= 1:
        return [1.0 for _ in records]
    return [1.0 / float(counts[bucket]) for bucket in buckets]


def _intersection_bucket(record: Dict) -> str:
    return _first_nonempty_path(record, ("intersection_id", "metadata.intersection_id"), "__missing_intersection__")


def _phase_action_bucket(record: Dict) -> str:
    return _first_nonempty_path(
        record,
        ("action", "phase", "metadata.action", "metadata.phase", "metadata.phase_action"),
        "__missing_phase_action__",
    )


def _first_nonempty_path(record: Dict, paths: Iterable[str], default: str) -> str:
    for path in paths:
        value = get_path_value(record, path, None)
        if value is not None and str(value) != "":
            return str(value)
    return default


def _first_finite_path(record: Dict, paths: Iterable[str]) -> float | None:
    for path in paths:
        value = _finite_float_or_default(get_path_value(record, path, None), None)
        if value is not None:
            return value
    return None


def _finite_float_or_default(value: Any, default: float | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _objective_score(record: Dict, objective: str) -> float | None:
    for field in ("objective_values_norm", "objectives_t_norm", "objectives_tp1_norm"):
        values = record.get(field)
        if isinstance(values, dict) and objective in values:
            return _finite_float_or_default(values.get(objective), None)
    return None


def _balanced_objective_utility(record: Dict) -> float:
    values = [_objective_score(record, objective) for objective in OBJECTIVE_NAMES]
    finite = [value for value in values if value is not None]
    if not finite:
        return 1.0
    return sum(finite) / float(len(finite))


def _record_feature_vector(record: Dict) -> List[float]:
    raw_features = record.get("obs_features")
    if isinstance(raw_features, list):
        features = [
            parsed
            for parsed in (_finite_float_or_default(value, None) for value in raw_features)
            if parsed is not None
        ]
        if features:
            return features
    return [
        value if value is not None else 0.0
        for value in (_objective_score(record, objective) for objective in OBJECTIVE_NAMES)
    ]


def _normalize_er_weights(raw_weights: List[float], mode: str) -> List[float]:
    if not raw_weights:
        return []
    parsed: List[float] = []
    for idx, raw_weight in enumerate(raw_weights):
        weight = _finite_float_or_default(raw_weight, None)
        if weight is None or weight < 0.0:
            raise ValueError(f"er_baseline_mode={mode!r} produced invalid weight at row {idx}: {raw_weight!r}")
        parsed.append(float(weight))
    positives = [weight for weight in parsed if weight > 0.0]
    if not positives:
        return [1.0 for _ in parsed]
    floor = max(max(positives) * 1e-6, 1e-12)
    positive_weights = [max(weight, floor) for weight in parsed]
    mean_weight = sum(positive_weights) / float(len(positive_weights))
    return [weight / mean_weight for weight in positive_weights]


def _compose_er_r2v_weights(er_weights: List[float], r2v_weights: List[float], *, mode: str) -> List[float]:
    if len(er_weights) != len(r2v_weights):
        raise ValueError("ER and R2V sampling weights length mismatch")
    if mode == "replace":
        return [float(weight) for weight in r2v_weights]
    if mode == "multiply":
        return [float(er_weight) * float(r2v_weight) for er_weight, r2v_weight in zip(er_weights, r2v_weights)]
    raise ValueError(f"unsupported er_r2v_combine {mode!r}")


def _sampling_weight_summary(weights: List[float]) -> Dict[str, Any]:
    if not weights:
        return {
            "count": 0,
            "positive_count": 0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "weight_mean": 0.0,
        }
    return {
        "count": len(weights),
        "positive_count": sum(1 for weight in weights if weight > 0.0),
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "weight_mean": float(sum(weights) / len(weights)),
    }


def _prepare_r2v_sampling_records(
    records: List[Dict],
    *,
    r2v_weighted_transitions: str | Path | None,
    r2v_sampling_mode: str,
    r2v_weight_key: str,
    r2v_join_key: str,
    seed: int,
) -> tuple[List[Dict], Dict[str, Any]]:
    if r2v_sampling_mode not in SUPPORTED_R2V_SAMPLING_MODES:
        raise ValueError(f"r2v_sampling_mode must be one of {sorted(SUPPORTED_R2V_SAMPLING_MODES)}")
    if r2v_sampling_mode == "off":
        if r2v_weighted_transitions is not None:
            raise ValueError("r2v_weighted_transitions requires r2v_sampling_mode != 'off'")
        return [_strip_internal_sampling_fields(record) for record in records], {}
    if r2v_weighted_transitions is None:
        raise ValueError("r2v_sampling_mode requires --r2v_weighted_transitions")

    weighted_path = Path(r2v_weighted_transitions)
    weighted_rows = load_records([weighted_path])
    traffic_artifact_summary = validate_r2v_traffic_artifact(weighted_rows)
    weight_map, weight_summary = build_r2v_weight_map(
        weighted_rows,
        join_key=r2v_join_key,
        weight_key=r2v_weight_key,
    )
    validate_weighted_transition_rows(weighted_rows, join_key=r2v_join_key, weight_key=r2v_weight_key)
    weighted_by_join = {str(get_path_value(row, r2v_join_key)): row for row in weighted_rows}
    missing = [
        str(record.get(r2v_join_key))
        for record in records
        if str(record.get(r2v_join_key)) not in weight_map
    ]
    if missing:
        raise ValueError(
            f"missing R2V weights for {len(missing)} pair records using join key {r2v_join_key}: {missing[:5]}"
        )

    rng = random.Random(f"{seed}:r2v_sampling_mode:{r2v_sampling_mode}")
    copied_records = [dict(record) for record in records]
    sampling_weights = _sampling_weights_for_mode(
        copied_records,
        weighted_by_join=weighted_by_join,
        weight_map=weight_map,
        mode=r2v_sampling_mode,
        join_key=r2v_join_key,
        rng=rng,
    )
    for record in copied_records:
        join_id = str(record.get(r2v_join_key))
        record["_r2v_sampling_weight"] = sampling_weights[join_id]

    positive_count = sum(1 for weight in sampling_weights.values() if weight > 0.0)
    if positive_count < 2:
        raise ValueError(
            f"r2v_sampling_mode={r2v_sampling_mode!r} produced fewer than two positive-weight records"
        )
    report = {
        "strategy": "r2v_weighted_index_pairs",
        "r2v_sampling_enabled": True,
        "r2v_source_path": str(weighted_path),
        "r2v_sampling_mode": r2v_sampling_mode,
        "r2v_join_key": r2v_join_key,
        "r2v_weight_key": r2v_weight_key,
        "r2v_weight_summary": weight_summary,
        "r2v_traffic_artifact_summary": traffic_artifact_summary,
        "r2v_positive_sampling_record_count": positive_count,
    }
    return copied_records, report


def _sampling_weights_for_mode(
    records: List[Dict],
    *,
    weighted_by_join: Dict[str, Dict],
    weight_map: Dict[str, float],
    mode: str,
    join_key: str,
    rng: random.Random,
) -> Dict[str, float]:
    if mode == "full_r2v":
        return dict(weight_map)

    max_weight = max(weight_map.values()) if weight_map else 1.0
    join_ids = [str(record.get(join_key)) for record in records]
    admitted_ids = [
        join_id
        for join_id in join_ids
        if bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_admitted", False))
    ]

    if mode == "admitted_only":
        return {join_id: weight_map[join_id] if join_id in admitted_ids else 0.0 for join_id in join_ids}
    if mode == "rare_only":
        return {
            join_id: max_weight if bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_gates.rare", False)) else 0.0
            for join_id in join_ids
        }
    if mode == "value_only":
        return {
            join_id: max_weight if bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_gates.value", False)) else 0.0
            for join_id in join_ids
        }
    if mode == "inverted_rarity":
        return {
            join_id: max_weight if not bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_gates.rare", False)) else 0.0
            for join_id in join_ids
        }
    if mode == "random_same_count":
        shuffled = list(join_ids)
        rng.shuffle(shuffled)
        selected = set(shuffled[: len(admitted_ids)])
        return {join_id: max_weight if join_id in selected else 0.0 for join_id in join_ids}
    if mode == "same_candidates_random_weights":
        low = min(weight_map.values()) if weight_map else 1.0
        high = max_weight
        return {
            join_id: rng.uniform(low, high) if join_id in admitted_ids else 0.0
            for join_id in join_ids
        }
    if mode == "shuffled_value":
        value_flags = [
            bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_gates.value", False))
            for join_id in join_ids
        ]
        rng.shuffle(value_flags)
        shuffled_value_by_id = dict(zip(join_ids, value_flags))
        return {
            join_id: max_weight
            if bool(get_path_value(weighted_by_join[join_id], "metadata.r2v_gates.rare", False))
            and shuffled_value_by_id[join_id]
            else 0.0
            for join_id in join_ids
        }
    raise ValueError(f"unsupported r2v_sampling_mode {mode!r}")


def _make_objective_row(
    a: Dict,
    b: Dict,
    objective: str,
    margin: float,
    pair_id: str,
    scenario: str,
    sampling_strategy: str,
    rng: random.Random,
    split: str,
) -> Dict:
    label = int(margin > 0)
    aa, bb, yy = _maybe_flip(a, b, label, rng)
    signed_margin = float(margin if aa is a else -margin)
    return ObjectivePairLabel(
        pair_id=pair_id,
        split=split,
        scenario=scenario,
        a_id=aa["sample_id"],
        b_id=bb["sample_id"],
        objective=objective,
        label=yy,
        is_tie=False,
        margin_raw=signed_margin,
        margin_norm=signed_margin,
        source="rule",
        rule_version="objective_norm_v1",
        sampling_strategy=sampling_strategy,
    ).to_dict()


def _valid_for_objectives(a: Dict, b: Dict, objectives: Iterable[str]) -> bool:
    return all(_valid(a, obj) and _valid(b, obj) for obj in objectives)


def _build_efficiency_controlled_rows(
    candidates: Iterable[tuple[Dict, Dict]],
    target_objective: str,
    quota: int,
    scenario: str,
    tie_margin: float,
    rng: random.Random,
    split: str,
    eps_efficiency: float = 0.15,
) -> List[Dict]:
    rows = []
    needed = ("efficiency", target_objective)
    for a, b in candidates:
        if not _valid_for_objectives(a, b, needed):
            continue
        efficiency_gap = abs(_obj(a, "efficiency") - _obj(b, "efficiency"))
        target_margin = _obj(a, target_objective) - _obj(b, target_objective)
        if efficiency_gap > eps_efficiency or abs(target_margin) < tie_margin:
            continue
        rows.append(_make_objective_row(
            a,
            b,
            target_objective,
            target_margin,
            pair_id=f"obj_eff_controlled_{target_objective}_{len(rows)}",
            scenario=scenario,
            sampling_strategy=f"eff_controlled_{target_objective}",
            rng=rng,
            split=split,
        ))
        if len(rows) >= quota:
            break
    return rows


def _build_efficiency_safety_conflict_rows(
    candidates: Iterable[tuple[Dict, Dict]],
    quota: int,
    scenario: str,
    tie_margin: float,
    rng: random.Random,
    split: str,
) -> List[Dict]:
    rows = []
    for a, b in candidates:
        if not _valid_for_objectives(a, b, ("efficiency", "safety")):
            continue
        efficiency_margin = _obj(a, "efficiency") - _obj(b, "efficiency")
        safety_margin = _obj(a, "safety") - _obj(b, "safety")
        if abs(efficiency_margin) < tie_margin or abs(safety_margin) < tie_margin:
            continue
        if efficiency_margin * safety_margin >= 0:
            continue
        rows.append(_make_objective_row(
            a,
            b,
            "safety",
            safety_margin,
            pair_id=f"obj_efficiency_safety_conflict_{len(rows)}",
            scenario=scenario,
            sampling_strategy="efficiency_safety_conflict",
            rng=rng,
            split=split,
        ))
        if len(rows) >= quota:
            break
    return rows


def _build_efficiency_stability_conflict_rows(
    candidates: Iterable[tuple[Dict, Dict]],
    quota: int,
    scenario: str,
    tie_margin: float,
    rng: random.Random,
    split: str,
) -> List[Dict]:
    rows = []
    for a, b in candidates:
        if not _valid_for_objectives(a, b, ("efficiency", "stability")):
            continue
        efficiency_margin = _obj(a, "efficiency") - _obj(b, "efficiency")
        stability_margin = _obj(a, "stability") - _obj(b, "stability")
        if abs(efficiency_margin) < tie_margin or abs(stability_margin) < tie_margin:
            continue
        if efficiency_margin * stability_margin >= 0:
            continue
        rows.append(_make_objective_row(
            a,
            b,
            "stability",
            stability_margin,
            pair_id=f"obj_efficiency_stability_conflict_{len(rows)}",
            scenario=scenario,
            sampling_strategy="efficiency_stability_conflict",
            rng=rng,
            split=split,
        ))
        if len(rows) >= quota:
            break
    return rows


def _build_generic_objective_rows(
    candidates: Iterable[tuple[Dict, Dict]],
    objective: str,
    quota: int,
    scenario: str,
    tie_margin: float,
    rng: random.Random,
    split: str,
    start_idx: int,
) -> tuple[List[Dict], int, int]:
    rows = []
    invalid_count = 0
    tie_count = 0
    for a, b in candidates:
        if not (_valid(a, objective) and _valid(b, objective)):
            invalid_count += 1
            continue
        margin = _obj(a, objective) - _obj(b, objective)
        if abs(margin) < tie_margin:
            tie_count += 1
            continue
        rows.append(_make_objective_row(
            a,
            b,
            objective,
            margin,
            pair_id=f"obj_{objective}_{start_idx + len(rows)}",
            scenario=scenario,
            sampling_strategy="objective_contrast",
            rng=rng,
            split=split,
        ))
        if len(rows) >= quota:
            break
    return rows, invalid_count, tie_count


def _preference_reversal_candidates(a: Dict, b: Dict, tie_margin: float) -> List[Dict]:
    pref_results = []
    for name, w in PREFERENCE_TEMPLATES.items():
        margin = _utility(a, w) - _utility(b, w)
        if abs(margin) >= tie_margin:
            pref_results.append((name, w, int(margin > 0), margin))
    rows = []
    for idx, left in enumerate(pref_results):
        for right in pref_results[idx + 1:]:
            if left[2] == right[2]:
                continue
            rows.append({
                "w_1_name": left[0],
                "w_1": left[1],
                "label_1": left[2],
                "margin_1": float(left[3]),
                "w_2_name": right[0],
                "w_2": right[1],
                "label_2": right[2],
                "margin_2": float(right[3]),
                "template_key": f"{left[0]}__{right[0]}",
            })
    return rows


def _make_reversal_row(
    a: Dict,
    b: Dict,
    candidate: Dict,
    pair_id: str,
    scenario: str,
    split: str,
) -> Dict:
    return {
        "pair_id": pair_id,
        "split": split,
        "scenario": scenario,
        "a_id": a["sample_id"],
        "b_id": b["sample_id"],
        "w_1_name": candidate["w_1_name"],
        "w_1": candidate["w_1"],
        "label_1": candidate["label_1"],
        "margin_1": candidate["margin_1"],
        "w_2_name": candidate["w_2_name"],
        "w_2": candidate["w_2"],
        "label_2": candidate["label_2"],
        "margin_2": candidate["margin_2"],
        "sampling_strategy": "reversal",
    }


def build_pairs_from_records(
    records: List[Dict],
    out_dir: str | Path,
    num_objective_pairs: int,
    num_preference_pairs: int,
    num_dominance_pairs: int,
    num_reversal_pairs: int,
    seed: int = 0,
    tie_margin: float = 0.05,
    split: str = "train",
    min_efficiency_stability_conflict: int = 0,
    reversal_template_quota: Dict[str, int] | None = None,
    er_baseline_mode: str = "uniform",
    er_priority_key: str = "metadata.td_error",
    er_r2v_combine: str = "multiply",
    r2v_weighted_transitions: str | Path | None = None,
    r2v_sampling_mode: str = "off",
    r2v_weight_key: str = "metadata.r2v_sample_weight",
    r2v_join_key: str = "sample_id",
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    scenario = records[0].get("scenario", "unknown") if records else "unknown"
    record_count = len(records)
    total_candidate_pairs = _total_candidate_pairs(record_count)
    sampling_records, sampling_report = _prepare_sampling_records(
        records,
        er_baseline_mode=er_baseline_mode,
        er_priority_key=er_priority_key,
        er_r2v_combine=er_r2v_combine,
        r2v_weighted_transitions=r2v_weighted_transitions,
        r2v_sampling_mode=r2v_sampling_mode,
        r2v_weight_key=r2v_weight_key,
        r2v_join_key=r2v_join_key,
        seed=seed,
    )
    tie_count = 0
    invalid_count = 0

    objective_rows = []
    objective_target = {
        name: num_objective_pairs // len(OBJECTIVE_NAMES)
        for name in OBJECTIVE_NAMES
    }
    for name in OBJECTIVE_NAMES[: num_objective_pairs % len(OBJECTIVE_NAMES)]:
        objective_target[name] += 1
    contrast_rows = {"fairness": [], "stability": [], "safety_conflict": [], "stability_conflict": []}
    objective_counts = {}
    for objective in OBJECTIVE_NAMES:
        target = objective_target[objective]
        rows_for_objective = []
        contrast_quota = min(target, max(1, target // 3)) if target else 0
        if objective == "fairness" and contrast_quota:
            contrast_rows["fairness"] = _build_efficiency_controlled_rows(
                _candidate_pass(
                    sampling_records,
                    seed,
                    "objective:fairness:efficiency_controlled",
                    contrast_quota,
                    multiplier=1500,
                ),
                "fairness",
                contrast_quota,
                scenario,
                tie_margin,
                rng,
                split,
            )
            rows_for_objective.extend(contrast_rows["fairness"])
        elif objective == "stability" and contrast_quota:
            contrast_rows["stability"] = _build_efficiency_controlled_rows(
                _candidate_pass(
                    sampling_records,
                    seed,
                    "objective:stability:efficiency_controlled",
                    contrast_quota,
                    multiplier=1500,
                ),
                "stability",
                contrast_quota,
                scenario,
                tie_margin,
                rng,
                split,
            )
            rows_for_objective.extend(contrast_rows["stability"])
            stability_conflict_quota = min(
                max(0, target - len(rows_for_objective)),
                max(0, int(min_efficiency_stability_conflict)),
            )
            if stability_conflict_quota:
                contrast_rows["stability_conflict"] = _build_efficiency_stability_conflict_rows(
                    _candidate_pass(
                        sampling_records,
                        seed,
                        "objective:stability:efficiency_conflict",
                        stability_conflict_quota,
                        multiplier=3000,
                        floor=50_000,
                        ceiling=1_000_000,
                    ),
                    stability_conflict_quota,
                    scenario,
                    tie_margin,
                    rng,
                    split,
                )
                rows_for_objective.extend(contrast_rows["stability_conflict"])
        elif objective == "safety" and contrast_quota:
            contrast_rows["safety_conflict"] = _build_efficiency_safety_conflict_rows(
                _candidate_pass(
                    sampling_records,
                    seed,
                    "objective:safety:efficiency_conflict",
                    contrast_quota,
                    multiplier=3000,
                    floor=50_000,
                    ceiling=1_000_000,
                ),
                contrast_quota,
                scenario,
                tie_margin,
                rng,
                split,
            )
            rows_for_objective.extend(contrast_rows["safety_conflict"])
        remaining = max(0, target - len(rows_for_objective))
        generic_rows, invalid_extra, tie_extra = _build_generic_objective_rows(
            _candidate_pass(
                sampling_records,
                seed,
                f"objective:{objective}:generic",
                remaining,
                multiplier=1000,
            ),
            objective,
            remaining,
            scenario,
            tie_margin,
            rng,
            split,
            start_idx=len(objective_rows) + len(rows_for_objective),
        )
        invalid_count += invalid_extra
        tie_count += tie_extra
        rows_for_objective.extend(generic_rows)
        objective_counts[objective] = len(rows_for_objective)
        objective_rows.extend(rows_for_objective[:target])

    preference_rows = []
    pref_items = list(PREFERENCE_TEMPLATES.items())
    for a, b in _candidate_pass(
        sampling_records,
        seed,
        "preference",
        num_preference_pairs,
        multiplier=1000,
    ):
        name, w = pref_items[len(preference_rows) % len(pref_items)]
        if not all(_valid(a, obj) and _valid(b, obj) for obj in OBJECTIVE_NAMES):
            invalid_count += 1
            continue
        utility_a = _utility(a, w)
        utility_b = _utility(b, w)
        margin = utility_a - utility_b
        if abs(margin) < tie_margin:
            tie_count += 1
            continue
        label = int(margin > 0)
        aa, bb, yy = _maybe_flip(a, b, label, rng)
        if aa is b:
            utility_a, utility_b, margin = utility_b, utility_a, -margin
        preference_rows.append(PreferencePairLabel(
            pair_id=f"pref_{len(preference_rows)}",
            split=split,
            scenario=scenario,
            a_id=aa["sample_id"],
            b_id=bb["sample_id"],
            w=w,
            label=yy,
            is_tie=False,
            source="rule",
            rule_utility_a=float(utility_a),
            rule_utility_b=float(utility_b),
            rule_margin=float(margin),
            sampling_strategy=f"preference_{name}",
        ).to_dict())
        if len(preference_rows) >= num_preference_pairs:
            break

    dominance_rows = []
    for a, b in _candidate_pass(
        sampling_records,
        seed,
        "dominance",
        num_dominance_pairs,
        multiplier=5000,
        floor=50_000,
        ceiling=2_000_000,
    ):
        if not all(_valid(a, obj) and _valid(b, obj) for obj in OBJECTIVE_NAMES):
            invalid_count += 1
            continue
        margins = {obj: _obj(a, obj) - _obj(b, obj) for obj in OBJECTIVE_NAMES}
        if all(value >= tie_margin for value in margins.values()):
            dom = "a"
        elif all(value <= -tie_margin for value in margins.values()):
            dom = "b"
        else:
            continue
        dominance_rows.append(DominancePairLabel(
            pair_id=f"dom_{len(dominance_rows)}",
            split=split,
            scenario=scenario,
            a_id=a["sample_id"],
            b_id=b["sample_id"],
            dominates=dom,
            objective_margins_norm={key: float(value) for key, value in margins.items()},
            min_margin_norm=float(min(abs(value) for value in margins.values())),
            dominance_threshold=tie_margin,
            source="rule",
        ).to_dict())
        if len(dominance_rows) >= num_dominance_pairs:
            break

    reversal_rows = []
    reversal_by_template_pair = {}
    used_reversal_keys = set()
    template_quota = dict(reversal_template_quota or {})
    for a, b in _candidate_pass(
        sampling_records,
        seed,
        "reversal:quota",
        num_reversal_pairs,
        multiplier=5000,
        floor=50_000,
        ceiling=2_000_000,
    ):
        if not all(_valid(a, obj) and _valid(b, obj) for obj in OBJECTIVE_NAMES):
            invalid_count += 1
            continue
        for candidate in _preference_reversal_candidates(a, b, tie_margin):
            template_key = candidate["template_key"]
            if template_key not in template_quota:
                continue
            if reversal_by_template_pair.get(template_key, 0) >= template_quota[template_key]:
                continue
            key = (a["sample_id"], b["sample_id"], template_key)
            if key in used_reversal_keys:
                continue
            reversal_rows.append(_make_reversal_row(
                a,
                b,
                candidate,
                pair_id=f"rev_{len(reversal_rows)}",
                scenario=scenario,
                split=split,
            ))
            used_reversal_keys.add(key)
            reversal_by_template_pair[template_key] = reversal_by_template_pair.get(template_key, 0) + 1
            break
        if len(reversal_rows) >= num_reversal_pairs:
            break
    for a, b in _candidate_pass(
        sampling_records,
        seed,
        "reversal:fallback",
        num_reversal_pairs,
        multiplier=5000,
        floor=50_000,
        ceiling=2_000_000,
    ):
        if len(reversal_rows) >= num_reversal_pairs:
            break
        if not all(_valid(a, obj) and _valid(b, obj) for obj in OBJECTIVE_NAMES):
            invalid_count += 1
            continue
        for candidate in _preference_reversal_candidates(a, b, tie_margin):
            template_key = candidate["template_key"]
            key = (a["sample_id"], b["sample_id"], template_key)
            if key in used_reversal_keys:
                continue
            reversal_rows.append(_make_reversal_row(
                a,
                b,
                candidate,
                pair_id=f"rev_{len(reversal_rows)}",
                scenario=scenario,
                split=split,
            ))
            used_reversal_keys.add(key)
            reversal_by_template_pair[template_key] = reversal_by_template_pair.get(template_key, 0) + 1
            break
        if len(reversal_rows) >= num_reversal_pairs:
            break

    _write_jsonl(out_dir / "objective_pairs.jsonl", objective_rows)
    _write_jsonl(out_dir / "objective_pairs_eff_controlled_fairness.jsonl", contrast_rows["fairness"])
    _write_jsonl(out_dir / "objective_pairs_eff_controlled_stability.jsonl", contrast_rows["stability"])
    _write_jsonl(out_dir / "objective_pairs_efficiency_safety_conflict.jsonl", contrast_rows["safety_conflict"])
    _write_jsonl(out_dir / "objective_pairs_efficiency_stability_conflict.jsonl", contrast_rows["stability_conflict"])
    _write_jsonl(out_dir / "preference_pairs.jsonl", preference_rows)
    _write_jsonl(out_dir / "dominance_pairs.jsonl", dominance_rows)
    _write_jsonl(out_dir / "reversal_pairs.jsonl", reversal_rows)
    candidate_sampling_report = {
        "strategy": "bounded_random_index_pairs",
        "record_count": record_count,
        "total_candidate_pairs": total_candidate_pairs,
        "max_enumerated_candidates": MAX_ENUMERATED_CANDIDATES,
        "default_candidate_ceiling": DEFAULT_CANDIDATE_CEILING,
    }
    if sampling_report:
        candidate_sampling_report.update(sampling_report)
        candidate_sampling_report["record_count"] = record_count
        candidate_sampling_report["total_candidate_pairs"] = total_candidate_pairs
        candidate_sampling_report["max_enumerated_candidates"] = MAX_ENUMERATED_CANDIDATES
        candidate_sampling_report["default_candidate_ceiling"] = DEFAULT_CANDIDATE_CEILING
    report = {
        "objective_pairs": len(objective_rows),
        "preference_pairs": len(preference_rows),
        "dominance_pairs": len(dominance_rows),
        "reversal_pairs": len(reversal_rows),
        "tie_count_before_filter": tie_count,
        "serialized_tie_count": 0,
        "invalid_candidate_count": invalid_count,
        "objective_counts": objective_counts,
        "contrast_counts": {
            "eff_controlled_fairness": len(contrast_rows["fairness"]),
            "eff_controlled_stability": len(contrast_rows["stability"]),
            "efficiency_safety_conflict": len(contrast_rows["safety_conflict"]),
            "efficiency_stability_conflict": len(contrast_rows["stability_conflict"]),
        },
        "reversal_by_template_pair": reversal_by_template_pair,
        "candidate_sampling": candidate_sampling_report,
    }
    write_json(out_dir / "pair_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffers", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_objective_pairs", type=int, default=1000)
    parser.add_argument("--num_preference_pairs", type=int, default=1000)
    parser.add_argument("--num_dominance_pairs", type=int, default=300)
    parser.add_argument("--num_reversal_pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie_margin", type=float, default=0.05)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--min_efficiency_stability_conflict", type=int, default=0)
    parser.add_argument("--reversal_template_quota", action="append")
    parser.add_argument(
        "--er_baseline_mode",
        "--er_sampling_mode",
        choices=sorted(SUPPORTED_ER_BASELINE_MODES),
        default="uniform",
    )
    parser.add_argument("--er_priority_key", default="metadata.td_error")
    parser.add_argument(
        "--er_r2v_combine",
        choices=sorted(SUPPORTED_ER_R2V_COMBINE_MODES),
        default="multiply",
    )
    parser.add_argument("--r2v_weighted_transitions", default=None)
    parser.add_argument(
        "--r2v_sampling_mode",
        choices=sorted(SUPPORTED_R2V_SAMPLING_MODES),
        default="off",
    )
    parser.add_argument("--r2v_weight_key", default="metadata.r2v_sample_weight")
    parser.add_argument("--r2v_join_key", default="sample_id")
    args = parser.parse_args()
    template_quota = {}
    for item in args.reversal_template_quota or []:
        if ":" not in item:
            raise ValueError(f"reversal template quota must be template:count, got {item}")
        template, count = item.split(":", 1)
        template_quota[template] = int(count)
    report = build_pairs_from_records(
        load_records(args.buffers),
        out_dir=args.out_dir,
        num_objective_pairs=args.num_objective_pairs,
        num_preference_pairs=args.num_preference_pairs,
        num_dominance_pairs=args.num_dominance_pairs,
        num_reversal_pairs=args.num_reversal_pairs,
        seed=args.seed,
        tie_margin=args.tie_margin,
        split=args.split,
        min_efficiency_stability_conflict=args.min_efficiency_stability_conflict,
        reversal_template_quota=template_quota,
        er_baseline_mode=args.er_baseline_mode,
        er_priority_key=args.er_priority_key,
        er_r2v_combine=args.er_r2v_combine,
        r2v_weighted_transitions=args.r2v_weighted_transitions,
        r2v_sampling_mode=args.r2v_sampling_mode,
        r2v_weight_key=args.r2v_weight_key,
        r2v_join_key=args.r2v_join_key,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
