from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class TtcRisk:
    violation_rate: float
    min_ttc: float
    pair_count: int
    valid: bool


@dataclass
class ObjectiveOutput:
    raw: Dict[str, float]
    norm: Dict[str, float]
    valid_mask: Dict[str, bool]
    debug: Dict[str, Any]


def gini(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.abs(arr)
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff_sum / (2.0 * arr.size * total))


def compute_efficiency(total_queue: float, total_wait: float, approaching: float) -> float:
    return -float(total_queue + 0.2 * total_wait + 0.1 * approaching)


def compute_fairness(lane_loads: Sequence[float]) -> float:
    return -gini(lane_loads)


def compute_stability(
    queue_now: Sequence[float],
    queue_prev: Sequence[float],
    action: Optional[int],
    prev_action: Optional[int],
    phase_duration: float,
    min_stable_duration: float = 5.0,
) -> float:
    now = np.asarray(list(queue_now), dtype=np.float64)
    prev = np.asarray(list(queue_prev), dtype=np.float64)
    if now.size == 0 or prev.size == 0 or now.shape != prev.shape:
        return 0.0
    queue_delta = np.mean(np.abs(now - prev) / np.maximum(prev + 1.0, 1.0))
    return -float(queue_delta)


def compute_transition_stability(
    queue_now: Sequence[float],
    queue_prev: Sequence[float],
    action: Optional[int],
    prev_action: Optional[int],
    phase_duration: float,
    min_stable_duration: float = 5.0,
) -> float:
    state_stability = compute_stability(
        queue_now=queue_now,
        queue_prev=queue_prev,
        action=None,
        prev_action=None,
        phase_duration=phase_duration,
        min_stable_duration=min_stable_duration,
    )
    switched = int(action is not None and prev_action is not None and action != prev_action)
    short_switch = int(switched and phase_duration < min_stable_duration)
    return float(state_stability - 0.2 * switched - 0.5 * short_switch)


def compute_ttc_risk(
    lane_vehicles: Dict[str, Sequence[str]],
    vehicle_distances: Dict[str, float],
    vehicle_speeds: Dict[str, float],
    ttc_threshold: float = 3.0,
    vehicle_length: float = 5.0,
    min_pairs: int = 1,
) -> TtcRisk:
    violations = 0
    pair_count = 0
    min_ttc = float("inf")

    for vehicles in lane_vehicles.values():
        sortable = [
            (veh, float(vehicle_distances[veh]), float(vehicle_speeds.get(veh, 0.0)))
            for veh in vehicles
            if veh in vehicle_distances
        ]
        sortable.sort(key=lambda item: item[1])
        for rear, front in zip(sortable, sortable[1:]):
            _, rear_distance, rear_speed = rear
            _, front_distance, front_speed = front
            gap = max(front_distance - rear_distance - vehicle_length, 0.0)
            closing_speed = rear_speed - front_speed
            pair_count += 1
            if closing_speed <= 1e-6:
                continue
            ttc = gap / closing_speed
            min_ttc = min(min_ttc, ttc)
            if ttc < ttc_threshold:
                violations += 1

    valid = pair_count >= min_pairs
    if not valid:
        return TtcRisk(violation_rate=0.0, min_ttc=float("inf"), pair_count=pair_count, valid=False)
    return TtcRisk(
        violation_rate=float(violations / max(pair_count, 1)),
        min_ttc=float(min_ttc),
        pair_count=pair_count,
        valid=True,
    )


def _movement_lane_loads(state_detail: Dict[str, Any]) -> list[float]:
    loads = []
    for lane in sorted(state_detail):
        item = state_detail[lane]
        loads.append(float(item.get("queue_len", 0.0)) + 0.1 * float(item.get("avg_wait_time", 0.0)))
    return loads


def _queue_values(state_detail: Dict[str, Any]) -> list[float]:
    return [float(state_detail[lane].get("queue_len", 0.0)) for lane in sorted(state_detail)]


def _approaching_count(state_detail: Dict[str, Any]) -> float:
    total = 0.0
    for item in state_detail.values():
        total += sum(float(v) for v in item.get("cells", []))
    return total


def compute_objectives_from_snapshot(
    snapshot,
    prev_snapshot=None,
    action: Optional[int] = None,
    prev_action: Optional[int] = None,
    normalizer=None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ObjectiveOutput:
    state = snapshot.state_detail
    queues = _queue_values(state)
    lane_loads = _movement_lane_loads(state)
    total_queue = float(sum(queues))
    total_wait = float(sum(float(item.get("queue_len", 0.0)) * float(item.get("avg_wait_time", 0.0)) for item in state.values()))
    approaching = _approaching_count(state)

    ttc = compute_ttc_risk(
        snapshot.lane_vehicles,
        snapshot.vehicle_distances,
        snapshot.vehicle_speeds,
        min_pairs=int((cfg or {}).get("ttc_min_pairs", 1)),
    )

    if prev_snapshot is not None:
        prev_queues = _queue_values(prev_snapshot.state_detail)
        stability = compute_stability(queues, prev_queues, action, prev_action, snapshot.time_this_phase)
        stability_valid = True
    else:
        stability = 0.0
        stability_valid = False

    raw = {
        "efficiency": compute_efficiency(total_queue, total_wait, approaching),
        "safety": -float(ttc.violation_rate),
        "fairness": compute_fairness(lane_loads),
        "stability": stability,
    }
    if normalizer is not None:
        norm = normalizer.transform(raw)
    else:
        norm = dict(raw)
    return ObjectiveOutput(
        raw=raw,
        norm=norm,
        valid_mask={
            "efficiency": True,
            "safety": bool(ttc.valid),
            "fairness": len(lane_loads) >= 4,
            "stability": stability_valid,
        },
        debug={
            "total_queue": total_queue,
            "total_wait": total_wait,
            "approaching": approaching,
            "local_lane_count": len(getattr(snapshot, "local_lane_ids", [])),
            "local_ttc_pair_count": ttc.pair_count,
            "ttc_pair_count": ttc.pair_count,
            "ttc_min": ttc.min_ttc,
            "ttc_valid": ttc.valid,
        },
    )
