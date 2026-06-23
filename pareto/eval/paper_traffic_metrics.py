from __future__ import annotations

from typing import Sequence

from pareto.data.objectives import gini


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ValueError(f"{label} observations are required")
    return float(sum(float(value) for value in values) / len(values))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("TTC values are required")
    xs = sorted(float(value) for value in values)
    if len(xs) == 1:
        return xs[0]
    pos = float(q) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def compute_ttc_metrics(ttc_values: Sequence[float], *, threshold: float = 3.0) -> dict[str, float]:
    if not ttc_values:
        raise ValueError("TTC values are required")
    xs = [float(value) for value in ttc_values]
    return {
        "ttc_p10": _quantile(xs, 0.10),
        "ttc_p50": _quantile(xs, 0.50),
        "ttc_violation_rate": float(sum(value < float(threshold) for value in xs) / len(xs)),
    }


def compute_harsh_brake_rate(speed_history: Sequence[Sequence[float]], *, threshold: float = 2.0) -> float:
    total = 0
    harsh = 0
    for row in speed_history:
        speeds = [float(value) for value in row]
        for prev, current in zip(speeds, speeds[1:]):
            total += 1
            if prev - current >= float(threshold):
                harsh += 1
    if total == 0:
        raise ValueError("speed history is required for harsh braking")
    return float(harsh / total)


def compute_phase_switch_rate(phase_sequence: Sequence[int]) -> float:
    if len(phase_sequence) < 2:
        raise ValueError("phase/action log is required for stability metrics")
    switches = sum(int(a != b) for a, b in zip(phase_sequence, phase_sequence[1:]))
    return float(switches / (len(phase_sequence) - 1))


def compute_oscillation_index(queue_volatility: Sequence[float]) -> float:
    if len(queue_volatility) < 3:
        raise ValueError("queue volatility trace length below minimum")
    diffs = [abs(float(b) - float(a)) for a, b in zip(queue_volatility, queue_volatility[1:])]
    return float(sum(diffs) / len(diffs))


def compute_paper_traffic_metrics(trace: dict) -> dict[str, float]:
    waiting_times = [float(value) for value in trace.get("waiting_times") or []]
    if not waiting_times:
        raise ValueError("waiting-time observations are required")
    metrics = {
        "average_travel_time": _mean(trace.get("completed_travel_times") or [], "completed travel time"),
        "average_waiting_time": _mean(waiting_times, "waiting-time"),
        "average_queue_length": _mean(trace.get("queue_lengths") or [], "queue"),
        "waiting_time_imbalance": gini(waiting_times),
        "harsh_brake_rate": compute_harsh_brake_rate(trace.get("speed_history") or []),
        "phase_switch_rate": compute_phase_switch_rate(trace.get("phase_sequence") or []),
        "oscillation_index": compute_oscillation_index(trace.get("queue_volatility") or []),
    }
    metrics.update(compute_ttc_metrics(trace.get("ttc_values") or []))
    return metrics
