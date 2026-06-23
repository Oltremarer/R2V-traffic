from __future__ import annotations

import math
from typing import Sequence


def _points(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    if not points:
        raise ValueError("Pareto point set cannot be empty")
    return [(float(point[0]), float(point[1])) for point in points]


def compute_hypervolume_2d(points: Sequence[Sequence[float]], *, reference: tuple[float, float]) -> float:
    xs = sorted(_points(points), key=lambda item: item[0], reverse=True)
    ref_x, ref_y = map(float, reference)
    hv = 0.0
    best_y = ref_y
    prev_x = ref_x
    for x, y in sorted(xs, key=lambda item: item[0]):
        width = max(x - prev_x, 0.0)
        height = max(y - best_y, 0.0)
        hv += width * height
        best_y = max(best_y, y)
        prev_x = max(prev_x, x)
    return float(hv)


def compute_coverage(points: Sequence[Sequence[float]], *, reference_points: Sequence[Sequence[float]]) -> float:
    point_set = set(_points(points))
    refs = _points(reference_points)
    return float(sum(ref in point_set for ref in refs) / len(refs))


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])


def compute_dominance_violation(points: Sequence[Sequence[float]]) -> int:
    xs = _points(points)
    count = 0
    for i, a in enumerate(xs):
        for j, b in enumerate(xs):
            if i == j:
                continue
            if _dominates(a, b):
                count += 1
    return int(count)


def compute_utility(objective_vector: Sequence[float], preference: Sequence[float]) -> float:
    values = [float(value) for value in objective_vector]
    weights = [float(value) for value in preference]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("preference vector must have positive sum")
    weights = [value / total for value in weights]
    return float(sum(value * weight for value, weight in zip(values, weights)))


def compute_alignment(requested_preference: Sequence[float], realized_response: Sequence[float]) -> float:
    requested = [float(value) for value in requested_preference]
    realized = [float(value) for value in realized_response]
    if len(requested) != len(realized):
        raise ValueError("requested and realized preferences must have same length")
    distance = sum(abs(a - b) for a, b in zip(requested, realized)) / 2.0
    return float(max(0.0, 1.0 - distance))
