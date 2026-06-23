from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bootstrap_mean_ci(
    values: Sequence[float | int | bool],
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap_mean_ci requires at least one value")
    rng = random.Random(seed)
    xs = [float(value) for value in values]
    means = []
    for _ in range(int(n_boot)):
        sample_sum = 0.0
        for _ in xs:
            sample_sum += xs[rng.randrange(len(xs))]
        means.append(sample_sum / len(xs))
    means.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(sum(xs) / len(xs)),
        "low": _quantile(means, alpha),
        "high": _quantile(means, 1.0 - alpha),
        "n": len(xs),
        "n_boot": int(n_boot),
        "confidence": float(confidence),
    }


def bootstrap_metric_ci(
    items: Sequence[T],
    metric_fn: Callable[[list[T]], float],
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if not items:
        raise ValueError("bootstrap_metric_ci requires at least one item")
    rng = random.Random(seed)
    boot_values = []
    items = list(items)
    for _ in range(int(n_boot)):
        sample = [items[rng.randrange(len(items))] for _ in items]
        boot_values.append(float(metric_fn(sample)))
    boot_values.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(metric_fn(list(items))),
        "low": _quantile(boot_values, alpha),
        "high": _quantile(boot_values, 1.0 - alpha),
        "n": len(items),
        "n_boot": int(n_boot),
        "confidence": float(confidence),
    }
