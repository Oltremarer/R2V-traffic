from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _action_entropy(action_counts: Counter, total_actions: int) -> float:
    if total_actions <= 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        prob = float(count) / float(total_actions)
        entropy -= prob * math.log(prob)
    return float(entropy)


def summarize_action_distribution_guard(
    action_counts: Counter,
    intersection_action_counts: list[Counter],
    *,
    max_global_single_action_rate: float = 0.95,
    max_intersection_single_action_rate: float = 0.98,
    context_label: str = "formal PPO rollout",
) -> dict[str, Any]:
    total_actions = int(sum(action_counts.values()))
    if total_actions <= 0:
        raise ValueError(f"{context_label} produced no actions")
    most_common_action, most_common_count = action_counts.most_common(1)[0]
    single_action_rate = float(most_common_count / total_actions)
    per_intersection_rates: list[float] = []
    collapsed_intersections: list[int] = []
    for idx, counter in enumerate(intersection_action_counts):
        total = int(sum(counter.values()))
        if total <= 0:
            per_intersection_rates.append(0.0)
            continue
        rate = float(counter.most_common(1)[0][1] / total)
        per_intersection_rates.append(rate)
        if rate > max_intersection_single_action_rate:
            collapsed_intersections.append(idx)
    summary = {
        "action_guard_pass": True,
        "total_actions": total_actions,
        "action_histogram": {str(key): int(value) for key, value in sorted(action_counts.items())},
        "unique_actions_used": int(len(action_counts)),
        "action_entropy": _action_entropy(action_counts, total_actions),
        "global_dominant_action": int(most_common_action),
        "global_single_action_rate": single_action_rate,
        "max_single_action_rate_allowed": float(max_global_single_action_rate),
        "max_global_single_action_rate_allowed": float(max_global_single_action_rate),
        "max_intersection_single_action_rate_allowed": float(max_intersection_single_action_rate),
        "per_intersection_single_action_rate_max": float(max(per_intersection_rates) if per_intersection_rates else 0.0),
        "collapsed_intersections": collapsed_intersections,
    }
    if single_action_rate > max_global_single_action_rate:
        raise ValueError(
            f"{context_label} action collapse: global single-action rate "
            f"{single_action_rate:.4f} exceeds {max_global_single_action_rate:.4f}"
        )
    if collapsed_intersections:
        raise ValueError(
            f"{context_label} action collapse: per-intersection single-action rate "
            f"{summary['per_intersection_single_action_rate_max']:.4f} exceeds "
            f"{max_intersection_single_action_rate:.4f}"
        )
    return summary


def summarize_paper_final_action_diversity(
    action_counts: Counter,
    action_sequence_hashes: list[str],
    *,
    method_kind: str,
    city: str,
    method: str,
    preference_id: str,
    max_single_action_rate: float = 0.95,
) -> dict[str, Any]:
    if method_kind not in {"learned_policy", "reference_policy"}:
        raise ValueError(f"unknown paper-final method_kind: {method_kind}")
    total_actions = int(sum(action_counts.values()))
    if total_actions <= 0:
        raise ValueError("paper-final action diagnostics produced no actions")
    if not action_sequence_hashes:
        raise ValueError("paper-final action diagnostics require action sequence hashes")
    repeated_hash_count = len(action_sequence_hashes) - len(set(action_sequence_hashes))
    deterministic_reference_repeat = bool(method_kind == "reference_policy" and repeated_hash_count > 0)
    if method_kind == "learned_policy" and len(set(action_sequence_hashes)) == 1 and len(action_sequence_hashes) > 1:
        raise ValueError(
            f"learned policy action-sequence collapse for {city}/{method}/{preference_id}: "
            "all sequence hashes are identical"
        )
    dominant_action, dominant_count = action_counts.most_common(1)[0]
    single_action_rate = float(dominant_count / total_actions)
    if method_kind == "learned_policy" and single_action_rate > float(max_single_action_rate):
        raise ValueError(
            f"learned policy action collapse for {city}/{method}/{preference_id}: "
            f"single-action rate {single_action_rate:.4f} exceeds {float(max_single_action_rate):.4f}"
        )
    return {
        "action_guard_pass": True,
        "city": city,
        "method": method,
        "method_kind": method_kind,
        "preference_id": preference_id,
        "total_actions": total_actions,
        "unique_actions_used": int(len(action_counts)),
        "action_histogram": {str(key): int(value) for key, value in sorted(action_counts.items())},
        "global_dominant_action": int(dominant_action),
        "global_single_action_rate": single_action_rate,
        "max_single_action_rate_allowed": float(max_single_action_rate),
        "sequence_hash_count": int(len(action_sequence_hashes)),
        "unique_sequence_hash_count": int(len(set(action_sequence_hashes))),
        "repeated_sequence_hash_count": int(repeated_hash_count),
        "deterministic_reference_repeat": deterministic_reference_repeat,
    }
