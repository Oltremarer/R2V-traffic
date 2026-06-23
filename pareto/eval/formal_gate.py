from __future__ import annotations

from typing import Any


FORMAL_THRESHOLDS = {
    "rev_acc": 0.60,
    "dpr_head": 0.75,
    "dpr_utility": 0.85,
    "pref_acc": 0.70,
    "obj_acc_mean": 0.68,
    "head_leakage_diag_offdiag_gap": 0.15,
}

WIRING_THRESHOLDS = {
    "rev_acc": 0.55,
    "dpr_head": 0.60,
    "dpr_utility": 0.85,
    "pref_acc": 0.70,
}


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, dict):
        value = value.get("mean", default)
    return float(value)


def _ci(metrics: dict[str, Any], key: str) -> tuple[float, float] | None:
    for ci_key in (f"{key}_ci", f"{key}_bootstrap_ci"):
        value = metrics.get(ci_key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
    return None


def _film_clearly_stronger(vector_metrics: dict[str, Any], film_metrics: dict[str, Any] | None) -> bool:
    if not film_metrics:
        return False
    vector_rev = _metric(vector_metrics, "rev_acc")
    film_rev = _metric(film_metrics, "rev_acc")
    vector_ci = _ci(vector_metrics, "rev_acc")
    film_ci = _ci(film_metrics, "rev_acc")
    if film_ci and vector_ci and film_ci[0] > vector_ci[1]:
        return True
    return (film_rev - vector_rev) >= 0.05


def evaluate_formal_gate(
    vector_metrics: dict[str, Any],
    film_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed_reasons: list[str] = []
    for key, threshold in FORMAL_THRESHOLDS.items():
        if _metric(vector_metrics, key) < threshold:
            failed_reasons.append(f"{key} below formal threshold")

    film_stronger = _film_clearly_stronger(vector_metrics, film_metrics)
    if film_stronger:
        failed_reasons.append("film condscalar clearly stronger on rev_acc")

    representation_gate_pass = not failed_reasons
    wiring_failed = [
        f"{key} below wiring threshold"
        for key, threshold in WIRING_THRESHOLDS.items()
        if _metric(vector_metrics, key) < threshold
    ]
    wiring_smoke_allowed = not wiring_failed

    if representation_gate_pass:
        claim_mode = "vector_superiority"
    elif film_stronger:
        claim_mode = "film_scalar_candidate"
    else:
        claim_mode = "diagnostics_only"

    return {
        "representation_gate_pass": representation_gate_pass,
        "ppo_formal_allowed": representation_gate_pass,
        "wiring_smoke_allowed": wiring_smoke_allowed,
        "claim_mode": claim_mode,
        "failed_reasons": failed_reasons,
        "wiring_failed_reasons": wiring_failed,
        "thresholds": {
            "formal": dict(FORMAL_THRESHOLDS),
            "wiring": dict(WIRING_THRESHOLDS),
            "film_rev_point_margin": 0.05,
        },
    }
