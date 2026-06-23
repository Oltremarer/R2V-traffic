from __future__ import annotations

from typing import Any

from pareto.rl.paper_learned_artifact_inventory import learned_artifact_blockers
from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC, REQUIRED_PAPER_BASELINES


ALLOWED_BASELINE_STATUSES = {"implemented", "reference_only", "missing_blocker"}
ALL_PAPER_CITIES = tuple(REQUIRED_CITY_TRAFFIC)


BASELINE_REGISTRY: dict[str, dict[str, Any]] = {
    "Random": {
        "status": "implemented",
        "method_id": "Random",
        "command_family": "run_random.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "FixedTime": {
        "status": "implemented",
        "method_id": "FixedTime",
        "command_family": "run_fixedtime.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "MaxPressure": {
        "status": "implemented",
        "method_id": "MaxPressure",
        "command_family": "reference_eval",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "PressLight": {
        "status": "implemented",
        "method_id": "PressLight",
        "command_family": "run_presslight.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "MPLight": {
        "status": "implemented",
        "method_id": "MPLight",
        "command_family": "run_mplight.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "CoLight": {
        "status": "implemented",
        "method_id": "CoLight",
        "command_family": "run_colight.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "Advanced-Co": {
        "status": "implemented",
        "method_id": "Advanced-Co",
        "command_family": "run_advanced_colight.py",
        "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        "preference_aware": False,
    },
    "C2T-scalar": {
        "status": "missing_blocker",
        "method_id": "C2T-scalar",
        "command_family": "scalar_reward_training",
        "city_support": [],
        "preference_aware": False,
        "blocker": "C2T-scalar command or reviewer-approved exclusion is missing",
    },
    "Cond-Scalar-RL": {
        "status": "missing_blocker",
        "method_id": "film_scalar_potential",
        "command_family": "formal_pilot_runner",
        "city_support": [],
        "preference_aware": True,
        "blocker": "Cond-Scalar-RL all-city paper_final model and objective normalizer hashes are missing",
    },
    "Weighted-RL": {
        "status": "missing_blocker",
        "method_id": "weighted_proxy",
        "command_family": "formal_pilot_runner",
        "city_support": [],
        "preference_aware": True,
        "blocker": "Weighted-RL -> weighted_proxy exact reviewer approval phrase is missing",
    },
    "VectorQ-PPO": {
        "status": "missing_blocker",
        "method_id": "vector_quality_potential",
        "command_family": "formal_pilot_runner",
        "city_support": [],
        "preference_aware": True,
        "blocker": "VectorQ-PPO all-city paper_final model and objective normalizer hashes are missing",
    },
}


def validate_baseline_registry(
    registry: dict[str, dict[str, Any]],
    *,
    require_all_cities_for_implemented: bool = False,
) -> dict[str, dict[str, Any]]:
    missing = sorted(set(REQUIRED_PAPER_BASELINES) - set(registry))
    extra = sorted(set(registry) - set(REQUIRED_PAPER_BASELINES))
    if missing:
        raise ValueError(f"missing paper baselines: {missing}")
    if extra:
        raise ValueError(f"unknown paper baselines: {extra}")
    for name, row in registry.items():
        status = row.get("status")
        if status not in ALLOWED_BASELINE_STATUSES:
            raise ValueError(f"unknown baseline status for {name}: {status}")
        if not row.get("command_family"):
            raise ValueError(f"baseline {name} missing command_family")
        if "env_reward" in {name, row.get("method_id")}:
            raise ValueError("Stage-A env_reward diagnostic is not a paper baseline")
        if require_all_cities_for_implemented and status == "implemented":
            missing_cities = sorted(set(ALL_PAPER_CITIES) - set(row.get("city_support") or []))
            if missing_cities:
                raise ValueError(f"baseline {name} missing city support: {missing_cities}")
    return {name: dict(row) for name, row in registry.items()}


def baseline_blockers(registry: dict[str, dict[str, Any]]) -> list[str]:
    validate_baseline_registry(registry)
    blockers: list[str] = []
    for name, row in registry.items():
        missing_cities = sorted(set(ALL_PAPER_CITIES) - set(row.get("city_support") or []))
        if row.get("status") == "missing_blocker":
            blockers.append(f"{name}: {row.get('blocker') or 'implementation missing'}")
        elif missing_cities:
            blockers.append(f"{name}: missing city support {missing_cities}")
    return blockers


def registry_with_learned_artifact_inventory(
    registry: dict[str, dict[str, Any]],
    learned_artifact_audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    updated = validate_baseline_registry(registry)
    if learned_artifact_audit.get("coverage_status") != "complete" or learned_artifact_blockers(learned_artifact_audit):
        return updated

    for name in ("Cond-Scalar-RL", "VectorQ-PPO"):
        row = dict(updated[name])
        row["status"] = "implemented"
        row["city_support"] = list(ALL_PAPER_CITIES)
        row["command_family"] = row.get("command_family") or "formal_pilot_runner"
        row.pop("blocker", None)
        updated[name] = row
    return validate_baseline_registry(updated)
