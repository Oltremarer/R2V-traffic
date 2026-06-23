from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any


REQUIRED_CITY_TRAFFIC = {
    "jinan": "anon_3_4_jinan_real.json",
    "hangzhou": "anon_4_4_hangzhou_real.json",
    "newyork_28x7": "anon_28_7_newyork_real_double.json",
}
PAPER_FINAL_SEEDS = (0, 1, 2, 3, 4)
REQUIRED_PAPER_BASELINES = (
    "Random",
    "FixedTime",
    "MaxPressure",
    "PressLight",
    "MPLight",
    "CoLight",
    "Advanced-Co",
    "C2T-scalar",
    "Cond-Scalar-RL",
    "Weighted-RL",
    "VectorQ-PPO",
)
BASELINE_STATUSES = {"implemented", "reference_only", "missing_blocker"}
REQUIRED_PREFERENCE_TEMPLATES = {
    "efficiency_focused": [0.7, 0.1, 0.1, 0.1],
    "safety_focused": [0.1, 0.7, 0.1, 0.1],
    "fairness_focused": [0.1, 0.1, 0.7, 0.1],
    "stability_focused": [0.1, 0.1, 0.1, 0.7],
    "balanced": [0.25, 0.25, 0.25, 0.25],
}
REQUIRED_METRIC_FAMILIES = (
    "efficiency",
    "safety",
    "fairness",
    "stability",
    "representation",
    "pareto",
    "controllability",
    "generalization",
)
REPRESENTATION_SOURCE_METRICS = ("obj_acc", "pref_acc", "rev_acc", "dpr")
FORBIDDEN_FINAL_ACTIONS = {
    "ranking",
    "plot",
    "paper_table",
    "paper_result_text",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_weights(values: Any, *, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{label} must be a 4-element list")
    weights = [float(value) for value in values]
    if any(value < 0.0 for value in weights):
        raise ValueError(f"{label} contains negative preference weight")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"{label} preference weights must sum to 1")
    return weights


def _validate_cities(packet: dict[str, Any]) -> None:
    rows = packet.get("cities")
    _require(isinstance(rows, list), "cities must be a list")
    observed = {str(row.get("scenario")): str(row.get("traffic_file")) for row in rows if isinstance(row, dict)}
    for city, traffic in REQUIRED_CITY_TRAFFIC.items():
        if city not in observed:
            raise ValueError(f"missing required city: {city}")
        if observed[city] != traffic:
            raise ValueError(f"traffic file mismatch for {city}: {observed[city]}")


def _validate_seeds(packet: dict[str, Any]) -> None:
    seeds = tuple(int(seed) for seed in packet.get("main_seed_ids") or ())
    if seeds != PAPER_FINAL_SEEDS:
        raise ValueError(f"main_seed_ids must be {list(PAPER_FINAL_SEEDS)}")
    binding = packet.get("seed_binding") or {}
    for key in ("cityflow_seed", "policy_seed", "model_seed", "reference_policy_seed"):
        if binding.get(key) != "seed_id":
            raise ValueError(f"seed_binding.{key} must bind to seed_id")


def _validate_preferences(packet: dict[str, Any]) -> None:
    templates = packet.get("preference_templates") or {}
    for name, expected in REQUIRED_PREFERENCE_TEMPLATES.items():
        if name not in templates:
            raise ValueError(f"missing preference template: {name}")
        observed = _normalize_weights(templates[name], label=f"preference_templates.{name}")
        if observed != expected:
            raise ValueError(f"preference template mismatch: {name}")


def _validate_paper_scale_ppo(packet: dict[str, Any]) -> None:
    model = packet.get("model") or {}
    ppo = packet.get("ppo") or {}
    checks = {
        "model.hidden_dim": int(model.get("hidden_dim", -1)) == 256,
        "ppo.hidden_dim": int(ppo.get("hidden_dim", -1)) == 256,
        "ppo.minibatch_size": int(ppo.get("minibatch_size", -1)) == 2048,
        "ppo.update_epochs": int(ppo.get("update_epochs", -1)) == 10,
        "ppo.rollout_horizon_sim_steps": int(ppo.get("rollout_horizon_sim_steps", -1)) == 3600,
        "ppo.total_env_steps_per_seed": int(ppo.get("total_env_steps_per_seed", -1)) == 1_000_000,
    }
    if ppo.get("algorithm_label") != "PPO":
        raise ValueError("paper-scale PPO requires algorithm_label=PPO")
    if abs(float(ppo.get("lr", -1.0)) - 0.0003) > 1e-12:
        raise ValueError("paper-scale PPO requires lr=3e-4")
    if abs(float(ppo.get("clip_eps", -1.0)) - 0.2) > 1e-12:
        raise ValueError("paper-scale PPO requires clip_eps=0.2")
    if abs(float(ppo.get("shaping_warmup_end", -1.0)) - 0.4) > 1e-12:
        raise ValueError("paper-scale PPO requires shaping_warmup_end=0.4")
    if abs(float(ppo.get("shaping_warmup_fraction", -1.0)) - 0.3) > 1e-12:
        raise ValueError("paper-scale PPO requires shaping_warmup_fraction=0.3")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"paper-scale PPO mismatch: {failed}")


def _validate_baselines(packet: dict[str, Any]) -> None:
    rows = packet.get("baselines")
    _require(isinstance(rows, list), "baselines must be a list")
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    for name in REQUIRED_PAPER_BASELINES:
        if name not in by_name:
            raise ValueError(f"missing required baseline: {name}")
        status = by_name[name].get("status")
        if status not in BASELINE_STATUSES:
            raise ValueError(f"unknown baseline status for {name}: {status}")
        if not by_name[name].get("command_family"):
            raise ValueError(f"baseline {name} missing command_family")


def _validate_metric_families(packet: dict[str, Any]) -> None:
    families = packet.get("metric_families")
    _require(isinstance(families, dict), "metric_families must be an object")
    for family in REQUIRED_METRIC_FAMILIES:
        if family not in families:
            raise ValueError(f"missing metric family: {family}")
        row = families[family]
        if row.get("status") not in BASELINE_STATUSES:
            raise ValueError(f"unknown metric family status for {family}: {row.get('status')}")
        if not row.get("metrics"):
            raise ValueError(f"metric family {family} must list metrics")


def _validate_roots_and_forbidden_actions(packet: dict[str, Any]) -> None:
    roots = packet.get("output_roots") or {}
    for key in ("train", "eval", "diagnostics", "preflight"):
        value = str(roots.get(key, ""))
        if not value.startswith("records/paper_final/"):
            raise ValueError(f"output_roots.{key} must be under records/paper_final")
    forbidden = set(packet.get("forbidden_actions") or [])
    missing = sorted(FORBIDDEN_FINAL_ACTIONS - forbidden)
    if missing:
        raise ValueError(f"forbidden_actions missing: {missing}")


def validate_paper_final_manifest(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("packet_type") != "paper_final_experiment_manifest":
        raise ValueError("packet_type must be paper_final_experiment_manifest")
    if packet.get("execution_allowed_now") is not False:
        raise ValueError("execution_allowed_now must be false")
    _validate_cities(packet)
    _validate_seeds(packet)
    _validate_preferences(packet)
    _validate_paper_scale_ppo(packet)
    _validate_baselines(packet)
    _validate_metric_families(packet)
    _validate_roots_and_forbidden_actions(packet)
    return dict(packet)


def paper_final_blockers(packet: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for row in packet.get("baselines") or []:
        if row.get("status") == "missing_blocker":
            reason = row.get("blocker") or "implementation missing"
            blockers.append(f"baseline {row.get('name')}: {reason}")
    for family, row in (packet.get("metric_families") or {}).items():
        if row.get("status") == "missing_blocker":
            reason = row.get("blocker") or "metric source missing"
            blockers.append(f"metric family {family}: {reason}")
    return blockers


def assert_ready_for_final_execution(packet: dict[str, Any]) -> dict[str, Any]:
    validated = validate_paper_final_manifest(packet)
    blockers = paper_final_blockers(validated)
    if blockers:
        raise ValueError(f"final execution blocked: {blockers}")
    return validated


def load_paper_final_manifest(path: str | Path) -> dict[str, Any]:
    return validate_paper_final_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def manifest_with_baseline_registry(
    packet: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(validate_paper_final_manifest(packet))
    registry_missing = sorted(set(REQUIRED_PAPER_BASELINES) - set(registry))
    if registry_missing:
        raise ValueError(f"baseline registry missing required baselines: {registry_missing}")
    baseline_rows = []
    for name in REQUIRED_PAPER_BASELINES:
        row = dict(registry[name])
        baseline_row = {
            "name": name,
            "status": row.get("status"),
            "command_family": row.get("command_family"),
            "city_support": list(row.get("city_support") or []),
        }
        if row.get("blocker"):
            baseline_row["blocker"] = row["blocker"]
        baseline_rows.append(baseline_row)
    updated["baselines"] = baseline_rows
    return validate_paper_final_manifest(updated)


def manifest_with_metric_source_policy(
    packet: dict[str, Any],
    metric_source_policy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(validate_paper_final_manifest(packet))
    representation_ready = all(
        (metric_source_policy.get(metric) or {}).get("status") == "implemented"
        for metric in REPRESENTATION_SOURCE_METRICS
    )
    if representation_ready:
        row = dict(updated["metric_families"]["representation"])
        row["status"] = "implemented"
        row["metrics"] = list(REPRESENTATION_SOURCE_METRICS)
        row.pop("blocker", None)
        updated["metric_families"]["representation"] = row
    return validate_paper_final_manifest(updated)
