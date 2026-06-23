from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from pareto.constants import OBJECTIVE_NAMES
from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.rl.formal_experiment_spec import FormalExperimentSpec
from pareto.rl.formal_reward_adapter import (
    EnvRewardAdapter,
    FiLMScalarPotentialRewardAdapter,
    build_weighted_proxy_adapter,
)
from pareto.train_common import load_checkpoint


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else Path(root) / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_obs(input_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    obs_t = torch.zeros(input_dim, dtype=torch.float32)
    obs_tp1 = torch.linspace(-1.0, 1.0, steps=input_dim, dtype=torch.float32)
    return obs_t, obs_tp1


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_film_model(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(model_dir / "model.pt", device)
    config = checkpoint["config"]
    model = build_conditioned_scalar_model(
        config.get("architecture", "film"),
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        preference_dim=config.get("preference_dim", len(OBJECTIVE_NAMES)),
        film_layers=config.get("film_layers", 2),
        head_layers=config.get("head_layers", 2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def validate_shared_budget(specs: Sequence[FormalExperimentSpec]) -> dict[str, Any]:
    if len(specs) < 2:
        return {"spec_count": len(specs), "shared": True}
    shared_fields = (
        "scenario",
        "traffic_file",
        "cityflow_seed",
        "state_encoder_id",
        "state_encoder_hash",
        "objective_norm_path",
        "objective_normalizer_hash",
        "min_action_time",
        "action_interval_source",
        "forbid_inner_step_direct_call",
        "preference_sampling",
        "train_preferences",
        "eval_preferences",
        "ppo_budget",
    )
    reference = specs[0]
    mismatches: list[dict[str, Any]] = []
    for spec in specs[1:]:
        for field_name in shared_fields:
            expected = getattr(reference, field_name)
            actual = getattr(spec, field_name)
            if actual != expected:
                mismatches.append(
                    {
                        "field": field_name,
                        "expected": expected,
                        "actual": actual,
                        "reward_adapter": spec.reward_adapter,
                    }
                )
    if mismatches:
        first = mismatches[0]
        raise ValueError(f"{first['field']} mismatch across formal specs")
    return {"spec_count": len(specs), "shared_fields": list(shared_fields), "shared": True}


def _check_stage_validation(specs: Sequence[FormalExperimentSpec]) -> PreflightCheck:
    details: dict[str, Any] = {"spec_count": len(specs)}
    for spec in specs:
        spec.validate_for_stage("preflight")
    return PreflightCheck("stage_validation", True, details)


def _check_paths_and_hashes(specs: Sequence[FormalExperimentSpec], root: str | Path) -> PreflightCheck:
    details: dict[str, Any] = {"artifacts": []}
    for spec in specs:
        objective_norm = _resolve(root, spec.objective_norm_path)
        gate = _resolve(root, spec.formal_gate_decision_path)
        if objective_norm is None or not objective_norm.exists():
            raise FileNotFoundError(f"objective_norm_path does not exist: {spec.objective_norm_path}")
        if gate is None or not gate.exists():
            raise FileNotFoundError(f"formal_gate_decision_path does not exist: {spec.formal_gate_decision_path}")
        artifact = {
            "reward_adapter": spec.reward_adapter,
            "objective_norm_path": str(objective_norm),
            "formal_gate_decision_path": str(gate),
        }
        if spec.reward_adapter == "film_scalar_potential":
            film_dir = _resolve(root, spec.film_model_dir)
            if film_dir is None or not film_dir.exists():
                raise FileNotFoundError(f"film_model_dir does not exist: {spec.film_model_dir}")
            model_path = film_dir / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"FiLM model checkpoint does not exist: {model_path}")
            actual_hash = _sha256(model_path)
            if actual_hash != spec.film_model_hash:
                raise ValueError(f"film_model_hash mismatch: expected {spec.film_model_hash}, got {actual_hash}")
            selection_report = _resolve(root, spec.film_model_selection_report)
            if selection_report is not None and not selection_report.exists():
                raise FileNotFoundError(f"film_model_selection_report does not exist: {selection_report}")
            artifact.update(
                {
                    "film_model_dir": str(film_dir),
                    "film_model_hash": actual_hash,
                    "film_model_selection_report": str(selection_report) if selection_report else None,
                }
            )
        details["artifacts"].append(artifact)
    return PreflightCheck("path_and_hash_validation", True, details)


def _check_shared_budget(specs: Sequence[FormalExperimentSpec]) -> PreflightCheck:
    return PreflightCheck("shared_budget", True, validate_shared_budget(specs))


def _check_actor_critic_conditioning(specs: Sequence[FormalExperimentSpec]) -> PreflightCheck:
    offenders = [
        spec.reward_adapter
        for spec in specs
        if not spec.policy_conditioned_on_w or not spec.critic_conditioned_on_w
    ]
    if offenders:
        raise ValueError(f"actor/critic must both condition on w: {offenders}")
    return PreflightCheck(
        "actor_critic_preference_conditioning",
        True,
        {"policy_conditioned_on_w": True, "critic_conditioned_on_w": True},
    )


def _film_specs(specs: Iterable[FormalExperimentSpec]) -> list[FormalExperimentSpec]:
    return [spec for spec in specs if spec.reward_adapter == "film_scalar_potential"]


def _compute_film_rewards(
    spec: FormalExperimentSpec,
    root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    film_dir = _resolve(root, spec.film_model_dir)
    if film_dir is None:
        raise ValueError("film_model_dir is required for FiLM reward check")
    model, config = _load_film_model(film_dir, device)
    obs_t, obs_tp1 = _toy_obs(int(config["input_dim"]))
    objectives = {name: 0.0 for name in OBJECTIVE_NAMES}
    preferences = {
        "efficiency": [1.0, 0.0, 0.0, 0.0],
        "safety": [0.0, 1.0, 0.0, 0.0],
        "fairness": [0.0, 0.0, 1.0, 0.0],
        "stability": [0.0, 0.0, 0.0, 1.0],
        "balanced": [0.25, 0.25, 0.25, 0.25],
    }
    rewards: dict[str, float] = {}
    for name, weights in preferences.items():
        adapter = FiLMScalarPotentialRewardAdapter(
            model,
            weights,
            gamma=spec.potential_gamma,
            device=device,
        )
        reward, debug = adapter.compute(obs_t, obs_tp1, objectives, objectives, done=False)
        values = [reward] + [float(value) for key, value in debug.items() if isinstance(value, (int, float))]
        if not all(torch.isfinite(torch.tensor(values))):
            raise ValueError(f"non-finite FiLM reward components for preference {name}")
        rewards[name] = float(reward)
    max_delta = max(rewards.values()) - min(rewards.values())
    return {"rewards": rewards, "max_delta": float(max_delta), "input_dim": int(config["input_dim"])}


def _preference_templates() -> dict[str, list[float]]:
    return {
        "efficiency": [1.0, 0.0, 0.0, 0.0],
        "safety": [0.0, 1.0, 0.0, 0.0],
        "fairness": [0.0, 0.0, 1.0, 0.0],
        "stability": [0.0, 0.0, 0.0, 1.0],
        "balanced": [0.25, 0.25, 0.25, 0.25],
    }


def _linked_record_pairs(records_path: str | Path, limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = _read_jsonl(records_path)
    by_id = {row["sample_id"]: row for row in rows if row.get("sample_id")}
    linked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        next_id = row.get("next_sample_id")
        if not next_id or next_id not in by_id:
            continue
        linked.append((row, by_id[next_id]))
        if len(linked) >= limit:
            break
    if not linked:
        raise ValueError(f"no linked real records found in {records_path}")
    return linked


def _check_real_record_film_reward_sensitivity(
    specs: Sequence[FormalExperimentSpec],
    root: str | Path,
    device: torch.device,
    records_path: str | Path,
    num_records: int,
    min_mean_delta: float,
    min_max_delta: float,
) -> PreflightCheck:
    film = _film_specs(specs)
    if not film:
        return PreflightCheck("real_record_film_reward_sensitivity", True, {"skipped": True})
    resolved_records = _resolve(root, str(records_path))
    if resolved_records is None or not resolved_records.exists():
        raise FileNotFoundError(f"real records path does not exist: {records_path}")
    spec = film[0]
    film_dir = _resolve(root, spec.film_model_dir)
    if film_dir is None:
        raise ValueError("film_model_dir is required for real-record sensitivity")
    model, config = _load_film_model(film_dir, device)
    input_dim = int(config["input_dim"])
    pairs = _linked_record_pairs(resolved_records, max(1, int(num_records)))
    templates = _preference_templates()
    per_record_delta: list[float] = []
    rewards_by_preference = {name: [] for name in templates}
    objectives = {name: 0.0 for name in OBJECTIVE_NAMES}
    for current, next_record in pairs:
        obs_t = torch.tensor(current["obs_features"], dtype=torch.float32)
        obs_tp1 = torch.tensor(next_record["obs_features"], dtype=torch.float32)
        if len(obs_t) != input_dim or len(obs_tp1) != input_dim:
            raise ValueError(
                f"record feature dim mismatch: expected {input_dim}, got {len(obs_t)} and {len(obs_tp1)}"
            )
        record_rewards: list[float] = []
        for name, weights in templates.items():
            adapter = FiLMScalarPotentialRewardAdapter(
                model,
                weights,
                gamma=spec.potential_gamma,
                device=device,
            )
            reward, debug = adapter.compute(obs_t, obs_tp1, objectives, objectives, done=False)
            values = [reward] + [float(value) for value in debug.values() if isinstance(value, (int, float))]
            if not all(torch.isfinite(torch.tensor(values))):
                raise ValueError(f"non-finite real-record reward component for preference {name}")
            reward_float = float(reward)
            record_rewards.append(reward_float)
            rewards_by_preference[name].append(reward_float)
        per_record_delta.append(max(record_rewards) - min(record_rewards))
    mean_delta = float(sum(per_record_delta) / len(per_record_delta))
    max_delta = float(max(per_record_delta))
    min_delta = float(min(per_record_delta))
    if mean_delta <= min_mean_delta:
        raise ValueError(f"real-record FiLM reward mean_delta too small: {mean_delta}")
    if max_delta <= min_max_delta:
        raise ValueError(f"real-record FiLM reward max_delta too small: {max_delta}")
    return PreflightCheck(
        "real_record_film_reward_sensitivity",
        True,
        {
            "records_source": str(resolved_records),
            "num_records": len(pairs),
            "input_dim": input_dim,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "min_mean_delta": float(min_mean_delta),
            "min_max_delta": float(min_max_delta),
            "preference_reward_means": {
                name: float(sum(values) / len(values)) for name, values in rewards_by_preference.items()
            },
        },
    )


def _check_film_reward_sensitivity(
    specs: Sequence[FormalExperimentSpec],
    root: str | Path,
    device: torch.device,
    min_delta: float,
) -> PreflightCheck:
    film = _film_specs(specs)
    if not film:
        return PreflightCheck("film_reward_sensitivity", True, {"skipped": True})
    details = _compute_film_rewards(film[0], root, device)
    if details["max_delta"] <= min_delta:
        raise ValueError(f"FiLM reward is insensitive to w: max_delta={details['max_delta']}")
    details["min_delta"] = min_delta
    return PreflightCheck("film_reward_sensitivity", True, details)


def _check_reward_components_finite(
    specs: Sequence[FormalExperimentSpec],
    root: str | Path,
    device: torch.device,
) -> PreflightCheck:
    obs_t, obs_tp1 = _toy_obs(6)
    objectives_t = {name: float(idx) / 10.0 for idx, name in enumerate(OBJECTIVE_NAMES)}
    objectives_tp1 = {name: float(idx + 1) / 10.0 for idx, name in enumerate(OBJECTIVE_NAMES)}
    details: list[dict[str, Any]] = []
    film_model_cache: dict[str, torch.nn.Module] = {}
    for spec in specs:
        if spec.reward_adapter == "film_scalar_potential":
            film_dir = _resolve(root, spec.film_model_dir)
            if film_dir is None:
                raise ValueError("film_model_dir is required for FiLM reward finite check")
            model, config = _load_film_model(film_dir, device)
            film_model_cache[str(film_dir)] = model
            obs_t_model, obs_tp1_model = _toy_obs(int(config["input_dim"]))
            adapter = FiLMScalarPotentialRewardAdapter(
                model,
                [0.25, 0.25, 0.25, 0.25],
                gamma=spec.potential_gamma,
                device=device,
            )
            reward, debug = adapter.compute(obs_t_model, obs_tp1_model, objectives_t, objectives_tp1, done=False)
        elif spec.reward_adapter == "weighted_proxy":
            adapter = build_weighted_proxy_adapter([0.25, 0.25, 0.25, 0.25])
            reward, debug = adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done=False)
        elif spec.reward_adapter == "env_reward":
            adapter = EnvRewardAdapter()
            reward, debug = adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done=False, env_reward=1.25)
        else:
            raise ValueError(f"preflight reward check does not support adapter {spec.reward_adapter}")
        numeric_values = [reward] + [float(value) for value in debug.values() if isinstance(value, (int, float))]
        if not all(torch.isfinite(torch.tensor(numeric_values))):
            raise ValueError(f"non-finite reward component for adapter {spec.reward_adapter}")
        details.append({"reward_adapter": spec.reward_adapter, "reward": float(reward), "debug": debug})
    del film_model_cache
    return PreflightCheck("reward_components_finite", True, {"adapters": details})


def run_preflight_checks(
    specs: Sequence[FormalExperimentSpec],
    root: str | Path = ".",
    device: str | torch.device = "cpu",
    min_film_reward_delta: float = 1e-7,
    real_records_path: str | Path | None = None,
    num_sensitivity_records: int = 64,
    min_real_mean_delta: float = 1e-4,
    min_real_max_delta: float = 1e-3,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checks = [
        _check_stage_validation,
        lambda values: _check_paths_and_hashes(values, root),
        _check_shared_budget,
        _check_actor_critic_conditioning,
        lambda values: _check_film_reward_sensitivity(values, root, torch_device, min_film_reward_delta),
        lambda values: _check_reward_components_finite(values, root, torch_device),
    ]
    if real_records_path is not None:
        checks.append(
            lambda values: _check_real_record_film_reward_sensitivity(
                values,
                root,
                torch_device,
                real_records_path,
                num_sensitivity_records,
                min_real_mean_delta,
                min_real_max_delta,
            )
        )
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for check_fn in checks:
        try:
            result = check_fn(specs)
        except Exception as exc:  # noqa: BLE001 - report all preflight failures as structured output.
            name = getattr(check_fn, "__name__", "preflight_check")
            result = PreflightCheck(name, False, {"error": str(exc), "error_type": type(exc).__name__})
            failures.append(str(exc))
        results.append(result.to_dict())
    return {
        "passed": not failures,
        "checks_only": True,
        "env_rollout": False,
        "ppo_training": False,
        "policy_update": False,
        "performance_claim": False,
        "formal_experiment_allowed": False,
        "spec_count": len(specs),
        "reward_adapters": [spec.reward_adapter for spec in specs],
        "failures": failures,
        "checks": results,
    }
