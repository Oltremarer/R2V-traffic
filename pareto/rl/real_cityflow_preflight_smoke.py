#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import append_jsonl, write_json
from pareto.common.scenario import build_llmlight_env_config
from pareto.constants import OBJECTIVE_NAMES
from pareto.data.normalization import RobustObjectiveNormalizer
from pareto.data.objectives import compute_objectives_from_snapshot
from pareto.data.snapshot import capture_snapshot
from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.rl.env_reward_source import ensure_nonzero_env_reward_info, select_cityflow_env_rewards
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig, load_formal_ppo_dryrun_config
from pareto.rl.formal_reward_adapter import EnvRewardAdapter, FiLMScalarPotentialRewardAdapter, build_weighted_proxy_adapter
from pareto.rl.ppo_actor_critic import PreferenceConditionedActorCritic
from pareto.rl.preference_sampler import EpisodeFixedPreferenceSampler
from pareto.rl.state_encoder import ParetoStateEncoder
from pareto.train_common import load_checkpoint


EXPECTED_HYBRID_V1_FEATURE_HASH = "4d1c2b4e276043ac"
LEARNING_ARTIFACT_NAMES = {
    "checkpoint_last.pt",
    "training_checkpoint_last.pt",
    "loss_debug.jsonl",
}
REWARD_READINESS_PREFERENCES = ("efficiency", "safety", "fairness", "stability", "balanced")


class SmokeScalarQualityModel(torch.nn.Module):
    def forward(self, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        obs_score = obs.float().mean(dim=-1, keepdim=True)
        w_score = w.float().matmul(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=w.device)).reshape(-1, 1)
        return obs_score + w_score


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(name: str, actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return False
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")
    return True


def _normalizer_hash_from_file(path: str | Path) -> str | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value = payload.get("hash")
    return str(value) if value is not None else None


def validate_real_cityflow_preflight_limits(
    real_env_preflight: bool,
    episodes: int,
    max_decision_steps_per_episode: int,
    reward_readiness: bool = False,
) -> None:
    if not real_env_preflight:
        raise ValueError("real CityFlow preflight requires --real_env_preflight")
    if reward_readiness:
        if int(episodes) != len(REWARD_READINESS_PREFERENCES):
            raise ValueError("episodes must be exactly 5 for real CityFlow reward readiness")
        if int(max_decision_steps_per_episode) != 1:
            raise ValueError("max_decision_steps_per_episode must be exactly 1 for real CityFlow reward readiness")
        return
    if int(episodes) != 1:
        raise ValueError("episodes must be exactly 1 for tiny real CityFlow preflight")
    if int(max_decision_steps_per_episode) <= 0 or int(max_decision_steps_per_episode) > 3:
        raise ValueError("max_decision_steps_per_episode must be in [1, 3] for tiny real CityFlow preflight")


def validate_real_cityflow_preflight_artifacts(
    *,
    method: str,
    reward_readiness: bool,
    objective_normalizer: str | Path | None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
) -> None:
    if not reward_readiness:
        return
    if objective_normalizer is None:
        raise ValueError("real reward readiness requires a train-only objective normalizer")
    objective_path = Path(objective_normalizer)
    if not objective_path.is_file():
        raise ValueError(f"objective normalizer not found: {objective_path}")
    if objective_normalizer_hash is None:
        raise ValueError("real reward readiness requires --objective_normalizer_hash")
    validate_hash(
        "objective_normalizer_hash",
        _normalizer_hash_from_file(objective_path),
        objective_normalizer_hash,
    )
    if method == "film_scalar_potential":
        if film_model_dir is None:
            raise ValueError("real reward readiness for film_scalar_potential requires a trained FiLM checkpoint")
        model_path = Path(film_model_dir) / "model.pt"
        if not model_path.is_file():
            raise ValueError(f"trained FiLM checkpoint not found: {model_path}")
        if film_model_hash is None:
            raise ValueError("real reward readiness for film_scalar_potential requires --film_model_hash")
        validate_hash("film_model_hash", sha256_file(model_path), film_model_hash)


def assert_no_learning_artifacts(run_dir: str | Path) -> None:
    root = Path(run_dir)
    if not root.exists():
        return
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in LEARNING_ARTIFACT_NAMES:
            found.append(str(path.relative_to(root)))
    if found:
        raise ValueError(f"learning artifacts are forbidden in real CityFlow preflight: {sorted(found)}")
    assert_no_forbidden_performance_artifacts(root)


def build_real_cityflow_preflight_metadata(
    *,
    method: str,
    scenario: str,
    traffic_file: str,
    cityflow_seed: int,
    policy_seed: int,
    model_seed: int,
    episodes: int,
    max_decision_steps_per_episode: int,
    min_action_time: int,
    state_encoder_id: str,
    state_encoder_hash: str,
    expected_feature_hash: str,
    objective_normalizer_path: str | None,
    objective_normalizer_hash: str | None,
    objective_normalizer_hash_expected: str | None,
    objective_normalizer_hash_verified: bool,
    objective_normalizer_file_sha256: str | None,
    objective_normalizer_loaded: bool,
    film_model_dir: str | None,
    film_model_hash: str | None,
    film_model_hash_expected: str | None,
    film_model_hash_verified: bool,
    film_model_source: str | None,
    film_model_loaded: bool,
    reward_readiness: bool,
    preference_name: str,
    w: Sequence[float],
    preference_coverage: Sequence[str],
) -> dict[str, Any]:
    return {
        "real_env_preflight": True,
        "tiny_cityflow_preflight": True,
        "real_reward_readiness": bool(reward_readiness),
        "reward_readiness_no_learning": bool(reward_readiness),
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "mock_env": False,
        "real_env_rollout": True,
        "cityflow_env_constructed": True,
        "cityflow_step_called": True,
        "ppo_training": False,
        "policy_update": False,
        "optimizer_step": False,
        "method": method,
        "scenario": scenario,
        "traffic_file": traffic_file,
        "cityflow_seed": int(cityflow_seed),
        "policy_seed": int(policy_seed),
        "model_seed": int(model_seed),
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "min_action_time": int(min_action_time),
        "sim_seconds_per_method": int(episodes * max_decision_steps_per_episode * min_action_time),
        "state_encoder_id": state_encoder_id,
        "state_encoder_hash": state_encoder_hash,
        "expected_feature_hash": expected_feature_hash,
        "objective_normalizer_path": objective_normalizer_path,
        "objective_normalizer_hash": objective_normalizer_hash,
        "objective_normalizer_hash_expected": objective_normalizer_hash_expected,
        "objective_normalizer_hash_verified": bool(objective_normalizer_hash_verified),
        "objective_normalizer_file_sha256": objective_normalizer_file_sha256,
        "objective_normalizer_loaded": bool(objective_normalizer_loaded),
        "film_model_dir": film_model_dir,
        "film_model_hash": film_model_hash,
        "film_model_hash_expected": film_model_hash_expected,
        "film_model_hash_verified": bool(film_model_hash_verified),
        "film_model_source": film_model_source,
        "film_model_loaded": bool(film_model_loaded),
        "preference_name": preference_name,
        "w": [float(value) for value in w],
        "preference_coverage": list(preference_coverage),
        "preference_coverage_count": len(set(preference_coverage)),
    }


def _clean_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "metadata.json",
        "status.json",
        "reward_components.jsonl",
        "action_debug.jsonl",
        "preflight_metrics.jsonl",
        "command.txt",
        "stdout.log",
        "stderr.log",
        "REAL_PREFLIGHT_DONE",
        "REAL_REWARD_READINESS_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()


def _prepare_work_dir(dic_agent_conf: dict[str, Any], dic_traffic_env_conf: dict[str, Any], dic_path: dict[str, str]) -> None:
    work_dir = Path(dic_path["PATH_TO_WORK_DIRECTORY"])
    model_dir = Path(dic_path["PATH_TO_MODEL"])
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "agent.conf").write_text(json.dumps(dic_agent_conf, indent=2, sort_keys=True), encoding="utf-8")
    (work_dir / "traffic_env.conf").write_text(json.dumps(dic_traffic_env_conf, indent=2, sort_keys=True), encoding="utf-8")
    data_dir = Path(dic_path["PATH_TO_DATA"])
    shutil.copy(data_dir / dic_traffic_env_conf["TRAFFIC_FILE"], work_dir / dic_traffic_env_conf["TRAFFIC_FILE"])
    shutil.copy(data_dir / dic_traffic_env_conf["ROADNET_FILE"], work_dir / dic_traffic_env_conf["ROADNET_FILE"])


def _normalizer_hash(normalizer: RobustObjectiveNormalizer | None) -> str | None:
    if normalizer is None:
        return None
    return normalizer.to_dict().get("hash")


def _load_normalizer(path: str | Path | None) -> RobustObjectiveNormalizer | None:
    if path is None:
        return None
    return RobustObjectiveNormalizer.load(path)


def _load_film_scalar_model(model_dir: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(Path(model_dir) / "model.pt", device)
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
    return model


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"non-finite {name}")


def _assert_finite_number(name: str, value: float) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"non-finite {name}")


def _compute_reward(
    method: str,
    scalar_model: torch.nn.Module,
    w: Sequence[float],
    obs_t: torch.Tensor,
    obs_tp1: torch.Tensor,
    objectives_t: Mapping[str, float],
    objectives_tp1: Mapping[str, float],
    done: bool,
    env_reward: float,
    gamma: float,
    device: torch.device | str = "cpu",
) -> tuple[float, dict[str, Any]]:
    if method == "film_scalar_potential":
        adapter = FiLMScalarPotentialRewardAdapter(scalar_model, w, gamma=gamma, device=device)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if method == "weighted_proxy":
        adapter = build_weighted_proxy_adapter(w)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if method == "env_reward":
        adapter = EnvRewardAdapter()
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done, env_reward=env_reward)
    raise ValueError(f"unsupported preflight method: {method}")


def _encode_snapshots(
    snapshots: Sequence[Any],
    encoder: ParetoStateEncoder,
    expected_obs_dim: int,
    expected_feature_hash: str,
) -> tuple[torch.Tensor, list[str], list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    names_ref: list[str] | None = None
    debug_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        encoded, names, debug = encoder.encode_snapshot(snapshot)
        if len(encoded) != int(expected_obs_dim):
            raise ValueError(f"state_encoder obs_dim mismatch: got {len(encoded)}, expected {expected_obs_dim}")
        if debug["feature_names_hash"] != expected_feature_hash:
            raise ValueError(
                f"state_encoder feature hash mismatch: got {debug['feature_names_hash']}, "
                f"expected {expected_feature_hash}"
            )
        if names_ref is None:
            names_ref = list(names)
        elif names != names_ref:
            raise ValueError("state_encoder feature names changed across intersections")
        features.append(encoded)
        debug_rows.append(debug)
    obs = torch.tensor(np.asarray(features), dtype=torch.float32)
    _assert_finite_tensor("obs_features", obs)
    return obs, names_ref or [], debug_rows


def run_real_cityflow_preflight_smoke(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    real_env_preflight: bool,
    episodes: int = 1,
    max_decision_steps_per_episode: int = 3,
    expected_feature_hash: str = EXPECTED_HYBRID_V1_FEATURE_HASH,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    reward_readiness: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    validate_real_cityflow_preflight_limits(
        real_env_preflight,
        episodes,
        max_decision_steps_per_episode,
        reward_readiness=reward_readiness,
    )
    if method not in config.pilot["methods"]:
        raise ValueError(f"method {method} is not in pilot methods")
    validate_real_cityflow_preflight_artifacts(
        method=method,
        reward_readiness=reward_readiness,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
    )

    random.seed(int(config.pilot.get("policy_seed", 0)))
    np.random.seed(int(config.pilot.get("policy_seed", 0)))
    torch.manual_seed(int(config.pilot.get("model_seed", 0)))
    device_obj = torch.device(device)

    out = Path(out_dir)
    _clean_output_dir(out)
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out / "stdout.log").write_text("", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")

    min_action_time = 30
    run_counts = int(max_decision_steps_per_episode * min_action_time)
    dic_agent_conf, dic_traffic_env_conf, dic_path = build_llmlight_env_config(
        scenario=str(config.pilot["scenario"]),
        traffic_file=str(config.pilot["traffic_file"]),
        seed=int(config.pilot.get("cityflow_seed", 0)),
        run_counts=run_counts,
        min_action_time=min_action_time,
        model_name="Random",
        work_dir=str(out / "llmlight_work"),
    )
    reward_info_debug = ensure_nonzero_env_reward_info(dic_traffic_env_conf, enable=method == "env_reward")
    _prepare_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)

    from utils.cityflow_env import CityFlowEnv

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
    )
    normalizer = _load_normalizer(objective_normalizer)
    normalizer_actual_hash = _normalizer_hash(normalizer)
    normalizer_file_sha256 = sha256_file(objective_normalizer) if objective_normalizer is not None else None
    normalizer_hash_verified = validate_hash(
        "objective_normalizer_hash",
        normalizer_actual_hash,
        objective_normalizer_hash,
    )
    encoder = ParetoStateEncoder("hybrid_v1")
    sampler = EpisodeFixedPreferenceSampler()
    film_model_loaded = False
    film_model_source: str | None = None
    film_model_actual_hash: str | None = None
    film_model_hash_verified = False
    if method == "film_scalar_potential" and film_model_dir is not None:
        film_model_actual_hash = sha256_file(Path(film_model_dir) / "model.pt")
        film_model_hash_verified = validate_hash("film_model_hash", film_model_actual_hash, film_model_hash)
        scalar_model = _load_film_scalar_model(film_model_dir, device_obj)
        film_model_loaded = True
        film_model_source = "trained_film_checkpoint"
    else:
        scalar_model = SmokeScalarQualityModel()
        if method == "film_scalar_potential":
            film_model_source = "smoke_scalar_quality_model"
    policy = PreferenceConditionedActorCritic(
        obs_dim=int(config.model["obs_dim"]),
        preference_dim=int(config.model["preference_dim"]),
        action_dim=int(config.model["action_dim"]),
        hidden_dim=int(config.model["hidden_dim"]),
    ).to(device_obj)
    policy.eval()

    total_steps = 0
    reward_row_count = 0
    observed_feature_hash: str | None = None
    observed_obs_dim: int | None = None
    observed_preferences: list[str] = []
    first_preference_name: str | None = None
    first_w_tuple: tuple[float, ...] | None = None

    for episode in range(int(episodes)):
        env.reset()
        prev_snapshots: list[Any | None] = [None for _ in env.list_intersection]
        prev_actions: list[int | None] = [None for _ in env.list_intersection]
        preference_name, w_tuple = sampler.preference_for_episode(episode)
        if first_preference_name is None:
            first_preference_name = preference_name
            first_w_tuple = tuple(w_tuple)
        if preference_name not in observed_preferences:
            observed_preferences.append(preference_name)
        w_tensor = torch.tensor(w_tuple, dtype=torch.float32, device=device_obj)
        for step in range(int(max_decision_steps_per_episode)):
            snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_t, names, debug_rows = _encode_snapshots(
                snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=expected_feature_hash,
            )
            observed_feature_hash = debug_rows[0]["feature_names_hash"]
            observed_obs_dim = int(obs_t.shape[-1])
            w_batch = w_tensor.reshape(1, -1).expand(obs_t.shape[0], -1)
            with torch.no_grad():
                actions, log_probs, values = policy.act(obs_t.to(device_obj), w_batch)
            _assert_finite_tensor("action_log_probs", log_probs.detach().cpu())
            _assert_finite_tensor("policy_values", values.detach().cpu())
            action_list = [int(value) for value in actions.detach().cpu().tolist()]
            if len(action_list) != len(env.list_intersection):
                raise ValueError("action_list length does not match num intersections")
            if any(action < 0 or action >= int(config.model["action_dim"]) for action in action_list):
                raise ValueError("action id out of range")
            sim_time_before = int(env.get_current_time())
            _, final_env_rewards, done, average_env_rewards = env.step(action_list)
            env_rewards, reward_source_debug = select_cityflow_env_rewards(final_env_rewards, average_env_rewards)
            sim_time_after = int(env.get_current_time())
            next_snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_tp1, _, _ = _encode_snapshots(
                next_snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=expected_feature_hash,
            )

            step_rewards: list[float] = []
            reward_rows: list[dict[str, Any]] = []
            for idx, snapshot in enumerate(snapshots):
                objective_t = compute_objectives_from_snapshot(
                    snapshot,
                    prev_snapshot=prev_snapshots[idx],
                    action=action_list[idx],
                    prev_action=prev_actions[idx],
                    normalizer=normalizer,
                )
                objective_tp1 = compute_objectives_from_snapshot(
                    next_snapshots[idx],
                    prev_snapshot=snapshot,
                    action=action_list[idx],
                    prev_action=prev_actions[idx],
                    normalizer=normalizer,
                )
                env_reward_i = float(env_rewards[idx]) if isinstance(env_rewards, (list, tuple, np.ndarray)) else float(env_rewards)
                reward, debug = _compute_reward(
                    method=method,
                    scalar_model=scalar_model,
                    w=w_tuple,
                    obs_t=obs_t[idx].cpu(),
                    obs_tp1=obs_tp1[idx].cpu(),
                    objectives_t=objective_t.norm,
                    objectives_tp1=objective_tp1.norm,
                    done=bool(done),
                    env_reward=env_reward_i,
                    gamma=float(config.ppo["gamma"]),
                    device=device_obj,
                )
                _assert_finite_number("total_reward", reward)
                step_rewards.append(float(reward))
                reward_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "intersection_idx": idx,
                        "preference_name": preference_name,
                        "w": [float(value) for value in w_tuple],
                        "action": action_list[idx],
                        "env_reward": env_reward_i,
                        "env_reward_source": reward_source_debug["env_reward_step_return_source"],
                        "total_reward": float(reward),
                        "objective_valid_mask_t": objective_t.valid_mask,
                        "objective_valid_mask_tp1": objective_tp1.valid_mask,
                        **reward_info_debug,
                        **debug,
                    }
                )
                prev_snapshots[idx] = snapshot
                prev_actions[idx] = action_list[idx]
            append_jsonl(out / "reward_components.jsonl", reward_rows)
            reward_row_count += len(reward_rows)
            append_jsonl(
                out / "action_debug.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "action_list_len": len(action_list),
                        "num_intersections": len(env.list_intersection),
                        "action_histogram": dict(Counter(action_list)),
                        "obs_dim": observed_obs_dim,
                        "feature_names_hash": observed_feature_hash,
                        "feature_name_count": len(names),
                        "finite_obs": bool(torch.isfinite(obs_t).all().item() and torch.isfinite(obs_tp1).all().item()),
                        "sim_time_before": sim_time_before,
                        "sim_time_after": sim_time_after,
                        "min_action_time": min_action_time,
                        **reward_source_debug,
                        **reward_info_debug,
                    }
                ],
            )
            append_jsonl(
                out / "preflight_metrics.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time": sim_time_after,
                        "preference_name": preference_name,
                        "reward_mean": float(np.mean(step_rewards)),
                        "reward_min": float(np.min(step_rewards)),
                        "reward_max": float(np.max(step_rewards)),
                        "row_count": len(reward_rows),
                    }
                ],
            )
            total_steps += 1
            if done:
                break
    env.batch_log_2()
    if reward_readiness and set(observed_preferences) != set(REWARD_READINESS_PREFERENCES):
        raise ValueError(f"reward readiness did not cover required preferences: {observed_preferences}")

    metadata = build_real_cityflow_preflight_metadata(
        method=method,
        scenario=str(config.pilot["scenario"]),
        traffic_file=str(config.pilot["traffic_file"]),
        cityflow_seed=int(config.pilot.get("cityflow_seed", 0)),
        policy_seed=int(config.pilot.get("policy_seed", 0)),
        model_seed=int(config.pilot.get("model_seed", 0)),
        episodes=int(episodes),
        max_decision_steps_per_episode=int(max_decision_steps_per_episode),
        min_action_time=min_action_time,
        state_encoder_id="hybrid_v1",
        state_encoder_hash=str(observed_feature_hash),
        expected_feature_hash=expected_feature_hash,
        objective_normalizer_path=str(objective_normalizer) if objective_normalizer is not None else None,
        objective_normalizer_hash=normalizer_actual_hash,
        objective_normalizer_hash_expected=objective_normalizer_hash,
        objective_normalizer_hash_verified=normalizer_hash_verified,
        objective_normalizer_file_sha256=normalizer_file_sha256,
        objective_normalizer_loaded=normalizer is not None,
        film_model_dir=str(film_model_dir) if film_model_dir is not None else None,
        film_model_hash=film_model_actual_hash,
        film_model_hash_expected=film_model_hash,
        film_model_hash_verified=film_model_hash_verified,
        film_model_source=film_model_source,
        film_model_loaded=film_model_loaded,
        reward_readiness=reward_readiness,
        preference_name="episode_fixed_5_templates" if reward_readiness else str(first_preference_name),
        w=[] if reward_readiness else list(first_w_tuple or ()),
        preference_coverage=observed_preferences,
    )
    metadata.update(reward_info_debug)
    status_name = "REAL_REWARD_READINESS_DONE" if reward_readiness else "REAL_PREFLIGHT_DONE"
    status = {
        "status": status_name,
        "steps": total_steps,
        "reward_row_count": reward_row_count,
        "obs_dim": observed_obs_dim,
        "feature_names_hash": observed_feature_hash,
        **metadata,
    }
    write_json(out / "metadata.json", metadata)
    write_json(out / "status.json", status)
    (out / status_name).write_text(
        "real CityFlow reward-readiness completed; no learning, no pilot, no performance claim\n"
        if reward_readiness
        else "real CityFlow preflight completed; no learning, no pilot, no performance claim\n",
        encoding="utf-8",
    )
    assert_no_learning_artifacts(out)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_decision_steps_per_episode", type=int, default=3)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--real_env_preflight", action="store_true")
    parser.add_argument("--reward_readiness", action="store_true")
    parser.add_argument("--expected_feature_hash", default=EXPECTED_HYBRID_V1_FEATURE_HASH)
    parser.add_argument("--objective_normalizer")
    parser.add_argument("--objective_normalizer_hash")
    parser.add_argument("--film_model_dir")
    parser.add_argument("--film_model_hash")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.real_env_preflight:
        raise SystemExit("real CityFlow preflight smoke requires --real_env_preflight")
    config = load_formal_ppo_dryrun_config(args.spec)
    payload = run_real_cityflow_preflight_smoke(
        config,
        args.method,
        args.out_dir,
        real_env_preflight=args.real_env_preflight,
        episodes=args.episodes,
        max_decision_steps_per_episode=args.max_decision_steps_per_episode,
        expected_feature_hash=args.expected_feature_hash,
        objective_normalizer=args.objective_normalizer,
        objective_normalizer_hash=args.objective_normalizer_hash,
        film_model_dir=args.film_model_dir,
        film_model_hash=args.film_model_hash,
        reward_readiness=args.reward_readiness,
        device=args.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
