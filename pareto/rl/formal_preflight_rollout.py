#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import append_jsonl, write_json
from pareto.common.scenario import build_llmlight_env_config, resolve_scenario
from pareto.constants import OBJECTIVE_NAMES
from pareto.data.abstraction import build_trajectory_record
from pareto.data.normalization import RobustObjectiveNormalizer
from pareto.data.snapshot import capture_snapshot
from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.rl.env_reward_source import ensure_nonzero_env_reward_info, select_cityflow_env_rewards
from pareto.rl.formal_experiment_spec import FormalExperimentSpec, load_formal_experiment_spec, write_formal_experiment_spec
from pareto.rl.formal_preflight_checks import run_preflight_checks
from pareto.rl.formal_reward_adapter import EnvRewardAdapter, FiLMScalarPotentialRewardAdapter, build_weighted_proxy_adapter
from pareto.rl.state_encoder import ParetoStateEncoder
from pareto.train_common import load_checkpoint, set_seed


PREFERENCE_WEIGHTS = {
    "efficiency": [1.0, 0.0, 0.0, 0.0],
    "safety": [0.0, 1.0, 0.0, 0.0],
    "fairness": [0.0, 0.0, 1.0, 0.0],
    "stability": [0.0, 0.0, 0.0, 1.0],
    "balanced": [0.25, 0.25, 0.25, 0.25],
}


class PreferenceActorCritic(nn.Module):
    def __init__(self, obs_dim: int, preference_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.preference_dim = int(preference_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.obs_dim + self.preference_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, self.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if w.ndim == 1:
            w = w.unsqueeze(0).expand(obs.shape[0], -1)
        inputs = torch.cat([obs.float(), w.float()], dim=-1)
        hidden = self.trunk(inputs)
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values


def save_preflight_policy_checkpoint(path: str | Path, model: PreferenceActorCritic, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "obs_dim": model.obs_dim,
                "preference_dim": model.preference_dim,
                "action_dim": model.action_dim,
                "hidden_dim": model.hidden_dim,
            },
            "metadata": metadata,
        },
        path,
    )


def load_preflight_policy_checkpoint(path: str | Path) -> tuple[PreferenceActorCritic, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=torch.device("cpu"))
    config = payload["config"]
    model = PreferenceActorCritic(
        obs_dim=config["obs_dim"],
        preference_dim=config["preference_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config.get("hidden_dim", 64),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def validate_preflight_rollout_limits(
    preflight_only: bool,
    episodes: int,
    max_decision_steps_per_episode: int,
) -> None:
    if not preflight_only:
        raise ValueError("real rollout preflight requires --preflight_only")
    if episodes <= 0 or episodes > 2:
        raise ValueError("episodes must be in [1, 2] for preflight rollout")
    if max_decision_steps_per_episode <= 0 or max_decision_steps_per_episode > 10:
        raise ValueError("max_decision_steps_per_episode must be in [1, 10] for preflight rollout")


def build_preflight_rollout_metadata(
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
    objective_norm_path: str,
    objective_normalizer_hash: str,
    reward_adapter: str,
    formal_gate_decision_path: str,
    film_model_dir: str | None = None,
    film_model_hash: str | None = None,
) -> dict[str, Any]:
    gate = json.loads(Path(formal_gate_decision_path).read_text(encoding="utf-8"))
    return {
        "method": method,
        "preflight_only": True,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "env_rollout": True,
        "ppo_training": True,
        "policy_update": True,
        "seed_expansion": False,
        "city_expansion": False,
        "scenario": scenario,
        "traffic_file": traffic_file,
        "cityflow_seed": int(cityflow_seed),
        "policy_seed": int(policy_seed),
        "model_seed": int(model_seed),
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "min_action_time": int(min_action_time),
        "sim_seconds_per_episode": int(max_decision_steps_per_episode * min_action_time),
        "state_encoder_id": state_encoder_id,
        "state_encoder_hash": state_encoder_hash,
        "objective_norm_path": objective_norm_path,
        "objective_normalizer_hash": objective_normalizer_hash,
        "film_model_dir": film_model_dir,
        "film_model_hash": film_model_hash,
        "formal_gate_decision_path": formal_gate_decision_path,
        "formal_gate_representation_gate_pass": bool(gate.get("representation_gate_pass", False)),
        "formal_gate_ppo_formal_allowed": bool(gate.get("ppo_formal_allowed", False)),
        "reward_adapter": reward_adapter,
        "reward_normalization": "none",
        "reward_scale": 1.0,
        "potential_gamma": 0.99,
        "policy_conditioned_on_w": True,
        "critic_conditioned_on_w": True,
        "preference_sampling": "episode_fixed",
    }


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


def _load_film_model(model_dir: str | Path, device: torch.device) -> torch.nn.Module:
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


def _preference_for_episode(spec: FormalExperimentSpec, episode: int) -> tuple[str, list[float]]:
    names = [name for name in spec.train_preferences if name in PREFERENCE_WEIGHTS]
    if not names:
        names = ["balanced"]
    name = names[episode % len(names)]
    return name, PREFERENCE_WEIGHTS[name]


def _reward_for_adapter(
    spec: FormalExperimentSpec,
    film_model: torch.nn.Module | None,
    w: Sequence[float],
    obs_t: torch.Tensor,
    obs_tp1: torch.Tensor,
    objectives_t: Mapping[str, float],
    objectives_tp1: Mapping[str, float],
    done: bool,
    env_reward: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    if spec.reward_adapter == "film_scalar_potential":
        if film_model is None:
            raise ValueError("FiLM model is required for film_scalar_potential")
        adapter = FiLMScalarPotentialRewardAdapter(film_model, w, gamma=spec.potential_gamma, device=device)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if spec.reward_adapter == "weighted_proxy":
        adapter = build_weighted_proxy_adapter(w)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if spec.reward_adapter == "env_reward":
        adapter = EnvRewardAdapter()
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done, env_reward=env_reward)
    raise ValueError(f"unsupported preflight rollout adapter: {spec.reward_adapter}")


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"non-finite {name}")


def _load_preflight_checks(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("passed", False):
        raise ValueError(f"preflight checks did not pass: {path}")
    names = {check.get("name"): check for check in payload.get("checks", [])}
    real_check = names.get("real_record_film_reward_sensitivity")
    if real_check is None or not real_check.get("passed", False):
        raise ValueError("preflight checks must include passing real_record_film_reward_sensitivity")
    return payload


def run_preflight_rollout(
    spec_path: str | Path,
    out_dir: str | Path,
    *,
    preflight_only: bool,
    episodes: int,
    max_decision_steps_per_episode: int,
    preflight_checks_json: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    validate_preflight_rollout_limits(preflight_only, episodes, max_decision_steps_per_episode)
    checks_payload = _load_preflight_checks(preflight_checks_json)
    spec = load_formal_experiment_spec(spec_path)
    spec.validate_for_stage("preflight")
    run_preflight_checks([spec], root=".", device=device)
    scenario_meta = resolve_scenario(spec.scenario)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out_dir / "stdout.log").write_text("", encoding="utf-8")
    (out_dir / "stderr.log").write_text("", encoding="utf-8")
    write_formal_experiment_spec(out_dir / "formal_experiment_spec.json", spec)
    shutil.copyfile(spec.formal_gate_decision_path, out_dir / "formal_gate_decision.json")

    metadata = build_preflight_rollout_metadata(
        method=spec.reward_adapter,
        scenario=spec.scenario,
        traffic_file=spec.traffic_file,
        cityflow_seed=spec.cityflow_seed,
        policy_seed=spec.policy_seed,
        model_seed=spec.model_seed,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        min_action_time=spec.min_action_time,
        state_encoder_id=spec.state_encoder_id,
        state_encoder_hash=spec.state_encoder_hash,
        objective_norm_path=spec.objective_norm_path,
        objective_normalizer_hash=spec.objective_normalizer_hash,
        reward_adapter=spec.reward_adapter,
        formal_gate_decision_path=spec.formal_gate_decision_path,
        film_model_dir=spec.film_model_dir,
        film_model_hash=spec.film_model_hash,
    )
    if checks_payload is not None:
        metadata["preflight_checks_json"] = str(preflight_checks_json)
        metadata["preflight_checks_passed"] = True
    write_json(out_dir / "metadata.json", metadata)

    set_seed(spec.model_seed)
    random.seed(spec.policy_seed)
    np.random.seed(spec.policy_seed)
    torch_device = torch.device(device)
    run_counts = max_decision_steps_per_episode * spec.min_action_time
    dic_agent_conf, dic_traffic_env_conf, dic_path = build_llmlight_env_config(
        scenario=spec.scenario,
        traffic_file=spec.traffic_file,
        seed=spec.cityflow_seed,
        run_counts=run_counts,
        min_action_time=spec.min_action_time,
        model_name="Random",
        work_dir=str(out_dir / "llmlight_work"),
    )
    reward_info_debug = ensure_nonzero_env_reward_info(dic_traffic_env_conf, enable=spec.reward_adapter == "env_reward")
    metadata.update(reward_info_debug)
    write_json(out_dir / "metadata.json", metadata)
    _prepare_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)

    from utils.cityflow_env import CityFlowEnv

    encoder = ParetoStateEncoder(spec.state_encoder_id)
    normalizer = RobustObjectiveNormalizer.load(spec.objective_norm_path)
    action_dim = len(dic_traffic_env_conf["PHASE"])
    film_model = _load_film_model(spec.film_model_dir, torch_device) if spec.reward_adapter == "film_scalar_potential" else None
    policy: PreferenceActorCritic | None = None
    optimizer: torch.optim.Optimizer | None = None
    step_count = 0

    for episode in range(episodes):
        env = CityFlowEnv(
            path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
        )
        state = env.reset()
        del state
        prev_snapshots = [None for _ in env.list_intersection]
        prev_actions = [None for _ in env.list_intersection]
        preference_name, w = _preference_for_episode(spec, episode)
        w_tensor_single = torch.tensor(w, dtype=torch.float32, device=torch_device)
        for decision_step in range(max_decision_steps_per_episode):
            snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            features_np = [encoder.encode_snapshot(snapshot)[0] for snapshot in snapshots]
            obs_tensor = torch.tensor(np.asarray(features_np), dtype=torch.float32, device=torch_device)
            if policy is None:
                policy = PreferenceActorCritic(
                    obs_dim=obs_tensor.shape[-1],
                    preference_dim=len(w),
                    action_dim=action_dim,
                    hidden_dim=64,
                ).to(torch_device)
                optimizer = torch.optim.Adam(policy.parameters(), lr=float(spec.ppo_budget.get("lr", 3e-4)))
            w_batch = w_tensor_single.unsqueeze(0).expand(obs_tensor.shape[0], -1)
            logits, values = policy(obs_tensor, w_batch)
            _assert_finite_tensor("action_logits", logits)
            _assert_finite_tensor("values", values)
            distribution = Categorical(logits=logits)
            actions_tensor = distribution.sample()
            log_probs = distribution.log_prob(actions_tensor)
            actions = [int(value) for value in actions_tensor.detach().cpu().tolist()]
            _, final_env_reward, done, average_env_reward = env.step(actions)
            env_reward, reward_source_debug = select_cityflow_env_rewards(final_env_reward, average_env_reward)
            next_snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            rewards: list[float] = []
            reward_debug_rows: list[dict[str, Any]] = []
            is_done = bool(done or decision_step == max_decision_steps_per_episode - 1)
            for idx, snapshot in enumerate(snapshots):
                record = build_trajectory_record(
                    snapshot=snapshot,
                    prev_snapshot=prev_snapshots[idx],
                    action=actions[idx],
                    prev_action=prev_actions[idx],
                    policy_id=f"preflight_{spec.reward_adapter}",
                    encoder=encoder,
                    objective_normalizer=normalizer,
                    metadata={
                        "run_id": f"preflight_{spec.reward_adapter}",
                        "scenario": spec.scenario,
                        "roadnet": scenario_meta["roadnet"],
                        "traffic_file": spec.traffic_file,
                        "seed": spec.cityflow_seed,
                        "episode": episode,
                        "step": decision_step,
                    },
                )
                next_record = build_trajectory_record(
                    snapshot=next_snapshots[idx],
                    prev_snapshot=snapshot,
                    action=actions[idx],
                    prev_action=prev_actions[idx],
                    policy_id=f"preflight_{spec.reward_adapter}",
                    encoder=encoder,
                    objective_normalizer=normalizer,
                    metadata={
                        "run_id": f"preflight_{spec.reward_adapter}",
                        "scenario": spec.scenario,
                        "roadnet": scenario_meta["roadnet"],
                        "traffic_file": spec.traffic_file,
                        "seed": spec.cityflow_seed,
                        "episode": episode,
                        "step": decision_step + 1,
                    },
                )
                obs_t = torch.tensor(record.obs_features, dtype=torch.float32)
                obs_tp1 = torch.tensor(next_record.obs_features, dtype=torch.float32)
                env_reward_i = float(env_reward[idx]) if isinstance(env_reward, (list, tuple, np.ndarray)) else float(env_reward)
                reward, debug = _reward_for_adapter(
                    spec,
                    film_model,
                    w,
                    obs_t,
                    obs_tp1,
                    record.objective_values_norm,
                    next_record.objective_values_norm,
                    is_done,
                    env_reward_i,
                    torch_device,
                )
                if not np.isfinite(reward):
                    raise ValueError("non-finite total_reward")
                rewards.append(float(reward))
                reward_debug_rows.append(
                    {
                        "episode": episode,
                        "decision_step": decision_step,
                        "intersection_idx": idx,
                        "preference_name": preference_name,
                        "action": actions[idx],
                        "env_reward": env_reward_i,
                        "env_reward_source": reward_source_debug["env_reward_step_return_source"],
                        **reward_info_debug,
                        **debug,
                    }
                )
                prev_snapshots[idx] = snapshot
                prev_actions[idx] = actions[idx]
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=torch_device)
            _assert_finite_tensor("rewards", reward_tensor)
            advantage = reward_tensor - values.detach()
            policy_loss = -(log_probs * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(values, reward_tensor)
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_loss - 0.01 * entropy
            _assert_finite_tensor("policy_loss", policy_loss)
            _assert_finite_tensor("value_loss", value_loss)
            _assert_finite_tensor("loss", loss)
            assert optimizer is not None and policy is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            append_jsonl(out_dir / "reward_components.jsonl", reward_debug_rows)
            append_jsonl(
                out_dir / "loss_debug.jsonl",
                [
                    {
                        "episode": episode,
                        "decision_step": decision_step,
                        "policy_loss": float(policy_loss.detach().cpu().item()),
                        "value_loss": float(value_loss.detach().cpu().item()),
                        "entropy": float(entropy.detach().cpu().item()),
                        "loss": float(loss.detach().cpu().item()),
                    }
                ],
            )
            append_jsonl(
                out_dir / "train_metrics.jsonl",
                [
                    {
                        "episode": episode,
                        "decision_step": decision_step,
                        "sim_time": env.get_current_time(),
                        "preference_name": preference_name,
                        "reward_mean": float(np.mean(rewards)),
                        "reward_min": float(np.min(rewards)),
                        "reward_max": float(np.max(rewards)),
                    }
                ],
            )
            step_count += 1
            if done:
                break
        env.batch_log_2()

    if policy is None:
        raise RuntimeError("preflight rollout did not initialize policy")
    checkpoint_path = out_dir / "checkpoint_last.pt"
    save_preflight_policy_checkpoint(checkpoint_path, policy.cpu(), metadata)
    loaded_policy, _ = load_preflight_policy_checkpoint(checkpoint_path)
    test_logits, test_values = loaded_policy(torch.zeros(2, policy.obs_dim), torch.ones(2, policy.preference_dim) / policy.preference_dim)
    _assert_finite_tensor("checkpoint_action_logits", test_logits)
    _assert_finite_tensor("checkpoint_values", test_values)
    write_json(out_dir / "status.json", {"status": "PREFLIGHT_DONE", "steps": step_count, **metadata})
    (out_dir / "PREFLIGHT_DONE").write_text("preflight rollout completed; not a formal experiment\n", encoding="utf-8")
    assert_no_forbidden_performance_artifacts(out_dir)
    return {"status": "PREFLIGHT_DONE", "steps": step_count, **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_decision_steps_per_episode", type=int, default=10)
    parser.add_argument("--preflight_checks_json")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise RuntimeError(
        "formal_preflight_rollout.py is deprecated. Use real_cityflow_preflight_smoke.py "
        "for no-learning reward readiness or future approved formal_pilot_runner.py."
    )
    payload = run_preflight_rollout(
        args.spec,
        args.out_dir,
        preflight_only=args.preflight_only,
        episodes=args.episodes,
        max_decision_steps_per_episode=args.max_decision_steps_per_episode,
        preflight_checks_json=args.preflight_checks_json,
        device=args.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
