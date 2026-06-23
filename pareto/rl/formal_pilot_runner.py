#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import append_jsonl, write_json
from pareto.common.scenario import build_llmlight_env_config
from pareto.data.objectives import compute_objectives_from_snapshot
from pareto.data.snapshot import capture_snapshot
from pareto.rl.action_diagnostics import summarize_action_distribution_guard
from pareto.rl.env_reward_source import ensure_nonzero_env_reward_info, select_cityflow_env_rewards
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig, load_formal_ppo_dryrun_config
from pareto.models.vector_quality import build_preference_scorer, build_vector_quality_model
from pareto.rl.formal_reward_adapter import EnvRewardAdapter, FiLMScalarPotentialRewardAdapter, VectorQRewardAdapter, build_weighted_proxy_adapter
from pareto.rl.paper_final_experiment_manifest import PAPER_FINAL_SEEDS, REQUIRED_CITY_TRAFFIC, REQUIRED_PREFERENCE_TEMPLATES
from pareto.rl.mock_llmlight_env import MockLLMLightEnv
from pareto.rl.ppo_actor_critic import (
    PreferenceConditionedActorCritic,
    load_actor_critic_checkpoint,
    load_training_checkpoint,
    save_actor_critic_checkpoint,
    save_training_checkpoint,
)
from pareto.rl.ppo_buffer import PPORolloutBuffer
from pareto.rl.ppo_loss import clipped_ppo_loss
from pareto.rl.preference_sampler import EpisodeFixedPreferenceSampler
from pareto.rl.real_cityflow_preflight_smoke import (
    assert_no_learning_artifacts,
    _compute_reward as compute_real_cityflow_reward,
    _encode_snapshots,
    _load_film_scalar_model,
    _load_normalizer,
    _normalizer_hash,
    _prepare_work_dir,
    run_real_cityflow_preflight_smoke,
    sha256_file,
    validate_hash,
    validate_real_cityflow_preflight_artifacts,
    validate_real_cityflow_preflight_limits,
)
from pareto.rl.state_encoder import ParetoStateEncoder
from pareto.train_common import load_checkpoint


FINAL_JINAN_PILOT_METHODS = ("film_scalar_potential", "weighted_proxy", "env_reward")
FORMAL_PILOT_WIRING_METHOD = "vector_quality_potential"
FORMAL_PILOT_WIRING_APPROVAL_PHRASE = "PARETO PPO FORMAL-PILOT DRY-RUN GO"
PAPER_FINAL_EXECUTION_APPROVAL_PHRASE = "PPTS PARETO PPO FINAL SCOPE-LIMITED EXECUTION GO"
PAPER_FINAL_PPO_METHODS = ("film_scalar_potential", "weighted_proxy", "vector_quality_potential")
PAPER_FINAL_STATE_ENCODER_HASH = "4d1c2b4e276043ac"
PAPER_FINAL_TOTAL_ENV_STEPS_PER_SEED = 1_000_000
PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE = 120
PAPER_FINAL_MIN_ACTION_TIME_SECONDS = 30
PAPER_FINAL_EPISODES_PER_SEED = math.ceil(
    PAPER_FINAL_TOTAL_ENV_STEPS_PER_SEED
    / (PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE * PAPER_FINAL_MIN_ACTION_TIME_SECONDS)
)
PAPER_FINAL_ACTUAL_SIM_SECONDS_PER_SEED = (
    PAPER_FINAL_EPISODES_PER_SEED * PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE * PAPER_FINAL_MIN_ACTION_TIME_SECONDS
)
FINAL_JINAN_PILOT_DISPLAY_NAMES = {
    "film_scalar_potential": "FiLMScalar-PPO",
    "weighted_proxy": "WeightedProxy-PPO",
    "env_reward": "EnvReward-QueuePenalty-PPO",
    FORMAL_PILOT_WIRING_METHOD: "VectorQ-PPO-wiring",
}
FINAL_JINAN_REFERENCE_ONLY_METHODS = ("MaxPressure", "AdvancedMaxPressure")
EXPLORATORY_PILOT_ALLOWED_OUTPUTS = {
    "metadata.json",
    "status.json",
    "train_metrics.jsonl",
    "reward_components.jsonl",
    "loss_debug.jsonl",
    "checkpoint_last.pt",
    "training_checkpoint_last.pt",
    "pilot_status.json",
    "pilot_guard_report.md",
    "stdout.txt",
    "stderr.txt",
}


class MockScalarQualityModel(torch.nn.Module):
    def forward(self, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        obs_score = obs.float().mean(dim=-1, keepdim=True)
        w_score = w.float().matmul(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=w.device)).reshape(-1, 1)
        return obs_score + w_score


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _short_sha256(path: str | Path) -> str:
    return _sha256(path)[:16]


def _clean_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "metadata.json",
        "ppo_config.json",
        "pilot_spec.json",
        "formal_gate_decision.json",
        "loss_debug.jsonl",
        "reward_components.jsonl",
        "train_metrics.jsonl",
        "checkpoint_last.pt",
        "training_checkpoint_last.pt",
        "status.json",
        "MOCK_DRY_RUN_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()


def compute_pilot_reward(
    method: str,
    obs_t: torch.Tensor,
    obs_tp1: torch.Tensor,
    objectives_t: dict[str, float],
    objectives_tp1: dict[str, float],
    w: list[float] | tuple[float, float, float, float],
    done: bool,
    env_reward: float,
    gamma: float = 0.99,
) -> tuple[float, dict]:
    if method == "film_scalar_potential":
        adapter = FiLMScalarPotentialRewardAdapter(MockScalarQualityModel(), w, gamma=gamma)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if method == "weighted_proxy":
        adapter = build_weighted_proxy_adapter(w)
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done)
    if method == "env_reward":
        adapter = EnvRewardAdapter()
        return adapter.compute(obs_t, obs_tp1, objectives_t, objectives_tp1, done, env_reward=env_reward)
    raise ValueError(f"unsupported pilot method: {method}")


def _metadata(
    config: FormalPPODryRunConfig,
    method: str,
    episodes: int,
    max_decision_steps_per_episode: int,
    resume_from: Path | None,
) -> dict[str, Any]:
    hash_metadata = _pilot_hash_metadata(config, method)
    return {
        "pilot_runner_skeleton": True,
        "pilot_dry_run": True,
        "result_mode": "engineering_mock_dry_run",
        "mock_env": True,
        "mock_env_rollout": True,
        "real_env_rollout": False,
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "cityflow_env_constructed": False,
        "cityflow_step_called": False,
        "real_ppo_update": False,
        "cityflow_optimizer_step": False,
        "mock_policy_update": True,
        "mock_optimizer_step": True,
        "method": method,
        "scenario": config.pilot["scenario"],
        "traffic_file": config.pilot["traffic_file"],
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "ppo_config_hash": config.ppo_config_hash(),
        "resume_loaded": resume_from is not None,
        "resume_from": str(resume_from) if resume_from is not None else None,
        **hash_metadata,
    }


def _method_display_name(method: str) -> str:
    return FINAL_JINAN_PILOT_DISPLAY_NAMES.get(method, method)


def _exploratory_pilot_gate_metadata(config: FormalPPODryRunConfig, method: str) -> dict[str, Any]:
    return {
        "exploratory_pilot": True,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "seed_expansion_allowed": False,
        "city_expansion_allowed": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "paper_result_allowed": False,
        "main_results_allowed": False,
        "offline_representation_gate": "partial_pass",
        "representation_status": "partial_pass_exploratory_pilot_only",
        "allowed_result_interpretation": "closed_loop_stability_only",
        "forbidden_result_interpretation": [
            "method_ranking",
            "performance_claim",
            "paper_result",
            "seed_expansion",
            "city_expansion",
        ],
        "reward_adapter_version": method,
        "objective_normalizer_version": str(config.pilot.get("objective_normalizer_hash")),
        "feature_schema_version": "hybrid_v1",
        "pair_reversal_data_source": str(config.pilot.get("pilot_spec_path")),
    }


def _action_entropy(action_counts: Counter, total_actions: int) -> float:
    if total_actions <= 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        prob = float(count) / float(total_actions)
        entropy -= prob * float(np.log(prob))
    return float(entropy)


def _summarize_action_distribution_guard(
    action_counts: Counter,
    intersection_action_counts: list[Counter],
    *,
    max_single_action_rate: float = 0.95,
) -> dict[str, Any]:
    return summarize_action_distribution_guard(
        action_counts,
        intersection_action_counts,
        max_global_single_action_rate=max_single_action_rate,
        max_intersection_single_action_rate=0.98,
        context_label="bounded exploratory pilot",
    )


def _verify_pilot_checkpoint_roundtrip(out: Path) -> dict[str, Any]:
    training_path = out / "training_checkpoint_last.pt"
    actor_path = out / "checkpoint_last.pt"
    if not training_path.is_file():
        raise ValueError("bounded exploratory pilot missing training checkpoint")
    if not actor_path.is_file():
        raise ValueError("bounded exploratory pilot missing actor checkpoint")
    try:
        _, _, training_payload = load_training_checkpoint(training_path)
        _, actor_payload = load_actor_critic_checkpoint(actor_path)
    except Exception as exc:  # pragma: no cover - defensive wrapper keeps user-facing error clear.
        raise ValueError(f"bounded exploratory pilot checkpoint roundtrip failed: {exc}") from exc
    return {
        "checkpoint_saved": True,
        "checkpoint_load_verified": True,
        "checkpoint_valid": True,
        "training_checkpoint_load_verified": True,
        "actor_checkpoint_load_verified": True,
        "training_checkpoint_step": int(training_payload.get("metadata", {}).get("step", 0)),
        "training_checkpoint_global_update": int(training_payload.get("metadata", {}).get("global_update", 0)),
        "actor_checkpoint_has_metadata": bool(actor_payload.get("metadata")),
    }


def _write_exploratory_pilot_guard_artifacts(out: Path, status: dict[str, Any], action_guard: dict[str, Any]) -> None:
    if not bool(status.get("checkpoint_load_verified", False)):
        raise ValueError("cannot write pilot_status checkpoint_valid=true before checkpoint load verification")
    pilot_status = {
        "status": status["status"],
        "exploratory_pilot": True,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "closed_loop_executed": True,
        "reward_finite": True,
        "loss_finite": bool(status.get("loss_debug_finite", False)),
        "checkpoint_saved": bool(status.get("checkpoint_saved", False)),
        "checkpoint_load_verified": True,
        "checkpoint_valid": True,
        "action_distribution_non_degenerate": True,
        "metrics_recorded": True,
        "policy_update_count": int(status.get("policy_update_count", 0)),
        "reward_row_count": int(status.get("reward_row_count", 0)),
        "action_guard": action_guard,
    }
    write_json(out / "pilot_status.json", pilot_status)
    report_lines = [
        "# Exploratory Pareto PPO Pilot Guard Report",
        "",
        "Status: `PASS`",
        "",
        "Scope: Jinan seed0 exploratory closed-loop check only.",
        "",
        "Allowed interpretation:",
        "- closed loop executed",
        "- reward/loss/checkpoint/action/logging guards passed",
        "",
        "Forbidden interpretation:",
        "- no performance claim",
        "- no method ranking",
        "- no seed or city expansion",
        "- no paper result",
        "",
        f"Policy updates: `{pilot_status['policy_update_count']}`",
        f"Reward rows: `{pilot_status['reward_row_count']}`",
        f"Action entropy: `{action_guard['action_entropy']:.6f}`",
        f"Global single-action rate: `{action_guard['global_single_action_rate']:.6f}`",
    ]
    (out / "pilot_guard_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _maybe_write_exploratory_pilot_guard_artifacts(
    out: Path,
    status: dict[str, Any],
    action_guard: dict[str, Any],
    formal_execution_context: dict[str, Any] | None,
) -> bool:
    if formal_execution_context is not None:
        return False
    _write_exploratory_pilot_guard_artifacts(out, status, action_guard)
    return True


def _finalize_exploratory_pilot_outputs(out: Path) -> list[str]:
    removed: list[str] = []
    for path in list(out.iterdir()):
        if path.name in EXPLORATORY_PILOT_ALLOWED_OUTPUTS:
            continue
        removed.append(path.name)
        _remove_exploratory_output(path)
    assert_no_forbidden_performance_artifacts(out)
    return sorted(removed)


def _remove_exploratory_output(path: Path, *, attempts: int = 20, delay_seconds: float = 0.25) -> None:
    last_error: OSError | None = None
    for attempt in range(max(1, attempts)):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            time.sleep(delay_seconds)
    if path.exists() and path.is_dir() and not path.is_symlink():
        quarantine = _quarantine_exploratory_output_path(path)
        try:
            path.rename(quarantine)
        except FileNotFoundError:
            return
        except OSError:
            if last_error is not None:
                raise last_error
            raise
        shutil.rmtree(quarantine, ignore_errors=True)
        return
    if last_error is not None:
        raise last_error


def _quarantine_exploratory_output_path(path: Path) -> Path:
    parent = path.parent
    timestamp = int(time.time() * 1000)
    for idx in range(100):
        suffix = f"{timestamp}_{idx}" if idx else str(timestamp)
        candidate = parent.parent / f".{parent.name}_{path.name}_cleanup_{suffix}"
        if not candidate.exists():
            return candidate
    return parent.parent / f".{parent.name}_{path.name}_cleanup_{timestamp}_fallback"


def _summarize_tiny_env_reward_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["env_reward"]) for row in rows if "env_reward" in row]
    finite = bool(values) and all(np.isfinite(value) for value in values)
    nonzero_count = sum(1 for value in values if np.isfinite(value) and abs(value) > 1e-12)
    all_zero = bool(values) and finite and nonzero_count == 0
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "finite": bool(finite),
        "nonzero_reward_count": int(nonzero_count),
        "env_reward_nonzero_rate": float(nonzero_count / len(values)) if values else 0.0,
        "all_zero_reward": bool(all_zero),
        "env_reward_sources": sorted({str(row.get("env_reward_source")) for row in rows if row.get("env_reward_source")}),
    }
    if values and finite:
        summary.update({"min": float(min(values)), "mean": float(sum(values) / len(values)), "max": float(max(values))})
    return summary


def _require_pilot_hash(config: FormalPPODryRunConfig, key: str) -> str:
    value = config.pilot.get(key)
    if not value:
        raise ValueError(f"{key} is required for formal pilot runner dry-run hash lock")
    return str(value)


def _pilot_hash_metadata(config: FormalPPODryRunConfig, method: str) -> dict[str, Any]:
    pilot_spec_path = Path(str(config.pilot["pilot_spec_path"]))
    if not pilot_spec_path.exists():
        raise FileNotFoundError(f"pilot_spec_path does not exist: {pilot_spec_path}")
    state_encoder_hash = _require_pilot_hash(config, "state_encoder_hash")
    objective_normalizer_hash = _require_pilot_hash(config, "objective_normalizer_hash")
    film_model_hash = str(config.pilot.get("film_model_hash") or "")
    if method == "film_scalar_potential" and not film_model_hash:
        raise ValueError("film_model_hash is required for film_scalar_potential dry-run hash lock")
    vector_model_hash = str(config.pilot.get("vector_model_hash") or "")
    if method == FORMAL_PILOT_WIRING_METHOD and not vector_model_hash:
        raise ValueError("vector_model_hash is required for formal-pilot wiring dry-run hash lock")
    return {
        "state_encoder_hash": state_encoder_hash,
        "objective_normalizer_hash": objective_normalizer_hash,
        "film_model_hash": film_model_hash or None,
        "vector_model_hash": vector_model_hash or None,
        "ppo_config_hash": config.ppo_config_hash(),
        "pilot_spec_hash": _short_sha256(pilot_spec_path),
        "pilot_spec_hash_verified": True,
    }


def _load_vector_quality_model(
    model_dir: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module | None, dict[str, Any]]:
    checkpoint = load_checkpoint(Path(model_dir) / "model.pt", device)
    config = checkpoint["config"]
    model = build_vector_quality_model(
        config.get("architecture", "shared_mlp"),
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        trunk_layers=config.get("trunk_layers", 2),
        head_layers=config.get("head_layers", 2),
        tower_residual_alpha=config.get("tower_residual_alpha", 0.5),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    scorer_config = checkpoint.get(
        "scorer_config",
        {
            "score_mode": config.get("score_mode", "linear"),
            "interaction_rank": config.get("interaction_rank", 4),
            "interaction_beta": config.get("interaction_beta", 0.3),
        },
    )
    scorer = build_preference_scorer(
        scorer_config.get("score_mode", "linear"),
        rank=scorer_config.get("interaction_rank", 4),
        beta=scorer_config.get("interaction_beta", 0.3),
    )
    if "scorer_state_dict" in checkpoint:
        scorer.load_state_dict(checkpoint["scorer_state_dict"])
    scorer.to(device)
    scorer.eval()
    return model, scorer, config


def _validate_vector_model_artifacts(
    *,
    vector_model_dir: str | Path | None,
    vector_model_hash: str | None,
) -> str:
    if vector_model_dir is None:
        raise ValueError("formal-pilot wiring dry-run requires --vector_model_dir")
    model_path = Path(vector_model_dir) / "model.pt"
    if not model_path.is_file():
        raise ValueError(f"VectorQ checkpoint not found: {model_path}")
    if vector_model_hash is None:
        raise ValueError("formal-pilot wiring dry-run requires --vector_model_hash")
    actual = sha256_file(model_path)
    validate_hash("vector_model_hash", actual, vector_model_hash)
    return actual


def _copy_or_write_json(source: str | Path, target: Path, fallback: dict[str, Any]) -> None:
    source_path = Path(source)
    if source_path.exists():
        shutil.copyfile(source_path, target)
    else:
        write_json(target, fallback)


def _update_from_buffer(
    model: PreferenceConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PPORolloutBuffer,
    config: FormalPPODryRunConfig,
    out: Path,
    global_update: int,
) -> int:
    if len(buffer) == 0:
        return global_update
    device = next(model.parameters()).device
    batch = buffer.compute_returns_and_advantages(
        last_value=0.0,
        gamma=float(config.ppo["gamma"]),
        gae_lambda=float(config.ppo["gae_lambda"]),
        normalize_advantages=bool(config.ppo["normalize_advantages"]),
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    num_items = len(buffer)
    batch_size = int(config.ppo["minibatch_size"])
    for epoch in range(int(config.ppo["update_epochs"])):
        order = torch.randperm(num_items, device=device)
        for start in range(0, num_items, batch_size):
            idx = order[start:start + batch_size]
            new_log_probs, entropy, values = model.evaluate_actions(batch["obs"][idx], batch["w"][idx], batch["actions"][idx])
            loss = clipped_ppo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch["old_log_probs"][idx],
                advantages=batch["advantages"][idx],
                values=values,
                returns=batch["returns"][idx],
                entropy=entropy,
                clip_eps=float(config.ppo["clip_eps"]),
                value_loss_coef=float(config.ppo["value_loss_coef"]),
                entropy_coef=float(config.ppo["entropy_coef"]),
            )
            optimizer.zero_grad()
            loss["total_loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.ppo["max_grad_norm"]))
            optimizer.step()
            row = {
                "global_update": global_update,
                "epoch": epoch,
                "policy_loss": float(loss["policy_loss"].detach().item()),
                "value_loss": float(loss["value_loss"].detach().item()),
                "entropy_bonus": float(loss["entropy_bonus"].detach().item()),
                "approx_kl": float(loss["approx_kl"].detach().item()),
                "clip_fraction": float(loss["clip_fraction"].detach().item()),
                "ratio_mean": float(loss["ratio_mean"].detach().item()),
                "ratio_min": float(loss["ratio_min"].detach().item()),
                "ratio_max": float(loss["ratio_max"].detach().item()),
                "total_loss": float(loss["total_loss"].detach().item()),
                "grad_norm": float(grad_norm),
            }
            if not all(torch.isfinite(torch.tensor(value)) for value in row.values() if isinstance(value, (int, float))):
                raise ValueError("non-finite pilot mock PPO loss")
            append_jsonl(out / "loss_debug.jsonl", [row])
            global_update += 1
    return global_update


def run_formal_pilot_mock_dry_run(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    episodes: int = 2,
    max_decision_steps_per_episode: int = 10,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    if method not in config.pilot["methods"]:
        raise ValueError(f"method {method} is not in pilot methods")
    random.seed(int(config.pilot.get("model_seed", 0)))
    torch.manual_seed(int(config.pilot.get("model_seed", 0)))

    out = Path(out_dir)
    _clean_output_dir(out)
    resume_path = Path(resume_from) if resume_from is not None else None
    metadata = _metadata(config, method, episodes, max_decision_steps_per_episode, resume_path)
    write_json(out / "metadata.json", metadata)
    write_json(out / "ppo_config.json", {"ppo": config.ppo, "ppo_config_hash": config.ppo_config_hash()})
    _copy_or_write_json(config.pilot["formal_gate_decision_path"], out / "formal_gate_decision.json", {"ppo_formal_allowed": False})
    _copy_or_write_json(config.pilot["pilot_spec_path"], out / "pilot_spec.json", {"missing_source": config.pilot["pilot_spec_path"]})

    if resume_path is not None:
        model, optimizer, payload = load_training_checkpoint(resume_path)
        if payload["metadata"].get("ppo_config_hash") != config.ppo_config_hash():
            raise ValueError("resume checkpoint ppo_config_hash mismatch")
    else:
        model = PreferenceConditionedActorCritic(
            obs_dim=int(config.model["obs_dim"]),
            preference_dim=int(config.model["preference_dim"]),
            action_dim=int(config.model["action_dim"]),
            hidden_dim=int(config.model["hidden_dim"]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.ppo["lr"]))

    env = MockLLMLightEnv(
        num_intersections=12,
        obs_dim=int(config.model["obs_dim"]),
        min_action_time=30,
        max_steps=int(max_decision_steps_per_episode),
        seed=int(config.pilot.get("policy_seed", 0)),
    )
    sampler = EpisodeFixedPreferenceSampler()
    buffer = PPORolloutBuffer()
    global_update = 0
    total_steps = 0

    for episode in range(int(episodes)):
        state = env.reset()
        preference_name, w_tuple = sampler.preference_for_episode(episode)
        w_tensor = torch.tensor(w_tuple, dtype=torch.float32)
        for step in range(int(max_decision_steps_per_episode)):
            w_batch = w_tensor.reshape(1, -1).expand(env.num_intersections, -1)
            with torch.no_grad():
                actions, log_probs, values = model.act(state.obs, w_batch)
            action_list = [int(action) for action in actions.tolist()]
            next_state, env_rewards, done, info = env.step(action_list)
            step_rewards = []
            for inter_idx in range(env.num_intersections):
                reward, debug = compute_pilot_reward(
                    method,
                    state.obs[inter_idx],
                    next_state.obs[inter_idx],
                    state.objectives_norm[inter_idx],
                    next_state.objectives_norm[inter_idx],
                    list(w_tuple),
                    done,
                    env_reward=float(env_rewards[inter_idx]),
                    gamma=float(config.ppo["gamma"]),
                )
                buffer.add(
                    obs=state.obs[inter_idx],
                    w=w_tensor,
                    action=action_list[inter_idx],
                    reward=reward,
                    done=done,
                    value=float(values[inter_idx].item()),
                    log_prob=float(log_probs[inter_idx].item()),
                )
                step_rewards.append(float(reward))
                append_jsonl(
                    out / "reward_components.jsonl",
                    [
                        {
                            "episode": episode,
                            "step": step,
                            "intersection_idx": inter_idx,
                            "preference_name": preference_name,
                            "w": list(w_tuple),
                            "env_reward": float(env_rewards[inter_idx]),
                            "total_reward": float(reward),
                            **debug,
                        }
                    ],
                )
            append_jsonl(
                out / "train_metrics.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time": info["sim_time"],
                        "preference_name": preference_name,
                        "mock_env": True,
                        "reward_mean": float(torch.tensor(step_rewards).mean().item()),
                        "reward_min": float(torch.tensor(step_rewards).min().item()),
                        "reward_max": float(torch.tensor(step_rewards).max().item()),
                    }
                ],
            )
            total_steps += 1
            if len(buffer) >= int(config.ppo["rollout_steps"]):
                global_update = _update_from_buffer(model, optimizer, buffer, config, out, global_update)
                buffer = PPORolloutBuffer()
            state = next_state
            if done:
                break
    if len(buffer):
        global_update = _update_from_buffer(model, optimizer, buffer, config, out, global_update)

    save_actor_critic_checkpoint(out / "checkpoint_last.pt", model, metadata)
    save_training_checkpoint(
        out / "training_checkpoint_last.pt",
        model,
        optimizer,
        metadata,
        step=total_steps,
        episode=int(episodes),
        global_update=global_update,
    )
    status = {"status": "MOCK_DRY_RUN_DONE", "steps": total_steps, "global_update": global_update, **metadata}
    write_json(out / "status.json", status)
    (out / "MOCK_DRY_RUN_DONE").write_text("mock pilot dry-run completed; no real env rollout\n", encoding="utf-8")
    assert_no_forbidden_performance_artifacts(out)
    return status


def _validate_formal_real_env_hash_lock(
    config: FormalPPODryRunConfig,
    method: str,
    objective_normalizer_hash: str | None,
    film_model_hash: str | None,
    vector_model_hash: str | None = None,
) -> None:
    expected_objective_hash = _require_pilot_hash(config, "objective_normalizer_hash")
    if objective_normalizer_hash != expected_objective_hash:
        raise ValueError(
            f"objective_normalizer_hash mismatch: expected config hash {expected_objective_hash}, "
            f"got {objective_normalizer_hash}"
        )
    if method == "film_scalar_potential":
        expected_film_hash = _require_pilot_hash(config, "film_model_hash")
        if film_model_hash != expected_film_hash:
            raise ValueError(f"film_model_hash mismatch: expected config hash {expected_film_hash}, got {film_model_hash}")
    if method == FORMAL_PILOT_WIRING_METHOD:
        expected_vector_hash = _require_pilot_hash(config, "vector_model_hash")
        if vector_model_hash != expected_vector_hash:
            raise ValueError(
                f"vector_model_hash mismatch: expected config hash {expected_vector_hash}, got {vector_model_hash}"
            )


def _rename_preflight_artifacts_for_pilot(out: Path) -> None:
    preflight_metrics = out / "preflight_metrics.jsonl"
    if preflight_metrics.exists():
        runner_metrics = out / "runner_metrics.jsonl"
        if runner_metrics.exists():
            runner_metrics.unlink()
        preflight_metrics.rename(runner_metrics)
    preflight_marker = out / "REAL_PREFLIGHT_DONE"
    if preflight_marker.exists():
        preflight_marker.unlink()


def _real_env_metadata_overlay(
    config: FormalPPODryRunConfig,
    method: str,
    episodes: int,
    max_decision_steps_per_episode: int,
) -> dict[str, Any]:
    hash_metadata = _pilot_hash_metadata(config, method)
    film_model_hash_config = hash_metadata.pop("film_model_hash", None)
    vector_model_hash_config = hash_metadata.pop("vector_model_hash", None)
    overlay: dict[str, Any] = {
        "pilot_runner_skeleton": True,
        "pilot_dry_run": True,
        "real_env_dry_run": True,
        "result_mode": "engineering_real_env_no_learning_dry_run",
        "mock_env": False,
        "mock_env_rollout": False,
        "real_env_rollout": True,
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "cityflow_env_constructed": True,
        "cityflow_step_called": True,
        "ppo_training": False,
        "policy_update": False,
        "optimizer_step": False,
        "real_ppo_update": False,
        "cityflow_optimizer_step": False,
        "mock_policy_update": False,
        "mock_optimizer_step": False,
        "method": method,
        "scenario": config.pilot["scenario"],
        "traffic_file": config.pilot["traffic_file"],
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "film_model_hash_config": film_model_hash_config,
        **hash_metadata,
    }
    if method == "film_scalar_potential":
        overlay["film_model_hash"] = film_model_hash_config
    return overlay


def run_formal_pilot_real_env_dry_run(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    episodes: int = 1,
    max_decision_steps_per_episode: int = 3,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    validate_real_cityflow_preflight_limits(
        real_env_preflight=True,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        reward_readiness=False,
    )
    if method not in config.pilot["methods"]:
        raise ValueError(f"method {method} is not in pilot methods")
    validate_real_cityflow_preflight_artifacts(
        method=method,
        reward_readiness=True,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
    )
    _validate_formal_real_env_hash_lock(config, method, objective_normalizer_hash, film_model_hash)

    out = Path(out_dir)
    payload = run_real_cityflow_preflight_smoke(
        config,
        method,
        out,
        real_env_preflight=True,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        expected_feature_hash=str(config.pilot["state_encoder_hash"]),
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
        reward_readiness=False,
        device=device,
    )
    _rename_preflight_artifacts_for_pilot(out)

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    overlay = _real_env_metadata_overlay(config, method, episodes, max_decision_steps_per_episode)
    metadata.update(overlay)
    metadata["film_model_source"] = metadata.get("film_model_source")
    metadata["smoke_scalar_quality_model"] = metadata.get("film_model_source") == "smoke_scalar_quality_model"
    if method == "film_scalar_potential" and metadata["smoke_scalar_quality_model"]:
        raise ValueError("film_scalar_potential formal real-env dry-run must use trained FiLM checkpoint")
    write_json(metadata_path, metadata)

    status = {
        "status": "REAL_ENV_DRY_RUN_DONE",
        "steps": int(payload.get("steps", 0)),
        "reward_row_count": int(payload.get("reward_row_count", 0)),
        "forbidden_artifacts": [],
        **metadata,
    }
    write_json(out / "status.json", status)
    (out / "REAL_ENV_DRY_RUN_DONE").write_text(
        "formal pilot real-env dry-run completed; no learning, no PPO update, no performance claim\n",
        encoding="utf-8",
    )
    assert_no_learning_artifacts(out)
    return status


def validate_tiny_real_ppo_update_limits(
    episodes: int,
    max_decision_steps_per_episode: int,
    max_policy_updates: int,
) -> None:
    if int(episodes) != 1:
        raise ValueError("episodes must be exactly 1 for tiny real PPO-update preflight")
    if int(max_decision_steps_per_episode) <= 0 or int(max_decision_steps_per_episode) > 2:
        raise ValueError("max_decision_steps_per_episode must be in [1, 2] for tiny real PPO-update preflight")
    if int(max_policy_updates) != 1:
        raise ValueError("max_policy_updates must be exactly 1 for tiny real PPO-update preflight")


def validate_bounded_jinan_pilot_dry_run_limits(
    config: FormalPPODryRunConfig,
    method: str,
    episodes: int,
    max_decision_steps_per_episode: int,
    *,
    approved_seed_ids: tuple[int, ...] = (0,),
    approved_methods: tuple[str, ...] = FINAL_JINAN_PILOT_METHODS,
    require_final_method_list: bool = True,
    approved_city_traffic: dict[str, str] | None = None,
    max_episodes: int = 5,
    max_decision_steps: int = 120,
    scope_label: str = "bounded Jinan pilot dry-run",
) -> None:
    approved_city_traffic = approved_city_traffic or {"jinan": "anon_3_4_jinan_real.json"}
    scenario = str(config.pilot.get("scenario"))
    traffic_file = str(config.pilot.get("traffic_file"))
    if scenario not in approved_city_traffic:
        if approved_city_traffic == {"jinan": "anon_3_4_jinan_real.json"}:
            raise ValueError("bounded pilot dry-run is limited to Jinan")
        raise ValueError(f"{scope_label} scenario is not approved: {scenario}")
    if traffic_file != approved_city_traffic[scenario]:
        if approved_city_traffic == {"jinan": "anon_3_4_jinan_real.json"}:
            raise ValueError("bounded pilot dry-run is limited to anon_3_4_jinan_real.json")
        raise ValueError(f"{scope_label} traffic_file mismatch for {scenario}")
    approved = tuple(int(value) for value in approved_seed_ids)
    cityflow_seed = int(config.pilot.get("cityflow_seed", -1))
    policy_seed = int(config.pilot.get("policy_seed", -1))
    model_seed = int(config.pilot.get("model_seed", -1))
    if cityflow_seed not in approved:
        if approved == (0,):
            raise ValueError("bounded pilot dry-run requires cityflow_seed=0")
        raise ValueError(f"bounded pilot dry-run requires cityflow_seed in {list(approved)}")
    if policy_seed != cityflow_seed:
        raise ValueError("bounded pilot dry-run requires policy_seed to match cityflow_seed")
    if model_seed != cityflow_seed:
        raise ValueError("bounded pilot dry-run requires model_seed to match cityflow_seed")
    if require_final_method_list and tuple(config.pilot.get("methods", ())) != FINAL_JINAN_PILOT_METHODS:
        raise ValueError("bounded pilot dry-run requires the final Jinan method list")
    if method not in approved_methods:
        raise ValueError(f"method {method} is not in the approved method list for {scope_label}")
    if int(episodes) <= 0 or int(episodes) > int(max_episodes):
        raise ValueError(f"episodes must be in [1, {int(max_episodes)}] for {scope_label}")
    if int(max_decision_steps_per_episode) <= 0 or int(max_decision_steps_per_episode) > int(max_decision_steps):
        raise ValueError(f"max_decision_steps_per_episode must be in [1, {int(max_decision_steps)}] for {scope_label}")


def validate_formal_pilot_wiring_dry_run_limits(
    config: FormalPPODryRunConfig,
    *,
    episodes: int,
    max_decision_steps_per_episode: int,
) -> None:
    validate_bounded_jinan_pilot_dry_run_limits(
        config,
        FORMAL_PILOT_WIRING_METHOD,
        episodes,
        max_decision_steps_per_episode,
        approved_methods=(FORMAL_PILOT_WIRING_METHOD,),
        require_final_method_list=False,
    )
    if int(episodes) != 1:
        raise ValueError("episodes must be exactly 1 for formal-pilot wiring dry-run")
    if int(max_decision_steps_per_episode) <= 0 or int(max_decision_steps_per_episode) > 3:
        raise ValueError("max_decision_steps_per_episode must be in [1, 3] for formal-pilot wiring dry-run")


def _formal_pilot_wiring_context(
    config: FormalPPODryRunConfig,
    *,
    vector_model_dir: str | Path,
    vector_model_hash: str,
) -> dict[str, Any]:
    packet_path = str(config.pilot.get("representation_gate_packet_path", ""))
    if not packet_path:
        raise ValueError("representation_gate_packet_path is required for formal-pilot wiring dry-run")
    return {
        "approval_phrase": FORMAL_PILOT_WIRING_APPROVAL_PHRASE,
        "formal_pilot_wiring_dry_run": True,
        "bounded_jinan_1seed_pilot_dry_run": True,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "traffic_result_value_reading_executed": False,
        "method_ranking_executed": False,
        "paper_result_claim": False,
        "seed_expansion_allowed": False,
        "city_expansion_allowed": False,
        "formal_experiment_requires_new_pro_approval": True,
        "representation_run_id": str(config.pilot.get("representation_run_id", "")),
        "representation_gate_packet_path": packet_path,
        "vector_model_dir": str(vector_model_dir),
        "vector_model_hash_expected": str(vector_model_hash),
        "status_label": "FORMAL_PILOT_WIRING_DRY_RUN_DONE",
        "allowed_interpretation": "ppo_reward_wiring_only",
        "forbidden_interpretation": [
            "formal_experiment",
            "traffic_result_value_reading",
            "method_ranking",
            "performance_claim",
            "paper_result",
        ],
    }


def validate_formal_jinan_3seed_execution_limits(
    config: FormalPPODryRunConfig,
    method: str,
    *,
    seed_id: int,
    approval_phrase: str,
    episodes: int,
    max_decision_steps_per_episode: int,
    rollout_steps: int,
) -> dict[str, Any]:
    from pareto.rl.formal_jinan_3seed_execution_guard import (
        APPROVED_FORMAL_JINAN_PPO_METHODS,
        FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
        FORMAL_JINAN_SEEDS,
        FORMAL_JINAN_SCENARIO,
        FORMAL_JINAN_TRAFFIC_FILE,
        FILM_MODEL_HASH,
        OBJECTIVE_NORMALIZER_HASH,
        STATE_ENCODER_HASH,
        VECTORQ_MODEL_HASH,
    )

    if approval_phrase != FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE:
        raise ValueError("formal Jinan 3-seed execution requires the exact external approval phrase")
    if method not in APPROVED_FORMAL_JINAN_PPO_METHODS:
        raise ValueError(f"method {method} is not approved for formal Jinan 3-seed execution")
    if int(seed_id) not in FORMAL_JINAN_SEEDS:
        raise ValueError(f"seed_id {seed_id} is not approved for formal Jinan 3-seed execution")
    if int(episodes) != 5:
        raise ValueError("episodes must be exactly 5 for formal Jinan 3-seed execution")
    if int(max_decision_steps_per_episode) != 120:
        raise ValueError("max_decision_steps_per_episode must be exactly 120 for formal Jinan 3-seed execution")
    if int(rollout_steps) != 120:
        raise ValueError("rollout_steps must be exactly 120 for formal Jinan 3-seed execution")
    if int(config.ppo["rollout_steps"]) != int(rollout_steps):
        raise ValueError("config ppo.rollout_steps must match --rollout_steps for formal Jinan 3-seed execution")
    if str(config.pilot.get("scenario")) != FORMAL_JINAN_SCENARIO:
        raise ValueError("formal Jinan 3-seed execution is limited to Jinan")
    if str(config.pilot.get("traffic_file")) != FORMAL_JINAN_TRAFFIC_FILE:
        raise ValueError("formal Jinan 3-seed execution traffic_file mismatch")
    if str(config.pilot.get("state_encoder_hash")) != STATE_ENCODER_HASH:
        raise ValueError("formal Jinan 3-seed execution state_encoder_hash mismatch")
    if str(config.pilot.get("objective_normalizer_hash")) != OBJECTIVE_NORMALIZER_HASH:
        raise ValueError("formal Jinan 3-seed execution objective_normalizer_hash mismatch")
    if method == FORMAL_PILOT_WIRING_METHOD and str(config.pilot.get("vector_model_hash")) != VECTORQ_MODEL_HASH:
        raise ValueError("formal Jinan 3-seed execution vector_model_hash mismatch")
    if method == "film_scalar_potential" and str(config.pilot.get("film_model_hash")) != FILM_MODEL_HASH:
        raise ValueError("formal Jinan 3-seed execution film_model_hash mismatch")
    return {
        "approval_phrase": approval_phrase,
        "method": method,
        "seed_id": int(seed_id),
        "approved_methods": list(APPROVED_FORMAL_JINAN_PPO_METHODS),
        "approved_seeds": list(FORMAL_JINAN_SEEDS),
        "rollout_steps": int(rollout_steps),
    }


def _formal_seed_bound_config(config: FormalPPODryRunConfig, *, seed_id: int) -> FormalPPODryRunConfig:
    payload = config.to_dict()
    payload.pop("source_path", None)
    payload["pilot"] = dict(payload["pilot"])
    payload["pilot"]["cityflow_seed"] = int(seed_id)
    payload["pilot"]["policy_seed"] = int(seed_id)
    payload["pilot"]["model_seed"] = int(seed_id)
    return FormalPPODryRunConfig.from_dict(payload, source_path=config.source_path)


def _formal_jinan_3seed_execution_context(request: dict[str, Any], *, guard_packet: str | None = None) -> dict[str, Any]:
    seed_id = int(request["seed_id"])
    return {
        "formal_jinan_3seed_execution": True,
        "formal_experiment": True,
        "pilot_only": False,
        "pilot_dry_run_execution": False,
        "bounded_jinan_1seed_pilot_dry_run": False,
        "bounded_pilot_not_formal": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "exclude_from_analysis": False,
        "traffic_result_value_reading_executed": False,
        "numeric_aggregation_executed": False,
        "method_ranking_executed": False,
        "performance_table_generated": False,
        "paper_result_claim": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "checkpoint_use": "formal_jinan_3seed_guarded_execution_last_only",
        "result_mode": "formal_jinan_3seed_guarded_execution_raw_no_analysis",
        "status_label": "FORMAL_JINAN_3SEED_RUN_DONE",
        "external_approval_phrase_verified": True,
        "external_approval_phrase": str(request["approval_phrase"]),
        "formal_execution_guard_packet": guard_packet,
        "cityflow_seed": seed_id,
        "policy_seed": seed_id,
        "model_seed": seed_id,
        "rollout_steps": int(request["rollout_steps"]),
    }


def _finalize_formal_jinan_3seed_outputs(run_dir: str | Path) -> None:
    from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_ALLOWED_EXECUTION_OUTPUTS

    root = Path(run_dir)
    for stale_name in (
        "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
        "formal_gate_decision.json",
        "pilot_spec.json",
        "ppo_config.json",
    ):
        path = root / stale_name
        if path.exists():
            path.unlink()
    work_dir = root / "llmlight_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    assert_no_forbidden_performance_artifacts(root)
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in FORMAL_ALLOWED_EXECUTION_OUTPUTS)
    if unexpected:
        raise ValueError(f"formal Jinan 3-seed execution produced non-allowlisted root artifacts: {unexpected}")


def run_formal_jinan_3seed_execution(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    seed_id: int,
    approval_phrase: str,
    episodes: int = 5,
    max_decision_steps_per_episode: int = 120,
    rollout_steps: int = 120,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    vector_model_dir: str | Path | None = None,
    vector_model_hash: str | None = None,
    device: str = "cpu",
    guard_packet: str | None = None,
) -> dict[str, Any]:
    from pareto.rl.formal_jinan_3seed_execution_guard import APPROVED_FORMAL_JINAN_PPO_METHODS, FORMAL_JINAN_SEEDS

    request = validate_formal_jinan_3seed_execution_limits(
        config,
        method,
        seed_id=seed_id,
        approval_phrase=approval_phrase,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        rollout_steps=rollout_steps,
    )
    seed_config = _formal_seed_bound_config(config, seed_id=seed_id)
    context = _formal_jinan_3seed_execution_context(request, guard_packet=guard_packet)
    payload = run_bounded_jinan_pilot_dry_run(
        seed_config,
        method,
        out_dir,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
        vector_model_dir=vector_model_dir,
        vector_model_hash=vector_model_hash,
        device=device,
        approved_seed_ids=tuple(FORMAL_JINAN_SEEDS),
        approved_methods=tuple(APPROVED_FORMAL_JINAN_PPO_METHODS),
        require_final_method_list=False,
        formal_execution_context=context,
    )
    _finalize_formal_jinan_3seed_outputs(out_dir)
    return payload


def validate_paper_final_execution_limits(
    config: FormalPPODryRunConfig,
    method: str,
    *,
    seed_id: int,
    approval_phrase: str,
    rollout_steps: int,
    total_env_steps_per_seed: int,
    episodes: int,
    max_decision_steps_per_episode: int,
    fixed_preference_template: str,
) -> dict[str, Any]:
    if approval_phrase != PAPER_FINAL_EXECUTION_APPROVAL_PHRASE:
        raise ValueError("paper-final execution requires the exact external approval phrase")
    if method not in PAPER_FINAL_PPO_METHODS:
        raise ValueError(f"method {method} is not approved for paper-final learned PPO execution")
    scenario = str(config.pilot.get("scenario"))
    traffic_file = str(config.pilot.get("traffic_file"))
    if scenario not in REQUIRED_CITY_TRAFFIC:
        raise ValueError(f"paper-final execution scenario is not approved: {scenario}")
    if traffic_file != REQUIRED_CITY_TRAFFIC[scenario]:
        raise ValueError(f"paper-final execution traffic_file mismatch for {scenario}")
    if int(seed_id) not in PAPER_FINAL_SEEDS:
        raise ValueError(f"seed_id {seed_id} is not approved for paper-final execution")
    if int(rollout_steps) != 3600:
        raise ValueError("rollout_steps must be exactly 3600 for paper-final execution")
    if int(config.ppo["rollout_steps"]) != int(rollout_steps):
        raise ValueError("config ppo.rollout_steps must match --rollout_steps for paper-final execution")
    if int(total_env_steps_per_seed) != PAPER_FINAL_TOTAL_ENV_STEPS_PER_SEED:
        raise ValueError("total_env_steps_per_seed must be exactly 1000000 for paper-final execution")
    if int(config.ppo.get("total_env_steps_per_seed", -1)) != int(total_env_steps_per_seed):
        raise ValueError("config ppo.total_env_steps_per_seed must match --total_env_steps_per_seed")
    actual_sim_seconds = int(episodes) * int(max_decision_steps_per_episode) * PAPER_FINAL_MIN_ACTION_TIME_SECONDS
    if int(episodes) != PAPER_FINAL_EPISODES_PER_SEED or int(max_decision_steps_per_episode) != PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE:
        raise ValueError(
            "paper-final execution run shape must be "
            f"episodes={PAPER_FINAL_EPISODES_PER_SEED} and "
            f"max_decision_steps_per_episode={PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE}; "
            f"got episodes={episodes}, max_decision_steps_per_episode={max_decision_steps_per_episode}"
        )
    if actual_sim_seconds < int(total_env_steps_per_seed):
        raise ValueError("paper-final execution run shape does not cover total_env_steps_per_seed")
    if fixed_preference_template not in REQUIRED_PREFERENCE_TEMPLATES:
        raise ValueError(f"unknown paper-final fixed preference template: {fixed_preference_template}")
    return {
        "approval_phrase": approval_phrase,
        "method": method,
        "seed_id": int(seed_id),
        "scenario": scenario,
        "traffic_file": traffic_file,
        "approved_methods": list(PAPER_FINAL_PPO_METHODS),
        "approved_seeds": list(PAPER_FINAL_SEEDS),
        "rollout_steps": int(rollout_steps),
        "total_env_steps_per_seed": int(total_env_steps_per_seed),
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "min_action_time_seconds": PAPER_FINAL_MIN_ACTION_TIME_SECONDS,
        "actual_sim_seconds_per_seed": int(actual_sim_seconds),
        "paper_final_run_length_policy": "ceil_to_cover_total_env_steps_per_seed",
        "fixed_preference_template": fixed_preference_template,
        "fixed_preference_weights": list(REQUIRED_PREFERENCE_TEMPLATES[fixed_preference_template]),
    }


def _paper_final_seed_bound_config(
    config: FormalPPODryRunConfig,
    *,
    seed_id: int,
    method: str,
    objective_normalizer_hash: str,
    film_model_hash: str | None = None,
    vector_model_hash: str | None = None,
    guard_packet: str | None = None,
) -> FormalPPODryRunConfig:
    payload = config.to_dict()
    payload.pop("source_path", None)
    payload["pilot"] = dict(payload["pilot"])
    payload["pilot"]["cityflow_seed"] = int(seed_id)
    payload["pilot"]["policy_seed"] = int(seed_id)
    payload["pilot"]["model_seed"] = int(seed_id)
    payload["pilot"]["state_encoder_hash"] = str(payload["pilot"].get("state_encoder_hash") or PAPER_FINAL_STATE_ENCODER_HASH)
    payload["pilot"]["objective_normalizer_hash"] = str(objective_normalizer_hash)
    payload["pilot"]["formal_gate_decision_path"] = str(guard_packet or "docs/pro_reviews/paper_final_execution_guard_not_supplied.json")
    payload["pilot"]["pilot_spec_path"] = str(config.source_path or "configs/formal/paper_final_unknown_5seed_ppo.json")
    if method == "film_scalar_potential":
        payload["pilot"]["film_model_hash"] = str(film_model_hash or "")
    if method == FORMAL_PILOT_WIRING_METHOD:
        payload["pilot"]["vector_model_hash"] = str(vector_model_hash or "")
    return FormalPPODryRunConfig.from_dict(payload, source_path=config.source_path)


def _paper_final_execution_context(request: dict[str, Any], *, guard_packet: str | None = None) -> dict[str, Any]:
    seed_id = int(request["seed_id"])
    return {
        "paper_final_execution": True,
        "paper_final_scope_limited_execution": True,
        "formal_experiment": True,
        "pilot_only": False,
        "pilot_dry_run_execution": False,
        "bounded_jinan_1seed_pilot_dry_run": False,
        "bounded_pilot_not_formal": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "exclude_from_analysis": False,
        "traffic_result_value_reading_executed": False,
        "numeric_aggregation_executed": False,
        "method_ranking_executed": False,
        "performance_table_generated": False,
        "paper_result_claim": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "checkpoint_use": "paper_final_guarded_execution_last_only",
        "result_mode": "paper_final_scope_limited_guarded_execution_raw_no_analysis",
        "status_label": "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE",
        "external_approval_phrase_verified": True,
        "external_approval_phrase": str(request["approval_phrase"]),
        "formal_execution_guard_packet": guard_packet,
        "scenario": str(request["scenario"]),
        "traffic_file": str(request["traffic_file"]),
        "cityflow_seed": seed_id,
        "policy_seed": seed_id,
        "model_seed": seed_id,
        "rollout_steps": int(request["rollout_steps"]),
        "total_env_steps_per_seed": int(request["total_env_steps_per_seed"]),
        "episodes": int(request["episodes"]),
        "max_decision_steps_per_episode": int(request["max_decision_steps_per_episode"]),
        "min_action_time_seconds": int(request["min_action_time_seconds"]),
        "actual_sim_seconds_per_seed": int(request["actual_sim_seconds_per_seed"]),
        "paper_final_run_length_policy": str(request["paper_final_run_length_policy"]),
        "fixed_preference_template": str(request["fixed_preference_template"]),
        "fixed_preference_weights": list(request["fixed_preference_weights"]),
    }


def _finalize_paper_final_execution_outputs(run_dir: str | Path) -> None:
    from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_ALLOWED_EXECUTION_OUTPUTS

    root = Path(run_dir)
    for stale_name in (
        "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
        "formal_gate_decision.json",
        "pilot_spec.json",
        "ppo_config.json",
    ):
        path = root / stale_name
        if path.exists():
            path.unlink()
    work_dir = root / "llmlight_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    assert_no_forbidden_performance_artifacts(root)
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in FORMAL_ALLOWED_EXECUTION_OUTPUTS)
    if unexpected:
        raise ValueError(f"paper-final execution produced non-allowlisted root artifacts: {unexpected}")


def run_paper_final_execution(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    seed_id: int,
    approval_phrase: str,
    rollout_steps: int,
    total_env_steps_per_seed: int,
    fixed_preference_template: str,
    episodes: int = 5,
    max_decision_steps_per_episode: int = 120,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    vector_model_dir: str | Path | None = None,
    vector_model_hash: str | None = None,
    device: str = "cpu",
    guard_packet: str | None = None,
) -> dict[str, Any]:
    if objective_normalizer_hash is None:
        raise ValueError("paper-final execution requires --objective_normalizer_hash")
    request = validate_paper_final_execution_limits(
        config,
        method,
        seed_id=seed_id,
        approval_phrase=approval_phrase,
        rollout_steps=rollout_steps,
        total_env_steps_per_seed=total_env_steps_per_seed,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        fixed_preference_template=fixed_preference_template,
    )
    seed_config = _paper_final_seed_bound_config(
        config,
        seed_id=seed_id,
        method=method,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_hash=film_model_hash,
        vector_model_hash=vector_model_hash,
        guard_packet=guard_packet,
    )
    context = _paper_final_execution_context(request, guard_packet=guard_packet)
    payload = run_bounded_jinan_pilot_dry_run(
        seed_config,
        method,
        out_dir,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
        vector_model_dir=vector_model_dir,
        vector_model_hash=vector_model_hash,
        device=device,
        approved_seed_ids=tuple(PAPER_FINAL_SEEDS),
        approved_methods=tuple(PAPER_FINAL_PPO_METHODS),
        require_final_method_list=False,
        formal_execution_context=context,
        approved_city_traffic=dict(REQUIRED_CITY_TRAFFIC),
        max_episodes=PAPER_FINAL_EPISODES_PER_SEED,
        max_decision_steps=PAPER_FINAL_MAX_DECISION_STEPS_PER_EPISODE,
        scope_label="paper-final learned PPO execution",
        fixed_preference_template=fixed_preference_template,
    )
    _finalize_paper_final_execution_outputs(out_dir)
    return payload


def _clean_tiny_update_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "metadata.json",
        "status.json",
        "formal_gate_decision.json",
        "pilot_spec.json",
        "ppo_config.json",
        "reward_components.jsonl",
        "train_metrics.jsonl",
        "loss_debug.jsonl",
        "action_debug.jsonl",
        "training_checkpoint_last.pt",
        "TINY_PPO_PREFLIGHT_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()


def _clean_bounded_pilot_output_dir(out: Path) -> None:
    _clean_tiny_update_output_dir(out)
    for stale_name in (
        "eval_metrics.jsonl",
        "pilot_debug_metrics.csv",
        "checkpoint_last.pt",
        "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()
    work_dir = out / "llmlight_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _single_real_ppo_update_from_buffer(
    model: PreferenceConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PPORolloutBuffer,
    config: FormalPPODryRunConfig,
    out: Path,
) -> dict[str, float]:
    if len(buffer) == 0:
        raise ValueError("tiny real PPO-update preflight requires a non-empty rollout buffer")
    device = next(model.parameters()).device
    batch = buffer.compute_returns_and_advantages(
        last_value=0.0,
        gamma=float(config.ppo["gamma"]),
        gae_lambda=float(config.ppo["gae_lambda"]),
        normalize_advantages=bool(config.ppo["normalize_advantages"]),
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    new_log_probs, entropy, values = model.evaluate_actions(batch["obs"], batch["w"], batch["actions"])
    loss = clipped_ppo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=batch["old_log_probs"],
        advantages=batch["advantages"],
        values=values,
        returns=batch["returns"],
        entropy=entropy,
        clip_eps=float(config.ppo["clip_eps"]),
        value_loss_coef=float(config.ppo["value_loss_coef"]),
        entropy_coef=float(config.ppo["entropy_coef"]),
    )
    optimizer.zero_grad()
    loss["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.ppo["max_grad_norm"]))
    optimizer.step()
    row = {
        "policy_update": 1,
        "policy_loss": float(loss["policy_loss"].detach().item()),
        "value_loss": float(loss["value_loss"].detach().item()),
        "entropy_bonus": float(loss["entropy_bonus"].detach().item()),
        "approx_kl": float(loss["approx_kl"].detach().item()),
        "clip_fraction": float(loss["clip_fraction"].detach().item()),
        "ratio_mean": float(loss["ratio_mean"].detach().item()),
        "ratio_min": float(loss["ratio_min"].detach().item()),
        "ratio_max": float(loss["ratio_max"].detach().item()),
        "total_loss": float(loss["total_loss"].detach().item()),
        "grad_norm": float(grad_norm),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in row.values()):
        raise ValueError("non-finite tiny real PPO-update loss")
    append_jsonl(out / "loss_debug.jsonl", [row])
    return row


def _save_tiny_training_checkpoint(
    path: str | Path,
    model: PreferenceConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    *,
    total_steps: int,
    episodes: int,
    max_policy_updates: int,
) -> None:
    save_training_checkpoint(
        path,
        model,
        optimizer,
        metadata,
        step=int(total_steps),
        episode=int(episodes),
        global_update=int(max_policy_updates),
    )


def _tiny_update_metadata(
    config: FormalPPODryRunConfig,
    method: str,
    episodes: int,
    max_decision_steps_per_episode: int,
    max_policy_updates: int,
    *,
    objective_normalizer_path: str | None,
    objective_normalizer_hash: str | None,
    objective_normalizer_hash_expected: str | None,
    objective_normalizer_hash_verified: bool,
    objective_normalizer_used_by_reward: bool,
    film_model_dir: str | None,
    film_model_hash: str | None,
    film_model_hash_expected: str | None,
    film_model_hash_verified: bool,
    film_model_source: str | None,
    film_model_loaded: bool,
    vector_model_dir: str | None = None,
    vector_model_hash: str | None = None,
    vector_model_hash_expected: str | None = None,
    vector_model_hash_verified: bool = False,
    vector_model_loaded: bool = False,
    vector_model_config: dict[str, Any] | None = None,
    observed_feature_hash: str | None = None,
    observed_obs_dim: int | None = None,
) -> dict[str, Any]:
    hash_metadata = _pilot_hash_metadata(config, method)
    film_model_hash_config = hash_metadata.pop("film_model_hash", None)
    vector_model_hash_config = hash_metadata.pop("vector_model_hash", None)
    film_model_required = method == "film_scalar_potential"
    vector_model_required = method == FORMAL_PILOT_WIRING_METHOD
    return {
        "tiny_real_ppo_update_preflight": True,
        "tiny_preflight_not_pilot": True,
        "exclude_from_analysis": True,
        "checkpoint_use": "preflight_resume_test_only",
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "mock_env": False,
        "real_env_rollout": True,
        "cityflow_env_constructed": True,
        "cityflow_step_called": True,
        "ppo_training": True,
        "real_ppo_update": True,
        "policy_update": True,
        "optimizer_step": True,
        "max_policy_updates": int(max_policy_updates),
        "policy_update_count": int(max_policy_updates),
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "min_action_time": 30,
        "sim_seconds_per_method": int(episodes * max_decision_steps_per_episode * 30),
        "method": method,
        "method_display_name": _method_display_name(method),
        "reward_adapter": method,
        "reward_adapter_semantics": "queue_length_penalty_proxy" if method == "env_reward" else method,
        "scenario": config.pilot["scenario"],
        "traffic_file": config.pilot["traffic_file"],
        "state_encoder_hash": observed_feature_hash or str(config.pilot["state_encoder_hash"]),
        "obs_dim": observed_obs_dim,
        "objective_normalizer_path": objective_normalizer_path,
        "objective_normalizer_hash": objective_normalizer_hash,
        "objective_normalizer_hash_expected": objective_normalizer_hash_expected,
        "objective_normalizer_hash_verified": bool(objective_normalizer_hash_verified),
        "objective_normalizer_loaded": True,
        "objective_normalizer_used_by_reward": bool(objective_normalizer_used_by_reward),
        "film_model_required": film_model_required,
        "film_model_dir": film_model_dir,
        "film_model_hash": film_model_hash,
        "film_model_hash_config": film_model_hash_config,
        "film_model_hash_expected": film_model_hash_expected,
        "film_model_hash_verified": bool(film_model_hash_verified),
        "film_model_source": film_model_source,
        "film_model_loaded": bool(film_model_loaded),
        "smoke_scalar_quality_model": film_model_source == "smoke_scalar_quality_model",
        "vector_model_required": bool(vector_model_required),
        "vector_model_dir": vector_model_dir,
        "vector_model_hash": vector_model_hash,
        "vector_model_hash_expected": vector_model_hash_expected,
        "vector_model_hash_config": vector_model_hash_config,
        "vector_model_hash_verified": bool(vector_model_hash_verified),
        "vector_model_loaded": bool(vector_model_loaded),
        "vector_model_architecture": (vector_model_config or {}).get("architecture"),
        "vector_model_score_mode": (vector_model_config or {}).get("score_mode"),
        **hash_metadata,
    }


def _run_tiny_real_ppo_update_core(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    episodes: int,
    max_decision_steps_per_episode: int,
    max_policy_updates: int,
    objective_normalizer: str | Path,
    objective_normalizer_hash: str,
    film_model_dir: str | Path | None,
    film_model_hash: str | None,
    device: str = "cpu",
) -> dict[str, Any]:
    random.seed(int(config.pilot.get("policy_seed", 0)))
    np.random.seed(int(config.pilot.get("policy_seed", 0)))
    torch.manual_seed(int(config.pilot.get("model_seed", 0)))
    device_obj = torch.device(device)

    out = Path(out_dir)
    _clean_tiny_update_output_dir(out)
    write_json(out / "ppo_config.json", {"ppo": config.ppo, "ppo_config_hash": config.ppo_config_hash()})
    _copy_or_write_json(config.pilot["formal_gate_decision_path"], out / "formal_gate_decision.json", {"ppo_formal_allowed": False})
    _copy_or_write_json(config.pilot["pilot_spec_path"], out / "pilot_spec.json", {"missing_source": config.pilot["pilot_spec_path"]})

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
    normalizer_hash_verified = validate_hash("objective_normalizer_hash", normalizer_actual_hash, objective_normalizer_hash)
    if method == "film_scalar_potential":
        film_model_actual_hash = sha256_file(Path(film_model_dir or "") / "model.pt")
        film_model_hash_verified = validate_hash("film_model_hash", film_model_actual_hash, film_model_hash)
        scalar_model = _load_film_scalar_model(str(film_model_dir), device_obj)
        film_model_source = "trained_film_checkpoint"
        film_model_loaded = True
    else:
        film_model_actual_hash = None
        film_model_hash_verified = False
        scalar_model = MockScalarQualityModel()
        film_model_source = None
        film_model_loaded = False

    encoder = ParetoStateEncoder("hybrid_v1")
    sampler = EpisodeFixedPreferenceSampler()
    policy = PreferenceConditionedActorCritic(
        obs_dim=int(config.model["obs_dim"]),
        preference_dim=int(config.model["preference_dim"]),
        action_dim=int(config.model["action_dim"]),
        hidden_dim=int(config.model["hidden_dim"]),
    ).to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.ppo["lr"]))
    buffer = PPORolloutBuffer()
    total_steps = 0
    reward_row_count = 0
    env_reward_rows: list[dict[str, Any]] = []
    observed_feature_hash: str | None = None
    observed_obs_dim: int | None = None

    for episode in range(int(episodes)):
        env.reset()
        prev_snapshots: list[Any | None] = [None for _ in env.list_intersection]
        prev_actions: list[int | None] = [None for _ in env.list_intersection]
        preference_name, w_tuple = sampler.preference_for_episode(episode)
        w_tensor = torch.tensor(w_tuple, dtype=torch.float32, device=device_obj)
        for step in range(int(max_decision_steps_per_episode)):
            snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_t, names, debug_rows = _encode_snapshots(
                snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=str(config.pilot["state_encoder_hash"]),
            )
            observed_feature_hash = debug_rows[0]["feature_names_hash"]
            observed_obs_dim = int(obs_t.shape[-1])
            w_batch = w_tensor.reshape(1, -1).expand(obs_t.shape[0], -1)
            with torch.no_grad():
                actions, log_probs, values = policy.act(obs_t.to(device_obj), w_batch)
            action_list = [int(value) for value in actions.detach().cpu().tolist()]
            if len(action_list) != len(env.list_intersection):
                raise ValueError("action_list length does not match num intersections")
            sim_time_before = int(env.get_current_time())
            _, final_env_rewards, done, average_env_rewards = env.step(action_list)
            env_rewards, reward_source_debug = select_cityflow_env_rewards(final_env_rewards, average_env_rewards)
            sim_time_after = int(env.get_current_time())
            next_snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_tp1, _, _ = _encode_snapshots(
                next_snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=str(config.pilot["state_encoder_hash"]),
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
                reward, debug = compute_real_cityflow_reward(
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
                buffer.add(
                    obs=obs_t[idx],
                    w=w_tensor.detach().cpu(),
                    action=action_list[idx],
                    reward=float(reward),
                    done=bool(done),
                    value=float(values[idx].detach().cpu().item()),
                    log_prob=float(log_probs[idx].detach().cpu().item()),
                )
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
            if method == "env_reward":
                env_reward_rows.extend(reward_rows)
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
                        "sim_time_before": sim_time_before,
                        "sim_time_after": sim_time_after,
                        **reward_source_debug,
                        **reward_info_debug,
                    }
                ],
            )
            append_jsonl(
                out / "train_metrics.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time": sim_time_after,
                        "preference_name": preference_name,
                        "reward_mean": float(np.mean(step_rewards)),
                        "reward_min": float(np.min(step_rewards)),
                        "reward_max": float(np.max(step_rewards)),
                    }
                ],
            )
            total_steps += 1
            if done:
                break
    env.batch_log_2()
    loss_row = _single_real_ppo_update_from_buffer(policy, optimizer, buffer, config, out)
    env_reward_summary: dict[str, Any] | None = None
    if method == "env_reward":
        env_reward_summary = _summarize_tiny_env_reward_rows(env_reward_rows)
        if not env_reward_summary["finite"]:
            raise ValueError("fixed env_reward tiny preflight produced non-finite reward")
        if env_reward_summary["all_zero_reward"]:
            raise ValueError("fixed env_reward tiny preflight still produced all-zero reward")
    metadata = _tiny_update_metadata(
        config,
        method,
        episodes,
        max_decision_steps_per_episode,
        max_policy_updates,
        objective_normalizer_path=str(objective_normalizer),
        objective_normalizer_hash=normalizer_actual_hash,
        objective_normalizer_hash_expected=objective_normalizer_hash,
        objective_normalizer_hash_verified=normalizer_hash_verified,
        objective_normalizer_used_by_reward=method == "weighted_proxy",
        film_model_dir=str(film_model_dir) if film_model_dir is not None else None,
        film_model_hash=film_model_actual_hash,
        film_model_hash_expected=film_model_hash,
        film_model_hash_verified=film_model_hash_verified,
        film_model_source=film_model_source,
        film_model_loaded=film_model_loaded,
        observed_feature_hash=observed_feature_hash,
        observed_obs_dim=observed_obs_dim,
    )
    metadata.update(reward_info_debug)
    if env_reward_summary is not None:
        metadata["env_reward_summary"] = env_reward_summary
    _save_tiny_training_checkpoint(
        out / "training_checkpoint_last.pt",
        policy,
        optimizer,
        metadata,
        total_steps=total_steps,
        episodes=int(episodes),
        max_policy_updates=int(max_policy_updates),
    )
    write_json(out / "metadata.json", metadata)
    status = {
        "status": "TINY_PPO_PREFLIGHT_DONE",
        "steps": total_steps,
        "reward_row_count": reward_row_count,
        "policy_update_count": int(max_policy_updates),
        "loss_debug_finite": True,
        "forbidden_artifacts": [],
        "last_loss": loss_row,
        **metadata,
    }
    write_json(out / "status.json", status)
    (out / "TINY_PPO_PREFLIGHT_DONE").write_text(
        "tiny real PPO-update preflight completed; not a pilot, not a performance result\n",
        encoding="utf-8",
    )
    assert_no_forbidden_performance_artifacts(out)
    return status


def run_tiny_real_ppo_update_preflight(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    episodes: int = 1,
    max_decision_steps_per_episode: int = 2,
    max_policy_updates: int = 1,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    device: str = "cpu",
    allow_nonfilm_tiny_preflight: bool = False,
) -> dict[str, Any]:
    validate_tiny_real_ppo_update_limits(episodes, max_decision_steps_per_episode, max_policy_updates)
    if method not in config.pilot["methods"]:
        raise ValueError(f"method {method} is not in pilot methods")
    if method != "film_scalar_potential" and not allow_nonfilm_tiny_preflight:
        raise ValueError("the first tiny real PPO-update preflight is limited to film_scalar_potential")
    validate_real_cityflow_preflight_artifacts(
        method=method,
        reward_readiness=True,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
    )
    _validate_formal_real_env_hash_lock(config, method, objective_normalizer_hash, film_model_hash)
    payload = _run_tiny_real_ppo_update_core(
        config,
        method,
        out_dir,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        max_policy_updates=max_policy_updates,
        objective_normalizer=str(objective_normalizer),
        objective_normalizer_hash=str(objective_normalizer_hash),
        film_model_dir=film_model_dir,
        film_model_hash=film_model_hash,
        device=device,
    )
    out = Path(out_dir)
    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "tiny_real_ppo_update_preflight": True,
            "tiny_preflight_not_pilot": True,
            "exclude_from_analysis": True,
            "checkpoint_use": "preflight_resume_test_only",
            "pilot_execution": False,
            "formal_experiment": False,
            "performance_claim": False,
            "not_for_main_results": True,
            "real_env_rollout": True,
            "cityflow_env_constructed": True,
            "cityflow_step_called": True,
            "ppo_training": True,
            "real_ppo_update": True,
            "policy_update": True,
            "optimizer_step": True,
            "max_policy_updates": int(max_policy_updates),
            "policy_update_count": int(payload.get("policy_update_count", max_policy_updates)),
            "episodes": int(episodes),
            "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
            "min_action_time": 30,
            "sim_seconds_per_method": int(episodes * max_decision_steps_per_episode * 30),
            "method": method,
            "objective_normalizer_used_by_reward": method == "weighted_proxy",
            "film_model_required": method == "film_scalar_potential",
        }
    )
    if "film_model_hash_config" not in metadata:
        metadata["film_model_hash_config"] = config.pilot.get("film_model_hash")
    write_json(metadata_path, metadata)
    status_path = out / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(
        {
            "status": "TINY_PPO_PREFLIGHT_DONE",
            "steps": int(payload.get("steps", status.get("steps", 0))),
            "policy_update_count": int(payload.get("policy_update_count", metadata["policy_update_count"])),
            "forbidden_artifacts": [],
            **metadata,
        }
    )
    write_json(status_path, status)
    assert_no_forbidden_performance_artifacts(out)
    return status


def run_bounded_jinan_pilot_dry_run(
    config: FormalPPODryRunConfig,
    method: str,
    out_dir: str | Path,
    *,
    episodes: int = 5,
    max_decision_steps_per_episode: int = 120,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    film_model_dir: str | Path | None = None,
    film_model_hash: str | None = None,
    vector_model_dir: str | Path | None = None,
    vector_model_hash: str | None = None,
    device: str = "cpu",
    approved_seed_ids: tuple[int, ...] = (0,),
    approved_methods: tuple[str, ...] = FINAL_JINAN_PILOT_METHODS,
    require_final_method_list: bool = True,
    formal_execution_context: dict[str, Any] | None = None,
    approved_city_traffic: dict[str, str] | None = None,
    max_episodes: int = 5,
    max_decision_steps: int = 120,
    scope_label: str = "bounded Jinan pilot dry-run",
    fixed_preference_template: str | None = None,
) -> dict[str, Any]:
    validate_bounded_jinan_pilot_dry_run_limits(
        config,
        method,
        episodes,
        max_decision_steps_per_episode,
        approved_seed_ids=approved_seed_ids,
        approved_methods=approved_methods,
        require_final_method_list=require_final_method_list,
        approved_city_traffic=approved_city_traffic,
        max_episodes=max_episodes,
        max_decision_steps=max_decision_steps,
        scope_label=scope_label,
    )
    if fixed_preference_template is not None and fixed_preference_template not in REQUIRED_PREFERENCE_TEMPLATES:
        raise ValueError(f"unknown fixed preference template for {scope_label}: {fixed_preference_template}")
    if method == FORMAL_PILOT_WIRING_METHOD:
        _validate_vector_model_artifacts(vector_model_dir=vector_model_dir, vector_model_hash=vector_model_hash)
        validate_real_cityflow_preflight_artifacts(
            method=method,
            reward_readiness=True,
            objective_normalizer=objective_normalizer,
            objective_normalizer_hash=objective_normalizer_hash,
        )
        _validate_formal_real_env_hash_lock(
            config,
            method,
            objective_normalizer_hash,
            film_model_hash,
            vector_model_hash=vector_model_hash,
        )
    else:
        validate_real_cityflow_preflight_artifacts(
            method=method,
            reward_readiness=True,
            objective_normalizer=objective_normalizer,
            objective_normalizer_hash=objective_normalizer_hash,
            film_model_dir=film_model_dir,
            film_model_hash=film_model_hash,
        )
        _validate_formal_real_env_hash_lock(config, method, objective_normalizer_hash, film_model_hash)

    random.seed(int(config.pilot.get("policy_seed", 0)))
    np.random.seed(int(config.pilot.get("policy_seed", 0)))
    torch.manual_seed(int(config.pilot.get("model_seed", 0)))
    device_obj = torch.device(device)
    out = Path(out_dir)
    _clean_bounded_pilot_output_dir(out)
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(out / "ppo_config.json", {"ppo": config.ppo, "ppo_config_hash": config.ppo_config_hash()})
    _copy_or_write_json(config.pilot["formal_gate_decision_path"], out / "formal_gate_decision.json", {"ppo_formal_allowed": False})
    _copy_or_write_json(config.pilot["pilot_spec_path"], out / "pilot_spec.json", {"missing_source": config.pilot["pilot_spec_path"]})

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
    normalizer_hash_verified = validate_hash("objective_normalizer_hash", normalizer_actual_hash, objective_normalizer_hash)
    if method == "film_scalar_potential":
        film_model_actual_hash = sha256_file(Path(film_model_dir or "") / "model.pt")
        film_model_hash_verified = validate_hash("film_model_hash", film_model_actual_hash, film_model_hash)
        scalar_model = _load_film_scalar_model(str(film_model_dir), device_obj)
        film_model_source = "trained_film_checkpoint"
        film_model_loaded = True
        vector_model_actual_hash = None
        vector_model_hash_verified = False
        vector_model_loaded = False
        vector_model_config = None
        vector_model = None
        vector_scorer = None
    elif method == FORMAL_PILOT_WIRING_METHOD:
        vector_model_actual_hash = sha256_file(Path(vector_model_dir or "") / "model.pt")
        vector_model_hash_verified = validate_hash("vector_model_hash", vector_model_actual_hash, vector_model_hash)
        vector_model, vector_scorer, vector_model_config = _load_vector_quality_model(str(vector_model_dir), device_obj)
        vector_model_loaded = True
        film_model_actual_hash = None
        film_model_hash_verified = False
        scalar_model = MockScalarQualityModel()
        film_model_source = None
        film_model_loaded = False
    else:
        film_model_actual_hash = None
        film_model_hash_verified = False
        scalar_model = MockScalarQualityModel()
        film_model_source = None
        film_model_loaded = False
        vector_model_actual_hash = None
        vector_model_hash_verified = False
        vector_model_loaded = False
        vector_model_config = None
        vector_model = None
        vector_scorer = None

    encoder = ParetoStateEncoder("hybrid_v1")
    sampler = EpisodeFixedPreferenceSampler()
    policy = PreferenceConditionedActorCritic(
        obs_dim=int(config.model["obs_dim"]),
        preference_dim=int(config.model["preference_dim"]),
        action_dim=int(config.model["action_dim"]),
        hidden_dim=int(config.model["hidden_dim"]),
    ).to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.ppo["lr"]))
    buffer = PPORolloutBuffer()
    rollout_transition_budget = int(config.ppo["rollout_steps"])
    global_update = 0
    total_steps = 0
    reward_row_count = 0
    env_reward_rows: list[dict[str, Any]] = []
    reward_values: list[float] = []
    global_action_counts: Counter = Counter()
    intersection_action_counts: list[Counter] | None = None
    observed_feature_hash: str | None = None
    observed_obs_dim: int | None = None

    for episode in range(int(episodes)):
        env.reset()
        if intersection_action_counts is None:
            intersection_action_counts = [Counter() for _ in env.list_intersection]
        prev_snapshots: list[Any | None] = [None for _ in env.list_intersection]
        prev_actions: list[int | None] = [None for _ in env.list_intersection]
        if fixed_preference_template is None:
            preference_name, w_tuple = sampler.preference_for_episode(episode)
        else:
            preference_name = fixed_preference_template
            w_tuple = tuple(float(value) for value in REQUIRED_PREFERENCE_TEMPLATES[fixed_preference_template])
        w_tensor = torch.tensor(w_tuple, dtype=torch.float32, device=device_obj)
        for step in range(int(max_decision_steps_per_episode)):
            snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_t, names, debug_rows = _encode_snapshots(
                snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=str(config.pilot["state_encoder_hash"]),
            )
            observed_feature_hash = debug_rows[0]["feature_names_hash"]
            observed_obs_dim = int(obs_t.shape[-1])
            w_batch = w_tensor.reshape(1, -1).expand(obs_t.shape[0], -1)
            with torch.no_grad():
                actions, log_probs, values = policy.act(obs_t.to(device_obj), w_batch)
            action_list = [int(value) for value in actions.detach().cpu().tolist()]
            if len(action_list) != len(env.list_intersection):
                raise ValueError("action_list length does not match num intersections")
            global_action_counts.update(action_list)
            for idx, action in enumerate(action_list):
                intersection_action_counts[idx].update([action])
            sim_time_before = int(env.get_current_time())
            _, final_env_rewards, done, average_env_rewards = env.step(action_list)
            env_rewards, reward_source_debug = select_cityflow_env_rewards(final_env_rewards, average_env_rewards)
            sim_time_after = int(env.get_current_time())
            next_snapshots = [capture_snapshot(env, idx) for idx in range(len(env.list_intersection))]
            obs_tp1, _, _ = _encode_snapshots(
                next_snapshots,
                encoder,
                expected_obs_dim=int(config.model["obs_dim"]),
                expected_feature_hash=str(config.pilot["state_encoder_hash"]),
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
                if method == FORMAL_PILOT_WIRING_METHOD:
                    adapter = VectorQRewardAdapter(
                        vector_model,
                        w_tuple,
                        scorer=vector_scorer,
                        gamma=float(config.ppo["gamma"]),
                        device=device_obj,
                    )
                    reward, debug = adapter.compute(
                        obs_t[idx].cpu(),
                        obs_tp1[idx].cpu(),
                        objective_t.norm,
                        objective_tp1.norm,
                        bool(done),
                    )
                else:
                    reward, debug = compute_real_cityflow_reward(
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
                if not np.isfinite(reward):
                    raise ValueError("non-finite bounded pilot reward")
                reward_values.append(float(reward))
                buffer.add(
                    obs=obs_t[idx],
                    w=w_tensor.detach().cpu(),
                    action=action_list[idx],
                    reward=float(reward),
                    done=bool(done),
                    value=float(values[idx].detach().cpu().item()),
                    log_prob=float(log_probs[idx].detach().cpu().item()),
                )
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
            if method == "env_reward":
                env_reward_rows.extend(reward_rows)
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
                        "sim_time_before": sim_time_before,
                        "sim_time_after": sim_time_after,
                        **reward_source_debug,
                        **reward_info_debug,
                    }
                ],
            )
            append_jsonl(
                out / "train_metrics.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time": sim_time_after,
                        "preference_name": preference_name,
                        "reward_finite": bool(all(np.isfinite(value) for value in step_rewards)),
                        "row_count": len(reward_rows),
                    }
                ],
            )
            total_steps += 1
            if len(buffer) >= rollout_transition_budget:
                global_update = _update_from_buffer(policy, optimizer, buffer, config, out, global_update)
                buffer = PPORolloutBuffer()
            if done:
                break
    if len(buffer) > 0:
        global_update = _update_from_buffer(policy, optimizer, buffer, config, out, global_update)
    env.batch_log_2()
    if not reward_values:
        raise ValueError("bounded exploratory pilot produced no rewards")
    if not all(np.isfinite(value) for value in reward_values):
        raise ValueError("bounded exploratory pilot produced non-finite reward")
    if all(abs(value) <= 1e-12 for value in reward_values):
        raise ValueError("bounded exploratory pilot produced all-zero reward")
    action_guard = _summarize_action_distribution_guard(global_action_counts, intersection_action_counts or [])

    env_reward_summary: dict[str, Any] | None = None
    if method == "env_reward":
        env_reward_summary = _summarize_tiny_env_reward_rows(env_reward_rows)
        if not env_reward_summary["finite"]:
            raise ValueError("bounded EnvReward-QueuePenalty dry-run produced non-finite env_reward")
        if env_reward_summary["all_zero_reward"]:
            raise ValueError("bounded EnvReward-QueuePenalty dry-run produced all-zero env_reward")

    metadata = _tiny_update_metadata(
        config,
        method,
        episodes,
        max_decision_steps_per_episode,
        global_update,
        objective_normalizer_path=str(objective_normalizer),
        objective_normalizer_hash=normalizer_actual_hash,
        objective_normalizer_hash_expected=objective_normalizer_hash,
        objective_normalizer_hash_verified=normalizer_hash_verified,
        objective_normalizer_used_by_reward=method == "weighted_proxy",
        film_model_dir=str(film_model_dir) if film_model_dir is not None else None,
        film_model_hash=film_model_actual_hash,
        film_model_hash_expected=film_model_hash,
        film_model_hash_verified=film_model_hash_verified,
        film_model_source=film_model_source,
        film_model_loaded=film_model_loaded,
        vector_model_dir=str(vector_model_dir) if vector_model_dir is not None else None,
        vector_model_hash=vector_model_actual_hash,
        vector_model_hash_expected=vector_model_hash,
        vector_model_hash_verified=vector_model_hash_verified,
        vector_model_loaded=vector_model_loaded,
        vector_model_config=vector_model_config,
        observed_feature_hash=observed_feature_hash,
        observed_obs_dim=observed_obs_dim,
    )
    metadata.update(
        {
            "tiny_real_ppo_update_preflight": False,
            "tiny_preflight_not_pilot": False,
            "bounded_jinan_1seed_pilot_dry_run": True,
            "bounded_pilot_not_formal": True,
            "pilot_dry_run_execution": True,
            "pilot_only": True,
            "exclude_from_analysis": True,
            "checkpoint_use": "bounded_pilot_dry_run_resume_test_only",
            "result_mode": "bounded_jinan_1seed_pilot_dry_run",
            "method_display_name": _method_display_name(method),
            "reference_only_methods": list(FINAL_JINAN_REFERENCE_ONLY_METHODS),
            "action_guard": action_guard,
            "cityflow_seed": int(config.pilot.get("cityflow_seed", 0)),
            "policy_seed": int(config.pilot.get("policy_seed", 0)),
            "model_seed": int(config.pilot.get("model_seed", 0)),
            "policy_update_count": int(global_update),
            "max_policy_updates": int(global_update),
            "sim_seconds_per_method": int(total_steps * min_action_time),
            **reward_info_debug,
        }
    )
    if formal_execution_context is not None:
        metadata.update(formal_execution_context)
    else:
        metadata.update(_exploratory_pilot_gate_metadata(config, method))
    if env_reward_summary is not None:
        metadata["env_reward_summary"] = env_reward_summary
    _save_tiny_training_checkpoint(
        out / "training_checkpoint_last.pt",
        policy,
        optimizer,
        metadata,
        total_steps=total_steps,
        episodes=int(episodes),
        max_policy_updates=int(global_update),
    )
    save_actor_critic_checkpoint(out / "checkpoint_last.pt", policy.cpu(), metadata)
    checkpoint_guard = _verify_pilot_checkpoint_roundtrip(out)
    metadata.update(checkpoint_guard)
    write_json(out / "metadata.json", metadata)
    status_label = (
        str(formal_execution_context.get("status_label"))
        if formal_execution_context is not None and formal_execution_context.get("status_label")
        else "BOUNDED_JINAN_PILOT_DRY_RUN_DONE"
    )
    status = {
        "status": status_label,
        "steps": total_steps,
        "reward_row_count": reward_row_count,
        "policy_update_count": int(global_update),
        "loss_debug_finite": True,
        "forbidden_artifacts": [],
        **metadata,
    }
    write_json(out / "status.json", status)
    if formal_execution_context is None:
        (out / "BOUNDED_JINAN_PILOT_DRY_RUN_DONE").write_text(
            "bounded Jinan 1-seed pilot dry-run completed; not formal, not performance, not ranking\n",
            encoding="utf-8",
        )
    exploratory_artifacts_written = _maybe_write_exploratory_pilot_guard_artifacts(
        out,
        status,
        action_guard,
        formal_execution_context,
    )
    if exploratory_artifacts_written:
        removed_outputs = _finalize_exploratory_pilot_outputs(out)
        status["exploratory_removed_non_allowlisted_outputs"] = removed_outputs
        write_json(out / "status.json", status)
    assert_no_forbidden_performance_artifacts(out)
    return status


def run_formal_pilot_wiring_dry_run(
    config: FormalPPODryRunConfig,
    out_dir: str | Path,
    *,
    episodes: int = 1,
    max_decision_steps_per_episode: int = 3,
    objective_normalizer: str | Path | None = None,
    objective_normalizer_hash: str | None = None,
    vector_model_dir: str | Path | None = None,
    vector_model_hash: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    validate_formal_pilot_wiring_dry_run_limits(
        config,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
    )
    if vector_model_hash is None:
        raise ValueError("formal-pilot wiring dry-run requires --vector_model_hash")
    if vector_model_dir is None:
        raise ValueError("formal-pilot wiring dry-run requires --vector_model_dir")
    context = _formal_pilot_wiring_context(
        config,
        vector_model_dir=vector_model_dir,
        vector_model_hash=vector_model_hash,
    )
    return run_bounded_jinan_pilot_dry_run(
        config,
        FORMAL_PILOT_WIRING_METHOD,
        out_dir,
        episodes=episodes,
        max_decision_steps_per_episode=max_decision_steps_per_episode,
        objective_normalizer=objective_normalizer,
        objective_normalizer_hash=objective_normalizer_hash,
        vector_model_dir=vector_model_dir,
        vector_model_hash=vector_model_hash,
        device=device,
        approved_methods=(FORMAL_PILOT_WIRING_METHOD,),
        require_final_method_list=False,
        formal_execution_context=context,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mock_env", action="store_true")
    parser.add_argument("--real_env", action="store_true")
    parser.add_argument("--real_env_dry_run", action="store_true")
    parser.add_argument("--tiny_real_ppo_update_preflight", action="store_true")
    parser.add_argument("--i_understand_this_runs_one_real_ppo_update", action="store_true")
    parser.add_argument("--bounded_jinan_pilot_dry_run", action="store_true")
    parser.add_argument("--i_understand_this_runs_bounded_jinan_pilot_dry_run", action="store_true")
    parser.add_argument("--formal_pilot_wiring_dry_run", action="store_true")
    parser.add_argument("--i_understand_this_runs_formal_pilot_wiring_dry_run", action="store_true")
    parser.add_argument("--formal_jinan_3seed_execution", action="store_true")
    parser.add_argument("--paper_final_execution", action="store_true")
    parser.add_argument("--approval_phrase")
    parser.add_argument("--seed_id", type=int)
    parser.add_argument("--rollout_steps", type=int)
    parser.add_argument("--total_env_steps_per_seed", type=int)
    parser.add_argument("--fixed_preference_template")
    parser.add_argument("--guard_packet")
    parser.add_argument("--allow_nonfilm_tiny_preflight", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_decision_steps_per_episode", type=int, default=10)
    parser.add_argument("--max_policy_updates", type=int, default=1)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resume_from")
    parser.add_argument("--objective_normalizer")
    parser.add_argument("--objective_normalizer_hash")
    parser.add_argument("--film_model_dir")
    parser.add_argument("--film_model_hash")
    parser.add_argument("--vector_model_dir")
    parser.add_argument("--vector_model_hash")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_env:
        raise SystemExit("real env pilot execution is not allowed; use --real_env_dry_run only when approved")
    if args.paper_final_execution:
        if (
            args.mock_env
            or args.real_env_dry_run
            or args.tiny_real_ppo_update_preflight
            or args.bounded_jinan_pilot_dry_run
            or args.formal_pilot_wiring_dry_run
            or args.formal_jinan_3seed_execution
        ):
            raise SystemExit("--paper_final_execution cannot be combined with other run modes")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for paper-final execution")
        if args.approval_phrase is None:
            raise SystemExit("--paper_final_execution requires --approval_phrase")
        if args.seed_id is None:
            raise SystemExit("--paper_final_execution requires --seed_id")
        if args.rollout_steps is None:
            raise SystemExit("--paper_final_execution requires --rollout_steps")
        if args.total_env_steps_per_seed is None:
            raise SystemExit("--paper_final_execution requires --total_env_steps_per_seed")
        if args.fixed_preference_template is None:
            raise SystemExit("--paper_final_execution requires --fixed_preference_template")
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_paper_final_execution(
            config,
            args.method,
            args.out_dir,
            seed_id=args.seed_id,
            approval_phrase=args.approval_phrase,
            rollout_steps=args.rollout_steps,
            total_env_steps_per_seed=args.total_env_steps_per_seed,
            fixed_preference_template=args.fixed_preference_template,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            film_model_dir=args.film_model_dir,
            film_model_hash=args.film_model_hash,
            vector_model_dir=args.vector_model_dir,
            vector_model_hash=args.vector_model_hash,
            device=args.device,
            guard_packet=args.guard_packet,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.formal_jinan_3seed_execution:
        if (
            args.mock_env
            or args.real_env_dry_run
            or args.tiny_real_ppo_update_preflight
            or args.bounded_jinan_pilot_dry_run
            or args.formal_pilot_wiring_dry_run
        ):
            raise SystemExit("--formal_jinan_3seed_execution cannot be combined with other run modes")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for formal Jinan 3-seed execution")
        if args.approval_phrase is None:
            raise SystemExit("--formal_jinan_3seed_execution requires --approval_phrase")
        if args.seed_id is None:
            raise SystemExit("--formal_jinan_3seed_execution requires --seed_id")
        if args.rollout_steps is None:
            raise SystemExit("--formal_jinan_3seed_execution requires --rollout_steps")
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_formal_jinan_3seed_execution(
            config,
            args.method,
            args.out_dir,
            seed_id=args.seed_id,
            approval_phrase=args.approval_phrase,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            rollout_steps=args.rollout_steps,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            film_model_dir=args.film_model_dir,
            film_model_hash=args.film_model_hash,
            vector_model_dir=args.vector_model_dir,
            vector_model_hash=args.vector_model_hash,
            device=args.device,
            guard_packet=args.guard_packet,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.formal_pilot_wiring_dry_run:
        if args.mock_env or args.real_env_dry_run or args.tiny_real_ppo_update_preflight or args.bounded_jinan_pilot_dry_run:
            raise SystemExit("--formal_pilot_wiring_dry_run cannot be combined with other run modes")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for formal-pilot wiring dry-run")
        if args.method != FORMAL_PILOT_WIRING_METHOD:
            raise SystemExit(f"--formal_pilot_wiring_dry_run requires --method {FORMAL_PILOT_WIRING_METHOD}")
        if not args.i_understand_this_runs_formal_pilot_wiring_dry_run:
            raise SystemExit(
                "Refusing to run: formal-pilot wiring dry-run performs a bounded real CityFlow PPO update "
                "for VectorQ reward wiring only. Pass --i_understand_this_runs_formal_pilot_wiring_dry_run to continue."
            )
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_formal_pilot_wiring_dry_run(
            config,
            args.out_dir,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            vector_model_dir=args.vector_model_dir,
            vector_model_hash=args.vector_model_hash,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.bounded_jinan_pilot_dry_run:
        if args.mock_env or args.real_env_dry_run or args.tiny_real_ppo_update_preflight:
            raise SystemExit("--bounded_jinan_pilot_dry_run cannot be combined with other run modes")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for bounded Jinan pilot dry-run")
        if not args.i_understand_this_runs_bounded_jinan_pilot_dry_run:
            raise SystemExit(
                "Refusing to run: bounded Jinan pilot dry-run performs real CityFlow PPO updates. "
                "Pass --i_understand_this_runs_bounded_jinan_pilot_dry_run to continue."
            )
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_bounded_jinan_pilot_dry_run(
            config,
            args.method,
            args.out_dir,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            film_model_dir=args.film_model_dir,
            film_model_hash=args.film_model_hash,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.tiny_real_ppo_update_preflight:
        if args.mock_env or args.real_env_dry_run:
            raise SystemExit("--tiny_real_ppo_update_preflight cannot be combined with mock/no-learning dry-run flags")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for tiny real PPO-update preflight")
        if not args.i_understand_this_runs_one_real_ppo_update:
            raise SystemExit(
                "Refusing to run: this mode performs one real CityFlow PPO optimizer step. "
                "Pass --i_understand_this_runs_one_real_ppo_update to continue."
            )
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_tiny_real_ppo_update_preflight(
            config,
            args.method,
            args.out_dir,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            max_policy_updates=args.max_policy_updates,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            film_model_dir=args.film_model_dir,
            film_model_hash=args.film_model_hash,
            device=args.device,
            allow_nonfilm_tiny_preflight=args.allow_nonfilm_tiny_preflight,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.real_env_dry_run:
        if args.mock_env:
            raise SystemExit("--real_env_dry_run cannot be combined with --mock_env")
        if args.resume_from:
            raise SystemExit("--resume_from is not allowed for no-learning real-env dry-run")
        config = load_formal_ppo_dryrun_config(args.spec)
        payload = run_formal_pilot_real_env_dry_run(
            config,
            args.method,
            args.out_dir,
            episodes=args.episodes,
            max_decision_steps_per_episode=args.max_decision_steps_per_episode,
            objective_normalizer=args.objective_normalizer,
            objective_normalizer_hash=args.objective_normalizer_hash,
            film_model_dir=args.film_model_dir,
            film_model_hash=args.film_model_hash,
            device=args.device,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not args.mock_env:
        raise SystemExit("formal pilot runner requires --mock_env in this stage; real env pilot execution is not allowed")
    config = load_formal_ppo_dryrun_config(args.spec)
    payload = run_formal_pilot_mock_dry_run(
        config,
        args.method,
        args.out_dir,
        episodes=args.episodes,
        max_decision_steps_per_episode=args.max_decision_steps_per_episode,
        resume_from=args.resume_from,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
