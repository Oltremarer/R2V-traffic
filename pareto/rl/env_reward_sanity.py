#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import append_jsonl, write_json
from pareto.common.scenario import build_llmlight_env_config
from pareto.rl.env_reward_source import ensure_nonzero_env_reward_info, select_cityflow_env_rewards
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig, load_formal_ppo_dryrun_config
from pareto.rl.formal_reward_adapter import EnvRewardAdapter
from pareto.rl.real_cityflow_preflight_smoke import _prepare_work_dir


def validate_env_reward_sanity_limits(episodes: int, max_decision_steps_per_episode: int) -> None:
    if int(episodes) != 1:
        raise ValueError("episodes must be exactly 1 for env_reward sanity")
    if int(max_decision_steps_per_episode) <= 0 or int(max_decision_steps_per_episode) > 20:
        raise ValueError("max_decision_steps_per_episode must be in [1, 20] for env_reward sanity")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def summarize_env_rewards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    env_values = [float(row["env_reward"]) for row in rows if "env_reward" in row]
    total_values = [float(row["total_reward"]) for row in rows if "total_reward" in row]
    values = total_values or env_values
    finite = bool(values) and all(_finite(value) for value in values)
    if not values:
        warnings.append("no_reward_rows")
    if values and not finite:
        warnings.append("non_finite_reward")
    nonzero_count = sum(1 for value in values if _finite(value) and abs(value) > 1e-12)
    all_zero = bool(values) and finite and nonzero_count == 0
    if all_zero:
        warnings.append("all_zero_reward")
    source_names = sorted({str(row.get("env_reward_source")) for row in rows if row.get("env_reward_source")})
    recommendation = "env_reward_source_has_signal"
    if not values:
        recommendation = "rerun_with_reward_rows"
    elif not finite:
        recommendation = "fix_non_finite_env_reward_source"
    elif all_zero:
        recommendation = "env_reward_source_has_no_signal"
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "finite": finite,
        "nonzero_reward_count": nonzero_count,
        "env_reward_nonzero_rate": float(nonzero_count / len(values)) if values else 0.0,
        "all_zero_reward": all_zero,
        "env_reward_sources": source_names,
        "recommendation": recommendation,
        "warnings": warnings,
    }
    if values and finite:
        summary.update(
            {
                "min": min(values),
                "mean": float(sum(values) / len(values)),
                "max": max(values),
            }
        )
    return summary


def _clean_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "metadata.json",
        "status.json",
        "reward_components.jsonl",
        "action_debug.jsonl",
        "env_reward_sanity.json",
        "ENV_REWARD_SANITY_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()
    work_dir = out / "llmlight_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)


def run_env_reward_sanity(
    config: FormalPPODryRunConfig,
    out_dir: str | Path,
    *,
    episodes: int = 1,
    max_decision_steps_per_episode: int = 20,
    device: str = "cpu",
) -> dict[str, Any]:
    del device
    validate_env_reward_sanity_limits(episodes, max_decision_steps_per_episode)
    random.seed(int(config.pilot.get("policy_seed", 0)))
    np.random.seed(int(config.pilot.get("policy_seed", 0)))

    out = Path(out_dir)
    _clean_output_dir(out)
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
    reward_info_debug = ensure_nonzero_env_reward_info(dic_traffic_env_conf, enable=True)
    _prepare_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)

    from utils.cityflow_env import CityFlowEnv

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
    )
    adapter = EnvRewardAdapter()
    reward_rows: list[dict[str, Any]] = []
    total_steps = 0

    for episode in range(int(episodes)):
        env.reset()
        for step in range(int(max_decision_steps_per_episode)):
            action_list = [random.randrange(int(config.model["action_dim"])) for _ in env.list_intersection]
            sim_time_before = int(env.get_current_time())
            _, final_env_rewards, done, average_env_rewards = env.step(action_list)
            env_rewards, reward_source_debug = select_cityflow_env_rewards(final_env_rewards, average_env_rewards)
            sim_time_after = int(env.get_current_time())
            rows: list[dict[str, Any]] = []
            for idx, env_reward in enumerate(env_rewards):
                reward, debug = adapter.compute(None, None, {}, {}, done=bool(done), env_reward=float(env_reward))
                row = {
                    "episode": episode,
                    "step": step,
                    "intersection_idx": idx,
                    "action": int(action_list[idx]),
                    "env_reward": float(env_reward),
                    "env_reward_source": reward_source_debug["env_reward_step_return_source"],
                    **reward_info_debug,
                    "total_reward": float(reward),
                    **debug,
                }
                rows.append(row)
            append_jsonl(out / "reward_components.jsonl", rows)
            reward_rows.extend(rows)
            append_jsonl(
                out / "action_debug.jsonl",
                [
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time_before": sim_time_before,
                        "sim_time_after": sim_time_after,
                        "action_list_len": len(action_list),
                        "num_intersections": len(env.list_intersection),
                        **reward_source_debug,
                        **reward_info_debug,
                    }
                ],
            )
            total_steps += 1
            if done:
                break

    env.batch_log_2()
    reward_summary = summarize_env_rewards(reward_rows)
    metadata = {
        "env_reward_sanity": True,
        "method": "env_reward",
        "reward_adapter": "env_reward",
        "no_learning": True,
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "exclude_from_analysis": True,
        "real_env_rollout": True,
        "cityflow_env_constructed": True,
        "cityflow_step_called": True,
        "ppo_training": False,
        "real_ppo_update": False,
        "policy_update": False,
        "optimizer_step": False,
        "env_reward_signal_required": True,
        "episodes": int(episodes),
        "max_decision_steps_per_episode": int(max_decision_steps_per_episode),
        "steps": total_steps,
        "reward_row_count": len(reward_rows),
        "scenario": config.pilot["scenario"],
        "traffic_file": config.pilot["traffic_file"],
        "cityflow_seed": int(config.pilot.get("cityflow_seed", 0)),
        "policy_seed": int(config.pilot.get("policy_seed", 0)),
        "min_action_time": min_action_time,
        "sim_seconds": int(total_steps * min_action_time),
        **reward_info_debug,
    }
    report = {"metadata": metadata, "reward_summary": reward_summary}
    write_json(out / "metadata.json", metadata)
    write_json(out / "env_reward_sanity.json", report)
    if reward_summary["all_zero_reward"]:
        failed_status = {
            "status": "ENV_REWARD_SANITY_FAILED",
            **metadata,
            "reward_summary": reward_summary,
            "failure_reason": "all_zero_reward",
        }
        write_json(out / "status.json", failed_status)
        raise ValueError("env_reward sanity still produced all-zero reward")
    status = {"status": "ENV_REWARD_SANITY_DONE", **metadata, "reward_summary": reward_summary}
    write_json(out / "status.json", status)
    (out / "ENV_REWARD_SANITY_DONE").write_text(
        "env_reward no-learning sanity completed; not a pilot, not a performance result\n",
        encoding="utf-8",
    )
    assert_no_forbidden_performance_artifacts(out)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_decision_steps_per_episode", type=int, default=20)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_formal_ppo_dryrun_config(args.spec)
    payload = run_env_reward_sanity(
        config,
        args.out_dir,
        episodes=args.episodes,
        max_decision_steps_per_episode=args.max_decision_steps_per_episode,
        device=args.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
