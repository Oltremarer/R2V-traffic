#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_jsonl, append_metrics_csv, write_json
from pareto.common.run_metadata import write_run_metadata
from pareto.common.scenario import build_llmlight_env_config, resolve_scenario
from pareto.data.abstraction import build_trajectory_record
from pareto.data.schema import TransitionRecord
from pareto.data.snapshot import capture_snapshot
from pareto.rl.env_reward_source import select_cityflow_env_rewards
from pareto.rl.state_encoder import ParetoStateEncoder


POLICY_TO_MODEL = {
    "fixedtime": "Fixedtime",
    "maxpressure": "MaxPressure",
    "advanced_maxpressure": "AdvancedMaxPressure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Pareto trajectory records from LLMLight CityFlowEnv.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--traffic_file", default=None)
    parser.add_argument("--policy", choices=["random", "fixedtime", "maxpressure", "advanced_maxpressure"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300, help="Simulation seconds per episode.")
    parser.add_argument("--encoder_id", default="hybrid_v1", choices=["llm_abstraction", "llmlight_feature", "hybrid_v1"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--transitions_out", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--work_dir", default=None)
    return parser.parse_args()


def prepare_work_dir(dic_agent_conf: Dict[str, Any], dic_traffic_env_conf: Dict[str, Any], dic_path: Dict[str, str]) -> None:
    work_dir = Path(dic_path["PATH_TO_WORK_DIRECTORY"])
    model_dir = Path(dic_path["PATH_TO_MODEL"])
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "agent.conf").write_text(json.dumps(dic_agent_conf, indent=2, sort_keys=True), encoding="utf-8")
    (work_dir / "traffic_env.conf").write_text(json.dumps(dic_traffic_env_conf, indent=2, sort_keys=True), encoding="utf-8")
    data_dir = Path(dic_path["PATH_TO_DATA"])
    shutil.copy(data_dir / dic_traffic_env_conf["TRAFFIC_FILE"], work_dir / dic_traffic_env_conf["TRAFFIC_FILE"])
    shutil.copy(data_dir / dic_traffic_env_conf["ROADNET_FILE"], work_dir / dic_traffic_env_conf["ROADNET_FILE"])


def build_agents(policy: str, dic_agent_conf: Dict[str, Any], dic_traffic_env_conf: Dict[str, Any], dic_path: Dict[str, str]):
    if policy == "random":
        return None
    if policy == "fixedtime":
        from models.fixedtime_agent import FixedtimeAgent as AgentClass
    elif policy == "maxpressure":
        from models.maxpressure_agent import MaxPressureAgent as AgentClass
    elif policy == "advanced_maxpressure":
        from models.advanced_maxpressure_agent import AdvancedMaxPressureAgent as AgentClass
    else:
        raise ValueError(f"unsupported policy {policy}")
    return [
        AgentClass(
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i),
        )
        for i in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])
    ]


def choose_actions(policy: str, agents, state: List[Dict[str, Any]], step_num: int, phase_count: int) -> List[int]:
    if policy == "random":
        return [random.randint(0, phase_count - 1) for _ in state]
    return [agents[i].choose_action(step_num, state[i]) for i in range(len(state))]


def collect(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scenario = resolve_scenario(args.scenario)
    traffic_file = args.traffic_file or scenario["default_traffic_file"]
    model_name = POLICY_TO_MODEL.get(args.policy, "Random")
    work_dir = args.work_dir or str(Path("records") / "pareto_buffer" / args.scenario / f"{args.policy}_seed{args.seed}")

    dic_agent_conf, dic_traffic_env_conf, dic_path = build_llmlight_env_config(
        scenario=args.scenario,
        traffic_file=traffic_file,
        seed=args.seed,
        run_counts=args.max_steps,
        model_name=model_name,
        work_dir=work_dir,
    )
    prepare_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)
    write_run_metadata(
        work_dir,
        " ".join(sys.argv),
        {
            "scenario": args.scenario,
            "traffic_file": traffic_file,
            "seed": args.seed,
            "policy": args.policy,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "encoder_id": args.encoder_id,
        },
    )

    from utils.cityflow_env import CityFlowEnv

    encoder = ParetoStateEncoder(args.encoder_id)
    output_path = Path(args.out)
    transitions_path = Path(args.transitions_out) if args.transitions_out else None
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    if output_path.exists():
        output_path.unlink()
    if transitions_path and transitions_path.exists() and not args.overwrite:
        raise FileExistsError(f"{transitions_path} exists; pass --overwrite to replace it")
    if transitions_path and transitions_path.exists():
        transitions_path.unlink()
    phase_count = len(dic_traffic_env_conf["PHASE"])
    decision_steps = max(1, args.max_steps // dic_traffic_env_conf["MIN_ACTION_TIME"])
    record_count = 0

    for episode in range(args.episodes):
        env = CityFlowEnv(
            path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
        )
        state = env.reset()
        agents = build_agents(args.policy, dic_agent_conf, dic_traffic_env_conf, dic_path)
        prev_snapshots: list[Optional[object]] = [None for _ in state]
        prev_actions: list[Optional[int]] = [None for _ in state]
        prev_sample_ids: list[Optional[str]] = [None for _ in state]

        for step_num in range(decision_steps):
            snapshots = [capture_snapshot(env, idx) for idx in range(len(state))]
            actions = choose_actions(args.policy, agents, state, step_num, phase_count)
            next_state, final_reward, done, average_reward = env.step(actions)
            reward, reward_source_debug = select_cityflow_env_rewards(final_reward, average_reward)
            next_snapshots = [capture_snapshot(env, idx) for idx in range(len(next_state))]
            is_terminal = bool(done or step_num == decision_steps - 1)
            rows = []
            transition_rows = []
            for idx, snapshot in enumerate(snapshots):
                next_sample_id = None
                if not is_terminal:
                    next_sample_id = f"{args.scenario}_{args.policy}_seed{args.seed}:{episode}:{snapshot.inter_name}:{next_snapshots[idx].sim_time}"
                metadata = {
                    "run_id": f"{args.scenario}_{args.policy}_seed{args.seed}",
                    "scenario": args.scenario,
                    "roadnet": scenario["roadnet"],
                    "traffic_file": traffic_file,
                    "seed": args.seed,
                    "episode": episode,
                    "step": step_num,
                    "prev_sample_id": prev_sample_ids[idx],
                    "next_sample_id": next_sample_id,
                }
                record = build_trajectory_record(
                    snapshot=snapshot,
                    prev_snapshot=prev_snapshots[idx],
                    action=int(actions[idx]),
                    prev_action=prev_actions[idx],
                    policy_id=args.policy,
                    encoder=encoder,
                    metadata=metadata,
                )
                next_record = build_trajectory_record(
                    snapshot=next_snapshots[idx],
                    prev_snapshot=snapshot,
                    action=int(actions[idx]),
                    prev_action=prev_actions[idx],
                    policy_id=args.policy,
                    encoder=encoder,
                    metadata={
                        **metadata,
                        "step": step_num + 1,
                        "prev_sample_id": record.sample_id,
                        "next_sample_id": None,
                    },
                )
                rows.append(record.to_dict())
                if transitions_path is not None:
                    transition = TransitionRecord(
                        schema_version="pareto-transition-v1",
                        run_id=record.run_id,
                        transition_id=f"{record.sample_id}->{next_record.sample_id}",
                        sample_id=record.sample_id,
                        next_sample_id=next_record.sample_id,
                        scenario=record.scenario,
                        traffic_file=record.traffic_file,
                        seed=record.seed,
                        episode=record.episode,
                        step=record.step,
                        intersection_id=record.intersection_id,
                        obs_features=record.obs_features,
                        next_obs_features=next_record.obs_features,
                        action=int(actions[idx]),
                        env_reward=float(reward[idx]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward),
                        objectives_t_norm=record.objective_values_norm,
                        objectives_tp1_norm=next_record.objective_values_norm,
                        done=is_terminal,
                        policy_id=args.policy,
                        metadata={
                            "state_feature_names_hash": record.metadata.get("encoder_debug", {}).get("feature_names_hash"),
                            "next_state_feature_names_hash": next_record.metadata.get("encoder_debug", {}).get("feature_names_hash"),
                            **reward_source_debug,
                        },
                    )
                    transition_rows.append(transition.to_dict())
                prev_snapshots[idx] = snapshot
                prev_actions[idx] = int(actions[idx])
                prev_sample_ids[idx] = record.sample_id
            append_jsonl(output_path, rows)
            if transitions_path is not None:
                append_jsonl(transitions_path, transition_rows)
            record_count += len(rows)

            append_metrics_csv(
                Path(work_dir) / "metrics.csv",
                {
                    "episode": episode,
                    "step": step_num,
                    "sim_time": env.get_current_time(),
                    "reward_sum": float(sum(reward)),
                    "record_count": record_count,
                },
            )
            state = next_state
            if done:
                break
        env.batch_log_2()

    write_json(Path(work_dir) / "status.json", {"status": "DONE", "records": record_count, "out": str(output_path)})


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
