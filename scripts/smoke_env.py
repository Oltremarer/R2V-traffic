#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_metrics_csv, write_json
from pareto.common.run_metadata import write_run_metadata
from pareto.common.scenario import build_llmlight_env_config, resolve_scenario


POLICY_TO_MODEL = {
    "fixedtime": "Fixedtime",
    "maxpressure": "MaxPressure",
    "advanced_maxpressure": "AdvancedMaxPressure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short LLMLight CityFlow smoke test.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--traffic_file", default=None)
    parser.add_argument("--policy", choices=["random", "fixedtime", "maxpressure", "advanced_maxpressure"], required=True)
    parser.add_argument("--steps", type=int, default=200, help="Simulation seconds to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
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


def summarize_step(env, reward: List[float], actions: List[int], step_num: int) -> Dict[str, Any]:
    queues = [sum(inter.dic_feature["lane_num_waiting_vehicle_in"]) for inter in env.list_intersection]
    waiting_times = [item["time"] for item in env.waiting_vehicle_list.values()]
    return {
        "step": step_num,
        "sim_time": env.get_current_time(),
        "reward_sum": float(sum(reward)),
        "avg_queue": float(np.mean(queues) if queues else 0.0),
        "total_queue": float(sum(queues)),
        "avg_waiting_time": float(np.mean(waiting_times) if waiting_times else 0.0),
        "action_histogram": json.dumps(dict(Counter(int(action) for action in actions)), sort_keys=True),
    }


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scenario = resolve_scenario(args.scenario)
    traffic_file = args.traffic_file or scenario["default_traffic_file"]
    run_counts = int(args.steps)
    model_name = POLICY_TO_MODEL.get(args.policy, "Random")
    out_dir = args.out_dir or str(Path("records") / "smoke" / "env" / f"{args.policy}_{args.scenario}_seed{args.seed}")

    dic_agent_conf, dic_traffic_env_conf, dic_path = build_llmlight_env_config(
        scenario=args.scenario,
        traffic_file=traffic_file,
        seed=args.seed,
        run_counts=run_counts,
        model_name=model_name,
        work_dir=out_dir,
    )
    prepare_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)
    write_run_metadata(
        out_dir,
        " ".join(sys.argv),
        {
            "scenario": args.scenario,
            "traffic_file": traffic_file,
            "seed": args.seed,
            "steps": args.steps,
            "policy": args.policy,
            "cityflow_seed": dic_traffic_env_conf["CITYFLOW_SEED"],
        },
    )

    from utils.cityflow_env import CityFlowEnv

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
    )
    state = env.reset()
    agents = build_agents(args.policy, dic_agent_conf, dic_traffic_env_conf, dic_path)
    phase_count = len(dic_traffic_env_conf["PHASE"])
    total_steps = max(1, run_counts // dic_traffic_env_conf["MIN_ACTION_TIME"])

    for step_num in range(total_steps):
        actions = choose_actions(args.policy, agents, state, step_num, phase_count)
        next_state, reward, done, _ = env.step(actions)
        append_metrics_csv(Path(out_dir) / "metrics.csv", summarize_step(env, reward, actions, step_num))
        state = next_state
        if done:
            break

    write_json(Path(out_dir) / "status.json", {"status": "DONE", "steps": total_steps})
    env.batch_log_2()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
