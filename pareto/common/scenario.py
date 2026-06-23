from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

from pareto.constants import SCENARIOS


FALLBACK_TRAFFIC_ENV_CONF = {
    "LIST_MODEL_NEED_TO_UPDATE": [
        "EfficientPressLight",
        "EfficientColight",
        "EfficientMPLight",
        "AdvancedMPLight",
        "AdvancedColight",
        "AdvancedDQN",
        "Attend",
    ],
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,
    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 30,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],
    "INTERVAL": 1,
    "LIST_STATE_FEATURE": [
        "cur_phase",
        "traffic_movement_pressure_queue",
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0],
    },
    "PHASE_LIST": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
}

FALLBACK_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30],
    "W": 1.0,
}

FALLBACK_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}


def _merge(base: dict, extra: dict) -> dict:
    result = deepcopy(base)
    result.update(extra)
    return result


def _load_llmlight_config():
    try:
        from utils import config as llmlight_config

        return (
            deepcopy(llmlight_config.DIC_BASE_AGENT_CONF),
            deepcopy(llmlight_config.dic_traffic_env_conf),
            deepcopy(llmlight_config.DIC_PATH),
        )
    except ModuleNotFoundError:
        return (
            deepcopy(FALLBACK_AGENT_CONF),
            deepcopy(FALLBACK_TRAFFIC_ENV_CONF),
            deepcopy(FALLBACK_PATH),
        )


def resolve_scenario(scenario: str) -> dict:
    key = scenario.lower()
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}; expected one of {sorted(SCENARIOS)}")
    return dict(SCENARIOS[key])


def build_llmlight_env_config(
    scenario: str,
    traffic_file: Optional[str],
    seed: int,
    run_counts: int,
    min_action_time: Optional[int] = None,
    model_name: str = "MaxPressure",
    work_dir: Optional[str] = None,
) -> tuple[dict, dict, dict]:
    meta = resolve_scenario(scenario)
    selected_traffic = traffic_file or meta["default_traffic_file"]
    if selected_traffic not in meta["traffic_files"]:
        raise ValueError(f"traffic_file {selected_traffic} is not registered for {scenario}")

    num_row, num_col = [int(part) for part in meta["roadnet"].split("_")]
    env_extra = {
        "MODEL_NAME": model_name,
        "PROJECT_NAME": "pareto-llmlight",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "NUM_INTERSECTIONS": num_row * num_col,
        "NUM_AGENTS": num_row * num_col,
        "TRAFFIC_FILE": selected_traffic,
        "ROADNET_FILE": f"roadnet_{meta['roadnet']}.json",
        "RUN_COUNTS": int(run_counts),
        "CITYFLOW_SEED": int(seed),
    }
    if min_action_time is not None:
        env_extra["MIN_ACTION_TIME"] = int(min_action_time)

    dic_agent_conf, base_env_conf, dic_path = _load_llmlight_config()
    if model_name in {"MaxPressure", "AdvancedMaxPressure", "Fixedtime", "Random"}:
        dic_agent_conf["FIXED_TIME"] = [30, 30, 30, 30]
    if model_name == "AdvancedMaxPressure":
        dic_agent_conf["W"] = dic_agent_conf.get("W", 1.0)
        env_extra["W"] = env_extra.get("W", 1.0)
        env_extra["LIST_STATE_FEATURE"] = [
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "cur_phase",
            "time_this_phase",
        ]
    dic_traffic_env_conf = _merge(base_env_conf, env_extra)

    path_root = Path(work_dir or Path("records") / "pareto" / scenario.lower() / f"{model_name}_seed{seed}")
    dic_path["PATH_TO_WORK_DIRECTORY"] = str(path_root)
    dic_path["PATH_TO_MODEL"] = str(Path("model") / "pareto" / scenario.lower() / f"{model_name}_seed{seed}")
    dic_path["PATH_TO_DATA"] = str(Path("data") / meta["template"] / meta["roadnet"])
    return dic_agent_conf, dic_traffic_env_conf, dic_path
