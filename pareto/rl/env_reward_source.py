from __future__ import annotations

import math
from typing import Any


DEFAULT_PARETO_ENV_REWARD_INFO = {
    "queue_length": -1.0,
    "pressure": 0.0,
}


def _is_nonzero_reward_info(reward_info: dict[str, Any]) -> bool:
    for value in reward_info.values():
        try:
            if abs(float(value)) > 1e-12:
                return True
        except (TypeError, ValueError):
            continue
    return False


def ensure_nonzero_env_reward_info(dic_traffic_env_conf: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    """Make the env_reward baseline explicit when LLMLight's reward config is zero.

    LLMLight's public config ships with DIC_REWARD_INFO weights set to zero. That is
    fine for heuristic/LLM controllers that do not train from env.step rewards, but it
    makes an EnvReward-PPO guardrail baseline silently receive a null reward. We keep
    the original config unless the caller is explicitly running an env_reward sanity or
    env_reward adapter path.
    """

    original = dict(dic_traffic_env_conf.get("DIC_REWARD_INFO", {}))
    if _is_nonzero_reward_info(original):
        return {
            "original_env_reward_info": original,
            "active_env_reward_info": original,
            "overrode_zero_reward_info": False,
            "env_reward_info_source": "existing_nonzero_config",
        }
    if not enable:
        return {
            "original_env_reward_info": original,
            "active_env_reward_info": original,
            "overrode_zero_reward_info": False,
            "env_reward_info_source": "existing_zero_config",
        }

    active = dict(DEFAULT_PARETO_ENV_REWARD_INFO)
    dic_traffic_env_conf["DIC_REWARD_INFO"] = active
    return {
        "original_env_reward_info": original,
        "active_env_reward_info": active,
        "overrode_zero_reward_info": True,
        "env_reward_info_source": "pareto_nonzero_queue_length_proxy",
    }


def _as_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return None
    try:
        iterator = iter(values)
    except TypeError:
        result = [float(values)]
    else:
        # CityFlow returns Python lists; tests and callers may pass numpy arrays.
        # Treat any non-string iterable as a per-intersection reward sequence.
        del iterator
        result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def select_cityflow_env_rewards(final_rewards: Any, average_rewards: Any) -> tuple[list[float], dict[str, Any]]:
    average = _as_float_list(average_rewards)
    if average is not None:
        return average, {
            "env_reward_step_return_source": "cityflow_average_reward",
            "env_reward_average_available": True,
        }
    final = _as_float_list(final_rewards)
    if final is None:
        raise ValueError("CityFlow env.step did not return finite final or average rewards")
    return final, {
        "env_reward_step_return_source": "cityflow_final_second_reward",
        "env_reward_average_available": False,
    }
