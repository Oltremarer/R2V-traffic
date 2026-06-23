from __future__ import annotations

from pareto.rl.env_reward_source import (
    DEFAULT_PARETO_ENV_REWARD_INFO,
    ensure_nonzero_env_reward_info,
    select_cityflow_env_rewards,
)


def test_env_reward_info_override_replaces_zero_llmlight_config():
    conf = {"DIC_REWARD_INFO": {"queue_length": 0, "pressure": 0}}

    debug = ensure_nonzero_env_reward_info(conf, enable=True)

    assert conf["DIC_REWARD_INFO"] == DEFAULT_PARETO_ENV_REWARD_INFO
    assert debug["overrode_zero_reward_info"] is True
    assert debug["env_reward_info_source"] == "pareto_nonzero_queue_length_proxy"


def test_env_reward_info_override_preserves_existing_nonzero_config():
    conf = {"DIC_REWARD_INFO": {"queue_length": 0, "pressure": -1.0}}

    debug = ensure_nonzero_env_reward_info(conf, enable=True)

    assert conf["DIC_REWARD_INFO"] == {"queue_length": 0, "pressure": -1.0}
    assert debug["overrode_zero_reward_info"] is False
    assert debug["env_reward_info_source"] == "existing_nonzero_config"


def test_env_reward_info_override_can_leave_zero_config_disabled():
    conf = {"DIC_REWARD_INFO": {"queue_length": 0, "pressure": 0}}

    debug = ensure_nonzero_env_reward_info(conf, enable=False)

    assert conf["DIC_REWARD_INFO"] == {"queue_length": 0, "pressure": 0}
    assert debug["overrode_zero_reward_info"] is False
    assert debug["env_reward_info_source"] == "existing_zero_config"


def test_select_cityflow_env_rewards_prefers_average_reward_return():
    selected, debug = select_cityflow_env_rewards(
        final_rewards=[0.0, 0.0],
        average_rewards=[-1.0, -2.0],
    )

    assert selected == [-1.0, -2.0]
    assert debug["env_reward_step_return_source"] == "cityflow_average_reward"


def test_select_cityflow_env_rewards_falls_back_to_final_reward():
    selected, debug = select_cityflow_env_rewards(
        final_rewards=[-3.0, -4.0],
        average_rewards=None,
    )

    assert selected == [-3.0, -4.0]
    assert debug["env_reward_step_return_source"] == "cityflow_final_second_reward"
