from __future__ import annotations

import pytest

from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_binding import ReferenceEvalCityFlowEnvWrapper


class _FakeEnv:
    def __init__(self, num_intersections: int):
        self.list_intersection = [object() for _ in range(num_intersections)]
        self.actions: list[list[int]] = []

    def reset(self):
        return [{"cur_phase": [idx % 4]} for idx, _item in enumerate(self.list_intersection)]

    def step(self, action):
        self.actions.append(list(action))
        return (
            [{"cur_phase": [idx % 4]} for idx, _item in enumerate(self.list_intersection)],
            [0.0 for _item in self.list_intersection],
            False,
            {},
        )


def _wrapper(num_intersections: int = 2) -> ReferenceEvalCityFlowEnvWrapper:
    return ReferenceEvalCityFlowEnvWrapper(
        _FakeEnv(num_intersections),
        dic_traffic_env_conf={
            "NUM_INTERSECTIONS": num_intersections,
            "PHASE": {1: [], 2: [], 3: [], 4: []},
        },
        min_action_time=30,
    )


def test_formal_jinan_action_mapping_accepts_joint_zero_based_phase_ids():
    env = _wrapper(num_intersections=2)
    env.reset(episode=0, seed=0)

    env.step([0, 3], min_action_time=30)

    assert env._env.actions == [[0, 3]]


def test_formal_jinan_action_mapping_rejects_wrong_joint_length():
    env = _wrapper(num_intersections=2)
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="joint action length"):
        env.step([1], min_action_time=30)


def test_formal_jinan_action_mapping_rejects_out_of_range_phase_id():
    env = _wrapper(num_intersections=1)
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="invalid action id"):
        env.step([4], min_action_time=30)


def test_formal_jinan_action_mapping_rejects_bool_phase_id():
    env = _wrapper(num_intersections=1)
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="deterministic policy action"):
        env.step([True], min_action_time=30)
