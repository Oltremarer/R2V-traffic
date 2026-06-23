from __future__ import annotations

import math

import pytest

from pareto.rl.paper_final_learned_eval_runner import (
    build_learned_eval_command,
    default_learned_eval_out_dir,
    validate_learned_eval_request,
    _travel_time_mean,
)


def test_learned_eval_request_uses_reference_eval_settings_and_default_out_dir():
    request = validate_learned_eval_request(
        spec="configs/formal/paper_final_jinan_5seed_ppo.json",
        method="Cond-Scalar-RL",
        seed_id=0,
        train_dir="records/paper_final/train_20260602_v1/jinan/anon_3_4_jinan_real/Cond-Scalar-RL/seed0",
        eval_out_dir=None,
        fixed_preference_template="balanced",
        execute=False,
    )

    assert request["method"] == "Cond-Scalar-RL"
    assert request["method_id"] == "film_scalar_potential"
    assert request["city"] == "jinan"
    assert request["traffic_file"] == "anon_3_4_jinan_real.json"
    assert request["run_counts"] == 3600
    assert request["min_action_time"] == 30
    assert request["eval_out_dir"] == default_learned_eval_out_dir(
        city="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        method="Cond-Scalar-RL",
        seed_id=0,
        fixed_preference_template="balanced",
    )
    command = build_learned_eval_command(request, python_bin="python3")
    assert command[:2] == ["python3", "pareto/rl/paper_final_learned_eval_runner.py"]
    assert "--execute" in command
    assert "--fixed_preference_template" in command
    assert "balanced" in command


def test_learned_eval_request_rejects_non_reference_eval_budget():
    with pytest.raises(ValueError, match="RUN_COUNTS=3600"):
        validate_learned_eval_request(
            spec="configs/formal/paper_final_hangzhou_5seed_ppo.json",
            method="weighted_proxy",
            seed_id=0,
            train_dir="records/paper_final/train_20260602_v1/hangzhou/anon_4_4_hangzhou_real/Weighted-RL/seed0",
            eval_out_dir=None,
            fixed_preference_template="balanced",
            run_counts=7200,
            execute=False,
        )


class _FakeIntersection:
    def __init__(self):
        self.dic_vehicle_arrive_leave_time = {
            "veh_done": {"enter_time": 0.0, "leave_time": 30.0},
            "veh_open": {"enter_time": 10.0, "leave_time": math.nan},
            "shadow_veh": {"enter_time": 0.0, "leave_time": 999.0},
        }


class _FakeEnv:
    list_intersection = [_FakeIntersection()]


def test_travel_time_mean_matches_legacy_incomplete_vehicle_semantics():
    assert _travel_time_mean(_FakeEnv(), run_counts=3600) == pytest.approx((30.0 + 3590.0) / 2.0)
