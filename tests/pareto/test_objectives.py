from pareto.data.objectives import (
    compute_objectives_from_snapshot,
    compute_efficiency,
    compute_fairness,
    compute_stability,
    compute_ttc_risk,
    gini,
)
from pareto.data.snapshot import TrafficSnapshot


def test_efficiency_is_higher_when_queue_and_wait_are_lower():
    low_load = compute_efficiency(total_queue=5.0, total_wait=10.0, approaching=3.0)
    high_load = compute_efficiency(total_queue=20.0, total_wait=80.0, approaching=9.0)
    assert low_load > high_load


def test_fairness_is_higher_for_balanced_lane_loads_with_same_total():
    balanced = compute_fairness([5.0, 5.0, 5.0, 5.0])
    imbalanced = compute_fairness([0.0, 0.0, 10.0, 10.0])
    assert balanced > imbalanced
    assert gini([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_stability_is_higher_for_smaller_queue_delta_and_no_switch():
    stable = compute_stability(
        queue_now=[5.0, 5.0, 5.0, 5.0],
        queue_prev=[5.0, 5.0, 5.0, 5.0],
        action=1,
        prev_action=1,
        phase_duration=30.0,
    )
    unstable = compute_stability(
        queue_now=[10.0, 0.0, 10.0, 0.0],
        queue_prev=[0.0, 10.0, 0.0, 10.0],
        action=2,
        prev_action=1,
        phase_duration=3.0,
    )
    assert stable > unstable


def test_ttc_risk_detects_closing_vehicle_pair():
    risk = compute_ttc_risk(
        lane_vehicles={"lane0": ["rear", "front"]},
        vehicle_distances={"rear": 20.0, "front": 30.0},
        vehicle_speeds={"rear": 10.0, "front": 5.0},
        ttc_threshold=3.0,
    )
    safe = compute_ttc_risk(
        lane_vehicles={"lane0": ["rear", "front"]},
        vehicle_distances={"rear": 20.0, "front": 30.0},
        vehicle_speeds={"rear": 5.0, "front": 10.0},
        ttc_threshold=3.0,
    )
    assert risk.valid
    assert risk.violation_rate > safe.violation_rate


def _snapshot(
    inter_name="intersection_1_1",
    state_detail=None,
    lane_vehicles=None,
    action_feature=None,
) -> TrafficSnapshot:
    return TrafficSnapshot(
        sim_time=30,
        inter_idx=0,
        inter_name=inter_name,
        local_lane_ids=list((lane_vehicles or {}).keys()),
        current_phase=0,
        time_this_phase=3,
        lane_vehicles=lane_vehicles or {},
        lane_waiting_counts={},
        vehicle_speeds={"rear": 10.0, "front": 5.0, "other_rear": 10.0, "other_front": 5.0},
        vehicle_distances={"rear": 20.0, "front": 30.0, "other_rear": 20.0, "other_front": 30.0},
        waiting_vehicle_list={},
        dic_feature=action_feature or {},
        state_detail=state_detail or {
            "lane_a": {"queue_len": 5, "avg_wait_time": 2, "cells": [1, 0]},
            "lane_b": {"queue_len": 3, "avg_wait_time": 1, "cells": [0, 1]},
            "lane_c": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0]},
            "lane_d": {"queue_len": 1, "avg_wait_time": 1, "cells": [0, 0]},
        },
        state_incoming={},
        mean_speed=5.0,
    )


def test_objective_stability_is_state_only_for_q_labels():
    prev = _snapshot(state_detail={
        "lane_a": {"queue_len": 4, "avg_wait_time": 2, "cells": [0]},
        "lane_b": {"queue_len": 4, "avg_wait_time": 1, "cells": [0]},
        "lane_c": {"queue_len": 0, "avg_wait_time": 0, "cells": [0]},
        "lane_d": {"queue_len": 1, "avg_wait_time": 1, "cells": [0]},
    })
    current = _snapshot()

    stay = compute_objectives_from_snapshot(current, prev_snapshot=prev, action=1, prev_action=1)
    switch = compute_objectives_from_snapshot(current, prev_snapshot=prev, action=2, prev_action=1)

    assert stay.raw["stability"] == switch.raw["stability"]


def test_ttc_risk_uses_only_local_lane_vehicles():
    local_risk = compute_ttc_risk(
        lane_vehicles={"local": ["rear", "front"]},
        vehicle_distances={"rear": 20.0, "front": 30.0, "other_rear": 20.0, "other_front": 30.0},
        vehicle_speeds={"rear": 10.0, "front": 5.0, "other_rear": 10.0, "other_front": 5.0},
    )
    global_risk = compute_ttc_risk(
        lane_vehicles={"local": ["rear", "front"], "other": ["other_rear", "other_front"]},
        vehicle_distances={"rear": 20.0, "front": 30.0, "other_rear": 20.0, "other_front": 30.0},
        vehicle_speeds={"rear": 10.0, "front": 5.0, "other_rear": 10.0, "other_front": 5.0},
    )

    assert local_risk.pair_count == 1
    assert global_risk.pair_count == 2
