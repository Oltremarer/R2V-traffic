from pareto.constants import OBJECTIVE_NAMES
from pareto.data.abstraction import build_trajectory_record
from pareto.data.snapshot import TrafficSnapshot
from pareto.rl.state_encoder import ParetoStateEncoder


def _snapshot(time_this_phase=10) -> TrafficSnapshot:
    return TrafficSnapshot(
        sim_time=30,
        inter_idx=0,
        inter_name="intersection_1_1",
        local_lane_ids=["lane0"],
        current_phase=1,
        time_this_phase=time_this_phase,
        lane_vehicles={"lane0": ["rear", "front"]},
        lane_waiting_counts={},
        vehicle_speeds={"rear": 10.0, "front": 5.0},
        vehicle_distances={"rear": 20.0, "front": 30.0},
        waiting_vehicle_list={},
        dic_feature={
            "cur_phase": [1],
            "time_this_phase": [time_this_phase],
            "lane_num_vehicle": [2],
            "lane_num_waiting_vehicle_in": [1],
            "pressure": [1],
            "traffic_movement_pressure_queue": [1],
        },
        state_detail={
            "NT": {"queue_len": 1, "avg_wait_time": 2, "cells": [1, 0, 0, 0]},
            "ST": {"queue_len": 2, "avg_wait_time": 3, "cells": [0, 1, 0, 0]},
            "ET": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 1, 0]},
            "WT": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 1]},
        },
        state_incoming={},
        mean_speed=4.0,
    )


def test_build_trajectory_record_contains_features_and_objectives():
    record = build_trajectory_record(
        snapshot=_snapshot(),
        prev_snapshot=None,
        action=0,
        prev_action=None,
        policy_id="maxpressure",
        encoder=ParetoStateEncoder("hybrid_v1"),
        metadata={
            "run_id": "run0",
            "scenario": "jinan",
            "roadnet": "3_4",
            "traffic_file": "anon_3_4_jinan_real.json",
            "seed": 0,
            "episode": 0,
            "step": 0,
        },
    )
    assert record.sample_id == "run0:0:intersection_1_1:30"
    assert len(record.obs_features) == len(record.obs_feature_names)
    assert set(record.objective_values_raw) == set(OBJECTIVE_NAMES)
    assert record.objective_valid_mask["stability"] is False
