from pareto.data.snapshot import TrafficSnapshot
from pareto.rl.state_encoder import ParetoStateEncoder


def _snapshot() -> TrafficSnapshot:
    return TrafficSnapshot(
        sim_time=0,
        inter_idx=0,
        inter_name="intersection_1_1",
        local_lane_ids=[],
        current_phase=1,
        time_this_phase=10,
        lane_vehicles={},
        lane_waiting_counts={},
        vehicle_speeds={},
        vehicle_distances={},
        waiting_vehicle_list={},
        dic_feature={
            "cur_phase": [1],
            "time_this_phase": [10],
            "lane_num_vehicle": [1, 2, 3],
            "lane_num_waiting_vehicle_in": [0, 1, 2],
            "pressure": [1, -1, 0],
            "traffic_movement_pressure_queue": [2, -2, 0],
        },
        state_detail={
            "NT": {"queue_len": 3, "avg_wait_time": 4, "cells": [1, 2, 3, 4]},
            "ST": {"queue_len": 1, "avg_wait_time": 2, "cells": [0, 1, 0, 1]},
        },
        state_incoming={"N": {"queue_len": 2, "cells": [1, 0, 0, 0]}},
        mean_speed=5.0,
    )


def test_hybrid_encoder_has_stable_feature_names_and_values():
    encoder = ParetoStateEncoder("hybrid_v1")
    features, names, debug = encoder.encode_snapshot(_snapshot())
    features2, names2, debug2 = encoder.encode_snapshot(_snapshot())
    assert names == names2
    assert features.tolist() == features2.tolist()
    assert len(features) == len(names)
    assert debug["encoder_id"] == "hybrid_v1"
    assert debug2["feature_count"] == len(names)


def test_hybrid_encoder_schema_is_invariant_to_missing_policy_features():
    base = _snapshot()
    advanced = _snapshot()
    advanced.dic_feature = {
        "cur_phase": [1],
        "time_this_phase": [10],
        "lane_num_vehicle": [1, 2, 3],
        "lane_num_waiting_vehicle_in": [0, 1, 2],
        "pressure": [1, -1, 0],
        "traffic_movement_pressure_queue_efficient": [3, 4],
        "lane_enter_running_part": [0.5, 0.25],
    }

    encoder = ParetoStateEncoder("hybrid_v1")
    base_features, base_names, base_debug = encoder.encode_snapshot(base)
    adv_features, adv_names, adv_debug = encoder.encode_snapshot(advanced)

    assert base_names == adv_names
    assert len(base_features) == len(adv_features)
    assert base_debug["feature_names_hash"] == adv_debug["feature_names_hash"]
    assert "traffic_movement_pressure_queue" in adv_debug["missing_feature_keys"]
    assert "traffic_movement_pressure_queue_efficient" in base_debug["missing_feature_keys"]


def test_llm_abstraction_schema_is_invariant_to_missing_state_keys():
    full = _snapshot()
    sparse = _snapshot()
    sparse.state_detail = {
        "ET": {"queue_len": 1, "avg_wait_time": 2, "cells": [1]},
    }
    sparse.state_incoming = {}

    encoder = ParetoStateEncoder("llm_abstraction")
    full_features, full_names, full_debug = encoder.encode_snapshot(full)
    sparse_features, sparse_names, sparse_debug = encoder.encode_snapshot(sparse)

    assert full_names == sparse_names
    assert len(full_features) == len(sparse_features)
    assert full_debug["feature_names_hash"] == sparse_debug["feature_names_hash"]
