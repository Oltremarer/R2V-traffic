from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrafficSnapshot:
    sim_time: int
    inter_idx: int
    inter_name: str
    local_lane_ids: List[str]
    current_phase: int
    time_this_phase: int
    lane_vehicles: Dict[str, List[str]]
    lane_waiting_counts: Dict[str, int]
    vehicle_speeds: Dict[str, float]
    vehicle_distances: Dict[str, float]
    waiting_vehicle_list: Dict[str, Dict[str, Any]]
    dic_feature: Dict[str, Any]
    state_detail: Dict[str, Any]
    state_incoming: Dict[str, Any]
    mean_speed: float


def capture_snapshot(env, inter_idx: int) -> TrafficSnapshot:
    from utils.my_utils import get_state_detail

    inter = env.list_intersection[inter_idx]
    inter_name = inter.inter_name
    intersection = env.intersection_dict[inter_name]
    roads = deepcopy(intersection["roads"])
    state_detail, state_incoming, mean_speed = get_state_detail(roads, env)
    local_lane_ids = list(getattr(inter, "list_lanes", []))
    all_lane_vehicles = env.system_states.get("get_lane_vehicles", {})
    all_lane_waiting_counts = env.system_states.get("get_lane_waiting_vehicle_count", {})
    return TrafficSnapshot(
        sim_time=int(env.get_current_time()),
        inter_idx=inter_idx,
        inter_name=inter_name,
        local_lane_ids=local_lane_ids,
        current_phase=int(inter.current_phase_index),
        time_this_phase=int(inter.current_phase_duration),
        lane_vehicles={lane: deepcopy(all_lane_vehicles.get(lane, [])) for lane in local_lane_ids},
        lane_waiting_counts={lane: int(all_lane_waiting_counts.get(lane, 0)) for lane in local_lane_ids},
        vehicle_speeds=deepcopy(env.system_states.get("get_vehicle_speed", {})),
        vehicle_distances=deepcopy(env.system_states.get("get_vehicle_distance", {})),
        waiting_vehicle_list=deepcopy(env.waiting_vehicle_list),
        dic_feature=deepcopy(inter.dic_feature),
        state_detail=state_detail,
        state_incoming=state_incoming,
        mean_speed=float(mean_speed),
    )
