from __future__ import annotations

from typing import Optional

from pareto.data.objectives import compute_objectives_from_snapshot
from pareto.data.schema import TrajectoryRecord


def build_trajectory_record(
    snapshot,
    prev_snapshot=None,
    action: Optional[int] = None,
    prev_action: Optional[int] = None,
    policy_id: str = "unknown",
    encoder=None,
    objective_normalizer=None,
    metadata: Optional[dict] = None,
) -> TrajectoryRecord:
    if encoder is None:
        from pareto.rl.state_encoder import ParetoStateEncoder

        encoder = ParetoStateEncoder("hybrid_v1")
    metadata = dict(metadata or {})
    features, feature_names, encoder_debug = encoder.encode_snapshot(snapshot)
    objectives = compute_objectives_from_snapshot(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        action=action,
        prev_action=prev_action,
        normalizer=objective_normalizer,
    )
    run_id = metadata.get("run_id", "run")
    episode = int(metadata.get("episode", 0))
    sample_id = f"{run_id}:{episode}:{snapshot.inter_name}:{snapshot.sim_time}"
    record = TrajectoryRecord(
        schema_version="pareto-trajectory-v1",
        run_id=run_id,
        sample_id=sample_id,
        scenario=metadata.get("scenario", "unknown"),
        roadnet=metadata.get("roadnet", "unknown"),
        traffic_file=metadata.get("traffic_file", "unknown"),
        seed=int(metadata.get("seed", 0)),
        episode=episode,
        step=int(metadata.get("step", 0)),
        sim_time_sec=float(snapshot.sim_time),
        intersection_id=snapshot.inter_name,
        policy_id=policy_id,
        action=action,
        action_name=metadata.get("action_name"),
        phase_id=int(snapshot.current_phase),
        phase_duration=float(snapshot.time_this_phase),
        obs_features=features.astype(float).tolist(),
        obs_feature_names=feature_names,
        llm_state=snapshot.state_detail,
        incoming_state=snapshot.state_incoming,
        mean_speed=float(snapshot.mean_speed),
        objective_values_raw=objectives.raw,
        objective_values_norm=objectives.norm,
        objective_valid_mask=objectives.valid_mask,
        prev_sample_id=metadata.get("prev_sample_id"),
        next_sample_id=metadata.get("next_sample_id"),
        metadata={**metadata, "encoder_debug": encoder_debug, "objective_debug": objectives.debug},
    )
    record.validate()
    return record
