import json

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.schema import TrajectoryRecord, TransitionRecord
from pareto.data.validate_schema import validate_files


def _record(sample_id: str, next_sample_id=None, feature_names=None):
    feature_names = feature_names or ["f0", "f1"]
    return TrajectoryRecord(
        schema_version="pareto-trajectory-v1",
        run_id="run0",
        sample_id=sample_id,
        scenario="jinan",
        roadnet="3_4",
        traffic_file="anon_3_4_jinan_real.json",
        seed=0,
        episode=0,
        step=0,
        sim_time_sec=0.0,
        intersection_id="intersection_1_1",
        policy_id="maxpressure",
        action=0,
        action_name=None,
        phase_id=0,
        phase_duration=30.0,
        obs_features=[0.0 for _ in feature_names],
        obs_feature_names=feature_names,
        llm_state={},
        incoming_state={},
        mean_speed=0.0,
        objective_values_raw={name: 0.0 for name in OBJECTIVE_NAMES},
        objective_values_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        objective_valid_mask={name: True for name in OBJECTIVE_NAMES},
        prev_sample_id=None,
        next_sample_id=next_sample_id,
        metadata={},
    )


def _write_jsonl(path, records):
    path.write_text("\n".join(record.to_json() for record in records) + "\n", encoding="utf-8")


def test_validate_files_checks_feature_schema_across_inputs(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [_record("a0", feature_names=["f0", "f1"])])
    _write_jsonl(b, [_record("b0", feature_names=["f0", "different"])])

    report = validate_files([a, b], check_feature_schema_same=True)

    assert report["error_count"] == 1
    assert "feature schema differs" in report["errors"][0]["error"]


def test_validate_files_checks_next_links_when_required(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [_record("s0", next_sample_id="s1"), _record("s1", next_sample_id=None)])

    report = validate_files([path], require_next_links=True)

    assert report["error_count"] == 0
    assert report["files"][0]["missing_next_link_count"] == 0


def test_validate_transition_checks_state_feature_hash_match(tmp_path):
    path = tmp_path / "transitions.jsonl"
    transition = TransitionRecord(
        schema_version="pareto-transition-v1",
        run_id="run0",
        transition_id="tr0",
        sample_id="s0",
        next_sample_id="s1",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed=0,
        episode=0,
        step=0,
        intersection_id="intersection_1_1",
        obs_features=[1.0, 2.0],
        next_obs_features=[2.0, 3.0],
        action=0,
        env_reward=0.0,
        objectives_t_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        objectives_tp1_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        done=False,
        policy_id="maxpressure",
        metadata={"state_feature_names_hash": "a", "next_state_feature_names_hash": "b"},
    )
    path.write_text(transition.to_json() + "\n", encoding="utf-8")

    report = validate_files([path], schema="transition")

    assert report["error_count"] == 1
    assert "feature_names_hash mismatch" in report["errors"][0]["error"]


def test_validate_transition_requires_feature_hashes(tmp_path):
    path = tmp_path / "transitions.jsonl"
    transition = TransitionRecord(
        schema_version="pareto-transition-v1",
        run_id="run0",
        transition_id="tr0",
        sample_id="s0",
        next_sample_id="s1",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed=0,
        episode=0,
        step=0,
        intersection_id="intersection_1_1",
        obs_features=[1.0, 2.0],
        next_obs_features=[2.0, 3.0],
        action=0,
        env_reward=0.0,
        objectives_t_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        objectives_tp1_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        done=False,
        policy_id="maxpressure",
        metadata={},
    )
    path.write_text(transition.to_json() + "\n", encoding="utf-8")

    report = validate_files([path], schema="transition")

    assert report["error_count"] == 1
    assert "missing state/next_state feature_names_hash" in report["errors"][0]["error"]
