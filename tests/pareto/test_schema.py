import pytest

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.schema import (
    DominancePairLabel,
    ObjectivePairLabel,
    PreferencePairLabel,
    TrajectoryRecord,
    TransitionRecord,
    validate_preference,
)


def _record() -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version="pareto-trajectory-v1",
        run_id="run0",
        sample_id="run0:0:intersection_1_1:0",
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
        action_name="ETWT",
        phase_id=1,
        phase_duration=30.0,
        obs_features=[1.0, 2.0, 3.0],
        obs_feature_names=["a", "b", "c"],
        llm_state={},
        incoming_state={},
        mean_speed=3.0,
        objective_values_raw={name: 0.0 for name in OBJECTIVE_NAMES},
        objective_values_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        objective_valid_mask={name: True for name in OBJECTIVE_NAMES},
        prev_sample_id=None,
        next_sample_id=None,
        metadata={"feature_schema_version": "hybrid_v1"},
    )


def test_trajectory_record_round_trips_json():
    record = _record()
    restored = TrajectoryRecord.from_json(record.to_json())
    assert restored == record


def test_trajectory_record_requires_matching_feature_names():
    record = _record()
    record.obs_feature_names = ["only_one_name"]
    with pytest.raises(ValueError, match="feature"):
        record.validate()


def test_preference_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum"):
        validate_preference([0.5, 0.5, 0.5, 0.5])


def test_pair_labels_round_trip():
    pair = PreferencePairLabel(
        pair_id="p0",
        split="train",
        scenario="jinan",
        a_id="a",
        b_id="b",
        w=[0.25, 0.25, 0.25, 0.25],
        label=1,
        is_tie=False,
        source="rule",
        rule_utility_a=1.0,
        rule_utility_b=0.0,
        rule_margin=1.0,
        sampling_strategy="toy",
    )
    assert PreferencePairLabel.from_json(pair.to_json()) == pair

    dom = DominancePairLabel(
        pair_id="d0",
        split="train",
        scenario="jinan",
        a_id="a",
        b_id="b",
        dominates="a",
        objective_margins_norm={name: 1.0 for name in OBJECTIVE_NAMES},
        min_margin_norm=1.0,
        dominance_threshold=0.1,
        source="rule",
    )
    assert DominancePairLabel.from_json(dom.to_json()) == dom


def test_tie_pair_labels_are_rejected_in_foundation_schema():
    pair = ObjectivePairLabel(
        pair_id="p0",
        split="train",
        scenario="jinan",
        a_id="a",
        b_id="b",
        objective="efficiency",
        label=1,
        is_tie=True,
        margin_raw=0.0,
        margin_norm=0.0,
        source="rule",
        rule_version="v1",
        sampling_strategy="toy_tie",
    )

    with pytest.raises(ValueError, match="tie"):
        pair.validate()


def test_transition_record_round_trips_json():
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
        action=1,
        env_reward=0.0,
        objectives_t_norm={name: 0.0 for name in OBJECTIVE_NAMES},
        objectives_tp1_norm={name: 1.0 for name in OBJECTIVE_NAMES},
        done=False,
        policy_id="maxpressure",
        metadata={},
    )

    assert TransitionRecord.from_json(transition.to_json()) == transition
