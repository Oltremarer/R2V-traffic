from pathlib import Path
import json
import math

import pareto.data.build_pairs as build_pairs_module
import pytest
from pareto.data.build_pairs import build_pairs_from_records
from pareto.data.validate_pairs import summarize_pairs_dir
from pareto.r2v.traffic_artifact_schema import upgrade_weighted_row_to_v2_metadata


def _record(sample_id, eff, safety, fairness, stability, valid=True):
    values = {
        "efficiency": eff,
        "safety": safety,
        "fairness": fairness,
        "stability": stability,
    }
    return {
        "sample_id": sample_id,
        "scenario": "jinan",
        "objective_values_norm": values,
        "objective_valid_mask": {key: valid for key in values},
    }


def test_build_pairs_filters_ties_and_writes_balanced_outputs(tmp_path: Path):
    records = [
        _record("fast", 2.0, -1.0, 0.0, 0.0),
        _record("safe", -1.0, 2.0, 0.0, 0.0),
        _record("fair", 0.0, 0.0, 2.0, -1.0),
        _record("stable", 0.0, 0.0, -1.0, 2.0),
    ]

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=8,
        num_preference_pairs=8,
        num_dominance_pairs=2,
        num_reversal_pairs=2,
        seed=0,
        tie_margin=0.01,
    )
    summary = summarize_pairs_dir(tmp_path)

    assert report["serialized_tie_count"] == 0
    assert report["candidate_sampling"]["strategy"] == "bounded_random_index_pairs"
    assert summary["invalid_objective_pair_count"] == 0
    assert summary["serialized_tie_count"] == 0
    assert (tmp_path / "objective_pairs.jsonl").exists()
    assert (tmp_path / "preference_pairs.jsonl").exists()
    assert (tmp_path / "reversal_pairs.jsonl").exists()


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_objective_pair_margin_matches_serialized_order(tmp_path: Path):
    records = [
        _record("a", 1.0, 0.0, 0.0, 0.0),
        _record("b", -1.0, 0.0, 0.0, 0.0),
        _record("c", 0.0, 1.0, 0.0, 0.0),
    ]
    build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=4,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=1,
        tie_margin=0.01,
    )

    record_by_id = {row["sample_id"]: row for row in records}
    rows = _read_jsonl(tmp_path / "objective_pairs.jsonl")
    assert rows
    for row in rows:
        expected = (
            record_by_id[row["a_id"]]["objective_values_norm"][row["objective"]]
            - record_by_id[row["b_id"]]["objective_values_norm"][row["objective"]]
        )
        assert row["margin_norm"] == expected
        assert row["label"] == int(expected > 0)


def test_build_pairs_emits_contrast_subsets_and_reversal_breakdown(tmp_path: Path):
    records = [
        _record("fair_good", 1.0, 0.0, 2.0, 0.0),
        _record("fair_bad", 1.02, 0.0, -2.0, 0.0),
        _record("stable_good", 0.0, 0.0, 0.0, 2.0),
        _record("stable_bad", 0.02, 0.0, 0.0, -2.0),
        _record("fast", 2.5, -2.0, 0.0, 0.0),
        _record("safe", -2.0, 2.5, 0.0, 0.0),
    ]

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=12,
        num_preference_pairs=8,
        num_dominance_pairs=2,
        num_reversal_pairs=4,
        seed=0,
        tie_margin=0.01,
    )

    assert (tmp_path / "objective_pairs_eff_controlled_fairness.jsonl").exists()
    assert (tmp_path / "objective_pairs_eff_controlled_stability.jsonl").exists()
    assert report["contrast_counts"]["eff_controlled_fairness"] >= 1
    assert report["contrast_counts"]["eff_controlled_stability"] >= 1
    assert report["reversal_by_template_pair"]


def test_build_pairs_honors_split_and_per_objective_targets(tmp_path: Path):
    records = []
    for idx in range(12):
        records.append(_record(
            f"s{idx}",
            eff=float(idx % 4),
            safety=float((idx + 1) % 4),
            fairness=float((idx % 2) * 2 - 1),
            stability=float(((idx + 1) % 2) * 2 - 1),
        ))

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=16,
        num_preference_pairs=4,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=3,
        tie_margin=0.01,
        split="val",
    )
    rows = _read_jsonl(tmp_path / "objective_pairs.jsonl")
    objective_counts = {}
    for row in rows:
        objective_counts[row["objective"]] = objective_counts.get(row["objective"], 0) + 1
        assert row["split"] == "val"

    assert report["objective_counts"] == objective_counts
    assert all(count >= 4 for count in objective_counts.values())


def test_build_pairs_targets_efficiency_stability_conflicts_and_reversal_quota(tmp_path: Path):
    records = [
        _record("fast_unstable_0", 2.0, 0.0, 0.0, -2.0),
        _record("slow_stable_0", -2.0, 0.0, 0.0, 2.0),
        _record("fast_unstable_1", 2.2, 0.0, 0.0, -2.2),
        _record("slow_stable_1", -2.2, 0.0, 0.0, 2.2),
        _record("safe", 0.0, 2.0, 0.0, 0.0),
        _record("fair", 0.0, 0.0, 2.0, 0.0),
    ]

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=16,
        num_preference_pairs=8,
        num_dominance_pairs=0,
        num_reversal_pairs=4,
        seed=2,
        tie_margin=0.01,
        min_efficiency_stability_conflict=2,
        reversal_template_quota={"efficiency__stability": 2},
    )
    summary = summarize_pairs_dir(tmp_path)

    assert (tmp_path / "objective_pairs_efficiency_stability_conflict.jsonl").exists()
    assert report["contrast_counts"]["efficiency_stability_conflict"] >= 2
    assert summary["sampling_strategy_counts"]["efficiency_stability_conflict"] >= 2
    assert report["reversal_by_template_pair"].get("efficiency__stability", 0) >= 2


def test_build_pairs_does_not_materialize_all_combinations(tmp_path: Path, monkeypatch):
    def fail_combinations(*args, **kwargs):
        raise AssertionError("build_pairs must not materialize all pair combinations")

    monkeypatch.setattr(build_pairs_module, "combinations", fail_combinations, raising=False)
    records = [
        _record(
            f"s{idx}",
            eff=float(idx % 11),
            safety=float((idx * 3) % 11),
            fairness=float((idx * 5) % 11),
            stability=float((idx * 7) % 11),
        )
        for idx in range(200)
    ]

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path,
        num_objective_pairs=32,
        num_preference_pairs=16,
        num_dominance_pairs=4,
        num_reversal_pairs=4,
        seed=5,
        tie_margin=0.01,
    )

    assert report["objective_pairs"] == 32
    assert report["preference_pairs"] == 16
    assert (tmp_path / "objective_pairs.jsonl").exists()


def _write_r2v_weighted_rows(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _weighted_row(sample_id, weight, *, admitted=True, rare=True, value=True):
    return upgrade_weighted_row_to_v2_metadata(
        _legacy_weighted_row(sample_id, weight, admitted=admitted, rare=rare, value=value)
    )


def _legacy_weighted_row(sample_id, weight, *, admitted=True, rare=True, value=True):
    return {
        "sample_id": sample_id,
        "transition_id": f"transition_{sample_id}",
        "metadata": {
            "r2v_schema_version": "r2v-tsc-weighted-transition-v1",
            "r2v_sample_weight": weight,
            "r2v_admitted": admitted,
            "r2v_gates": {
                "rare": rare,
                "value": value,
                "support": True,
                "safety": True,
            },
        },
    }


def _positive_weight_ids(records):
    return {record["sample_id"] for record in records if record.get("_r2v_sampling_weight", 0.0) > 0.0}


def test_internal_r2v_weight_field_is_ignored_when_mode_is_off(tmp_path: Path):
    clean_records = [_record(f"s{idx}", float(idx), float(5 - idx), float(idx % 3), float((idx + 1) % 3)) for idx in range(6)]
    sentinel_records = [dict(record) for record in clean_records]
    for record in sentinel_records:
        record["_r2v_sampling_weight"] = 50.0 if record["sample_id"] == "s5" else 1.0

    build_pairs_from_records(
        clean_records,
        out_dir=tmp_path / "clean",
        num_objective_pairs=16,
        num_preference_pairs=4,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=17,
        tie_margin=0.01,
    )
    build_pairs_from_records(
        sentinel_records,
        out_dir=tmp_path / "sentinel",
        num_objective_pairs=16,
        num_preference_pairs=4,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=17,
        tie_margin=0.01,
    )

    assert (tmp_path / "clean" / "objective_pairs.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "sentinel" / "objective_pairs.jsonl"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "clean" / "preference_pairs.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "sentinel" / "preference_pairs.jsonl"
    ).read_text(encoding="utf-8")


def test_r2v_weighted_pair_sampling_is_opt_in_and_reported(tmp_path: Path):
    records = [_record(f"s{idx}", float(idx), float(7 - idx), float(idx % 3), float((idx + 1) % 3)) for idx in range(8)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row(record["sample_id"], 10.0 if record["sample_id"] == "s7" else 1.0) for record in records],
    )

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path / "pairs",
        num_objective_pairs=24,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=9,
        tie_margin=0.01,
        r2v_weighted_transitions=weighted_path,
        r2v_sampling_mode="full_r2v",
    )

    assert report["candidate_sampling"]["strategy"] == "r2v_weighted_index_pairs"
    assert report["candidate_sampling"]["r2v_sampling_mode"] == "full_r2v"
    assert report["candidate_sampling"]["r2v_join_key"] == "sample_id"
    assert report["candidate_sampling"]["r2v_weight_summary"]["weighted_row_count"] == 8
    assert report["candidate_sampling"]["r2v_weight_summary"]["weight_max"] == 10.0


def test_r2v_sampling_modes_assign_expected_positive_record_sets(tmp_path: Path):
    records = [_record(f"s{idx}", float(idx), float(4 - idx), float(idx % 2), float((idx + 1) % 2)) for idx in range(4)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [
            _weighted_row("s0", 5.0, admitted=True, rare=True, value=True),
            _weighted_row("s1", 4.0, admitted=True, rare=True, value=False),
            _weighted_row("s2", 3.0, admitted=False, rare=False, value=True),
            _weighted_row("s3", 1.0, admitted=False, rare=False, value=False),
        ],
    )

    expected_positive = {
        "admitted_only": {"s0", "s1"},
        "rare_only": {"s0", "s1"},
        "value_only": {"s0", "s2"},
        "inverted_rarity": {"s2", "s3"},
        "same_candidates_random_weights": {"s0", "s1"},
    }
    for mode, expected_ids in expected_positive.items():
        sampling_records, report = build_pairs_module._prepare_r2v_sampling_records(
            records,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode=mode,
            r2v_weight_key="metadata.r2v_sample_weight",
            r2v_join_key="sample_id",
            seed=19,
        )
        assert _positive_weight_ids(sampling_records) == expected_ids
        assert report["r2v_sampling_mode"] == mode

    random_records, random_report = build_pairs_module._prepare_r2v_sampling_records(
        records,
        r2v_weighted_transitions=weighted_path,
        r2v_sampling_mode="random_same_count",
        r2v_weight_key="metadata.r2v_sample_weight",
        r2v_join_key="sample_id",
        seed=19,
    )
    assert len(_positive_weight_ids(random_records)) == 2
    assert random_report["r2v_sampling_mode"] == "random_same_count"

    shuffled_weighted_path = tmp_path / "weighted_all_rare.jsonl"
    _write_r2v_weighted_rows(
        shuffled_weighted_path,
        [
            _weighted_row("s0", 5.0, admitted=True, rare=True, value=True),
            _weighted_row("s1", 4.0, admitted=True, rare=True, value=False),
            _weighted_row("s2", 3.0, admitted=False, rare=True, value=True),
            _weighted_row("s3", 1.0, admitted=False, rare=True, value=False),
        ],
    )
    shuffled_records, shuffled_report = build_pairs_module._prepare_r2v_sampling_records(
        records,
        r2v_weighted_transitions=shuffled_weighted_path,
        r2v_sampling_mode="shuffled_value",
        r2v_weight_key="metadata.r2v_sample_weight",
        r2v_join_key="sample_id",
        seed=19,
    )
    assert len(_positive_weight_ids(shuffled_records)) == 2
    assert shuffled_report["r2v_sampling_mode"] == "shuffled_value"


def test_r2v_weighted_pair_sampling_is_deterministic(tmp_path: Path):
    records = [_record(f"s{idx}", float(idx), float(7 - idx), float(idx % 4), float((idx + 2) % 4)) for idx in range(8)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row(record["sample_id"], 6.0 if record["sample_id"] in {"s6", "s7"} else 1.0) for record in records],
    )

    kwargs = dict(
        records=records,
        num_objective_pairs=24,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=11,
        tie_margin=0.01,
        r2v_weighted_transitions=weighted_path,
        r2v_sampling_mode="full_r2v",
    )
    build_pairs_from_records(out_dir=tmp_path / "run_a", **kwargs)
    build_pairs_from_records(out_dir=tmp_path / "run_b", **kwargs)

    assert (tmp_path / "run_a" / "objective_pairs.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "run_b" / "objective_pairs.jsonl"
    ).read_text(encoding="utf-8")


def test_r2v_weighted_pair_sampling_biases_toward_high_weight_records(tmp_path: Path):
    records = [_record(f"s{idx}", float(idx), float(10 - idx), float((idx * 2) % 5), float((idx * 3) % 5)) for idx in range(10)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row(record["sample_id"], 30.0 if record["sample_id"] == "s9" else 1.0) for record in records],
    )

    build_pairs_from_records(
        records,
        out_dir=tmp_path / "pairs",
        num_objective_pairs=80,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=13,
        tie_margin=0.01,
        r2v_weighted_transitions=weighted_path,
        r2v_sampling_mode="full_r2v",
    )
    rows = _read_jsonl(tmp_path / "pairs" / "objective_pairs.jsonl")
    hot_count = sum(1 for row in rows if row["a_id"] == "s9" or row["b_id"] == "s9")
    cold_count = sum(1 for row in rows if row["a_id"] == "s0" or row["b_id"] == "s0")

    assert hot_count > cold_count


def test_er_recent_pair_sampling_is_opt_in_and_reported(tmp_path: Path):
    records = []
    for idx in range(10):
        record = _record(f"s{idx}", float(idx), float(10 - idx), float((idx * 2) % 5), float((idx * 3) % 5))
        record["episode"] = 0
        record["step"] = idx
        records.append(record)

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path / "pairs",
        num_objective_pairs=80,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=31,
        tie_margin=0.01,
        er_baseline_mode="recent",
    )
    rows = _read_jsonl(tmp_path / "pairs" / "objective_pairs.jsonl")
    hot_count = sum(1 for row in rows if row["a_id"] == "s9" or row["b_id"] == "s9")
    cold_count = sum(1 for row in rows if row["a_id"] == "s0" or row["b_id"] == "s0")

    assert report["candidate_sampling"]["strategy"] == "er_weighted_index_pairs"
    assert report["candidate_sampling"]["er_baseline_mode"] == "recent"
    assert report["candidate_sampling"]["r2v_sampling_enabled"] is False
    assert hot_count > cold_count


def test_er_td_error_priority_requires_priority_key(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]

    with pytest.raises(ValueError, match="requires finite er_priority_key"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            er_baseline_mode="td_error_priority",
        )


def test_er_td_error_priority_uses_metadata_key(tmp_path: Path):
    records = []
    for idx in range(10):
        record = _record(f"s{idx}", float(idx), float(10 - idx), float(idx % 4), float((idx + 1) % 4))
        record["metadata"] = {"td_error": 40.0 if idx == 9 else 1.0}
        records.append(record)

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path / "pairs",
        num_objective_pairs=80,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=37,
        tie_margin=0.01,
        er_baseline_mode="td_error_priority",
    )
    rows = _read_jsonl(tmp_path / "pairs" / "objective_pairs.jsonl")
    hot_count = sum(1 for row in rows if row["a_id"] == "s9" or row["b_id"] == "s9")
    cold_count = sum(1 for row in rows if row["a_id"] == "s0" or row["b_id"] == "s0")

    assert report["candidate_sampling"]["strategy"] == "er_weighted_index_pairs"
    assert report["candidate_sampling"]["er_baseline_mode"] == "td_error_priority"
    assert report["candidate_sampling"]["er_weight_summary"]["weight_max"] > 1.0
    assert hot_count > cold_count


def test_r2v_overlay_combines_with_same_er_baseline(tmp_path: Path):
    records = []
    for idx in range(8):
        record = _record(f"s{idx}", float(idx), float(8 - idx), float(idx % 3), float((idx + 1) % 3))
        record["episode"] = 0
        record["step"] = idx
        records.append(record)
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row(record["sample_id"], 10.0 if record["sample_id"] == "s7" else 1.0) for record in records],
    )

    report = build_pairs_from_records(
        records,
        out_dir=tmp_path / "pairs",
        num_objective_pairs=32,
        num_preference_pairs=0,
        num_dominance_pairs=0,
        num_reversal_pairs=0,
        seed=41,
        tie_margin=0.01,
        er_baseline_mode="recent",
        er_r2v_combine="multiply",
        r2v_weighted_transitions=weighted_path,
        r2v_sampling_mode="full_r2v",
    )

    assert report["candidate_sampling"]["strategy"] == "r2v_er_weighted_index_pairs"
    assert report["candidate_sampling"]["er_baseline_mode"] == "recent"
    assert report["candidate_sampling"]["r2v_sampling_mode"] == "full_r2v"
    assert report["candidate_sampling"]["er_r2v_combine"] == "multiply"
    assert report["candidate_sampling"]["combined_weight_summary"]["weight_max"] > 10.0


def test_r2v_weighted_pair_sampling_rejects_missing_join_key(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [{"transition_id": "t0", "metadata": {"r2v_sample_weight": 1.0}}],
    )

    with pytest.raises(ValueError, match="missing join key sample_id"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="full_r2v",
        )


def test_r2v_weighted_pair_sampling_rejects_off_mode_with_weighted_file(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(weighted_path, [_weighted_row("s0", 1.0), _weighted_row("s1", 1.0)])

    with pytest.raises(ValueError, match="requires r2v_sampling_mode != 'off'"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="off",
        )


def test_r2v_weighted_pair_sampling_rejects_legacy_weighted_artifact(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "legacy_weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_legacy_weighted_row("s0", 1.0), _legacy_weighted_row("s1", 1.0)],
    )

    with pytest.raises(ValueError, match="r2v_traffic_schema_version"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="full_r2v",
        )


def test_r2v_weighted_pair_sampling_rejects_enabled_mode_without_weighted_file(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]

    with pytest.raises(ValueError, match="requires --r2v_weighted_transitions"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_sampling_mode="full_r2v",
        )


def test_r2v_weighted_pair_sampling_rejects_missing_weights_for_pair_records(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(weighted_path, [_weighted_row("s0", 1.0), _weighted_row("other", 1.0)])

    with pytest.raises(ValueError, match="missing R2V weights"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="full_r2v",
        )


def test_r2v_weighted_pair_sampling_rejects_fewer_than_two_positive_records(tmp_path: Path):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [
            _weighted_row("s0", 2.0, admitted=True),
            _weighted_row("s1", 1.0, admitted=False),
        ],
    )

    with pytest.raises(ValueError, match="fewer than two positive-weight records"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="admitted_only",
        )


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, math.inf, math.nan])
def test_r2v_weighted_pair_sampling_rejects_invalid_weights(tmp_path: Path, bad_weight: float):
    records = [_record("s0", 0.0, 0.0, 0.0, 0.0), _record("s1", 1.0, 0.0, 0.0, 0.0)]
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row("s0", bad_weight), _weighted_row("s1", 1.0)],
    )

    with pytest.raises(ValueError, match="invalid r2v weight"):
        build_pairs_from_records(
            records,
            out_dir=tmp_path / "pairs",
            num_objective_pairs=2,
            num_preference_pairs=0,
            num_dominance_pairs=0,
            num_reversal_pairs=0,
            r2v_weighted_transitions=weighted_path,
            r2v_sampling_mode="full_r2v",
        )


def test_pair_builder_cli_writes_r2v_pair_report(tmp_path: Path, monkeypatch):
    records = [_record(f"s{idx}", float(idx), float(5 - idx), float(idx % 3), float((idx + 1) % 3)) for idx in range(6)]
    buffer_path = tmp_path / "records.jsonl"
    buffer_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records), encoding="utf-8")
    weighted_path = tmp_path / "weighted.jsonl"
    _write_r2v_weighted_rows(
        weighted_path,
        [_weighted_row(record["sample_id"], 8.0 if record["sample_id"] == "s5" else 1.0) for record in records],
    )
    out_dir = tmp_path / "pairs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_pairs.py",
            "--buffers",
            str(buffer_path),
            "--out_dir",
            str(out_dir),
            "--num_objective_pairs",
            "12",
            "--num_preference_pairs",
            "4",
            "--num_dominance_pairs",
            "0",
            "--num_reversal_pairs",
            "0",
            "--seed",
            "23",
            "--tie_margin",
            "0.01",
            "--r2v_weighted_transitions",
            str(weighted_path),
            "--r2v_sampling_mode",
            "full_r2v",
        ],
    )

    build_pairs_module.main()

    report = json.loads((out_dir / "pair_report.json").read_text(encoding="utf-8"))
    assert report["candidate_sampling"]["strategy"] == "r2v_weighted_index_pairs"
    assert report["candidate_sampling"]["r2v_source_path"] == str(weighted_path)
