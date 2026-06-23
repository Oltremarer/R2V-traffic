from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.r2v.build_r2v_candidates import build_candidates_from_files, main as build_candidates_cli_main, parse_args
from pareto.r2v.traffic_candidate_selector import (
    R2VTrafficSelectorConfig,
    select_r2v_candidates,
)
from pareto.r2v.traffic_artifact_schema import validate_r2v_traffic_artifact


def _transition(
    transition_id: str,
    obs: list[float],
    next_obs: list[float],
    *,
    value_delta: float,
    safety_tp1: float = 0.0,
    frozen_target_utility: float | None = None,
    metadata: dict | None = None,
) -> dict:
    objectives_t = {
        "efficiency": 0.0,
        "safety": 0.0,
        "fairness": 0.0,
        "stability": 0.0,
    }
    objectives_tp1 = {
        "efficiency": float(value_delta),
        "safety": float(safety_tp1),
        "fairness": 0.0,
        "stability": 0.0,
    }
    row = {
        "schema_version": "pareto-transition-v1",
        "run_id": "run0",
        "transition_id": transition_id,
        "sample_id": transition_id,
        "next_sample_id": f"{transition_id}_next",
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "seed": 0,
        "episode": 0,
        "step": 0,
        "intersection_id": "0",
        "obs_features": obs,
        "next_obs_features": next_obs,
        "action": 0,
        "env_reward": value_delta,
        "objectives_t_norm": objectives_t,
        "objectives_tp1_norm": objectives_tp1,
        "done": False,
        "policy_id": "mock",
        "metadata": dict(metadata or {}),
    }
    if frozen_target_utility is not None:
        row["frozen_target_utility"] = float(frozen_target_utility)
    return row


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _repair_metadata(source_gates: dict[str, bool], final_gates: dict[str, bool]) -> dict:
    return {
        "r2v_source_gates": dict(source_gates),
        "r2v_final_gates": dict(final_gates),
    }


def _all_gates(value: bool = True, rare: bool = True, support: bool = True, safety: bool = True) -> dict[str, bool]:
    return {
        "rare": bool(rare),
        "value": bool(value),
        "support": bool(support),
        "safety": bool(safety),
    }


def test_rare_not_valuable_transition_is_not_admitted():
    records = [
        _transition("common_good", [0.0, 0.0], [0.0, 0.0], value_delta=0.2),
        _transition("common_bad", [0.1, 0.0], [0.1, 0.0], value_delta=-0.1),
        _transition("rare_bad", [10.0, 10.0], [10.0, 10.0], value_delta=-2.0),
        _transition("rare_good", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["rare_bad"]["gates"]["rare"] is True
    assert by_id["rare_bad"]["gates"]["value"] is False
    assert by_id["rare_bad"]["admitted"] is False
    assert by_id["rare_bad"]["debug"]["rare_is_not_value"] is True
    assert by_id["common_good"]["gates"]["rare"] is False
    assert by_id["common_good"]["admitted"] is True
    assert by_id["rare_good"]["admitted"] is True
    assert "rare" not in summary["active_gates"]
    assert summary["admitted_count"] == 2


def test_full_gate_admission_does_not_require_rare_gate_when_value_support_dynamics_pass():
    records = [
        _transition(
            "ordinary_repaired_to_value",
            [0.0, 0.0],
            [0.0, 0.0],
            value_delta=1.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(rare=False, value=False, support=True, safety=True),
                final_gates=_all_gates(rare=False, value=True, support=True, safety=True),
            ),
        )
    ]

    candidates, _summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            repair_story="not_rare_to_val",
            gate_variant="full",
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    assert candidates[0]["gates"]["rare"] is False
    assert candidates[0]["admitted"] is True
    assert "rare" not in candidates[0]["active_gates"]


def test_zero_reward_transition_is_not_treated_as_corrupted_when_value_gate_passes():
    records = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("rare_zero_reward_good", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]
    records[1]["env_reward"] = 0.0

    candidates, _summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert records[1]["env_reward"] == 0.0
    assert by_id["rare_zero_reward_good"]["gates"]["value"] is True
    assert by_id["rare_zero_reward_good"]["admitted"] is True


def test_safety_gate_blocks_high_rarity_high_value_transition():
    records = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.1, safety_tp1=0.0),
        _transition("safe_rare", [8.0, 8.0], [8.0, 8.0], value_delta=2.0, safety_tp1=0.0),
        _transition("unsafe_rare", [10.0, 10.0], [10.0, 10.0], value_delta=3.0, safety_tp1=-1.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.34,
            value_quantile=0.34,
            support_min_quantile=0.0,
            safety_min=-0.5,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["unsafe_rare"]["gates"]["rare"] is True
    assert by_id["unsafe_rare"]["gates"]["value"] is True
    assert by_id["unsafe_rare"]["gates"]["safety"] is False
    assert by_id["unsafe_rare"]["admitted"] is False
    assert by_id["unsafe_rare"]["admission_score_components"]["safety_margin"] == 0.0
    assert by_id["safe_rare"]["admitted"] is True
    assert summary["gate_counts"]["safety"] == 2
    assert summary["gate_failure_counts"]["safety"] == 1


def test_no_dynamics_gate_variant_is_ablation_not_default():
    records = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.1, safety_tp1=0.0),
        _transition("safe_rare", [8.0, 8.0], [8.0, 8.0], value_delta=2.0, safety_tp1=0.0),
        _transition("unsafe_rare", [10.0, 10.0], [10.0, 10.0], value_delta=3.0, safety_tp1=-1.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.34,
            value_quantile=0.34,
            support_min_quantile=0.0,
            safety_min=-0.5,
            gate_variant="no_dynamics",
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["unsafe_rare"]["gates"]["safety"] is False
    assert by_id["unsafe_rare"]["admitted"] is True
    assert by_id["unsafe_rare"]["gate_variant"] == "no_dynamics"
    assert "safety" not in by_id["unsafe_rare"]["active_gates"]
    assert summary["gate_variant"] == "no_dynamics"
    assert "safety" not in summary["active_gates"]


def test_summary_reports_rarity_value_diagnostics_and_admission_components():
    records = [
        _transition("common_good", [0.0, 0.0], [0.0, 0.0], value_delta=1.0),
        _transition("common_bad", [0.1, 0.0], [0.1, 0.0], value_delta=-1.0),
        _transition("rare_bad", [9.0, 9.0], [9.0, 9.0], value_delta=-2.0),
        _transition("rare_good", [10.0, 10.0], [10.0, 10.0], value_delta=2.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    assert summary["value_mode"] == "objective_delta"
    assert isinstance(summary["rarity_value_correlation"], float)
    assert summary["gate_failure_counts"]["value"] >= 1
    assert summary["admission_score_component_means"]["value_margin"] >= 0.0
    assert all("admission_score_components" in row for row in candidates)
    assert "admitted_admission_score_component_means" in summary
    assert "admission_score_mean" in summary
    assert "admitted_admission_score_mean" in summary


def test_frozen_target_utility_mode_keeps_value_gate_independent_from_objective_delta():
    records = [
        _transition("common0", [0.0, 0.0], [0.0, 0.0], value_delta=0.0, frozen_target_utility=0.0),
        _transition("common1", [0.1, 0.0], [0.1, 0.0], value_delta=0.1, frozen_target_utility=0.1),
        _transition("rare_polluted_delta", [10.0, 10.0], [10.0, 10.0], value_delta=10.0, frozen_target_utility=-2.0),
        _transition("rare_frozen_good", [8.0, 8.0], [8.0, 8.0], value_delta=-1.0, frozen_target_utility=2.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
            value_mode="frozen_target_utility",
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert summary["value_mode"] == "frozen_target_utility"
    assert by_id["rare_polluted_delta"]["gates"]["rare"] is True
    assert by_id["rare_polluted_delta"]["gates"]["value"] is False
    assert by_id["rare_polluted_delta"]["debug"]["frozen_target_utility"] == -2.0
    assert by_id["rare_frozen_good"]["gates"]["value"] is True
    assert by_id["rare_frozen_good"]["admitted"] is True


def test_empty_input_keeps_summary_threshold_schema():
    candidates, summary = select_r2v_candidates(
        [],
        R2VTrafficSelectorConfig(),
    )

    assert candidates == []
    assert summary["thresholds"] == {
        "rarity_min": 0.0,
        "value_min": 0.0,
        "support_min": 0.0,
        "safety_min": -1.0,
    }
    assert summary["gate_failure_counts"] == {
        "rare": 0,
        "value": 0,
        "support": 0,
        "safety": 0,
    }


def test_invalid_utility_weights_raise_value_error():
    with pytest.raises(ValueError, match="utility_weights missing objective keys"):
        R2VTrafficSelectorConfig(
            utility_weights={
                "efficiency": 1.0,
                "safety": 0.0,
            },
        ).validate()


def test_frozen_target_utility_rejects_env_reward_mixing():
    with pytest.raises(ValueError, match="requires env_reward_weight=0.0"):
        R2VTrafficSelectorConfig(
            value_mode="frozen_target_utility",
            env_reward_weight=0.1,
        ).validate()


def test_missing_frozen_target_utility_raises_value_error():
    records = [
        _transition("common0", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("rare0", [10.0, 10.0], [10.0, 10.0], value_delta=1.0),
    ]

    with pytest.raises(ValueError, match="requires a frozen_target_utility field"):
        select_r2v_candidates(
            records,
            R2VTrafficSelectorConfig(value_mode="frozen_target_utility"),
        )


def test_repair_story_requires_source_and_final_gate_metadata():
    records = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("t1", [1.0, 1.0], [1.0, 1.0], value_delta=1.0),
    ]

    with pytest.raises(ValueError, match="requires source gates"):
        select_r2v_candidates(
            records,
            R2VTrafficSelectorConfig(repair_story="not_val_to_val"),
        )


def test_not_rare_to_val_proxy_policy_can_build_without_repair_metadata():
    records = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("rare_good", [10.0, 10.0], [10.0, 10.0], value_delta=2.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            repair_story="not_rare_to_val",
            repair_metadata_policy="metadata_or_proxy",
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["rare_good"]["admitted"] is True
    assert by_id["rare_good"]["repair_story_match"] is True
    assert by_id["rare_good"]["source_gates"]["rare"] is False
    assert by_id["rare_good"]["final_gates"]["rare"] is True
    assert by_id["rare_good"]["gate_source"] == "computed_proxy_repair_metadata"
    assert summary["repair_metadata_policy"] == "metadata_or_proxy"
    assert summary["gate_source"] == "computed_proxy_repair_metadata"


def test_not_val_to_val_repair_story_uses_final_gates_for_admission():
    records = [
        _transition(
            "repaired_value",
            [0.0, 0.0],
            [0.0, 0.0],
            value_delta=-2.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(value=False),
                final_gates=_all_gates(value=True),
            ),
        ),
        _transition(
            "source_already_value",
            [2.0, 2.0],
            [2.0, 2.0],
            value_delta=2.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(value=True),
                final_gates=_all_gates(value=True),
            ),
        ),
        _transition(
            "final_not_value",
            [4.0, 4.0],
            [4.0, 4.0],
            value_delta=0.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(value=False),
                final_gates=_all_gates(value=False),
            ),
        ),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            repair_story="not_val_to_val",
            rare_quantile=0.0,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["repaired_value"]["debug"]["computed_gates"]["value"] is False
    assert by_id["repaired_value"]["gates"]["value"] is True
    assert by_id["repaired_value"]["repair_story_match"] is True
    assert by_id["repaired_value"]["admitted"] is True
    assert by_id["source_already_value"]["repair_story_match"] is False
    assert by_id["source_already_value"]["admitted"] is False
    assert by_id["final_not_value"]["repair_story_match"] is False
    assert by_id["final_not_value"]["admitted"] is False
    assert summary["repair_story"] == "not_val_to_val"
    assert summary["repair_story_required"] is True
    assert summary["repair_story_match_count"] == 1
    assert summary["admitted_count"] == 1


def test_not_rare_to_val_repair_story_requires_source_not_rare_and_final_value():
    records = [
        _transition(
            "repaired_rare",
            [0.0, 0.0],
            [0.0, 0.0],
            value_delta=0.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(rare=False, value=False),
                final_gates=_all_gates(rare=True, value=True),
            ),
        ),
        _transition(
            "source_already_rare",
            [1.0, 1.0],
            [1.0, 1.0],
            value_delta=1.0,
            metadata=_repair_metadata(
                source_gates=_all_gates(rare=True, value=False),
                final_gates=_all_gates(rare=True, value=True),
            ),
        ),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            repair_story="not_rare_to_val",
            rare_quantile=0.0,
            value_quantile=0.0,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["repaired_rare"]["repair_story_match"] is True
    assert by_id["repaired_rare"]["admitted"] is True
    assert by_id["source_already_rare"]["repair_story_match"] is False
    assert by_id["source_already_rare"]["admitted"] is False
    assert summary["repair_story_match_count"] == 1


def test_support_gate_can_veto_rare_high_value_candidate():
    records = [
        _transition("common0", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("common1", [0.1, 0.0], [0.1, 0.0], value_delta=0.1),
        _transition("rare_unsupported", [100.0, 100.0], [100.0, 100.0], value_delta=10.0),
        _transition("rare_supported", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.5,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["rare_unsupported"]["gates"]["rare"] is True
    assert by_id["rare_unsupported"]["gates"]["value"] is True
    assert by_id["rare_unsupported"]["gates"]["support"] is False
    assert by_id["rare_unsupported"]["admitted"] is False
    assert summary["gate_failure_counts"]["support"] >= 1


def test_score_artifact_overrides_density_rarity_but_keeps_value_gate(tmp_path: Path):
    score_artifact = tmp_path / "scores.jsonl"
    score_artifact.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "transition_id": "common_good",
                    "rarity_score": 0.1,
                    "support_score": 1.0,
                    "model_checkpoint": "mock_diffusion.ckpt",
                    "config_hash": "cfg0",
                },
                {
                    "transition_id": "rare_bad",
                    "rarity_score": 10.0,
                    "support_score": 0.8,
                    "model_checkpoint": "mock_diffusion.ckpt",
                    "config_hash": "cfg0",
                },
                {
                    "transition_id": "rare_good",
                    "rarity_score": 9.0,
                    "support_score": 0.9,
                    "model_checkpoint": "mock_diffusion.ckpt",
                    "config_hash": "cfg0",
                },
            ]
        ),
        encoding="utf-8",
    )
    records = [
        _transition("common_good", [0.0, 0.0], [0.0, 0.0], value_delta=2.0),
        _transition("rare_bad", [0.1, 0.0], [0.1, 0.0], value_delta=-1.0),
        _transition("rare_good", [0.2, 0.0], [0.2, 0.0], value_delta=1.0),
    ]

    candidates, summary = select_r2v_candidates(
        records,
        R2VTrafficSelectorConfig(
            candidate_model="diffusion_score_artifact",
            score_artifact_path=str(score_artifact),
            score_artifact_backend="diffusion",
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )
    by_id = {row["transition_id"]: row for row in candidates}

    assert by_id["rare_bad"]["rarity_score"] == 10.0
    assert by_id["rare_bad"]["gates"]["rare"] is True
    assert by_id["rare_bad"]["gates"]["value"] is False
    assert by_id["rare_bad"]["admitted"] is False
    assert by_id["rare_good"]["admitted"] is True
    assert by_id["rare_good"]["debug"]["score_source"]["backend"] == "diffusion"
    assert by_id["rare_good"]["debug"]["score_source"]["model_checkpoint"] == "mock_diffusion.ckpt"
    assert summary["score_source"]["kind"] == "score_artifact"
    assert summary["score_source"]["backend"] == "diffusion"
    assert summary["score_source"]["matched_count"] == len(records)


def test_score_artifact_repaired_transition_feeds_weights_plus_repaired(tmp_path: Path):
    repaired = _transition("rare_good_repaired", [0.3, 0.0], [0.4, 0.0], value_delta=2.5)
    score_artifact = tmp_path / "diffusion_repair_scores.jsonl"
    score_artifact.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "transition_id": "common",
                    "rarity_score": 0.1,
                    "support_score": 1.0,
                    "model_checkpoint": "r2v_diffusion.pt",
                    "config_hash": "cfg0",
                    "normalization_id": "norm0",
                },
                {
                    "transition_id": "rare_good",
                    "rarity_score": 9.0,
                    "support_score": 0.9,
                    "model_checkpoint": "r2v_diffusion.pt",
                    "config_hash": "cfg0",
                    "normalization_id": "norm0",
                    "repaired_transition": repaired,
                },
            ]
        ),
        encoding="utf-8",
    )
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("rare_good", [0.2, 0.0], [0.2, 0.0], value_delta=2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            candidate_model="diffusion_score_artifact",
            score_artifact_path=str(score_artifact),
            score_artifact_backend="diffusion",
            admission_mode="weights_plus_repaired",
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)
    by_id = {row["transition_id"]: row for row in weighted_rows}

    assert set(by_id) == {"common", "rare_good", "rare_good_repaired"}
    assert by_id["rare_good_repaired"]["metadata"]["r2v_row_role"] == "repaired"
    assert by_id["rare_good_repaired"]["metadata"]["r2v_repaired_from_transition_id"] == "rare_good"
    assert by_id["rare_good_repaired"]["metadata"]["r2v_proposal_source"] == "score_artifact.repaired_transition"
    assert by_id["rare_good_repaired"]["metadata"]["r2v_generative_backend"] == "diffusion"
    assert validate_r2v_traffic_artifact(weighted_rows)["admission_modes"] == ["weights_plus_repaired"]


def test_exact_admitted_weight_overrides_bonus_weighting(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("rare_good", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            admitted_weight=4.0,
            admitted_weight_bonus=9.0,
            max_weight=10.0,
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)
    by_id = {row["transition_id"]: row for row in weighted_rows}

    assert by_id["rare_good"]["metadata"]["r2v_admitted"] is True
    assert by_id["rare_good"]["metadata"]["r2v_sample_weight"] == 4.0


def test_weights_plus_repaired_appends_rejected_repair_proposal_with_rejected_weight(tmp_path: Path):
    repaired = _transition("rare_bad_repaired", [0.4, 0.0], [0.5, 0.0], value_delta=0.5)
    score_artifact = tmp_path / "diffusion_repair_scores.jsonl"
    score_artifact.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "transition_id": "common",
                    "rarity_score": 0.1,
                    "support_score": 1.0,
                },
                {
                    "transition_id": "rare_bad",
                    "rarity_score": 9.0,
                    "support_score": 0.9,
                    "repaired_transition": repaired,
                },
            ]
        ),
        encoding="utf-8",
    )
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("common", [0.0, 0.0], [0.0, 0.0], value_delta=10.0, safety_tp1=-20.0),
        _transition("rare_bad", [0.2, 0.0], [0.2, 0.0], value_delta=-2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            candidate_model="diffusion_score_artifact",
            score_artifact_path=str(score_artifact),
            score_artifact_backend="diffusion",
            admission_mode="weights_plus_repaired",
            repair_rejected_weight=0.25,
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)
    by_id = {row["transition_id"]: row for row in weighted_rows}

    assert set(by_id) == {"common", "rare_bad", "rare_bad_repaired"}
    assert by_id["rare_bad"]["metadata"]["r2v_admitted"] is False
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_row_role"] == "repair_rejected"
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_repaired_from_transition_id"] == "rare_bad"
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_repair_rejected"] is True
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_admitted"] is False
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_sample_weight"] == 0.25
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_proposal_source"] == "score_artifact.repaired_transition"
    assert by_id["rare_bad_repaired"]["metadata"]["r2v_generative_backend"] == "diffusion"


def test_score_artifact_missing_transition_fails_closed(tmp_path: Path):
    score_artifact = tmp_path / "scores.jsonl"
    score_artifact.write_text(
        json.dumps({"transition_id": "present", "rarity_score": 1.0}) + "\n",
        encoding="utf-8",
    )
    records = [
        _transition("present", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("missing", [1.0, 0.0], [1.0, 0.0], value_delta=1.0),
    ]

    with pytest.raises(ValueError, match="missing score artifact row"):
        select_r2v_candidates(
            records,
            R2VTrafficSelectorConfig(score_artifact_path=str(score_artifact)),
        )


def test_cli_builds_candidate_jsonl_weighted_copy_and_summary(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition("t1", [0.2, 0.0], [0.2, 0.0], value_delta=0.1),
        _transition("t2", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
        _transition("t3", [9.0, 9.0], [9.0, 9.0], value_delta=3.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    out_path = tmp_path / "candidates.jsonl"
    weighted_path = tmp_path / "weighted_transitions.jsonl"
    summary_path = tmp_path / "summary.json"

    report = build_candidates_from_files(
        transitions=[transitions_path],
        output=out_path,
        summary_output=summary_path,
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    candidate_rows = _read_jsonl(out_path)
    weighted_rows = _read_jsonl(weighted_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert report["candidate_count"] == 4
    assert summary["record_count"] == 4
    assert any(row["admitted"] for row in candidate_rows)
    assert all("r2v_sample_weight" in row["metadata"] for row in weighted_rows)
    required_metadata = {
        "r2v_schema_version",
        "r2v_value_mode",
        "r2v_repair_story",
        "r2v_repair_story_match",
        "r2v_sample_weight",
        "r2v_admitted",
        "r2v_candidate_model",
        "r2v_admission_score",
        "r2v_candidate_rank",
        "r2v_gates",
        "r2v_final_gates",
        "r2v_computed_gates",
        "r2v_source_summary",
    }
    assert all(required_metadata.issubset(row["metadata"]) for row in weighted_rows)
    assert all(row["metadata"]["r2v_traffic_schema_version"] == "r2v-traffic-artifact-v2" for row in weighted_rows)
    assert all("r2v_traffic_gates" in row["metadata"] for row in weighted_rows)
    assert validate_r2v_traffic_artifact(weighted_rows)["row_count"] == 4
    assert max(row["metadata"]["r2v_sample_weight"] for row in weighted_rows) > 1.0


def test_weighted_output_records_weights_only_admission_mode_by_default(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition("t1", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)

    assert all(row["metadata"]["r2v_admission_mode"] == "weights_only" for row in weighted_rows)
    assert all(row["metadata"]["r2v_row_role"] == "source" for row in weighted_rows)
    assert all(row["metadata"]["r2v_repaired_from_transition_id"] is None for row in weighted_rows)


def test_weighted_output_records_proxy_backend_without_score_artifact(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition("t1", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            candidate_model="feature_density_proxy",
            score_artifact_backend="diffusion",
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)
    summary = validate_r2v_traffic_artifact(weighted_rows)

    assert summary["generative_backends"] == ["feature_density_proxy"]
    assert all(
        row["metadata"]["r2v_generative_backend"] == "feature_density_proxy"
        for row in weighted_rows
    )


def test_weights_plus_repaired_appends_admitted_repaired_transition(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    repaired = _transition("t1_repaired", [7.0, 7.0], [7.5, 7.5], value_delta=2.5)
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition(
            "t1",
            [8.0, 8.0],
            [8.0, 8.0],
            value_delta=2.0,
            metadata={"r2v_repaired_transition": repaired},
        ),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    weighted_path = tmp_path / "weighted_transitions.jsonl"

    build_candidates_from_files(
        transitions=[transitions_path],
        output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "summary.json",
        weighted_output=weighted_path,
        config=R2VTrafficSelectorConfig(
            admission_mode="weights_plus_repaired",
            rare_quantile=0.5,
            value_quantile=0.5,
            support_min_quantile=0.0,
            safety_min=-10.0,
        ),
    )

    weighted_rows = _read_jsonl(weighted_path)
    by_id = {row["transition_id"]: row for row in weighted_rows}

    assert set(by_id) == {"t0", "t1", "t1_repaired"}
    assert by_id["t1_repaired"]["metadata"]["r2v_admission_mode"] == "weights_plus_repaired"
    assert by_id["t1_repaired"]["metadata"]["r2v_row_role"] == "repaired"
    assert by_id["t1_repaired"]["metadata"]["r2v_repaired_from_transition_id"] == "t1"
    assert by_id["t1_repaired"]["metadata"]["r2v_admitted"] is True
    assert by_id["t1_repaired"]["metadata"]["r2v_traffic_schema_version"] == "r2v-traffic-artifact-v2"
    assert validate_r2v_traffic_artifact(weighted_rows)["row_count"] == 3


def test_weights_plus_repaired_requires_payload_for_admitted_candidate(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition("t1", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="requires repaired transition"):
        build_candidates_from_files(
            transitions=[transitions_path],
            output=tmp_path / "candidates.jsonl",
            summary_output=tmp_path / "summary.json",
            weighted_output=tmp_path / "weighted_transitions.jsonl",
            config=R2VTrafficSelectorConfig(
                admission_mode="weights_plus_repaired",
                rare_quantile=0.5,
                value_quantile=0.5,
                support_min_quantile=0.0,
                safety_min=-10.0,
            ),
        )


def test_weighted_output_rejects_missing_transition_id(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    row = _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=0.0)
    row.pop("transition_id")
    transitions_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="missing transition_id"):
        build_candidates_from_files(
            transitions=[transitions_path],
            output=tmp_path / "candidates.jsonl",
            summary_output=tmp_path / "summary.json",
            weighted_output=tmp_path / "weighted.jsonl",
            config=R2VTrafficSelectorConfig(),
        )


def test_weighted_output_rejects_duplicate_transition_id(tmp_path: Path):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("dup", [0.0, 0.0], [0.0, 0.0], value_delta=0.0),
        _transition("dup", [1.0, 1.0], [1.0, 1.0], value_delta=1.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate transition_id"):
        build_candidates_from_files(
            transitions=[transitions_path],
            output=tmp_path / "candidates.jsonl",
            summary_output=tmp_path / "summary.json",
            weighted_output=tmp_path / "weighted.jsonl",
            config=R2VTrafficSelectorConfig(),
        )


def test_cli_accepts_value_mode_argument(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_r2v_candidates.py",
            "--transitions",
            "transitions.jsonl",
            "--output",
            "candidates.jsonl",
            "--summary_output",
            "summary.json",
            "--value_mode",
            "frozen_target_utility",
        ],
    )

    args = parse_args()

    assert args.value_mode == "frozen_target_utility"


def test_cli_accepts_repair_story_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_r2v_candidates.py",
            "--transitions",
            "transitions.jsonl",
            "--output",
            "candidates.jsonl",
            "--summary_output",
            "summary.json",
            "--repair_story",
            "not_val_to_val",
            "--source_gates_key",
            "metadata.before",
            "--final_gates_key",
            "metadata.after",
        ],
    )

    args = parse_args()

    assert args.repair_story == "not_val_to_val"
    assert args.source_gates_key == "metadata.before"
    assert args.final_gates_key == "metadata.after"


def test_cli_accepts_repair_metadata_policy(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_r2v_candidates.py",
            "--transitions",
            "transitions.jsonl",
            "--output",
            "candidates.jsonl",
            "--summary_output",
            "summary.json",
            "--repair_story",
            "not_rare_to_val",
            "--repair_metadata_policy",
            "metadata_or_proxy",
        ],
    )

    args = parse_args()

    assert args.repair_metadata_policy == "metadata_or_proxy"


def test_cli_accepts_admission_mode_and_repaired_transition_key(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_r2v_candidates.py",
            "--transitions",
            "transitions.jsonl",
            "--output",
            "candidates.jsonl",
            "--summary_output",
            "summary.json",
            "--admission_mode",
            "weights_plus_repaired",
            "--admitted_weight",
            "4.0",
            "--repaired_transition_key",
            "metadata.repaired",
            "--repair_rejected_weight",
            "0.25",
        ],
    )

    args = parse_args()

    assert args.admission_mode == "weights_plus_repaired"
    assert args.admitted_weight == 4.0
    assert args.repaired_transition_key == "metadata.repaired"
    assert args.repair_rejected_weight == 0.25


def test_candidate_builder_cli_writes_weighted_output_and_summary(tmp_path: Path, monkeypatch):
    transitions_path = tmp_path / "transitions.jsonl"
    rows = [
        _transition("t0", [0.0, 0.0], [0.0, 0.0], value_delta=-0.1),
        _transition("t1", [0.2, 0.0], [0.2, 0.0], value_delta=0.1),
        _transition("t2", [8.0, 8.0], [8.0, 8.0], value_delta=2.0),
        _transition("t3", [9.0, 9.0], [9.0, 9.0], value_delta=3.0),
    ]
    transitions_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    out_path = tmp_path / "candidates.jsonl"
    weighted_path = tmp_path / "weighted.jsonl"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_r2v_candidates.py",
            "--transitions",
            str(transitions_path),
            "--output",
            str(out_path),
            "--summary_output",
            str(summary_path),
            "--weighted_output",
            str(weighted_path),
            "--rare_quantile",
            "0.5",
            "--value_quantile",
            "0.5",
            "--support_min_quantile",
            "0.0",
            "--safety_min",
            "-10.0",
        ],
    )

    build_candidates_cli_main()

    assert out_path.exists()
    assert weighted_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    weighted_rows = _read_jsonl(weighted_path)
    assert summary["candidate_count"] == 4
    assert all(row["metadata"]["r2v_schema_version"] == "r2v-tsc-weighted-transition-v1" for row in weighted_rows)
