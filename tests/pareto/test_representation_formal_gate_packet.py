import json
from pathlib import Path

import pytest

from pareto.eval.representation_formal_gate_packet import (
    build_packet,
    validate_packet,
    write_markdown,
)


OBJECTIVES = ("efficiency", "safety", "fairness", "stability")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metadata(param_count: int, architecture: str = "residual_tower") -> dict:
    return {
        "architecture": architecture,
        "score_mode": "low_rank_interaction",
        "isotonic_dominance_weight": 1.0,
        "dominance_coord_loss_weight": 0.5,
        "dominance_utility_loss_weight": 0.5,
        "param_count": param_count,
    }


def _metrics() -> dict:
    return {
        "rev_acc": 0.63,
        "dpr_head": 0.75,
        "dpr_utility": 0.95,
        "pref_acc": 0.76,
        "obj_acc_mean": 0.75,
        "head_leakage_diag_offdiag_gap": 0.24,
        "output_std_per_head": [2.0, 1.0, 1.8, 1.5],
    }


def _pair_report(split: str) -> dict:
    return {
        "counts": {
            "objective_pairs": 1000,
            "preference_pairs": 1000,
            "dominance_pairs": 180,
            "reversal_pairs": 200,
        },
        "error_count": 0,
        "serialized_tie_count": 0,
        "invalid_objective_pair_count": 0,
        "objective_counts": {name: 250 for name in OBJECTIVES},
        "positive_ratio_by_objective": {name: 0.5 for name in OBJECTIVES},
        "positive_ratio_by_strategy": {
            "eff_controlled_fairness": 0.5,
            "eff_controlled_stability": 0.5,
            "efficiency_safety_conflict": 0.5,
            "efficiency_stability_conflict": 0.5,
        },
        "sampling_strategy_counts": {
            "eff_controlled_fairness": 80,
            "eff_controlled_stability": 80,
            "efficiency_safety_conflict": 80,
            "efficiency_stability_conflict": 25,
        },
        "reversal_by_template_pair": {"efficiency__stability": 30},
        "split_counts": {split: 2380},
    }


def _write_evidence(root: Path) -> None:
    root.mkdir(parents=True)
    _write(root / "lowrank_iso_w100_coord050_diagnostics_test.json", _metrics())
    _write(root / "lowrank_iso_w100_coord050_diagnostics_val.json", _metrics())
    _write(root / "lowrank_iso_w100_coord050_metadata.json", _metadata(149800))
    _write(root / "film_rich_v2_metadata.json", _metadata(108161, architecture="film"))
    _write(root / "lowrank_iso_w100_coord050_formal_gate_decision.json", {
        "representation_gate_pass": True,
        "ppo_formal_allowed": True,
        "claim_mode": "vector_superiority",
        "failed_reasons": [],
    })
    _write(root / "lowrank_iso_w100_coord050_pair_bootstrap_test.json", {
        "vector": {
            "metrics": {
                "rev_acc": {"mean": 0.63, "low": 0.61, "high": 0.69, "n": 200, "n_boot": 1000},
                "dpr_head": {"mean": 0.75, "low": 0.75, "high": 0.81, "n": 180, "n_boot": 1000},
                "dpr_utility": {"mean": 0.95, "low": 0.94, "high": 0.96, "n": 900, "n_boot": 1000},
                "pref_acc": {"mean": 0.76, "low": 0.73, "high": 0.78, "n": 1000, "n_boot": 1000},
                "obj_acc_mean": {"mean": 0.75, "low": 0.72, "high": 0.77, "n": 1000, "n_boot": 1000},
                "head_leakage_diag_offdiag_gap": {"mean": 0.24, "low": 0.21, "high": 0.27, "n": 1000, "n_boot": 1000},
            }
        },
        "cond_scalar": {
            "metrics": {
                "rev_acc": {"mean": 0.605, "low": 0.535, "high": 0.67, "n": 200, "n_boot": 1000},
                "pref_acc": {"mean": 0.746, "low": 0.718, "high": 0.774, "n": 1000, "n_boot": 1000},
                "dpr_utility": {"mean": 0.913, "low": 0.894, "high": 0.931, "n": 900, "n_boot": 1000},
            }
        },
    })
    _write(root / "lowrank_iso_w100_coord050_dominance_error_audit.json", {
        "audit": {
            "DPR_head": 0.75,
            "DPR_head_by_objective": {name: 0.9 for name in OBJECTIVES},
            "DPR_utility_all_templates": 0.86,
            "violation_by_margin_bin": {},
            "violation_rate_by_objective": {},
        }
    })
    _write(root / "split_records_report.json", {
        "group_key": "time_block",
        "group_count": 36,
        "time_block_size": 300,
        "split_counts": {"train": 3000, "val": 600, "test": 720},
        "sample_overlap": {},
    })
    _write(root / "objective_norm_smoke3600.json", {
        "fit_input_files": ["data/pareto_records_split/jinan/smoke3600/train_raw.jsonl"],
        "zero_iqr_objectives": [],
        "valid_count": {name: 100 for name in OBJECTIVES},
        "hash": "abc123",
    })
    _write(root / "objective_sanity_v4_train.json", {
        "strict_failures": [],
        "warnings": [],
        "safety_valid_rate": 0.8,
        "objective_correlations": {},
    })
    for split in ("train", "val", "test"):
        _write(root / f"pair_report_{split}.json", _pair_report(split))


def test_representation_packet_passes_required_checks(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)

    packet = build_packet(evidence)

    assert packet["formal_representation_pass"] is True
    assert packet["formal_experiment_requires_new_pro_approval"] is True
    assert packet["split_leakage_check"]["pass"] is True
    assert packet["train_only_normalizer_check"]["pass"] is True
    assert packet["pair_coverage_check"]["pass"] is True
    assert validate_packet(packet) == []


def test_representation_packet_rejects_normalizer_eval_leakage(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    _write(evidence / "objective_norm_smoke3600.json", {
        "fit_input_files": [
            "data/pareto_records_split/jinan/smoke3600/train_raw.jsonl",
            "data/pareto_records_split/jinan/smoke3600/test_raw.jsonl",
        ],
        "zero_iqr_objectives": [],
        "valid_count": {name: 100 for name in OBJECTIVES},
    })

    packet = build_packet(evidence)

    assert packet["train_only_normalizer_check"]["pass"] is False


def test_representation_packet_flags_bootstrap_pair_count_mismatch(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    bootstrap_path = evidence / "lowrank_iso_w100_coord050_pair_bootstrap_test.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["vector"]["metrics"]["rev_acc"]["n"] = 100
    _write(bootstrap_path, bootstrap)

    packet = build_packet(evidence)

    assert packet["bootstrap_pair_count_consistency_check"]["pass"] is False
    assert packet["bootstrap_pair_count_consistency_check"]["metrics"]["rev_acc"] == {
        "observed_n": 100,
        "expected_n": 200,
        "pass": False,
    }
    assert packet["formal_representation_pass"] is False


def test_markdown_writer_names_boundary_without_formal_execution(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    packet = build_packet(evidence)
    out = tmp_path / "packet.md"

    write_markdown(packet, out)

    text = out.read_text(encoding="utf-8")
    assert "No new CityFlow PPO run" in text
    assert "formal experiment still requires new Pro approval" in text
    assert "DPR_head equals the formal threshold" in text


def test_representation_packet_supports_remediation_run_ids(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    run_id = "dpr_e40_m03_iso4_c2_u03"
    for suffix in (
        "diagnostics_test",
        "diagnostics_val",
        "metadata",
        "formal_gate_decision",
        "pair_bootstrap_test",
        "dominance_error_audit",
    ):
        src = evidence / f"lowrank_iso_w100_coord050_{suffix}.json"
        dst = evidence / f"{run_id}_{suffix}.json"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    diagnostics = _metrics()
    diagnostics["dpr_head"] = 0.8167
    _write(evidence / f"{run_id}_diagnostics_test.json", diagnostics)
    _write(evidence / f"{run_id}_pair_bootstrap_test.json", {
        "vector": {
            "metrics": {
                "rev_acc": {"mean": 0.63, "low": 0.565, "high": 0.70, "n": 200, "n_boot": 1000},
                "dpr_head": {"mean": 0.8167, "low": 0.7611, "high": 0.8724, "n": 180, "n_boot": 1000},
                "dpr_utility": {"mean": 0.95, "low": 0.94, "high": 0.96, "n": 900, "n_boot": 1000},
                "pref_acc": {"mean": 0.75, "low": 0.72, "high": 0.78, "n": 1000, "n_boot": 1000},
                "obj_acc_mean": {"mean": 0.73, "low": 0.70, "high": 0.76, "n": 1000, "n_boot": 1000},
                "head_leakage_diag_offdiag_gap": {"mean": 0.21, "low": 0.18, "high": 0.24, "n": 1000, "n_boot": 1000},
            }
        },
        "cond_scalar": {
            "metrics": {
                "rev_acc": {"mean": 0.605, "low": 0.535, "high": 0.67, "n": 200, "n_boot": 1000},
            }
        },
    })

    packet = build_packet(
        evidence,
        vector_run_id=run_id,
        approval_phrase="PARETO PPO OFFLINE REPRESENTATION DPR-HEAD MARGIN REMEDIATION GO",
    )

    assert packet["selected_vector_run"]["run_id"] == run_id
    assert packet["dpr_head_margin_status"]["bootstrap_low_pass"] is True
    assert packet["bootstrap_lower_bound_checks"]["dpr_head"]["pass"] is True
    assert validate_packet(packet) == []


def test_representation_packet_supports_city_and_paper_final_normalizer_filename(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    _write(evidence / "objective_norm_paper_final.json", {
        "fit_input_files": ["data/pareto_records_split/hangzhou/paper_final/train_raw.jsonl"],
        "zero_iqr_objectives": [],
        "valid_count": {name: 100 for name in OBJECTIVES},
        "hash": "paper-final-hash",
    })

    packet = build_packet(
        evidence,
        city="hangzhou",
        normalizer_filename="objective_norm_paper_final.json",
    )

    assert packet["scope"]["city"] == "hangzhou paper-final offline records only"
    assert packet["train_only_normalizer_check"]["normalizer_hash"] == "paper-final-hash"


def test_representation_packet_rejects_unknown_city(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)

    with pytest.raises(ValueError, match="unknown representation city"):
        build_packet(evidence, city="boston")
