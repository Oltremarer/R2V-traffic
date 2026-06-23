from __future__ import annotations

import json
from pathlib import Path

from pareto.r2v.jinan_pair_ablation_runner import (
    R2VJinanAblationConfig,
    build_command_plan,
    build_film_pilot_spec,
    command_plan_as_dicts,
    sha256_file,
)


def test_command_plan_keeps_baseline_and_r2v_differing_only_by_sampling(tmp_path: Path):
    config = R2VJinanAblationConfig(
        python_bin="/env/bin/python",
        records_root=Path("data/pareto_records_split_norm/jinan/paper_final"),
        transition_inputs=[Path("records/jinan/random/seed0/transitions_raw.jsonl")],
        normalizer_path=Path("data/normalizers/jinan/objective_norm_paper_final.json"),
        output_root=tmp_path / "run",
        epochs=3,
        device="cuda",
        repair_story="not_val_to_val",
        r2v_admitted_weight=4.0,
        r2v_repair_rejected_weight=0.25,
        run_bounded_ppo=False,
    )

    plan = command_plan_as_dicts(build_command_plan(config))
    names = [item["name"] for item in plan]

    assert "build_baseline_uniform_pairs_train" in names
    assert "build_r2v_full_r2v_pairs_train" in names
    assert "train_baseline_uniform_film_scalar" in names
    assert "train_r2v_full_r2v_film_scalar" in names

    baseline_train = next(item for item in plan if item["name"] == "build_baseline_uniform_pairs_train")
    r2v_train = next(item for item in plan if item["name"] == "build_r2v_full_r2v_pairs_train")

    assert "--r2v_sampling_mode" in baseline_train["argv"]
    assert baseline_train["argv"][baseline_train["argv"].index("--r2v_sampling_mode") + 1] == "off"
    assert "--r2v_weighted_transitions" not in baseline_train["argv"]

    assert "--r2v_sampling_mode" in r2v_train["argv"]
    assert r2v_train["argv"][r2v_train["argv"].index("--r2v_sampling_mode") + 1] == "full_r2v"
    assert "--r2v_weighted_transitions" in r2v_train["argv"]
    assert str(tmp_path / "run" / "artifacts" / "r2v_weighted_transitions.jsonl") in r2v_train["argv"]

    build_candidates = next(item for item in plan if item["name"] == "build_r2v_weighted_transitions")
    assert "--repair_story" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--repair_story") + 1] == "not_val_to_val"
    assert "--admission_mode" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--admission_mode") + 1] == "weights_only"
    assert "--admitted_weight" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--admitted_weight") + 1] == "4.0"
    assert "--admitted_weight_bonus" not in build_candidates["argv"]
    assert "--repair_rejected_weight" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--repair_rejected_weight") + 1] == "0.25"
    assert "--repair_metadata_policy" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--repair_metadata_policy") + 1] == "metadata_or_proxy"
    assert "--source_gates_key" in build_candidates["argv"]
    assert "--final_gates_key" in build_candidates["argv"]
    assert "--score_artifact" not in build_candidates["argv"]
    assert "--score_artifact_backend" not in build_candidates["argv"]

    baseline_train_cmd = next(item for item in plan if item["name"] == "train_baseline_uniform_film_scalar")
    r2v_train_cmd = next(item for item in plan if item["name"] == "train_r2v_full_r2v_film_scalar")
    for item in (baseline_train_cmd, r2v_train_cmd):
        assert "--architecture" in item["argv"]
        assert item["argv"][item["argv"].index("--architecture") + 1] == "film"
        assert "--training_schedule" in item["argv"]
        assert item["argv"][item["argv"].index("--training_schedule") + 1] == "joint"


def test_command_plan_pairs_each_er_baseline_with_r2v_overlay(tmp_path: Path):
    config = R2VJinanAblationConfig(
        python_bin="/env/bin/python",
        records_root=Path("data/pareto_records_split_norm/jinan/paper_final"),
        transition_inputs=[Path("records/jinan/random/seed0/transitions_raw.jsonl")],
        normalizer_path=Path("data/normalizers/jinan/objective_norm_paper_final.json"),
        output_root=tmp_path / "run",
        er_baseline_mode="recent",
        r2v_sampling_mode="full_r2v",
        run_bounded_ppo=False,
    )

    plan = command_plan_as_dicts(build_command_plan(config))
    names = [item["name"] for item in plan]

    assert "build_baseline_recent_pairs_train" in names
    assert "build_r2v_recent_full_r2v_pairs_train" in names
    assert "train_baseline_recent_film_scalar" in names
    assert "train_r2v_recent_full_r2v_film_scalar" in names

    baseline_train = next(item for item in plan if item["name"] == "build_baseline_recent_pairs_train")
    r2v_train = next(item for item in plan if item["name"] == "build_r2v_recent_full_r2v_pairs_train")
    for item in (baseline_train, r2v_train):
        assert "--er_baseline_mode" in item["argv"]
        assert item["argv"][item["argv"].index("--er_baseline_mode") + 1] == "recent"
        assert "--er_r2v_combine" in item["argv"]
        assert item["argv"][item["argv"].index("--er_r2v_combine") + 1] == "multiply"

    assert baseline_train["argv"][baseline_train["argv"].index("--r2v_sampling_mode") + 1] == "off"
    assert "--r2v_weighted_transitions" not in baseline_train["argv"]
    assert r2v_train["argv"][r2v_train["argv"].index("--r2v_sampling_mode") + 1] == "full_r2v"
    assert "--r2v_weighted_transitions" in r2v_train["argv"]


def test_command_plan_passes_backend_only_with_score_artifact(tmp_path: Path):
    score_artifact = tmp_path / "diffusion_scores.jsonl"
    config = R2VJinanAblationConfig(
        python_bin="/env/bin/python",
        records_root=Path("data/pareto_records_split_norm/jinan/paper_final"),
        transition_inputs=[Path("records/jinan/random/seed0/transitions_raw.jsonl")],
        normalizer_path=Path("data/normalizers/jinan/objective_norm_paper_final.json"),
        output_root=tmp_path / "run",
        r2v_artifact_path=str(score_artifact),
        run_bounded_ppo=False,
    )

    plan = command_plan_as_dicts(build_command_plan(config))
    build_candidates = next(item for item in plan if item["name"] == "build_r2v_weighted_transitions")

    assert "--score_artifact" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--score_artifact") + 1] == str(score_artifact)
    assert "--score_artifact_backend" in build_candidates["argv"]
    assert build_candidates["argv"][build_candidates["argv"].index("--score_artifact_backend") + 1] == "diffusion"


def test_command_plan_adds_bounded_ppo_commands_when_requested(tmp_path: Path):
    config = R2VJinanAblationConfig(
        python_bin="python",
        records_root=Path("data/pareto_records_split_norm/jinan/paper_final"),
        transition_inputs=[Path("records/jinan/random/seed0/transitions_raw.jsonl")],
        normalizer_path=Path("data/normalizers/jinan/objective_norm_paper_final.json"),
        output_root=tmp_path / "run",
        run_bounded_ppo=True,
        pilot_template_spec=Path("configs/formal/film_jinan_preflight.json"),
    )

    plan = command_plan_as_dicts(build_command_plan(config))
    ppo_names = [item["name"] for item in plan if item["name"].startswith("bounded_ppo_")]

    assert ppo_names == [
        "bounded_ppo_baseline_uniform",
        "bounded_ppo_r2v_full_r2v",
    ]
    for name in ppo_names:
        item = next(row for row in plan if row["name"] == name)
        assert "--bounded_jinan_pilot_dry_run" in item["argv"]
        assert "--i_understand_this_runs_bounded_jinan_pilot_dry_run" in item["argv"]
        assert "--method" in item["argv"]
        assert item["argv"][item["argv"].index("--method") + 1] == "film_scalar_potential"


def test_build_film_pilot_spec_updates_model_and_normalizer_hashes(tmp_path: Path):
    template = {
        "pilot": {
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "min_action_time": 30,
            "cityflow_seed": 0,
            "policy_seed": 0,
            "model_seed": 0,
            "state_encoder_hash": "statehash",
            "objective_norm_path": "old_norm.json",
            "objective_normalizer_hash": "old_norm_hash",
            "film_model_dir": "old_model",
            "film_model_hash": "old_model_hash",
            "pilot_spec_path": "old_spec.json",
            "formal_gate_decision_path": "old_gate.json",
        },
        "ppo": {
            "algorithm_label": "PPO",
            "requires_clipped_objective": True,
            "rollout_steps": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "update_epochs": 4,
            "minibatch_size": 128,
            "lr": 0.0003,
            "entropy_coef": 0.001,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantages": True,
        },
        "model": {
            "obs_dim": 10,
            "preference_dim": 4,
            "action_dim": 8,
            "hidden_dim": 64,
        },
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.pt"
    model_path.write_bytes(b"model bytes")
    normalizer_path = tmp_path / "norm.json"
    normalizer_path.write_text(json.dumps({"hash": "normhash"}), encoding="utf-8")
    spec_path = tmp_path / "spec.json"

    payload = build_film_pilot_spec(
        template_spec=template_path,
        output_spec=spec_path,
        model_dir=model_dir,
        normalizer_path=normalizer_path,
        formal_gate_decision_path=tmp_path / "formal_gate_decision.json",
    )

    assert spec_path.exists()
    assert payload["pilot"]["film_model_dir"] == str(model_dir)
    assert payload["pilot"]["film_model_hash"] == sha256_file(model_path)
    assert payload["pilot"]["objective_norm_path"] == str(normalizer_path)
    assert payload["pilot"]["objective_normalizer_hash"] == "normhash"
    assert payload["pilot"]["pilot_spec_path"] == str(spec_path)
