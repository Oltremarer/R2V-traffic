from __future__ import annotations

import json
import subprocess
import sys

from pareto.r2v.jinan_pair_ablation_runner import (
    build_command_plan,
    command_plan_as_dicts,
    config_from_args,
    parse_args,
)
from pareto.r2v.traffic_experiment_plan import (
    R2VTrafficExperimentSpec,
    build_ablation_commands,
    build_main_commands,
    build_main_pipeline_commands,
    build_paper_artifact_manifest_command,
    build_performance_readiness_command,
    build_result_aggregation_command,
    build_strict_paper_readiness_commands,
    build_smoke_commands,
    validate_experiment_plan,
)


def test_smoke_commands_pair_baseline_with_not_rare_to_val_full_r2v():
    spec = R2VTrafficExperimentSpec(
        python_bin="/env/bin/python",
        transition_glob="records/jinan/seed0/transitions_raw.jsonl",
        output_root="runs/smoke",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        smoke_seeds=(0,),
        main_seeds=(0, 1, 2),
    )

    commands = build_smoke_commands(spec)
    names = [command["name"] for command in commands]

    assert names == [
        "smoke_seed0_baseline",
        "smoke_seed0_r2v_diffusion_not_rare_to_val_full",
    ]
    assert commands[0]["metadata"]["r2v"] == "off"
    assert "--r2v_sampling_mode off" in commands[0]["shell"]
    assert commands[1]["metadata"]["repair_story"] == "not_rare_to_val"
    assert commands[1]["metadata"]["gate_variant"] == "full"
    assert "--generative_backend diffusion" in commands[1]["shell"]
    assert "--repair_metadata_policy metadata_or_proxy" in commands[1]["shell"]


def test_main_commands_use_three_seeds_and_keep_baseline_off():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    commands = build_main_commands(spec)

    assert len(commands) == 6
    baseline_commands = [command for command in commands if command["metadata"]["r2v"] == "off"]
    r2v_commands = [command for command in commands if command["metadata"]["r2v"] == "on"]
    assert len(baseline_commands) == 3
    assert len(r2v_commands) == 3
    assert all("--r2v_sampling_mode off" in command["shell"] for command in baseline_commands)
    assert all(command["metadata"]["generative_backend"] == "diffusion" for command in r2v_commands)
    assert all("--r2v_admission_mode weights_only" in command["shell"] for command in r2v_commands)
    assert all(command["metadata"]["r2v_admission_mode"] == "weights_only" for command in r2v_commands)


def test_strict_paper_readiness_commands_require_real_diffusion_and_repair_metadata():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    commands = build_strict_paper_readiness_commands(spec)

    assert [command["name"] for command in commands] == [
        "paper_readiness_seed0_r2v_diffusion_not_rare_to_val_full",
        "paper_readiness_seed1_r2v_diffusion_not_rare_to_val_full",
        "paper_readiness_seed2_r2v_diffusion_not_rare_to_val_full",
    ]
    for seed, command in zip((0, 1, 2), commands, strict=True):
        shell = command["shell"]
        assert "-m pareto.r2v.experiment_readiness" in shell
        assert "--require_diffusion_artifacts" in shell
        assert "--require_paper_claim_eligible_diffusion" in shell
        assert "--repair_metadata_policy require_metadata" in shell
        assert "--require_strict_repair_metadata_policy" in shell
        assert f"--diffusion_artifact {seed}:records/r2v_traffic/diffusion_seed{seed}_scores.jsonl" in shell
        assert command["metadata"]["repair_metadata_policy"] == "require_metadata"
        assert command["metadata"]["paper_claim_eligible_diffusion_required"] is True


def test_result_aggregation_command_uses_performance_rows_and_seed_integrity_artifacts():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    command = build_result_aggregation_command(spec)

    assert command["name"] == "aggregate_main_r2v_traffic_results"
    assert "-m pareto.r2v.result_aggregation" in command["shell"]
    assert "--performance_path runs/main/aggregation/r2v_performance_rows.jsonl" in command["shell"]
    for seed in (0, 1, 2):
        assert f"--integrity_path runs/main/main/seed{seed}/r2v/artifacts/r2v_summary.json" in command["shell"]
    assert "--output runs/main/aggregation/r2v_result_aggregation.json" in command["shell"]
    assert command["metadata"]["performance_path"] == "runs/main/aggregation/r2v_performance_rows.jsonl"
    assert command["metadata"]["integrity_paths"] == [
        "runs/main/main/seed0/r2v/artifacts/r2v_summary.json",
        "runs/main/main/seed1/r2v/artifacts/r2v_summary.json",
        "runs/main/main/seed2/r2v/artifacts/r2v_summary.json",
    ]


def test_paper_artifact_manifest_command_freezes_main_evidence_bundle():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    command = build_paper_artifact_manifest_command(spec)

    assert command["name"] == "build_main_paper_artifact_manifest"
    assert "-m pareto.r2v.paper_artifact_manifest" in command["shell"]
    assert "--artifact performance:main_performance_rows:runs/main/aggregation/r2v_performance_rows.jsonl" in command["shell"]
    assert "--artifact aggregation:main_result_aggregation:runs/main/aggregation/r2v_result_aggregation.json" in command["shell"]
    assert "--artifact readiness:main_performance_readiness:runs/main/aggregation/performance_readiness.json" in command["shell"]
    for seed in (0, 1, 2):
        assert f"--artifact readiness:paper_readiness_seed{seed}:runs/main/main/seed{seed}/r2v/r2v_paper_readiness.json" in command["shell"]
        assert f"--artifact integrity:seed{seed}_r2v_summary:runs/main/main/seed{seed}/r2v/artifacts/r2v_summary.json" in command["shell"]
        assert f"--artifact weighted_transitions:seed{seed}_r2v_weighted_transitions:runs/main/main/seed{seed}/r2v/artifacts/r2v_weighted_transitions.jsonl" in command["shell"]
        assert f"--artifact diffusion_score:seed{seed}_diffusion_scores:records/r2v_traffic/diffusion_seed{seed}_scores.jsonl" in command["shell"]
    assert "--output runs/main/aggregation/paper_artifact_manifest.json" in command["shell"]
    assert command["metadata"]["artifact_type_counts"] == {
        "aggregation": 1,
        "diffusion_score": 3,
        "integrity": 3,
        "performance": 1,
        "readiness": 4,
        "weighted_transitions": 3,
    }
    assert command["metadata"]["output"] == "runs/main/aggregation/paper_artifact_manifest.json"


def test_performance_readiness_command_requires_complete_baseline_r2v_seed_grid():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    command = build_performance_readiness_command(spec)

    assert command["name"] == "check_main_performance_readiness"
    assert "-m pareto.r2v.experiment_readiness" in command["shell"]
    assert "--no-require_cityflow_data" in command["shell"]
    assert "--performance_path runs/main/aggregation/r2v_performance_rows.jsonl" in command["shell"]
    assert "--require_performance_metrics" in command["shell"]
    assert "--expected_performance_method baseline_uniform" in command["shell"]
    assert "--expected_performance_method r2v_diffusion_not_rare_to_val_full" in command["shell"]
    for seed in (0, 1, 2):
        assert f"--expected_performance_seed {seed}" in command["shell"]
    assert "--require_completed_performance_status" in command["shell"]
    assert "--output runs/main/aggregation/performance_readiness.json" in command["shell"]
    assert command["metadata"]["expected_methods"] == [
        "baseline_uniform",
        "r2v_diffusion_not_rare_to_val_full",
    ]
    assert command["metadata"]["expected_seeds"] == [0, 1, 2]


def test_main_pipeline_commands_order_preflight_runner_readiness_and_aggregation():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    commands = build_main_pipeline_commands(spec)
    names = [command["name"] for command in commands]

    assert len(commands) == 12
    assert names[:3] == [
        "paper_readiness_seed0_r2v_diffusion_not_rare_to_val_full",
        "paper_readiness_seed1_r2v_diffusion_not_rare_to_val_full",
        "paper_readiness_seed2_r2v_diffusion_not_rare_to_val_full",
    ]
    assert names[3:9] == [
        "main_seed0_baseline",
        "main_seed0_r2v_diffusion_not_rare_to_val_full",
        "main_seed1_baseline",
        "main_seed1_r2v_diffusion_not_rare_to_val_full",
        "main_seed2_baseline",
        "main_seed2_r2v_diffusion_not_rare_to_val_full",
    ]
    assert names[-3:] == [
        "check_main_performance_readiness",
        "aggregate_main_r2v_traffic_results",
        "build_main_paper_artifact_manifest",
    ]
    assert commands[0]["metadata"]["label"] == "paper_readiness"
    assert commands[3]["metadata"]["r2v"] == "off"
    assert commands[4]["metadata"]["r2v"] == "on"
    assert commands[-3]["metadata"]["label"] == "performance_readiness"
    assert commands[-2]["metadata"]["label"] == "result_aggregation"
    assert commands[-1]["metadata"]["label"] == "paper_artifact_manifest"


def test_main_pipeline_r2v_runner_uses_strict_repair_metadata_policy():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )

    commands = build_main_pipeline_commands(spec)
    r2v_commands = [
        command
        for command in commands
        if command["name"].startswith("main_seed")
        and "_r2v_diffusion_not_rare_to_val_full" in command["name"]
    ]

    assert len(r2v_commands) == 3
    for seed, command in zip((0, 1, 2), r2v_commands, strict=True):
        expected_artifact = f"records/r2v_traffic/diffusion_seed{seed}_scores.jsonl"
        assert "--repair_metadata_policy require_metadata" in command["shell"]
        assert "--r2v_admission_mode weights_only" in command["shell"]
        assert f"--r2v_artifact_path {expected_artifact}" in command["shell"]
        assert command["metadata"]["repair_metadata_policy"] == "require_metadata"
        assert command["metadata"]["r2v_admission_mode"] == "weights_only"
        assert command["metadata"]["r2v_artifact_path"] == expected_artifact


def test_experiment_plan_cli_can_export_weights_plus_repaired_interface(tmp_path):
    output_path = tmp_path / "ablation_weights_plus_repaired.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.traffic_experiment_plan",
            "--plan",
            "ablation",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--python_bin",
            "python3",
            "--output_root",
            "runs/ablation",
            "--r2v_admission_mode",
            "weights_plus_repaired",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    r2v_commands = [command for command in payload["commands"] if command["metadata"]["r2v"] == "on"]
    assert r2v_commands
    assert all("--r2v_admission_mode weights_plus_repaired" in command["shell"] for command in r2v_commands)
    assert all(command["metadata"]["r2v_admission_mode"] == "weights_plus_repaired" for command in r2v_commands)


def test_validate_main_pipeline_checks_order_flags_and_manifest_artifacts():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )
    commands = build_main_pipeline_commands(spec)

    report = validate_experiment_plan("main_pipeline", commands, spec)

    assert report["schema_version"] == "r2v-traffic-experiment-plan-validation-v1"
    assert report["status"] == "READY"
    assert report["failed_count"] == 0
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["main_pipeline_order"]["status"] == "pass"
    assert checks["baseline_commands_disable_r2v"]["status"] == "pass"
    assert checks["r2v_commands_use_main_diffusion_config"]["status"] == "pass"
    assert checks["r2v_commands_use_strict_repair_metadata_policy"]["status"] == "pass"
    assert checks["r2v_commands_use_seed_diffusion_artifacts"]["status"] == "pass"
    assert checks["paper_artifact_manifest_coverage"]["status"] == "pass"


def test_validate_main_pipeline_blocks_missing_manifest_step():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        diffusion_artifact_template="records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
        output_root="runs/main",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        main_seeds=(0, 1, 2),
    )
    commands = build_main_pipeline_commands(spec)[:-1]

    report = validate_experiment_plan("main_pipeline", commands, spec)

    assert report["status"] == "BLOCKED"
    failure = next(check for check in report["failed_checks"] if check["name"] == "main_pipeline_order")
    assert failure["status"] == "fail"
    assert "build_main_paper_artifact_manifest" in failure["missing_names"]


def test_experiment_plan_cli_writes_main_pipeline_json(tmp_path):
    output_path = tmp_path / "main_pipeline.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.traffic_experiment_plan",
            "--plan",
            "main_pipeline",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--python_bin",
            "python3",
            "--transition_glob",
            "records/jinan/seed{seed}/transitions_raw.jsonl",
            "--diffusion_artifact_template",
            "records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
            "--output_root",
            "runs/main",
            "--main_seeds",
            "0,1,2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    names = [command["name"] for command in payload["commands"]]

    assert payload["plan"] == "main_pipeline"
    assert len(payload["commands"]) == 12
    assert names[0] == "paper_readiness_seed0_r2v_diffusion_not_rare_to_val_full"
    assert names[-3:] == [
        "check_main_performance_readiness",
        "aggregate_main_r2v_traffic_results",
        "build_main_paper_artifact_manifest",
    ]


def test_experiment_plan_cli_can_include_validation_report(tmp_path):
    output_path = tmp_path / "main_pipeline_with_validation.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.traffic_experiment_plan",
            "--plan",
            "main_pipeline",
            "--format",
            "json",
            "--include_validation",
            "--output",
            str(output_path),
            "--python_bin",
            "python3",
            "--transition_glob",
            "records/jinan/seed{seed}/transitions_raw.jsonl",
            "--diffusion_artifact_template",
            "records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
            "--output_root",
            "runs/main",
            "--main_seeds",
            "0,1,2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())

    assert payload["validation"]["status"] == "READY"
    assert payload["validation"]["failed_count"] == 0
    assert payload["validation"]["checks"][-1]["name"] == "paper_artifact_manifest_coverage"


def test_experiment_plan_cli_writes_paper_artifact_manifest_plan_json(tmp_path):
    output_path = tmp_path / "paper_manifest_plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.traffic_experiment_plan",
            "--plan",
            "paper_artifact_manifest",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--python_bin",
            "python3",
            "--diffusion_artifact_template",
            "records/r2v_traffic/diffusion_seed{seed}_scores.jsonl",
            "--output_root",
            "runs/main",
            "--main_seeds",
            "0,1,2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())

    assert payload["plan"] == "paper_artifact_manifest"
    assert [command["name"] for command in payload["commands"]] == [
        "build_main_paper_artifact_manifest"
    ]
    assert payload["commands"][0]["metadata"]["artifact_type_counts"]["weighted_transitions"] == 3


def test_experiment_plan_cli_writes_shell_script(tmp_path):
    output_path = tmp_path / "smoke.sh"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.traffic_experiment_plan",
            "--plan",
            "smoke",
            "--format",
            "shell",
            "--output",
            str(output_path),
            "--python_bin",
            "python3",
            "--transition_glob",
            "records/jinan/seed{seed}/transitions_raw.jsonl",
            "--output_root",
            "runs/smoke",
            "--smoke_seeds",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    script = output_path.read_text()

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "# smoke_seed0_baseline" in script
    assert "# smoke_seed0_r2v_diffusion_not_rare_to_val_full" in script
    assert "--r2v_sampling_mode off" in script
    assert "--r2v_sampling_mode full_r2v" in script


def test_ablation_commands_cover_required_gate_story_and_sampling_modes():
    spec = R2VTrafficExperimentSpec(
        python_bin="python3",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        output_root="runs/ablation",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        smoke_seeds=(0,),
    )

    commands = build_ablation_commands(spec, seed=0)
    labels = {command["metadata"]["label"] for command in commands}

    assert {
        "gate_full",
        "gate_no_support",
        "gate_no_ood",
        "gate_no_dynamics",
        "story_not_rare_to_val",
        "story_not_val_to_val",
        "sampling_admitted_only",
        "sampling_random_same_count",
        "sampling_shuffled_value",
        "sampling_inverted_rarity",
    } <= labels
    assert all(command["metadata"]["r2v"] == "on" for command in commands)


def test_generated_smoke_commands_parse_against_jinan_runner(monkeypatch):
    spec = R2VTrafficExperimentSpec(
        python_bin=sys.executable,
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        output_root="runs/smoke",
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        smoke_seeds=(0,),
    )

    for command in build_smoke_commands(spec):
        argv = command["argv"]
        module_idx = argv.index("pareto.r2v.jinan_pair_ablation_runner")
        monkeypatch.setattr(sys, "argv", ["jinan_pair_ablation_runner.py", *argv[module_idx + 1 :], "--dry_run"])
        args = parse_args()
        config = config_from_args(args)
        plan = command_plan_as_dicts(build_command_plan(config))

        assert config.r2v == command["metadata"]["r2v"]
        if command["metadata"]["r2v"] == "off":
            assert all(item["name"] != "build_r2v_weighted_transitions" for item in plan)
            assert {item["argv"][item["argv"].index("--r2v_sampling_mode") + 1] for item in plan if "--r2v_sampling_mode" in item["argv"]} == {"off"}
        else:
            assert any(item["name"] == "build_r2v_weighted_transitions" for item in plan)
            assert config.gate_variant == "full"
            assert config.repair_story == "not_rare_to_val"
