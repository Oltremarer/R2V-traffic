from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import shlex
from pathlib import Path
import sys
from typing import Any

from pareto.r2v.traffic_artifact_schema import R2VTrafficConfig


MAIN_PERFORMANCE_METHODS = ("baseline_uniform", "r2v_diffusion_not_rare_to_val_full")


@dataclass(frozen=True)
class R2VTrafficExperimentSpec:
    python_bin: str = "python3"
    transition_glob: str = "records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed{seed}/transitions_raw.jsonl"
    diffusion_artifact_template: str = "records/r2v_traffic/diffusion_seed{seed}_scores.jsonl"
    output_root: str = "records/r2v_traffic_runs"
    scenario: str = "jinan"
    traffic_file: str = "anon_3_4_jinan_real.json"
    smoke_seeds: tuple[int, ...] = (0,)
    main_seeds: tuple[int, ...] = (0, 1, 2)
    r2v_admitted_weight: float = 2.0
    r2v_repair_rejected_weight: float = 2.0
    r2v_admission_mode: str = "weights_only"
    rare_fraction: float = 0.2
    base_command: tuple[str, ...] = field(default_factory=lambda: ("-m", "pareto.r2v.jinan_pair_ablation_runner"))


def build_smoke_commands(spec: R2VTrafficExperimentSpec) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for seed in spec.smoke_seeds:
        commands.append(_command(spec, seed=seed, label="smoke", r2v_config=R2VTrafficConfig(r2v="off")))
        commands.append(
            _command(
                spec,
                seed=seed,
                label="smoke",
                r2v_config=R2VTrafficConfig(
                    r2v="on",
                    repair_story="not_rare_to_val",
                    gate_variant="full",
                    r2v_sampling_mode="full_r2v",
                    r2v_admitted_weight=spec.r2v_admitted_weight,
                    r2v_repair_rejected_weight=spec.r2v_repair_rejected_weight,
                    r2v_admission_mode=spec.r2v_admission_mode,
                    rare_fraction=spec.rare_fraction,
                ),
            )
        )
    return commands


def build_main_commands(
    spec: R2VTrafficExperimentSpec,
    *,
    repair_metadata_policy: str = "metadata_or_proxy",
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for seed in spec.main_seeds:
        commands.append(_command(spec, seed=seed, label="main", r2v_config=R2VTrafficConfig(r2v="off")))
        commands.append(
            _command(
                spec,
                seed=seed,
                label="main",
                r2v_config=R2VTrafficConfig(
                    r2v="on",
                    repair_story="not_rare_to_val",
                    repair_metadata_policy=repair_metadata_policy,
                    gate_variant="full",
                    r2v_sampling_mode="full_r2v",
                    r2v_artifact_path=spec.diffusion_artifact_template.format(seed=seed),
                    r2v_admitted_weight=spec.r2v_admitted_weight,
                    r2v_repair_rejected_weight=spec.r2v_repair_rejected_weight,
                    r2v_admission_mode=spec.r2v_admission_mode,
                    rare_fraction=spec.rare_fraction,
                ),
            )
        )
    return commands


def build_main_pipeline_commands(spec: R2VTrafficExperimentSpec) -> list[dict[str, Any]]:
    return [
        *build_strict_paper_readiness_commands(spec),
        *build_main_commands(spec, repair_metadata_policy="require_metadata"),
        build_performance_readiness_command(spec),
        build_result_aggregation_command(spec),
        build_paper_artifact_manifest_command(spec),
    ]


def build_strict_paper_readiness_commands(spec: R2VTrafficExperimentSpec) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for seed in spec.main_seeds:
        diffusion_artifact = spec.diffusion_artifact_template.format(seed=seed)
        output_path = (
            Path(spec.output_root)
            / "main"
            / f"seed{seed}"
            / "r2v"
            / "r2v_paper_readiness.json"
        )
        argv = [
            spec.python_bin,
            "-m",
            "pareto.r2v.experiment_readiness",
            "--scenario",
            spec.scenario,
            "--traffic_file",
            spec.traffic_file,
            "--transition_glob",
            spec.transition_glob.format(seed=seed),
            "--seed",
            str(seed),
            "--require_diffusion_artifacts",
            "--require_paper_claim_eligible_diffusion",
            "--repair_metadata_policy",
            "require_metadata",
            "--require_strict_repair_metadata_policy",
            "--diffusion_artifact",
            f"{seed}:{diffusion_artifact}",
            "--output",
            str(output_path),
        ]
        commands.append(
            {
                "name": f"paper_readiness_seed{seed}_r2v_diffusion_not_rare_to_val_full",
                "argv": argv,
                "shell": " ".join(shlex.quote(str(part)) for part in argv),
                "output_root": str(output_path.parent),
                "wandb": None,
                "metadata": {
                    "label": "paper_readiness",
                    "scenario": spec.scenario,
                    "traffic_file": spec.traffic_file,
                    "seed": seed,
                    "r2v": "on",
                    "generative_backend": "diffusion",
                    "repair_story": "not_rare_to_val",
                    "repair_metadata_policy": "require_metadata",
                    "gate_variant": "full",
                    "r2v_sampling_mode": "full_r2v",
                    "diffusion_artifact": diffusion_artifact,
                    "paper_claim_eligible_diffusion_required": True,
                    "strict_repair_metadata_policy_required": True,
                },
            }
        )
    return commands


def build_result_aggregation_command(spec: R2VTrafficExperimentSpec) -> dict[str, Any]:
    performance_path = Path(spec.output_root) / "aggregation" / "r2v_performance_rows.jsonl"
    integrity_paths = [
        Path(spec.output_root) / "main" / f"seed{seed}" / "r2v" / "artifacts" / "r2v_summary.json"
        for seed in spec.main_seeds
    ]
    output_path = Path(spec.output_root) / "aggregation" / "r2v_result_aggregation.json"
    argv = [
        spec.python_bin,
        "-m",
        "pareto.r2v.result_aggregation",
        "--performance_path",
        str(performance_path),
    ]
    for path in integrity_paths:
        argv.extend(["--integrity_path", str(path)])
    argv.extend(["--output", str(output_path)])
    return {
        "name": "aggregate_main_r2v_traffic_results",
        "argv": argv,
        "shell": " ".join(shlex.quote(str(part)) for part in argv),
        "output_root": str(output_path.parent),
        "wandb": None,
        "metadata": {
            "label": "result_aggregation",
            "scenario": spec.scenario,
            "traffic_file": spec.traffic_file,
            "performance_path": str(performance_path),
            "integrity_paths": [str(path) for path in integrity_paths],
            "output": str(output_path),
        },
    }


def build_performance_readiness_command(spec: R2VTrafficExperimentSpec) -> dict[str, Any]:
    performance_path = Path(spec.output_root) / "aggregation" / "r2v_performance_rows.jsonl"
    output_path = Path(spec.output_root) / "aggregation" / "performance_readiness.json"
    argv = [
        spec.python_bin,
        "-m",
        "pareto.r2v.experiment_readiness",
        "--no-require_cityflow_data",
        "--performance_path",
        str(performance_path),
        "--require_performance_metrics",
    ]
    for method in MAIN_PERFORMANCE_METHODS:
        argv.extend(["--expected_performance_method", method])
    for seed in spec.main_seeds:
        argv.extend(["--expected_performance_seed", str(seed)])
    argv.extend(
        [
            "--require_completed_performance_status",
            "--output",
            str(output_path),
        ]
    )
    return {
        "name": "check_main_performance_readiness",
        "argv": argv,
        "shell": " ".join(shlex.quote(str(part)) for part in argv),
        "output_root": str(output_path.parent),
        "wandb": None,
        "metadata": {
            "label": "performance_readiness",
            "scenario": spec.scenario,
            "traffic_file": spec.traffic_file,
            "performance_path": str(performance_path),
            "expected_methods": list(MAIN_PERFORMANCE_METHODS),
            "expected_seeds": list(spec.main_seeds),
            "require_completed_performance_status": True,
            "output": str(output_path),
        },
    }


def build_paper_artifact_manifest_command(spec: R2VTrafficExperimentSpec) -> dict[str, Any]:
    aggregation_dir = Path(spec.output_root) / "aggregation"
    output_path = aggregation_dir / "paper_artifact_manifest.json"
    artifact_specs = [
        ("performance", "main_performance_rows", aggregation_dir / "r2v_performance_rows.jsonl"),
        ("aggregation", "main_result_aggregation", aggregation_dir / "r2v_result_aggregation.json"),
        ("readiness", "main_performance_readiness", aggregation_dir / "performance_readiness.json"),
    ]
    for seed in spec.main_seeds:
        r2v_output_root = Path(spec.output_root) / "main" / f"seed{seed}" / "r2v"
        artifact_specs.extend(
            [
                ("readiness", f"paper_readiness_seed{seed}", r2v_output_root / "r2v_paper_readiness.json"),
                ("integrity", f"seed{seed}_r2v_summary", r2v_output_root / "artifacts" / "r2v_summary.json"),
                (
                    "weighted_transitions",
                    f"seed{seed}_r2v_weighted_transitions",
                    r2v_output_root / "artifacts" / "r2v_weighted_transitions.jsonl",
                ),
                ("diffusion_score", f"seed{seed}_diffusion_scores", spec.diffusion_artifact_template.format(seed=seed)),
            ]
        )
    argv = [
        spec.python_bin,
        "-m",
        "pareto.r2v.paper_artifact_manifest",
    ]
    for artifact_type, name, path in artifact_specs:
        argv.extend(["--artifact", f"{artifact_type}:{name}:{path}"])
    argv.extend(["--output", str(output_path)])
    artifact_type_counts: dict[str, int] = {}
    for artifact_type, _name, _path in artifact_specs:
        artifact_type_counts[artifact_type] = artifact_type_counts.get(artifact_type, 0) + 1
    return {
        "name": "build_main_paper_artifact_manifest",
        "argv": argv,
        "shell": " ".join(shlex.quote(str(part)) for part in argv),
        "output_root": str(output_path.parent),
        "wandb": None,
        "metadata": {
            "label": "paper_artifact_manifest",
            "scenario": spec.scenario,
            "traffic_file": spec.traffic_file,
            "artifact_type_counts": dict(sorted(artifact_type_counts.items())),
            "artifacts": [
                {"artifact_type": artifact_type, "name": name, "path": str(path)}
                for artifact_type, name, path in artifact_specs
            ],
            "output": str(output_path),
            "claim_boundary": "paper artifact manifest records evidence hashes and keeps performance separate from integrity/status",
        },
    }


def build_ablation_commands(spec: R2VTrafficExperimentSpec, *, seed: int = 0) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for gate_variant in ("full", "no_support", "no_ood", "no_dynamics"):
        commands.append(
            _command(
                spec,
                seed=seed,
                label=f"gate_{gate_variant}",
                r2v_config=R2VTrafficConfig(
                    r2v="on",
                    repair_story="not_rare_to_val",
                    gate_variant=gate_variant,
                    r2v_sampling_mode="full_r2v",
                    r2v_admission_mode=spec.r2v_admission_mode,
                ),
            )
        )
    for repair_story in ("not_rare_to_val", "not_val_to_val"):
        commands.append(
            _command(
                spec,
                seed=seed,
                label=f"story_{repair_story}",
                r2v_config=R2VTrafficConfig(
                    r2v="on",
                    repair_story=repair_story,
                    gate_variant="full",
                    r2v_sampling_mode="full_r2v",
                    r2v_admission_mode=spec.r2v_admission_mode,
                ),
            )
        )
    for sampling_mode in ("admitted_only", "random_same_count", "shuffled_value", "inverted_rarity"):
        commands.append(
            _command(
                spec,
                seed=seed,
                label=f"sampling_{sampling_mode}",
                r2v_config=R2VTrafficConfig(
                    r2v="on",
                    repair_story="not_rare_to_val",
                    gate_variant="full",
                    r2v_sampling_mode=sampling_mode,
                    r2v_admission_mode=spec.r2v_admission_mode,
                ),
            )
        )
    return commands


def build_plan_commands(
    plan: str,
    spec: R2VTrafficExperimentSpec,
    *,
    ablation_seed: int = 0,
) -> list[dict[str, Any]]:
    if plan == "smoke":
        return build_smoke_commands(spec)
    if plan == "main":
        return build_main_commands(spec)
    if plan == "main_pipeline":
        return build_main_pipeline_commands(spec)
    if plan == "strict_paper_readiness":
        return build_strict_paper_readiness_commands(spec)
    if plan == "performance_readiness":
        return [build_performance_readiness_command(spec)]
    if plan == "result_aggregation":
        return [build_result_aggregation_command(spec)]
    if plan == "paper_artifact_manifest":
        return [build_paper_artifact_manifest_command(spec)]
    if plan == "ablation":
        return build_ablation_commands(spec, seed=ablation_seed)
    raise ValueError(f"unsupported plan: {plan}")


def validate_experiment_plan(
    plan: str,
    commands: list[dict[str, Any]],
    spec: R2VTrafficExperimentSpec,
) -> dict[str, Any]:
    checks: list[dict[str, Any]]
    if plan == "main_pipeline":
        checks = [
            _main_pipeline_order_check(commands, spec),
            _baseline_commands_disable_r2v_check(commands),
            _r2v_commands_use_main_diffusion_config_check(commands),
            _r2v_commands_use_strict_repair_metadata_policy_check(commands),
            _r2v_commands_use_seed_diffusion_artifacts_check(commands, spec),
            _paper_artifact_manifest_coverage_check(commands, spec),
        ]
    else:
        checks = [
            {
                "name": "plan_has_commands",
                "status": "pass" if commands else "fail",
                "command_count": len(commands),
                "message": "plan has at least one command" if commands else "plan has no commands",
            }
        ]
    failed = [check for check in checks if check["status"] != "pass"]
    return {
        "schema_version": "r2v-traffic-experiment-plan-validation-v1",
        "plan": plan,
        "status": "BLOCKED" if failed else "READY",
        "command_count": len(commands),
        "failed_count": len(failed),
        "checks": checks,
        "failed_checks": failed,
    }


def render_plan_json(
    plan: str,
    spec: R2VTrafficExperimentSpec,
    commands: list[dict[str, Any]],
    *,
    include_validation: bool = False,
) -> str:
    payload = {
        "plan": plan,
        "scenario": spec.scenario,
        "traffic_file": spec.traffic_file,
        "output_root": spec.output_root,
        "commands": commands,
    }
    if include_validation:
        payload["validation"] = validate_experiment_plan(plan, commands, spec)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_plan_shell(commands: list[dict[str, Any]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for command in commands:
        lines.append(f"# {command['name']}")
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export R2V-Traffic experiment command plans.")
    parser.add_argument(
        "--plan",
        choices=(
            "smoke",
            "main",
            "main_pipeline",
            "strict_paper_readiness",
            "performance_readiness",
            "result_aggregation",
            "paper_artifact_manifest",
            "ablation",
        ),
        required=True,
    )
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--include_validation", action="store_true", help="Include a validation report in JSON output.")
    parser.add_argument("--output", help="Output file. Defaults to stdout when omitted.")
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument(
        "--transition_glob",
        default=R2VTrafficExperimentSpec.transition_glob,
        help="Transition input pattern. May include {seed}.",
    )
    parser.add_argument(
        "--diffusion_artifact_template",
        default=R2VTrafficExperimentSpec.diffusion_artifact_template,
        help="Diffusion score artifact pattern. May include {seed}.",
    )
    parser.add_argument("--output_root", default=R2VTrafficExperimentSpec.output_root)
    parser.add_argument("--scenario", default=R2VTrafficExperimentSpec.scenario)
    parser.add_argument("--traffic_file", default=R2VTrafficExperimentSpec.traffic_file)
    parser.add_argument("--smoke_seeds", default="0", help="Comma-separated smoke seeds.")
    parser.add_argument("--main_seeds", default="0,1,2", help="Comma-separated main seeds.")
    parser.add_argument("--ablation_seed", type=int, default=0)
    parser.add_argument("--r2v_admitted_weight", type=float, default=2.0)
    parser.add_argument("--r2v_repair_rejected_weight", type=float, default=2.0)
    parser.add_argument("--r2v_admission_mode", choices=("weights_only", "weights_plus_repaired"), default="weights_only")
    parser.add_argument("--rare_fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    spec = R2VTrafficExperimentSpec(
        python_bin=args.python_bin,
        transition_glob=args.transition_glob,
        diffusion_artifact_template=args.diffusion_artifact_template,
        output_root=args.output_root,
        scenario=args.scenario,
        traffic_file=args.traffic_file,
        smoke_seeds=_parse_seed_list(args.smoke_seeds),
        main_seeds=_parse_seed_list(args.main_seeds),
        r2v_admitted_weight=args.r2v_admitted_weight,
        r2v_repair_rejected_weight=args.r2v_repair_rejected_weight,
        r2v_admission_mode=args.r2v_admission_mode,
        rare_fraction=args.rare_fraction,
    )
    commands = build_plan_commands(args.plan, spec, ablation_seed=args.ablation_seed)
    if args.format == "json":
        output = render_plan_json(args.plan, spec, commands, include_validation=args.include_validation)
    else:
        output = render_plan_shell(commands)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
    else:
        sys.stdout.write(output)
    return 0


def _parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _main_pipeline_order_check(commands: list[dict[str, Any]], spec: R2VTrafficExperimentSpec) -> dict[str, Any]:
    expected_names = [command["name"] for command in build_main_pipeline_commands(spec)]
    observed_names = [command["name"] for command in commands]
    ok = observed_names == expected_names
    return {
        "name": "main_pipeline_order",
        "status": "pass" if ok else "fail",
        "expected_names": expected_names,
        "observed_names": observed_names,
        "missing_names": [name for name in expected_names if name not in observed_names],
        "extra_names": [name for name in observed_names if name not in expected_names],
        "message": "main pipeline command order matches expected contract" if ok else "main pipeline command order changed",
    }


def _baseline_commands_disable_r2v_check(commands: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_commands = [command for command in commands if command["name"].startswith("main_seed") and command["name"].endswith("_baseline")]
    failures = [
        command["name"]
        for command in baseline_commands
        if command["metadata"].get("r2v") != "off" or "--r2v_sampling_mode off" not in command["shell"]
    ]
    ok = bool(baseline_commands) and not failures
    return {
        "name": "baseline_commands_disable_r2v",
        "status": "pass" if ok else "fail",
        "checked_count": len(baseline_commands),
        "failed_command_names": failures,
        "message": "baseline commands keep R2V disabled" if ok else "baseline command R2V-off contract failed",
    }


def _r2v_commands_use_main_diffusion_config_check(commands: list[dict[str, Any]]) -> dict[str, Any]:
    r2v_commands = [
        command
        for command in commands
        if command["name"].startswith("main_seed") and "_r2v_diffusion_not_rare_to_val_full" in command["name"]
    ]
    failures = [
        command["name"]
        for command in r2v_commands
        if command["metadata"].get("r2v") != "on"
        or command["metadata"].get("generative_backend") != "diffusion"
        or command["metadata"].get("repair_story") != "not_rare_to_val"
        or command["metadata"].get("gate_variant") != "full"
        or command["metadata"].get("r2v_sampling_mode") != "full_r2v"
    ]
    ok = bool(r2v_commands) and not failures
    return {
        "name": "r2v_commands_use_main_diffusion_config",
        "status": "pass" if ok else "fail",
        "checked_count": len(r2v_commands),
        "failed_command_names": failures,
        "message": "R2V main commands use diffusion/not_rare_to_val/full" if ok else "R2V main command contract failed",
    }


def _r2v_commands_use_strict_repair_metadata_policy_check(commands: list[dict[str, Any]]) -> dict[str, Any]:
    r2v_commands = [
        command
        for command in commands
        if command["name"].startswith("main_seed") and "_r2v_diffusion_not_rare_to_val_full" in command["name"]
    ]
    failures = [
        command["name"]
        for command in r2v_commands
        if command["metadata"].get("repair_metadata_policy") != "require_metadata"
        or "--repair_metadata_policy require_metadata" not in command["shell"]
    ]
    ok = bool(r2v_commands) and not failures
    return {
        "name": "r2v_commands_use_strict_repair_metadata_policy",
        "status": "pass" if ok else "fail",
        "checked_count": len(r2v_commands),
        "failed_command_names": failures,
        "message": (
            "R2V main pipeline commands require repaired-source/final gate metadata"
            if ok
            else "R2V main pipeline commands allow proxy repair metadata"
        ),
    }


def _r2v_commands_use_seed_diffusion_artifacts_check(
    commands: list[dict[str, Any]],
    spec: R2VTrafficExperimentSpec,
) -> dict[str, Any]:
    r2v_commands = [
        command
        for command in commands
        if command["name"].startswith("main_seed") and "_r2v_diffusion_not_rare_to_val_full" in command["name"]
    ]
    failures = []
    for command in r2v_commands:
        seed = int(command["metadata"].get("seed"))
        expected_path = spec.diffusion_artifact_template.format(seed=seed)
        if (
            command["metadata"].get("r2v_artifact_path") != expected_path
            or f"--r2v_artifact_path {expected_path}" not in command["shell"]
        ):
            failures.append(command["name"])
    ok = bool(r2v_commands) and not failures
    return {
        "name": "r2v_commands_use_seed_diffusion_artifacts",
        "status": "pass" if ok else "fail",
        "checked_count": len(r2v_commands),
        "failed_command_names": failures,
        "message": (
            "R2V main commands consume the per-seed diffusion artifacts"
            if ok
            else "R2V main commands are not wired to per-seed diffusion artifacts"
        ),
    }


def _paper_artifact_manifest_coverage_check(commands: list[dict[str, Any]], spec: R2VTrafficExperimentSpec) -> dict[str, Any]:
    expected_counts = build_paper_artifact_manifest_command(spec)["metadata"]["artifact_type_counts"]
    manifest_commands = [command for command in commands if command["name"] == "build_main_paper_artifact_manifest"]
    observed_counts = manifest_commands[-1]["metadata"].get("artifact_type_counts", {}) if manifest_commands else {}
    ok = observed_counts == expected_counts
    return {
        "name": "paper_artifact_manifest_coverage",
        "status": "pass" if ok else "fail",
        "expected_artifact_type_counts": expected_counts,
        "observed_artifact_type_counts": observed_counts,
        "message": "paper artifact manifest covers expected evidence artifact types" if ok else "paper artifact manifest coverage changed",
    }


def _command(
    spec: R2VTrafficExperimentSpec,
    *,
    seed: int,
    label: str,
    r2v_config: R2VTrafficConfig,
) -> dict[str, Any]:
    if not r2v_config.r2v_enabled:
        name = f"{label}_seed{seed}_baseline"
        metadata_label = "baseline"
    else:
        name = (
            f"{label}_seed{seed}_r2v_"
            f"{r2v_config.generative_backend}_{r2v_config.repair_story}_{r2v_config.gate_variant}"
        )
        metadata_label = label
    output_dir = Path(spec.output_root) / label / f"seed{seed}" / ("r2v" if r2v_config.r2v_enabled else "baseline")
    transition_input = spec.transition_glob.format(seed=seed)
    argv = [
        spec.python_bin,
        *spec.base_command,
        "--seed",
        str(seed),
        "--transition_input",
        transition_input,
        "--output_root",
        str(output_dir),
        *r2v_config.to_cli_flags(),
    ]
    return {
        "name": name,
        "argv": argv,
        "shell": " ".join(shlex.quote(str(part)) for part in argv),
        "output_root": str(output_dir),
        "wandb": {
            "project": "R2V-Traffic",
            "group": f"{spec.scenario}-{label}",
            "name": f"{name}-{spec.traffic_file}",
        },
        "metadata": {
            "label": metadata_label,
            "scenario": spec.scenario,
            "traffic_file": spec.traffic_file,
            "seed": seed,
            "r2v": r2v_config.r2v,
            "generative_backend": r2v_config.generative_backend,
            "repair_story": r2v_config.repair_story,
            "repair_metadata_policy": r2v_config.repair_metadata_policy,
            "gate_variant": r2v_config.gate_variant,
            "r2v_sampling_mode": r2v_config.r2v_sampling_mode if r2v_config.r2v_enabled else "off",
            "r2v_admission_mode": r2v_config.r2v_admission_mode,
            "r2v_artifact_path": r2v_config.r2v_artifact_path,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
