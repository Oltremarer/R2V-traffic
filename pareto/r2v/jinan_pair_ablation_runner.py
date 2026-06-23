#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.data.build_pairs import SUPPORTED_ER_BASELINE_MODES, SUPPORTED_ER_R2V_COMBINE_MODES


SPLITS = ("train", "val", "test")
DEFAULT_TRANSITION_GLOB = (
    "records/paper_final_data_buffers/paper_final_20260602_v1/"
    "jinan/*/seed*/transitions_raw.jsonl"
)
DEFAULT_FINAL_JINAN_METHODS = (
    "film_scalar_potential",
    "weighted_proxy",
    "env_reward",
)


@dataclass(frozen=True)
class PairQuota:
    objective: int
    preference: int
    dominance: int
    reversal: int
    min_efficiency_stability_conflict: int
    min_efficiency_stability_reversal: int


@dataclass(frozen=True)
class MethodSpec:
    name: str
    er_baseline_mode: str
    r2v_sampling_mode: str
    uses_r2v: bool


@dataclass(frozen=True)
class R2VJinanAblationConfig:
    python_bin: str = sys.executable
    records_root: Path = Path("data/pareto_records_split_norm/jinan/paper_final")
    transition_inputs: list[Path] = field(default_factory=list)
    normalizer_path: Path = Path("data/normalizers/jinan/objective_norm_paper_final.json")
    output_root: Path = Path("records/r2v_tsc_runs/jinan_pair_ablation")
    r2v: str = "paired"
    r2v_mode: str = "traffic"
    generative_backend: str = "diffusion"
    gate_variant: str = "full"
    r2v_artifact_path: str = ""
    r2v_admitted_weight: float = 2.0
    r2v_repair_rejected_weight: float = 2.0
    r2v_admission_mode: str = "weights_only"
    support_gate: str = "on"
    ood_gate: str = "on"
    dynamics_gate: str = "on"
    seed: int = 0
    device: str = "cuda"
    epochs: int = 20
    batch_size: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    candidate_model: str = "feature_center_distance_proxy"
    er_baseline_mode: str = "uniform"
    er_priority_key: str = "metadata.td_error"
    er_r2v_combine: str = "multiply"
    r2v_sampling_mode: str = "full_r2v"
    rare_quantile: float = 0.8
    value_quantile: float = 0.6
    support_min_quantile: float = 0.02
    safety_min: float = -1.0
    repair_story: str = "none"
    repair_metadata_policy: str = "metadata_or_proxy"
    source_gates_key: str = "metadata.r2v_source_gates"
    final_gates_key: str = "metadata.r2v_final_gates"
    run_bounded_ppo: bool = False
    pilot_template_spec: Path = Path("configs/formal/jinan_1seed_film_pilot_dryrun.json")
    bounded_ppo_episodes: int = 2
    bounded_ppo_steps: int = 10
    force: bool = False

    def pair_quota(self, split: str) -> PairQuota:
        if split == "train":
            return PairQuota(
                objective=2400,
                preference=2400,
                dominance=500,
                reversal=600,
                min_efficiency_stability_conflict=80,
                min_efficiency_stability_reversal=80,
            )
        return PairQuota(
            objective=1000,
            preference=1000,
            dominance=180,
            reversal=200,
            min_efficiency_stability_conflict=25,
            min_efficiency_stability_reversal=25,
        )


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: list[str]
    cwd: Path = Path(".")

    def shell(self) -> str:
        return " ".join(shlex.quote(str(part)) for part in self.argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_plan_as_dicts(commands: Iterable[CommandSpec]) -> list[dict[str, Any]]:
    return [
        {"name": command.name, "cwd": str(command.cwd), "argv": [str(item) for item in command.argv], "shell": command.shell()}
        for command in commands
    ]


def _method_specs(config: R2VJinanAblationConfig) -> tuple[MethodSpec, ...]:
    baseline_name = "baseline_uniform" if config.er_baseline_mode == "uniform" else f"baseline_{config.er_baseline_mode}"
    r2v_name = (
        "r2v_full_r2v"
        if config.er_baseline_mode == "uniform" and config.r2v_sampling_mode == "full_r2v"
        else f"r2v_{config.er_baseline_mode}_{config.r2v_sampling_mode}"
    )
    baseline = MethodSpec(
        name=baseline_name,
        er_baseline_mode=config.er_baseline_mode,
        r2v_sampling_mode="off",
        uses_r2v=False,
    )
    r2v = MethodSpec(
        name=r2v_name,
        er_baseline_mode=config.er_baseline_mode,
        r2v_sampling_mode=config.r2v_sampling_mode,
        uses_r2v=True,
    )
    if config.r2v == "off":
        return (baseline,)
    if config.r2v == "on":
        return (r2v,)
    return (baseline, r2v)


def _method_names(config: R2VJinanAblationConfig) -> tuple[str, ...]:
    return tuple(method.name for method in _method_specs(config))


def build_command_plan(config: R2VJinanAblationConfig) -> list[CommandSpec]:
    method_specs = _method_specs(config)
    commands = []
    if any(method.uses_r2v for method in method_specs):
        commands.append(_build_r2v_candidates_command(config))
    for method in method_specs:
        for split in SPLITS:
            commands.append(_build_pairs_command(config, method, split))
            commands.append(_validate_pairs_command(config, method.name, split))
    for method in method_specs:
        commands.append(_train_scalar_command(config, method.name))
    if config.run_bounded_ppo:
        for method in method_specs:
            commands.append(_bounded_ppo_command(config, method.name))
    return commands


def build_film_pilot_spec(
    *,
    template_spec: str | Path,
    output_spec: str | Path,
    model_dir: str | Path,
    normalizer_path: str | Path,
    formal_gate_decision_path: str | Path,
) -> dict[str, Any]:
    template_path = Path(template_spec)
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if not {"pilot", "ppo", "model"} <= set(payload):
        raise ValueError(f"pilot template must contain pilot/ppo/model sections: {template_path}")

    output_path = Path(output_spec)
    model_path = Path(model_dir)
    norm_path = Path(normalizer_path)
    pilot = dict(payload["pilot"])
    pilot.update(
        {
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "cityflow_seed": int(pilot.get("cityflow_seed", 0)),
            "policy_seed": int(pilot.get("cityflow_seed", pilot.get("policy_seed", 0))),
            "model_seed": int(pilot.get("cityflow_seed", pilot.get("model_seed", 0))),
            "methods": list(DEFAULT_FINAL_JINAN_METHODS),
            "reward_adapter": "film_scalar_potential",
            "objective_norm_path": str(norm_path),
            "objective_normalizer_hash": _normalizer_hash_from_file(norm_path),
            "film_model_dir": str(model_path),
            "film_model_hash": sha256_file(model_path / "model.pt"),
            "pilot_spec_path": str(output_path),
            "formal_gate_decision_path": str(formal_gate_decision_path),
            "r2v_tsc_ablation_spec": True,
        }
    )
    updated = {
        "pilot": pilot,
        "ppo": dict(payload["ppo"]),
        "model": dict(payload["model"]),
    }
    write_json(output_path, updated)
    return updated


def run(config: R2VJinanAblationConfig, *, dry_run: bool = False) -> dict[str, Any]:
    _prepare_output_root(config, dry_run=dry_run)
    write_json(config.output_root / "run_config.json", _config_to_json(config))
    write_json(
        config.output_root / "setting.json",
        {
            "setting_label": "Jinan 3x4 paper-final pair-sampling ablation",
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "records_root": str(config.records_root),
            "transition_input_count": len(config.transition_inputs),
            "baseline": f"{config.er_baseline_mode} ER pair sampling with --r2v_sampling_mode off",
            "ours": (
                f"R2V + {config.er_baseline_mode} ER pair sampling with "
                f"--r2v_sampling_mode {config.r2v_sampling_mode}"
            ),
            "er_baseline_mode": config.er_baseline_mode,
            "er_priority_key": config.er_priority_key,
            "er_r2v_combine": config.er_r2v_combine,
            "candidate_model": config.candidate_model,
            "r2v": config.r2v,
            "r2v_mode": config.r2v_mode,
            "generative_backend": config.generative_backend,
            "gate_variant": config.gate_variant,
            "repair_story": config.repair_story,
            "r2v_admission_mode": config.r2v_admission_mode,
            "bounded_cityflow_ppo": bool(config.run_bounded_ppo),
            "claim_boundary": "engineering ablation run; not a final paper metric by itself",
        },
    )

    if dry_run:
        commands = build_command_plan(config)
        write_json(config.output_root / "command_plan.json", {"commands": command_plan_as_dicts(commands)})
        _write_status(config, "DRY_RUN_READY", "dry_run", {"command_count": len(commands)})
        return {"status": "DRY_RUN_READY", "command_count": len(commands), "output_root": str(config.output_root)}

    non_ppo_config = replace(config, run_bounded_ppo=False)
    non_ppo_commands = build_command_plan(non_ppo_config)
    _run_commands(config, non_ppo_commands)

    ppo_results: dict[str, Any] = {}
    if config.run_bounded_ppo:
        for method in _method_names(config):
            spec_payload = _prepare_dynamic_pilot_spec(config, method)
            command = _bounded_ppo_command(config, method)
            _run_commands(config, [command])
            ppo_results[method] = {
                "spec": str(_spec_path(config, method)),
                "film_model_hash": spec_payload["pilot"]["film_model_hash"],
                "out_dir": str(_ppo_root(config, method)),
            }

    summary = _build_summary(config, ppo_results)
    write_json(config.output_root / "summary.json", summary)
    _write_status(config, "COMPLETED", "summary", summary)
    return summary


def _build_r2v_candidates_command(config: R2VJinanAblationConfig) -> CommandSpec:
    argv = [
            config.python_bin,
            "-m",
            "pareto.r2v.build_r2v_candidates",
            "--transitions",
            *[str(path) for path in config.transition_inputs],
            "--output",
            str(_artifact_root(config) / "r2v_candidates.jsonl"),
            "--summary_output",
            str(_artifact_root(config) / "r2v_summary.json"),
            "--weighted_output",
            str(_weighted_transitions_path(config)),
            "--candidate_model",
            config.candidate_model,
            "--rare_quantile",
            str(config.rare_quantile),
            "--value_quantile",
            str(config.value_quantile),
            "--support_min_quantile",
            str(config.support_min_quantile),
            "--safety_min",
            str(config.safety_min),
            "--repair_story",
            config.repair_story,
            "--repair_metadata_policy",
            config.repair_metadata_policy,
            "--gate_variant",
            config.gate_variant,
            "--admission_mode",
            config.r2v_admission_mode,
            "--admitted_weight",
            str(config.r2v_admitted_weight),
            "--repair_rejected_weight",
            str(config.r2v_repair_rejected_weight),
            "--source_gates_key",
            config.source_gates_key,
            "--final_gates_key",
            config.final_gates_key,
        ]
    if config.r2v_artifact_path:
        argv.extend([
            "--score_artifact",
            config.r2v_artifact_path,
            "--score_artifact_backend",
            config.generative_backend,
        ])
    return CommandSpec(
        name="build_r2v_weighted_transitions",
        argv=argv,
        cwd=ROOT,
    )


def _build_pairs_command(config: R2VJinanAblationConfig, method: MethodSpec, split: str) -> CommandSpec:
    quota = config.pair_quota(split)
    argv = [
        config.python_bin,
        "-m",
        "pareto.data.build_pairs",
        "--buffers",
        str(config.records_root / f"{split}_raw.jsonl"),
        "--out_dir",
        str(_pairs_root(config, method.name) / split),
        "--split",
        split,
        "--num_objective_pairs",
        str(quota.objective),
        "--num_preference_pairs",
        str(quota.preference),
        "--num_dominance_pairs",
        str(quota.dominance),
        "--num_reversal_pairs",
        str(quota.reversal),
        "--min_efficiency_stability_conflict",
        str(quota.min_efficiency_stability_conflict),
        "--reversal_template_quota",
        f"efficiency__stability:{quota.min_efficiency_stability_reversal}",
        "--seed",
        str(config.seed + SPLITS.index(split)),
        "--er_baseline_mode",
        method.er_baseline_mode,
        "--er_priority_key",
        config.er_priority_key,
        "--er_r2v_combine",
        config.er_r2v_combine,
        "--r2v_sampling_mode",
        method.r2v_sampling_mode,
    ]
    if method.uses_r2v:
        argv.extend(["--r2v_weighted_transitions", str(_weighted_transitions_path(config))])
    return CommandSpec(name=f"build_{method.name}_pairs_{split}", argv=argv, cwd=ROOT)


def _validate_pairs_command(config: R2VJinanAblationConfig, method: str, split: str) -> CommandSpec:
    if split == "train":
        gates = {
            "min_objective_per_head": 400,
            "min_preference_pairs": 2000,
            "min_dominance_pairs": 400,
            "min_reversal_pairs": 500,
            "min_eff_controlled_stability": 80,
            "min_efficiency_stability_conflict": 60,
            "min_reversal_template_pair": "efficiency__stability:50",
        }
    else:
        gates = {
            "min_objective_per_head": 150,
            "min_preference_pairs": 700,
            "min_dominance_pairs": 120,
            "min_reversal_pairs": 150,
            "min_eff_controlled_stability": 20,
            "min_efficiency_stability_conflict": 15,
            "min_reversal_template_pair": "efficiency__stability:15",
        }
    return CommandSpec(
        name=f"validate_{method}_pairs_{split}",
        argv=[
            config.python_bin,
            "-m",
            "pareto.data.validate_pairs",
            "--pairs_dir",
            str(_pairs_root(config, method) / split),
            "--report",
            str(_report_root(config) / f"{method}_{split}_pair_validation.json"),
            "--strict",
            "--min_objective_per_head",
            str(gates["min_objective_per_head"]),
            "--min_preference_pairs",
            str(gates["min_preference_pairs"]),
            "--min_dominance_pairs",
            str(gates["min_dominance_pairs"]),
            "--min_reversal_pairs",
            str(gates["min_reversal_pairs"]),
            "--min_eff_controlled_stability",
            str(gates["min_eff_controlled_stability"]),
            "--min_efficiency_stability_conflict",
            str(gates["min_efficiency_stability_conflict"]),
            "--min_reversal_template_pair",
            str(gates["min_reversal_template_pair"]),
            "--positive_ratio_low",
            "0.20",
            "--positive_ratio_high",
            "0.80",
            "--positive_ratio_by_objective_low",
            "0.15",
            "--positive_ratio_by_objective_high",
            "0.85",
            "--require_no_ties",
        ],
        cwd=ROOT,
    )


def _train_scalar_command(config: R2VJinanAblationConfig, method: str) -> CommandSpec:
    return CommandSpec(
        name=f"train_{method}_film_scalar",
        argv=[
            config.python_bin,
            "-m",
            "pareto.train_conditioned_scalar",
            "--records_root",
            str(config.records_root),
            "--pairs_root",
            str(_pairs_root(config, method)),
            "--output_dir",
            str(_model_root(config, method)),
            "--epochs",
            str(config.epochs),
            "--batch_size",
            str(config.batch_size),
            "--seed",
            str(config.seed),
            "--device",
            config.device,
            "--hidden_dim",
            str(config.hidden_dim),
            "--num_layers",
            str(config.num_layers),
            "--dropout",
            "0.1",
            "--lr",
            "0.001",
            "--architecture",
            "film",
            "--training_schedule",
            "joint",
            "--preference_loss_weight",
            "1.0",
            "--reversal_loss_weight",
            "1.0",
            "--dominance_loss_weight",
            "0.2",
            "--dominance_margin",
            "0.1",
            "--reversal_sampler",
            "template_balanced",
            "--reversal_template_min_count",
            "20",
            "--pref_margin_loss_weight",
            "0.25",
            "--rev_margin_loss_weight",
            "0.25",
            "--pref_hinge_loss_weight",
            "0.25",
            "--rev_hinge_loss_weight",
            "0.25",
            "--classification_margin",
            "0.5",
            "--margin_clip",
            "2.0",
        ],
        cwd=ROOT,
    )


def _bounded_ppo_command(config: R2VJinanAblationConfig, method: str) -> CommandSpec:
    model_hash = _maybe_file_hash(_model_root(config, method) / "model.pt")
    normalizer_hash = _maybe_normalizer_hash(config.normalizer_path)
    return CommandSpec(
        name=f"bounded_ppo_{method}",
        argv=[
            config.python_bin,
            "-m",
            "pareto.rl.formal_pilot_runner",
            "--spec",
            str(_spec_path(config, method)),
            "--method",
            "film_scalar_potential",
            "--bounded_jinan_pilot_dry_run",
            "--i_understand_this_runs_bounded_jinan_pilot_dry_run",
            "--episodes",
            str(config.bounded_ppo_episodes),
            "--max_decision_steps_per_episode",
            str(config.bounded_ppo_steps),
            "--out_dir",
            str(_ppo_root(config, method)),
            "--objective_normalizer",
            str(config.normalizer_path),
            "--objective_normalizer_hash",
            normalizer_hash,
            "--film_model_dir",
            str(_model_root(config, method)),
            "--film_model_hash",
            model_hash,
            "--device",
            config.device,
        ],
        cwd=ROOT,
    )


def _run_commands(config: R2VJinanAblationConfig, commands: Iterable[CommandSpec]) -> None:
    for command in commands:
        _write_status(config, "RUNNING", command.name, {"command": command.shell()})
        with (config.output_root / "driver.log").open("a", encoding="utf-8") as log:
            log.write(f"\n[{_now_iso()}] START {command.name}\n{command.shell()}\n")
            log.flush()
            result = subprocess.run(
                [str(part) for part in command.argv],
                cwd=str(command.cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(f"[{_now_iso()}] END {command.name} returncode={result.returncode}\n")
        if result.returncode != 0:
            _write_status(config, "FAILED", command.name, {"returncode": result.returncode})
            raise subprocess.CalledProcessError(result.returncode, command.argv)
        _write_status(config, "STEP_DONE", command.name, {"returncode": result.returncode})


def _prepare_dynamic_pilot_spec(config: R2VJinanAblationConfig, method: str) -> dict[str, Any]:
    formal_gate = _report_root(config) / f"{method}_formal_gate_decision.json"
    if not formal_gate.exists():
        write_json(
            formal_gate,
            {
                "ppo_formal_allowed": False,
                "bounded_pilot_allowed": True,
                "source": "r2v_jinan_pair_ablation_runner",
                "method": method,
            },
        )
    return build_film_pilot_spec(
        template_spec=config.pilot_template_spec,
        output_spec=_spec_path(config, method),
        model_dir=_model_root(config, method),
        normalizer_path=config.normalizer_path,
        formal_gate_decision_path=formal_gate,
    )


def _build_summary(config: R2VJinanAblationConfig, ppo_results: dict[str, Any]) -> dict[str, Any]:
    method_summaries = {}
    for method in _method_names(config):
        method_summaries[method] = {
            "pairs_root": str(_pairs_root(config, method)),
            "model_dir": str(_model_root(config, method)),
            "diagnostics_val": str(_model_root(config, method) / "diagnostics_val.json"),
            "diagnostics_test": str(_model_root(config, method) / "diagnostics_test.json"),
            "pair_reports": {
                split: str(_pairs_root(config, method) / split / "pair_report.json")
                for split in SPLITS
            },
            "pair_validations": {
                split: str(_report_root(config) / f"{method}_{split}_pair_validation.json")
                for split in SPLITS
            },
        }
        if method in ppo_results:
            method_summaries[method]["bounded_ppo"] = ppo_results[method]
    return {
        "status": "COMPLETED",
        "completed_at": _now_iso(),
        "output_root": str(config.output_root),
        "setting": (
            "Jinan anon_3_4_jinan_real.json, paper-final records, "
            f"{config.er_baseline_mode} ER pairs vs R2V+{config.er_baseline_mode} pairs"
        ),
        "er_baseline_mode": config.er_baseline_mode,
        "er_priority_key": config.er_priority_key,
        "er_r2v_combine": config.er_r2v_combine,
        "r2v": config.r2v,
        "r2v_mode": config.r2v_mode,
        "generative_backend": config.generative_backend,
        "gate_variant": config.gate_variant,
        "repair_story": config.repair_story,
        "r2v_admission_mode": config.r2v_admission_mode,
        "candidate_summary": str(_artifact_root(config) / "r2v_summary.json")
        if any(method.uses_r2v for method in _method_specs(config))
        else None,
        "weighted_transitions": str(_weighted_transitions_path(config))
        if any(method.uses_r2v for method in _method_specs(config))
        else None,
        "methods": method_summaries,
        "bounded_cityflow_ppo": bool(config.run_bounded_ppo),
        "claim_boundary": "Use diagnostics and bounded PPO as run-health evidence before claiming final traffic performance.",
    }


def _prepare_output_root(config: R2VJinanAblationConfig, *, dry_run: bool) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    existing_status = config.output_root / "status.json"
    if existing_status.exists() and not (config.force or dry_run):
        raise FileExistsError(f"{existing_status} exists; choose a fresh --output_root or pass --force")
    for subdir in ("artifacts", "pairs", "models", "reports", "specs", "ppo"):
        (config.output_root / subdir).mkdir(parents=True, exist_ok=True)


def _write_status(config: R2VJinanAblationConfig, status: str, phase: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "phase": phase,
        "updated_at": _now_iso(),
        "updated_at_epoch": time.time(),
        "output_root": str(config.output_root),
    }
    if extra:
        payload.update(extra)
    write_json(config.output_root / "status.json", payload)
    write_json(config.output_root / "heartbeat.json", payload)


def _config_to_json(config: R2VJinanAblationConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return payload


def _normalizer_hash_from_file(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "hash" not in payload:
        raise ValueError(f"objective normalizer missing hash field: {path}")
    return str(payload["hash"])


def _maybe_file_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else "__MODEL_HASH_AFTER_TRAINING__"


def _maybe_normalizer_hash(path: Path) -> str:
    return _normalizer_hash_from_file(path) if path.exists() else "__OBJECTIVE_NORMALIZER_HASH__"


def _artifact_root(config: R2VJinanAblationConfig) -> Path:
    return config.output_root / "artifacts"


def _report_root(config: R2VJinanAblationConfig) -> Path:
    return config.output_root / "reports"


def _weighted_transitions_path(config: R2VJinanAblationConfig) -> Path:
    return _artifact_root(config) / "r2v_weighted_transitions.jsonl"


def _pairs_root(config: R2VJinanAblationConfig, method: str) -> Path:
    return config.output_root / "pairs" / method


def _model_root(config: R2VJinanAblationConfig, method: str) -> Path:
    return config.output_root / "models" / method / "film_scalar"


def _spec_path(config: R2VJinanAblationConfig, method: str) -> Path:
    return config.output_root / "specs" / f"{method}_film_pilot.json"


def _ppo_root(config: R2VJinanAblationConfig, method: str) -> Path:
    return config.output_root / "ppo" / method


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _resolve_transition_inputs(values: list[str], globs: list[str]) -> list[Path]:
    result = [Path(value) for value in values]
    for pattern in globs:
        result.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in result:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise FileNotFoundError("no transition inputs resolved; pass --transition_input or --transition_glob")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jinan paper-final uniform-vs-R2V pair ablation.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--records_root", default="data/pareto_records_split_norm/jinan/paper_final")
    parser.add_argument("--transition_input", action="append", default=[])
    parser.add_argument("--transition_glob", action="append", default=[DEFAULT_TRANSITION_GLOB])
    parser.add_argument("--normalizer_path", default="data/normalizers/jinan/objective_norm_paper_final.json")
    parser.add_argument("--output_root")
    parser.add_argument("--r2v", choices=("on", "off", "paired"), default="paired")
    parser.add_argument("--r2v_mode", choices=("traffic",), default="traffic")
    parser.add_argument("--generative_backend", choices=("diffusion",), default="diffusion")
    parser.add_argument("--gate_variant", choices=("full", "no_support", "no_ood", "no_dynamics"), default="full")
    parser.add_argument("--r2v_output_dir", default=None)
    parser.add_argument("--r2v_artifact_path", default="")
    parser.add_argument("--r2v_admitted_weight", type=float, default=2.0)
    parser.add_argument("--r2v_repair_rejected_weight", type=float, default=2.0)
    parser.add_argument("--r2v_admission_mode", choices=("weights_only", "weights_plus_repaired"), default="weights_only")
    parser.add_argument("--rare_fraction", type=float, default=None)
    parser.add_argument("--support_gate", choices=("on", "off"), default="on")
    parser.add_argument("--ood_gate", choices=("on", "off"), default="on")
    parser.add_argument("--dynamics_gate", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--candidate_model", default="feature_center_distance_proxy")
    parser.add_argument(
        "--er_baseline_mode",
        "--er_sampling_mode",
        choices=sorted(SUPPORTED_ER_BASELINE_MODES),
        default="uniform",
    )
    parser.add_argument("--er_priority_key", default="metadata.td_error")
    parser.add_argument(
        "--er_r2v_combine",
        choices=sorted(SUPPORTED_ER_R2V_COMBINE_MODES),
        default="multiply",
    )
    parser.add_argument("--r2v_sampling_mode", default="full_r2v")
    parser.add_argument("--rare_quantile", type=float, default=0.8)
    parser.add_argument("--value_quantile", type=float, default=0.6)
    parser.add_argument("--support_min_quantile", type=float, default=0.02)
    parser.add_argument("--safety_min", type=float, default=-1.0)
    parser.add_argument("--repair_story", choices=("none", "not_val_to_val", "not_rare_to_val"), default="none")
    parser.add_argument(
        "--repair_metadata_policy",
        choices=("require_metadata", "metadata_or_proxy"),
        default="metadata_or_proxy",
    )
    parser.add_argument("--source_gates_key", default="metadata.r2v_source_gates")
    parser.add_argument("--final_gates_key", default="metadata.r2v_final_gates")
    parser.add_argument("--run_bounded_ppo", action="store_true")
    parser.add_argument("--pilot_template_spec", default="configs/formal/jinan_1seed_film_pilot_dryrun.json")
    parser.add_argument("--bounded_ppo_episodes", type=int, default=2)
    parser.add_argument("--bounded_ppo_steps", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> R2VJinanAblationConfig:
    output_root = args.output_root or args.r2v_output_dir
    if output_root is None:
        output_root = f"records/r2v_tsc_runs/jinan_pair_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    gate_variant = _resolve_gate_variant(args.gate_variant, args.support_gate, args.ood_gate, args.dynamics_gate)
    rare_quantile = args.rare_quantile
    if args.rare_fraction is not None:
        if not 0.0 < float(args.rare_fraction) <= 1.0:
            raise ValueError("rare_fraction must be in (0, 1]")
        rare_quantile = 1.0 - float(args.rare_fraction)
    if args.r2v == "on" and args.r2v_sampling_mode == "off":
        raise ValueError("r2v_sampling_mode cannot be off when --r2v on")
    return R2VJinanAblationConfig(
        python_bin=args.python_bin,
        records_root=Path(args.records_root),
        transition_inputs=_resolve_transition_inputs(args.transition_input, args.transition_glob),
        normalizer_path=Path(args.normalizer_path),
        output_root=Path(output_root),
        r2v=args.r2v,
        r2v_mode=args.r2v_mode,
        generative_backend=args.generative_backend,
        gate_variant=gate_variant,
        r2v_artifact_path=args.r2v_artifact_path,
        r2v_admitted_weight=args.r2v_admitted_weight,
        r2v_repair_rejected_weight=args.r2v_repair_rejected_weight,
        r2v_admission_mode=args.r2v_admission_mode,
        support_gate=args.support_gate,
        ood_gate=args.ood_gate,
        dynamics_gate=args.dynamics_gate,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        candidate_model=args.candidate_model,
        er_baseline_mode=args.er_baseline_mode,
        er_priority_key=args.er_priority_key,
        er_r2v_combine=args.er_r2v_combine,
        r2v_sampling_mode=args.r2v_sampling_mode,
        rare_quantile=rare_quantile,
        value_quantile=args.value_quantile,
        support_min_quantile=args.support_min_quantile,
        safety_min=args.safety_min,
        repair_story=args.repair_story,
        repair_metadata_policy=args.repair_metadata_policy,
        source_gates_key=args.source_gates_key,
        final_gates_key=args.final_gates_key,
        run_bounded_ppo=args.run_bounded_ppo,
        pilot_template_spec=Path(args.pilot_template_spec),
        bounded_ppo_episodes=args.bounded_ppo_episodes,
        bounded_ppo_steps=args.bounded_ppo_steps,
        force=args.force,
    )


def _resolve_gate_variant(gate_variant: str, support_gate: str, ood_gate: str, dynamics_gate: str) -> str:
    if gate_variant != "full":
        return gate_variant
    disabled = [
        name
        for name, state in (
            ("support", support_gate),
            ("ood", ood_gate),
            ("dynamics", dynamics_gate),
        )
        if state == "off"
    ]
    if not disabled:
        return "full"
    if disabled == ["support"]:
        return "no_support"
    if disabled == ["ood"]:
        return "no_ood"
    if disabled == ["dynamics"]:
        return "no_dynamics"
    raise ValueError("only one gate can be disabled by support_gate/ood_gate/dynamics_gate in this runner")


def main() -> None:
    args = parse_args()
    result = run(config_from_args(args), dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
