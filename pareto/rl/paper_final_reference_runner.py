#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.paper_final_experiment_manifest import PAPER_FINAL_SEEDS, REQUIRED_CITY_TRAFFIC
from pareto.rl.paper_final_wandb_uploader import PAPER_FINAL_WANDB_LAYOUT_SCOPE, PAPER_FINAL_WANDB_MODE

REFERENCE_BASELINE_SCRIPTS = {
    "Random": "run_random.py",
    "FixedTime": "run_fixedtime.py",
    "MaxPressure": "run_maxpressure.py",
    "PressLight": "run_presslight.py",
    "MPLight": "run_mplight.py",
    "CoLight": "run_colight.py",
    "Advanced-Co": "run_advanced_colight.py",
}
PAPER_FINAL_REFERENCE_EXECUTION_APPROVAL_PHRASE = "PPTS PARETO PPO FINAL SCOPE-LIMITED EXECUTION GO"
PAPER_FINAL_REFERENCE_APPROVAL_ENV = "PPTS_PARETO_PPO_FINAL_EXECUTION_APPROVAL_PHRASE"
PAPER_FINAL_REFERENCE_MEMO = "paper_final_scope_limited_reference"
PAPER_FINAL_REFERENCE_PROJECT = "paper_final_scope_limited_reference"
PAPER_FINAL_REFERENCE_CLEAN_PROJECT = "paper_final_scope_limited"
PAPER_FINAL_REFERENCE_CLEAN_JOB_TYPE = "paper_final_scope_limited_reference"
PAPER_FINAL_REFERENCE_ROOT = "records/paper_final/"
REFERENCE_METRIC_KEYS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_queuing_vehicle_num_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
)


def _safe_wandb_text(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _scope_tags(scope: str) -> list[str]:
    return [_safe_wandb_text(item) for item in scope.split("/") if item.strip()]


def _require_paper_final_out_dir(out_dir: str) -> str:
    normalized = str(out_dir).strip()
    if not normalized.startswith(PAPER_FINAL_REFERENCE_ROOT):
        raise ValueError("paper-final reference out_dir must be under records/paper_final")
    parts = Path(normalized).parts
    if ".." in parts:
        raise ValueError("paper-final reference out_dir must not contain '..'")
    return normalized


def validate_reference_request(
    *,
    method: str,
    dataset: str,
    traffic_file: str,
    seed_id: int,
    out_dir: str,
    legacy_script: str,
    execute: bool,
    approval_phrase: str | None,
) -> dict[str, Any]:
    if method not in REFERENCE_BASELINE_SCRIPTS:
        raise ValueError(f"unknown paper-final reference method: {method}")
    expected_script = REFERENCE_BASELINE_SCRIPTS[method]
    if legacy_script != expected_script:
        raise ValueError(f"legacy_script mismatch for {method}: expected {expected_script}, got {legacy_script}")
    if dataset not in REQUIRED_CITY_TRAFFIC:
        raise ValueError(f"paper-final reference dataset is not approved: {dataset}")
    if traffic_file != REQUIRED_CITY_TRAFFIC[dataset]:
        raise ValueError(f"paper-final reference traffic_file mismatch for {dataset}")
    seed = int(seed_id)
    if seed not in PAPER_FINAL_SEEDS:
        raise ValueError(f"seed_id {seed_id} is not approved for paper-final reference execution")
    if execute and approval_phrase != PAPER_FINAL_REFERENCE_EXECUTION_APPROVAL_PHRASE:
        raise ValueError("paper-final reference execution requires the exact external approval phrase")
    script_path = ROOT / legacy_script
    if not script_path.is_file():
        raise ValueError(f"paper-final reference legacy script missing: {legacy_script}")
    return {
        "method": method,
        "dataset": dataset,
        "traffic_file": traffic_file,
        "seed_id": seed,
        "out_dir": _require_paper_final_out_dir(out_dir),
        "legacy_script": legacy_script,
        "execute": bool(execute),
        "approval_phrase_verified": bool(execute),
    }


def build_legacy_reference_command(request: dict[str, Any], *, python_bin: str = sys.executable) -> list[str]:
    return [
        python_bin,
        str(request["legacy_script"]),
        "--memo",
        PAPER_FINAL_REFERENCE_MEMO,
        "--dataset",
        str(request["dataset"]),
        "--traffic_file",
        str(request["traffic_file"]),
        "--proj_name",
        PAPER_FINAL_REFERENCE_PROJECT,
        "--workers",
        "1",
    ]


def paper_final_reference_env(request: dict[str, Any], *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(
        {
            "PAPER_FINAL_REFERENCE_RUN": "1",
            "PAPER_FINAL_REFERENCE_SEED_ID": str(request["seed_id"]),
            "PAPER_FINAL_REFERENCE_OUT_DIR": str(request["out_dir"]),
            "PAPER_FINAL_REFERENCE_MODEL_DIR": reference_model_dir(request).as_posix(),
            "PAPER_FINAL_REFERENCE_METHOD": str(request["method"]),
            "PAPER_FINAL_REFERENCE_TRAFFIC_FILE": str(request["traffic_file"]),
            "CUDA_VISIBLE_DEVICES": "",
            "TF_CPP_MIN_LOG_LEVEL": env.get("TF_CPP_MIN_LOG_LEVEL", "2"),
            "WANDB_MODE": "disabled",
        }
    )
    return env


def reference_model_dir(request: dict[str, Any]) -> Path:
    out_dir = Path(str(request["out_dir"]))
    return Path("model") / out_dir.relative_to("records")


def reference_output_is_complete(out_path: Path) -> bool:
    status_path = out_path / "paper_final_reference_status.json"
    metrics_path = out_path / "paper_final_reference_metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") == "PAPER_FINAL_REFERENCE_RUN_DONE"


def prepare_reference_paths_for_execution(request: dict[str, Any]) -> None:
    out_path = ROOT / str(request["out_dir"])
    if out_path.exists():
        if reference_output_is_complete(out_path):
            raise FileExistsError(f"paper-final reference out_dir is already complete: {request['out_dir']}")
        shutil.rmtree(out_path)
    model_path = ROOT / reference_model_dir(request)
    if model_path.exists():
        shutil.rmtree(model_path)


def _normalize_numpy_scalar_literals(text: str) -> str:
    return re.sub(r"\bnp\.(?:float16|float32|float64|int32|int64)\(([^()]+)\)", r"\1", text)


def extract_reference_metrics_from_stdout(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        if not all(key in line for key in REFERENCE_METRIC_KEYS):
            continue
        candidate = _normalize_numpy_scalar_literals(line.strip())
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < start:
            continue
        try:
            payload = ast.literal_eval(candidate[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        metrics: dict[str, float] = {}
        for key in REFERENCE_METRIC_KEYS:
            if key not in payload:
                break
            value = float(payload[key])
            if not math.isfinite(value):
                raise ValueError(f"paper-final reference metric is not finite: {key}={value}")
            metrics[key] = value
        if len(metrics) == len(REFERENCE_METRIC_KEYS):
            return metrics
    raise ValueError("paper-final reference metrics were not found in legacy stdout")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wandb_module(wandb_module: object | None = None) -> object:
    if wandb_module is not None:
        return wandb_module
    return importlib.import_module("wandb")


def upload_reference_metrics_to_wandb(
    request: dict[str, Any],
    metrics: dict[str, float],
    *,
    wandb_module: object | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    upload_env = dict(env or os.environ)
    if upload_env.get("WANDB_MODE", PAPER_FINAL_WANDB_MODE).lower() == "disabled":
        return {
            "status": "skipped_disabled",
            "project": upload_env.get("WANDB_REFERENCE_CLEAN_PROJECT")
            or upload_env.get("WANDB_PROJECT", PAPER_FINAL_REFERENCE_CLEAN_PROJECT),
        }

    selected_project = (
        upload_env.get("WANDB_REFERENCE_CLEAN_PROJECT")
        or upload_env.get("WANDB_PROJECT")
        or PAPER_FINAL_REFERENCE_CLEAN_PROJECT
    )
    selected_entity = upload_env.get("WANDB_ENTITY") or None
    layout_scope = upload_env.get("WANDB_LAYOUT_SCOPE", PAPER_FINAL_WANDB_LAYOUT_SCOPE)
    method = _safe_wandb_text(request["method"])
    dataset = _safe_wandb_text(request["dataset"])
    seed = int(request["seed_id"])
    run_name = f"reference__{dataset}__{method}__seed{seed:02d}"
    group = f"{layout_scope}/reference/{dataset}/{method}"
    tags = _scope_tags(layout_scope) + ["reference", dataset, method, f"seed_{seed}"]
    wandb = _wandb_module(wandb_module)
    init_kwargs = {
        "project": selected_project,
        "group": group,
        "name": run_name,
        "tags": tags,
        "job_type": PAPER_FINAL_REFERENCE_CLEAN_JOB_TYPE,
        "config": {
            "paper_final_scope_limited_reference": True,
            "wandb_layout_scope": layout_scope,
            "method": request["method"],
            "dataset": request["dataset"],
            "traffic_file": request["traffic_file"],
            "seed_id": request["seed_id"],
            "out_dir": request["out_dir"],
            "legacy_script": request["legacy_script"],
            "legacy_wandb_disabled": True,
            "reference_execution_device": "cpu",
        },
        "reinit": True,
    }
    if selected_entity:
        init_kwargs["entity"] = selected_entity

    wandb.init(**init_kwargs)
    saved_files: list[str] = []
    try:
        wandb.log(metrics, step=0)
        out_dir = Path(str(request["out_dir"]))
        for name in ("paper_final_reference_metrics.json", "paper_final_reference_status.json"):
            path = out_dir / name
            if path.is_file() and hasattr(wandb, "save"):
                wandb.save(path.as_posix(), base_path=out_dir.as_posix(), policy="now")
                saved_files.append(name)
    finally:
        wandb.finish()

    return {
        "status": "uploaded",
        "project": selected_project,
        "entity": selected_entity,
        "run_name": run_name,
        "group": group,
        "tags": tags,
        "layout_scope": layout_scope,
        "metric_keys": list(metrics),
        "files_saved": saved_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--traffic_file", required=True)
    parser.add_argument("--seed_id", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--legacy_script", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval_phrase")
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = validate_reference_request(
        method=args.method,
        dataset=args.dataset,
        traffic_file=args.traffic_file,
        seed_id=args.seed_id,
        out_dir=args.out_dir,
        legacy_script=args.legacy_script,
        execute=args.execute,
        approval_phrase=args.approval_phrase,
    )
    command = build_legacy_reference_command(request, python_bin=args.python_bin)
    if not args.execute:
        print(
            json.dumps(
                {
                    "paper_final_reference_adapter": True,
                    "executes_now": False,
                    "request": request,
                    "legacy_command": command,
                    "env_overrides": {
                        key: value
                        for key, value in paper_final_reference_env(request, base_env={}).items()
                        if key.startswith("PAPER_FINAL_REFERENCE_")
                        or key in {"WANDB_MODE", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL"}
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    prepare_reference_paths_for_execution(request)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=paper_final_reference_env(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        out_path = ROOT / str(request["out_dir"])
        _write_json(
            out_path / "paper_final_reference_status.json",
            {
                "status": "LEGACY_REFERENCE_FAILED",
                "returncode": completed.returncode,
                "method": request["method"],
                "dataset": request["dataset"],
                "seed_id": request["seed_id"],
                "legacy_wandb_disabled": True,
                "reference_execution_device": "cpu",
            },
        )
        raise SystemExit(completed.returncode)

    metrics = extract_reference_metrics_from_stdout(completed.stdout)
    out_path = ROOT / str(request["out_dir"])
    _write_json(out_path / "paper_final_reference_metrics.json", metrics)
    status = {
        "status": "PAPER_FINAL_REFERENCE_RUN_DONE",
        "returncode": completed.returncode,
        "method": request["method"],
        "dataset": request["dataset"],
        "traffic_file": request["traffic_file"],
        "seed_id": request["seed_id"],
        "legacy_script": request["legacy_script"],
        "legacy_wandb_disabled": True,
        "reference_execution_device": "cpu",
        "metrics_file": "paper_final_reference_metrics.json",
    }
    _write_json(out_path / "paper_final_reference_status.json", status)
    status["wandb_upload"] = upload_reference_metrics_to_wandb(request, metrics)
    _write_json(out_path / "paper_final_reference_status.json", status)
    print(json.dumps({"paper_final_reference_status": "DONE", "metrics": metrics}, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
