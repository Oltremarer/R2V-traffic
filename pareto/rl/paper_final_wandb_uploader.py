from __future__ import annotations

import importlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

PAPER_FINAL_WANDB_PROJECT = "paper_final_scope_limited"
PAPER_FINAL_WANDB_MODE = "online"
PAPER_FINAL_WANDB_JOB_TYPE = "paper_final_scope_limited_learned"
PAPER_FINAL_WANDB_LAYOUT_SCOPE = "paper_final/no_newyork"
_ARTIFACT_FILE_NAMES = ("metadata.json", "status.json", "train_metrics.jsonl", "loss_debug.jsonl")
_LEARNED_EVAL_ARTIFACT_FILE_NAMES = (
    "paper_final_learned_eval_metadata.json",
    "paper_final_learned_eval_status.json",
    "paper_final_learned_eval_metrics.json",
    "command.txt",
)
_REFERENCE_SCHEMA_METRIC_KEYS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_queuing_vehicle_num_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
)


def paper_final_wandb_env(*, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.setdefault("WANDB_MODE", PAPER_FINAL_WANDB_MODE)
    env.setdefault("WANDB_PROJECT", PAPER_FINAL_WANDB_PROJECT)
    env.setdefault("WANDB_LAYOUT_SCOPE", PAPER_FINAL_WANDB_LAYOUT_SCOPE)
    return env


def _safe_text(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _primitive(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"paper-final W&B payload must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"paper-final W&B JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _load_reference_schema_metrics(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    missing = [key for key in _REFERENCE_SCHEMA_METRIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"paper-final learned eval metrics missing keys: {missing}")
    return {key: float(payload[key]) for key in _REFERENCE_SCHEMA_METRIC_KEYS}


def _prefixed_primitives(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in payload.items() if _primitive(value)}


def _wandb_module(wandb_module: object | None) -> object:
    if wandb_module is not None:
        return wandb_module
    return importlib.import_module("wandb")


def _positive_int_env(env: dict[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _wandb_settings(wandb: object, env: dict[str, str]) -> object | None:
    settings_cls = getattr(wandb, "Settings", None)
    if settings_cls is None:
        return None
    settings_kwargs: dict[str, int] = {}
    init_timeout = _positive_int_env(env, "WANDB_INIT_TIMEOUT")
    service_wait = _positive_int_env(env, "WANDB__SERVICE_WAIT")
    if init_timeout is not None:
        settings_kwargs["init_timeout"] = init_timeout
    if service_wait is not None:
        settings_kwargs["_service_wait"] = service_wait
    if not settings_kwargs:
        return None
    try:
        return settings_cls(**settings_kwargs)
    except TypeError:
        if "init_timeout" in settings_kwargs:
            return settings_cls(init_timeout=settings_kwargs["init_timeout"])
        return None


def _scope_tags(scope: str) -> list[str]:
    return [_safe_text(item) for item in scope.split("/") if item.strip()]


def build_paper_final_wandb_identity(row: dict[str, Any], *, scope: str = PAPER_FINAL_WANDB_LAYOUT_SCOPE) -> dict[str, Any]:
    row_index = int(row.get("row_index", 0))
    method = _safe_text(row.get("method", "unknown_method"))
    city = _safe_text(row.get("city", "unknown_city"))
    seed = _safe_text(row.get("seed", "unknown_seed"))
    preference = _safe_text(row.get("preference_template", "default"))
    phase = "learned" if row.get("runner_family") == "formal_pilot_paper_final" else _safe_text(row.get("runner_family", "paper_final"))
    return {
        "name": f"{phase}__{city}__{method}__seed{seed}__{preference}__row{row_index:03d}",
        "group": f"{scope}/{phase}/{city}/{method}/{preference}",
        "tags": _scope_tags(scope) + [phase, city, method, f"seed_{seed}", preference],
    }


def upload_paper_final_run_to_wandb(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    project: str | None = None,
    entity: str | None = None,
    wandb_module: object | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    wandb_env = paper_final_wandb_env(base_env=env)
    if wandb_env.get("WANDB_MODE", PAPER_FINAL_WANDB_MODE).lower() == "disabled":
        return {
            "status": "skipped_disabled",
            "row_index": row.get("row_index"),
            "project": project or wandb_env["WANDB_PROJECT"],
        }

    out_dir = Path(str(row.get("out_dir")))
    if not out_dir.is_dir():
        raise ValueError(f"paper-final W&B upload requires existing out_dir: {out_dir}")

    metadata = _load_json(out_dir / "metadata.json")
    status = _load_json(out_dir / "status.json")
    train_metrics = _load_jsonl(out_dir / "train_metrics.jsonl")
    layout_scope = wandb_env.get("WANDB_LAYOUT_SCOPE", PAPER_FINAL_WANDB_LAYOUT_SCOPE)
    identity_row = dict(row)
    identity_row.setdefault("row_index", result.get("row_index", 0))
    identity = build_paper_final_wandb_identity(identity_row, scope=layout_scope)
    wandb = _wandb_module(wandb_module)
    selected_project = project or wandb_env["WANDB_PROJECT"]
    config = {
        "paper_final_scope_limited_execution": True,
        "row": {key: value for key, value in row.items() if _primitive(value)},
        "result": {key: value for key, value in result.items() if _primitive(value)},
        "metadata": {key: value for key, value in metadata.items() if _primitive(value)},
        "status": {key: value for key, value in status.items() if _primitive(value)},
        "wandb_layout_scope": layout_scope,
    }

    init_kwargs = {
        "project": selected_project,
        "group": identity["group"],
        "name": identity["name"],
        "tags": identity["tags"],
        "job_type": PAPER_FINAL_WANDB_JOB_TYPE,
        "config": config,
        "reinit": True,
    }
    selected_entity = entity or wandb_env.get("WANDB_ENTITY")
    if selected_entity:
        init_kwargs["entity"] = selected_entity
    settings = _wandb_settings(wandb, wandb_env)
    if settings is not None:
        init_kwargs["settings"] = settings

    attempts = _positive_int_env(wandb_env, "WANDB_UPLOAD_RETRIES") or 1
    retry_delay = _positive_int_env(wandb_env, "WANDB_UPLOAD_RETRY_DELAY_SECONDS") or 5
    last_error: Exception | None = None
    saved_files: list[str] = []
    completed_attempt = 0
    for attempt in range(1, attempts + 1):
        saved_files = []
        completed_attempt = attempt
        try:
            wandb.init(**init_kwargs)
            try:
                for step, metric in enumerate(train_metrics):
                    payload = _prefixed_primitives("train", metric)
                    if payload:
                        wandb.log(payload, step=step)
                status_payload = _prefixed_primitives("status", status)
                if status_payload:
                    wandb.log(status_payload, step=len(train_metrics))
                for name in _ARTIFACT_FILE_NAMES:
                    path = out_dir / name
                    if path.is_file() and hasattr(wandb, "save"):
                        wandb.save(path.as_posix(), base_path=out_dir.as_posix(), policy="now")
                        saved_files.append(name)
            finally:
                wandb.finish()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            try:
                wandb.finish()
            except Exception:
                pass
            if attempt >= attempts:
                raise
            time.sleep(retry_delay)
    if last_error is not None:
        raise last_error

    return {
        "status": "uploaded",
        "row_index": row.get("row_index"),
        "project": selected_project,
        "entity": selected_entity,
        "run_name": identity["name"],
        "group": identity["group"],
        "tags": identity["tags"],
        "layout_scope": layout_scope,
        "metric_rows_logged": len(train_metrics),
        "files_saved": saved_files,
        "attempt": completed_attempt,
    }


def upload_paper_final_learned_eval_to_wandb(
    row: dict[str, Any],
    *,
    project: str | None = None,
    entity: str | None = None,
    wandb_module: object | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    wandb_env = paper_final_wandb_env(base_env=env)
    selected_project = project or wandb_env["WANDB_PROJECT"]
    if wandb_env.get("WANDB_MODE", PAPER_FINAL_WANDB_MODE).lower() == "disabled":
        return {
            "status": "skipped_disabled",
            "row_index": row.get("row_index"),
            "project": selected_project,
        }

    eval_out_dir = Path(str(row.get("eval_out_dir") or row.get("out_dir")))
    if not eval_out_dir.is_dir():
        raise ValueError(f"paper-final learned eval W&B upload requires existing eval_out_dir: {eval_out_dir}")
    metadata = _load_json(eval_out_dir / "paper_final_learned_eval_metadata.json")
    status = _load_json(eval_out_dir / "paper_final_learned_eval_status.json")
    metrics = _load_reference_schema_metrics(eval_out_dir / "paper_final_learned_eval_metrics.json")
    layout_scope = wandb_env.get("WANDB_LAYOUT_SCOPE", PAPER_FINAL_WANDB_LAYOUT_SCOPE)
    row_index = int(row.get("row_index", 0))
    method = _safe_text(row.get("method") or metadata.get("method") or "unknown_method")
    city = _safe_text(row.get("city") or metadata.get("city") or "unknown_city")
    seed = _safe_text(row.get("seed") if row.get("seed") is not None else metadata.get("seed_id", "unknown_seed"))
    preference = _safe_text(
        row.get("preference_template") or metadata.get("fixed_preference_template") or "default"
    )
    identity = {
        "name": f"learned_eval__{city}__{method}__seed{seed}__{preference}__row{row_index:03d}",
        "group": f"{layout_scope}/learned_eval/{city}/{method}/{preference}",
        "tags": _scope_tags(layout_scope) + ["learned_eval", city, method, f"seed_{seed}", preference],
    }
    selected_entity = entity or wandb_env.get("WANDB_ENTITY") or None
    wandb = _wandb_module(wandb_module)
    init_kwargs = {
        "project": selected_project,
        "group": identity["group"],
        "name": identity["name"],
        "tags": identity["tags"],
        "job_type": "paper_final_scope_limited_learned_eval",
        "config": {
            "paper_final_scope_limited_learned_eval": True,
            "row": {key: value for key, value in row.items() if _primitive(value)},
            "metadata": {key: value for key, value in metadata.items() if _primitive(value)},
            "status": {key: value for key, value in status.items() if _primitive(value)},
            "wandb_layout_scope": layout_scope,
        },
        "reinit": True,
    }
    if selected_entity:
        init_kwargs["entity"] = selected_entity
    settings = _wandb_settings(wandb, wandb_env)
    if settings is not None:
        init_kwargs["settings"] = settings

    wandb.init(**init_kwargs)
    saved_files: list[str] = []
    try:
        wandb.log(metrics, step=0)
        status_payload = _prefixed_primitives("eval_status", status)
        if status_payload:
            wandb.log(status_payload, step=1)
        for name in _LEARNED_EVAL_ARTIFACT_FILE_NAMES:
            path = eval_out_dir / name
            if path.is_file() and hasattr(wandb, "save"):
                wandb.save(path.as_posix(), base_path=eval_out_dir.as_posix(), policy="now")
                saved_files.append(name)
    finally:
        wandb.finish()

    return {
        "status": "uploaded",
        "row_index": row.get("row_index"),
        "project": selected_project,
        "entity": selected_entity,
        "run_name": identity["name"],
        "group": identity["group"],
        "tags": identity["tags"],
        "layout_scope": layout_scope,
        "metric_keys": list(metrics),
        "files_saved": saved_files,
    }
