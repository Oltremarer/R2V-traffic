#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.paper_final_eval_aggregation import (
    DEFAULT_AGGREGATION_OUT_DIR,
    build_paper_final_eval_aggregation,
    write_paper_final_eval_aggregation_outputs,
)
from pareto.rl.paper_final_learned_eval_runner import (
    PAPER_FINAL_LEARNED_EVAL_DONE,
    default_learned_eval_out_dir,
)
from pareto.rl.paper_final_reference_runner import REFERENCE_METRIC_KEYS


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_set(value: str | Iterable[str]) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _safe_text(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _training_row_complete(row: dict[str, Any]) -> bool:
    out_dir = ROOT / str(row.get("out_dir"))
    status_path = out_dir / "status.json"
    if not status_path.is_file() or not (out_dir / "checkpoint_last.pt").is_file():
        return False
    status = _read_json(status_path)
    return (
        status.get("status") == "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE"
        and status.get("fixed_preference_template") == row.get("preference_template")
    )


def _learned_eval_complete(eval_out_dir: str | Path) -> bool:
    out_dir = ROOT / str(eval_out_dir)
    status_path = out_dir / "paper_final_learned_eval_status.json"
    metrics_path = out_dir / "paper_final_learned_eval_metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return False
    status = _read_json(status_path)
    if status.get("status") != PAPER_FINAL_LEARNED_EVAL_DONE:
        return False
    metrics = _read_json(metrics_path)
    return all(key in metrics for key in REFERENCE_METRIC_KEYS)


def _learned_training_rows(spec: dict[str, Any], *, excluded_cities: set[str]) -> list[tuple[int, dict[str, Any]]]:
    rows = spec.get("rows")
    if not isinstance(rows, list):
        raise ValueError("paper-final executable spec rows must be a list")
    learned_rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get("runner_family") != "formal_pilot_paper_final":
            continue
        if row.get("status") != "executable_preview":
            continue
        if row.get("city") in excluded_cities:
            continue
        learned_rows.append((index, row))
    return learned_rows


def _eval_command_for_row(
    *,
    source_row_index: int,
    row: dict[str, Any],
    eval_out_dir: str,
    python_bin: str,
    device: str,
) -> list[str]:
    return [
        python_bin,
        "pareto/rl/paper_final_learned_eval_runner.py",
        "--spec",
        str(row["spec_path"]),
        "--method",
        str(row["method"]),
        "--seed_id",
        str(row["seed"]),
        "--train_dir",
        str(row["out_dir"]),
        "--eval_out_dir",
        eval_out_dir,
        "--fixed_preference_template",
        str(row["preference_template"]),
        "--device",
        device,
        "--row_index",
        str(source_row_index),
        "--execute",
    ]


def build_learned_eval_backfill_plan(
    *,
    spec_json: str | Path,
    python_bin: str = sys.executable,
    device: str = "cpu",
    excluded_cities: str | Iterable[str] = (),
) -> dict[str, Any]:
    spec_path = Path(spec_json)
    spec = _read_json(spec_path)
    excluded = _csv_set(excluded_cities)
    plan_rows: list[dict[str, Any]] = []
    for source_row_index, row in _learned_training_rows(spec, excluded_cities=excluded):
        eval_out_dir = default_learned_eval_out_dir(
            city=str(row["city"]),
            traffic_file=str(row["traffic_file"]),
            method=str(row["method"]),
            seed_id=int(row["seed"]),
            fixed_preference_template=str(row["preference_template"]),
        )
        base = {
            "source_row_index": source_row_index,
            "method": row["method"],
            "city": row["city"],
            "traffic_file": row["traffic_file"],
            "seed": int(row["seed"]),
            "preference_template": row["preference_template"],
            "train_dir": row["out_dir"],
            "eval_out_dir": eval_out_dir,
        }
        if not _training_row_complete(row):
            plan_rows.append({**base, "status": "blocked_training_incomplete", "command_argv": []})
            continue
        if _learned_eval_complete(eval_out_dir):
            plan_rows.append({**base, "status": "already_complete", "command_argv": []})
            continue
        plan_rows.append(
            {
                **base,
                "status": "pending_eval",
                "command_argv": _eval_command_for_row(
                    source_row_index=source_row_index,
                    row=row,
                    eval_out_dir=eval_out_dir,
                    python_bin=python_bin,
                    device=device,
                ),
            }
        )
    return {
        "packet_type": "paper_final_learned_eval_backfill_plan",
        "spec_json": str(spec_json),
        "device": device,
        "excluded_cities": sorted(excluded),
        "rows": plan_rows,
        "counts": {
            "expected_eval_rows": len(plan_rows),
            "rows_to_run": sum(1 for row in plan_rows if row["status"] == "pending_eval"),
            "already_complete": sum(1 for row in plan_rows if row["status"] == "already_complete"),
            "blocked_training_incomplete": sum(
                1 for row in plan_rows if row["status"] == "blocked_training_incomplete"
            ),
        },
    }


def _run_eval_row(row: dict[str, Any], *, log_root: Path) -> dict[str, Any]:
    label = (
        f"{int(row['source_row_index']):03d}_"
        f"{_safe_text(row['method'])}_{_safe_text(row['city'])}_"
        f"seed{row['seed']}_{_safe_text(row['preference_template'])}"
    )
    stdout_path = log_root / f"{label}.stdout.txt"
    stderr_path = log_root / f"{label}.stderr.txt"
    argv = [str(item) for item in row["command_argv"]]
    env = dict(os.environ)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    return {
        "source_row_index": row["source_row_index"],
        "method": row["method"],
        "city": row["city"],
        "seed": row["seed"],
        "preference_template": row["preference_template"],
        "train_dir": row["train_dir"],
        "eval_out_dir": row["eval_out_dir"],
        "returncode": completed.returncode,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "command_argv": argv,
    }


def execute_learned_eval_backfill_plan(
    plan: dict[str, Any],
    *,
    audit_root: str | Path,
    parallelism: int = 4,
) -> dict[str, Any]:
    audit = ROOT / str(audit_root)
    log_root = audit / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit / "learned_eval_backfill_plan.json", plan)
    pending = [row for row in plan["rows"] if row["status"] == "pending_eval"]
    results: list[dict[str, Any]] = []
    if pending:
        if parallelism <= 1:
            for row in pending:
                result = _run_eval_row(row, log_root=log_root)
                results.append(result)
                _write_json(audit / "learned_eval_execution_progress.json", results)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
                future_map = {pool.submit(_run_eval_row, row, log_root=log_root): row for row in pending}
                for future in concurrent.futures.as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    results.sort(key=lambda item: int(item["source_row_index"]))
                    _write_json(audit / "learned_eval_execution_progress.json", results)
    failed = [result for result in results if result["returncode"] != 0]
    incomplete = [
        row
        for row in plan["rows"]
        if row["status"] != "blocked_training_incomplete" and not _learned_eval_complete(row["eval_out_dir"])
    ]
    blocked_training = [row for row in plan["rows"] if row["status"] == "blocked_training_incomplete"]
    status = "PASS" if not failed and not incomplete and not blocked_training else "BLOCKED"
    postflight = {
        "packet_type": "paper_final_learned_eval_backfill_postflight",
        "status": status,
        "counts": {
            **plan["counts"],
            "executed": len(results),
            "failed": len(failed),
            "incomplete_after_execution": len(incomplete),
        },
        "failed_rows": failed,
        "incomplete_rows": incomplete,
        "blocked_training_rows": blocked_training,
        "results": results,
    }
    _write_json(audit / "learned_eval_execution_results.json", results)
    _write_json(audit / "learned_eval_postflight.json", postflight)
    return postflight


def _plan_cities(plan: dict[str, Any]) -> tuple[str, ...]:
    cities: list[str] = []
    for row in plan["rows"]:
        city = str(row["city"])
        if city not in cities:
            cities.append(city)
    return tuple(cities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_json", required=True)
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--excluded_cities", default="")
    parser.add_argument("--aggregation_out_dir", default=DEFAULT_AGGREGATION_OUT_DIR)
    parser.add_argument("--aggregation_cities", default="")
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_learned_eval_backfill_plan(
        spec_json=args.spec_json,
        python_bin=args.python_bin,
        device=args.device,
        excluded_cities=args.excluded_cities,
    )
    audit_root = ROOT / str(args.audit_root)
    _write_json(audit_root / "learned_eval_backfill_plan.json", plan)
    if args.plan_only:
        print(json.dumps({"plan_only": True, "counts": plan["counts"]}, indent=2, sort_keys=True))
        return
    postflight = execute_learned_eval_backfill_plan(
        plan,
        audit_root=args.audit_root,
        parallelism=int(args.parallelism),
    )
    aggregation_cities = (
        tuple(part.strip() for part in args.aggregation_cities.split(",") if part.strip())
        if args.aggregation_cities
        else _plan_cities(plan)
    )
    aggregation = build_paper_final_eval_aggregation(cities=aggregation_cities)
    write_paper_final_eval_aggregation_outputs(aggregation, args.aggregation_out_dir)
    _write_json(
        audit_root / "learned_eval_aggregation_summary.json",
        {
            "aggregation_out_dir": args.aggregation_out_dir,
            "aggregation_ready": aggregation["aggregation_ready"],
            "counts": aggregation["counts"],
        },
    )
    payload = {
        "learned_eval_status": postflight["status"],
        "learned_eval_counts": postflight["counts"],
        "aggregation_ready": aggregation["aggregation_ready"],
        "aggregation_counts": aggregation["counts"],
        "aggregation_out_dir": args.aggregation_out_dir,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_complete and (postflight["status"] != "PASS" or not aggregation["aggregation_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
