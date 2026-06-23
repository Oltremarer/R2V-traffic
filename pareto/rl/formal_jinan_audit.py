#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import write_json
from pareto.rl.formal_jinan_execution import FORMAL_ROOT_ALLOWED_OUTPUTS
from pareto.rl.formal_pilot_runner import FINAL_JINAN_PILOT_DISPLAY_NAMES, FINAL_JINAN_PILOT_METHODS


APPROVED_FORMAL_JINAN_SEEDS = (0, 1, 2)
FORMAL_JINAN_TRAFFIC_FILE = "anon_3_4_jinan_real.json"
FORMAL_JINAN_SCENARIO = "jinan"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return True


def _check_checkpoint_loads(run_dir: Path) -> tuple[bool, str | None]:
    try:
        import torch

        torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
        torch.load(run_dir / "training_checkpoint_last.pt", map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - guard audit reports exact failure text.
        return False, str(exc)
    return True, None


def _root_artifact_failures(run_dir: Path) -> list[str]:
    failures: list[str] = []
    try:
        assert_no_forbidden_performance_artifacts(run_dir)
    except ValueError as exc:
        failures.append(str(exc))
    existing = {path.name for path in run_dir.iterdir()}
    unexpected = sorted(existing - FORMAL_ROOT_ALLOWED_OUTPUTS)
    if unexpected:
        failures.append(f"non-allowlisted root artifacts: {unexpected}")
    missing = sorted(FORMAL_ROOT_ALLOWED_OUTPUTS - existing)
    if missing:
        failures.append(f"missing allowed root artifacts: {missing}")
    return failures


def _audit_one_run(
    root: Path,
    *,
    seed: int,
    method: str,
    expected_steps: int,
    expected_reward_rows: int,
    check_checkpoints: bool,
) -> tuple[dict[str, Any], list[str]]:
    run_dir = root / f"seed{seed}" / method
    failures: list[str] = []
    if not run_dir.exists():
        return {"seed": seed, "method": method, "exists": False}, [f"missing run directory: seed{seed}/{method}"]
    failures.extend(f"seed{seed}/{method}: {failure}" for failure in _root_artifact_failures(run_dir))

    metadata = _read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    status = _read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
    train_rows = _read_jsonl(run_dir / "train_metrics.jsonl") if (run_dir / "train_metrics.jsonl").exists() else []
    reward_rows = _read_jsonl(run_dir / "reward_components.jsonl") if (run_dir / "reward_components.jsonl").exists() else []
    loss_rows = _read_jsonl(run_dir / "loss_debug.jsonl") if (run_dir / "loss_debug.jsonl").exists() else []

    prefix = f"seed{seed}/{method}"
    expected_display = FINAL_JINAN_PILOT_DISPLAY_NAMES[method]
    if metadata.get("formal_jinan_3seed_execution") is not True:
        failures.append(f"{prefix}: formal_jinan_3seed_execution flag missing")
    if metadata.get("formal_experiment") is not True:
        failures.append(f"{prefix}: formal_experiment flag missing")
    if metadata.get("performance_claim") is not False:
        failures.append(f"{prefix}: performance_claim must be false")
    if metadata.get("method_ranking_allowed") is not False:
        failures.append(f"{prefix}: method_ranking_allowed must be false")
    if metadata.get("performance_table_allowed") is not False:
        failures.append(f"{prefix}: performance_table_allowed must be false")
    if metadata.get("pro_approval_phrase_verified") is not True:
        failures.append(f"{prefix}: Pro approval phrase was not verified")
    if metadata.get("method") != method:
        failures.append(f"{prefix}: method metadata mismatch")
    if metadata.get("method_display_name") != expected_display:
        failures.append(f"{prefix}: method_display_name mismatch")
    if metadata.get("scenario") != FORMAL_JINAN_SCENARIO or metadata.get("traffic_file") != FORMAL_JINAN_TRAFFIC_FILE:
        failures.append(f"{prefix}: scenario or traffic file mismatch")
    if (
        metadata.get("cityflow_seed") != seed
        or metadata.get("policy_seed") != seed
        or metadata.get("model_seed") != seed
    ):
        failures.append(f"{prefix}: seed binding mismatch")
    if int(metadata.get("episodes", -1)) <= 0 or int(metadata.get("max_decision_steps_per_episode", -1)) <= 0:
        failures.append(f"{prefix}: missing episode or horizon metadata")
    if int(metadata.get("min_action_time", -1)) != 30:
        failures.append(f"{prefix}: min_action_time mismatch")
    if status.get("steps") != expected_steps:
        failures.append(f"{prefix}: step budget mismatch")
    if status.get("reward_row_count") != expected_reward_rows:
        failures.append(f"{prefix}: reward row budget mismatch")
    if len(train_rows) != expected_steps:
        failures.append(f"{prefix}: train_metrics row count mismatch")
    if len(reward_rows) != expected_reward_rows:
        failures.append(f"{prefix}: reward_components row count mismatch")
    if not loss_rows:
        failures.append(f"{prefix}: loss_debug is empty")
    if not _all_numbers_finite(train_rows) or not _all_numbers_finite(reward_rows) or not _all_numbers_finite(loss_rows):
        failures.append(f"{prefix}: non-finite numeric value in logs")
    if "Traceback" in (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="ignore"):
        failures.append(f"{prefix}: stderr contains Traceback")
    if method == "env_reward":
        env_summary = metadata.get("env_reward_summary") or {}
        if env_summary.get("finite") is not True:
            failures.append(f"{prefix}: EnvReward summary is not finite")
        if env_summary.get("all_zero_reward") is not False:
            failures.append(f"{prefix}: EnvReward all-zero guard failed")
        if env_summary.get("env_reward_sources") != ["cityflow_average_reward"]:
            failures.append(f"{prefix}: EnvReward source mismatch")
        if metadata.get("reward_adapter_semantics") != "queue_length_penalty_proxy":
            failures.append(f"{prefix}: EnvReward semantic lock mismatch")
    checkpoint_loads = None
    checkpoint_error = None
    if check_checkpoints:
        checkpoint_loads, checkpoint_error = _check_checkpoint_loads(run_dir)
        if not checkpoint_loads:
            failures.append(f"{prefix}: checkpoint load failed: {checkpoint_error}")

    summary = {
        "seed": seed,
        "method": method,
        "exists": True,
        "status": status.get("status"),
        "steps": status.get("steps"),
        "reward_row_count": status.get("reward_row_count"),
        "policy_update_count": status.get("policy_update_count"),
        "train_metrics_rows": len(train_rows),
        "reward_components_rows": len(reward_rows),
        "loss_debug_rows": len(loss_rows),
        "checkpoint_loads": checkpoint_loads,
        "root_artifact_count": len(list(run_dir.iterdir())),
        "performance_claim": metadata.get("performance_claim"),
        "method_ranking_allowed": metadata.get("method_ranking_allowed"),
        "performance_table_allowed": metadata.get("performance_table_allowed"),
    }
    return summary, failures


def audit_formal_jinan_runs(
    root: str | Path,
    *,
    expected_steps: int = 600,
    expected_reward_rows: int = 7200,
    check_checkpoints: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    run_summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for seed in APPROVED_FORMAL_JINAN_SEEDS:
        for method in FINAL_JINAN_PILOT_METHODS:
            summary, run_failures = _audit_one_run(
                root_path,
                seed=seed,
                method=method,
                expected_steps=expected_steps,
                expected_reward_rows=expected_reward_rows,
                check_checkpoints=check_checkpoints,
            )
            run_summaries.append(summary)
            failures.extend(run_failures)

    budget_tuples = {
        (summary.get("steps"), summary.get("reward_row_count"), summary.get("policy_update_count"))
        for summary in run_summaries
    }
    budget_consistent = len(budget_tuples) == 1
    if not budget_consistent:
        failures.append(f"budget mismatch across runs: {sorted(budget_tuples)}")

    report = {
        "report_status": "FORMAL_JINAN_3SEED_GUARD_FAIL" if failures else "FORMAL_JINAN_3SEED_GUARD_PASS",
        "scope": "formal_jinan_3seed_guard_audit_only_no_ranking_no_performance_table",
        "performance_claim": False,
        "ranking_generated": False,
        "performance_table_generated": False,
        "approved_seeds": list(APPROVED_FORMAL_JINAN_SEEDS),
        "approved_methods": list(FINAL_JINAN_PILOT_METHODS),
        "expected_steps_per_run": int(expected_steps),
        "expected_reward_rows_per_run": int(expected_reward_rows),
        "budget_consistent": budget_consistent,
        "runs": run_summaries,
        "failures": failures,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_steps", type=int, default=600)
    parser.add_argument("--expected_reward_rows", type=int, default=7200)
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_formal_jinan_runs(
        args.root,
        expected_steps=args.expected_steps,
        expected_reward_rows=args.expected_reward_rows,
        check_checkpoints=not args.skip_checkpoint_load,
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
