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

from pareto.common.io import write_json
from pareto.rl.formal_analysis_plan import (
    FORMAL_ANALYSIS_APPROVAL_PHRASE,
    FormalAnalysisPlan,
    load_formal_analysis_plan,
)


NO_RANKING_ANALYSIS_PASS = "FORMAL_JINAN_NO_RANKING_ANALYSIS_PASS"
NO_RANKING_ANALYSIS_FAIL = "FORMAL_JINAN_NO_RANKING_ANALYSIS_FAIL"
ALLOWED_ANALYSIS_OUTPUTS = {
    "formal_analysis_packet.md",
    "guard_audit_summary.json",
    "training_stability_sanity.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _root_matches_allowed(root: Path, allowed_roots: list[str]) -> bool:
    root_norm = root.as_posix().rstrip("/")
    root_abs = root.resolve().as_posix().rstrip("/") if root.exists() else root_norm
    for allowed in allowed_roots:
        allowed_norm = Path(allowed).as_posix().rstrip("/")
        if root_norm == allowed_norm or root_abs == allowed_norm:
            return True
        if root_norm.endswith("/" + allowed_norm) or root_abs.endswith("/" + allowed_norm):
            return True
        if root.name == allowed_norm:
            return True
    return False


def _validate_approval_and_plan(
    *,
    root: Path,
    guard_audit_json: Path,
    analysis_plan: Path,
    approval_phrase: str,
) -> FormalAnalysisPlan:
    if approval_phrase != FORMAL_ANALYSIS_APPROVAL_PHRASE:
        raise ValueError("missing exact Pro approval phrase for formal no-ranking analysis")
    plan = load_formal_analysis_plan(analysis_plan)
    payload = plan.to_dict()
    allowed_roots = [str(item) for item in (payload.get("inputs") or {}).get("allowed_future_input_roots") or []]
    if allowed_roots and not _root_matches_allowed(root, allowed_roots):
        raise ValueError(f"root is not in allowed future input roots: {root}")
    expected_audit = str((payload.get("inputs") or {}).get("guard_audit_json") or "")
    if expected_audit:
        audit_norm = guard_audit_json.as_posix()
        if audit_norm != expected_audit and not audit_norm.endswith("/" + expected_audit):
            raise ValueError(f"guard audit json does not match approved plan input: {guard_audit_json}")
    return plan


def _guard_audit_summary(guard_audit_json: Path) -> dict[str, Any]:
    audit = _read_json(guard_audit_json)
    runs = audit.get("runs") or []
    checkpoint_loads = [run.get("checkpoint_loads") is True for run in runs]
    summary = {
        "guard_pass_fail": audit.get("report_status"),
        "budget_consistency": bool(audit.get("budget_consistent") is True),
        "failure_count": len(audit.get("failures") or []),
        "run_count": len(runs),
        "checkpoint_load_status": {
            "all_loaded": bool(checkpoint_loads and all(checkpoint_loads)),
            "loaded_count": sum(1 for value in checkpoint_loads if value),
            "checked_count": len(checkpoint_loads),
        },
    }
    return summary


def _run_training_sanity(root: Path, audit_runs: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    run_summaries: list[dict[str, Any]] = []
    env_reward_source_ok = True
    env_reward_nonzero_ok = True

    for audit_run in audit_runs:
        seed = audit_run.get("seed")
        method = audit_run.get("method")
        run_dir = root / f"seed{seed}" / str(method)
        prefix = f"seed{seed}/{method}"
        metadata = _read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
        status = _read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
        loss_rows = _read_jsonl(run_dir / "loss_debug.jsonl")
        train_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
        reward_rows = _read_jsonl(run_dir / "reward_components.jsonl")

        finite_loss = bool(loss_rows) and _all_numbers_finite(loss_rows)
        finite_train = _all_numbers_finite(train_rows)
        finite_reward = _all_numbers_finite(reward_rows)
        stderr_has_traceback = False
        stderr_path = run_dir / "stderr.txt"
        if stderr_path.exists():
            stderr_has_traceback = "Traceback" in stderr_path.read_text(encoding="utf-8", errors="ignore")

        if not finite_loss:
            failures.append(f"{prefix}: loss_debug is empty or non-finite")
        if not finite_train:
            failures.append(f"{prefix}: train_metrics contains non-finite numeric values")
        if not finite_reward:
            failures.append(f"{prefix}: reward_components contains non-finite numeric values")
        if stderr_has_traceback:
            failures.append(f"{prefix}: stderr contains Traceback")

        env_summary = metadata.get("env_reward_summary") or {}
        if method == "env_reward":
            source_ok = env_summary.get("env_reward_sources") == ["cityflow_average_reward"]
            nonzero_ok = env_summary.get("all_zero_reward") is False
            env_reward_source_ok = env_reward_source_ok and source_ok
            env_reward_nonzero_ok = env_reward_nonzero_ok and nonzero_ok
            if not source_ok:
                failures.append(f"{prefix}: EnvReward source status failed")
            if not nonzero_ok:
                failures.append(f"{prefix}: EnvReward nonzero status failed")

        run_summaries.append(
            {
                "seed": seed,
                "method": method,
                "finite_training_logs": finite_loss and finite_train and finite_reward and not stderr_has_traceback,
                "loss_debug_rows": len(loss_rows),
                "train_metrics_rows": len(train_rows),
                "reward_components_rows": len(reward_rows),
                "policy_update_count": status.get("policy_update_count"),
                "checkpoint_load_status": audit_run.get("checkpoint_loads"),
            }
        )

    return {
        "finite_training_logs": not failures,
        "env_reward_source_nonzero_status": {
            "source_ok": env_reward_source_ok,
            "nonzero_ok": env_reward_nonzero_ok,
        },
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "failures": failures,
    }


def _assert_allowed_out_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    unexpected = sorted(path.name for path in out_dir.iterdir() if path.name not in ALLOWED_ANALYSIS_OUTPUTS)
    if unexpected:
        raise ValueError(f"analysis out_dir contains non-allowlisted files: {unexpected}")


def _write_packet(path: Path, *, guard_summary: dict[str, Any], training_sanity: dict[str, Any]) -> None:
    status = NO_RANKING_ANALYSIS_PASS
    if guard_summary.get("failure_count") or not guard_summary.get("budget_consistency") or training_sanity.get("failures"):
        status = NO_RANKING_ANALYSIS_FAIL
    lines = [
        "# Formal Jinan No-Ranking Analysis Packet",
        "",
        f"Status: `{status}`",
        "",
        "Scope: guard and training-stability sanity only. Comparative result outputs remain disabled.",
        "",
        "Allowed checks completed:",
        f"- Guard pass/fail: `{guard_summary.get('guard_pass_fail')}`",
        f"- Budget consistency: `{guard_summary.get('budget_consistency')}`",
        f"- Finite training logs: `{training_sanity.get('finite_training_logs')}`",
        f"- Checkpoint load status: `{guard_summary.get('checkpoint_load_status', {}).get('all_loaded')}`",
        "- EnvReward source/nonzero status: "
        f"`{training_sanity.get('env_reward_source_nonzero_status', {}).get('source_ok')}` / "
        f"`{training_sanity.get('env_reward_source_nonzero_status', {}).get('nonzero_ok')}`",
        "",
        "Generated files:",
        "- `guard_audit_summary.json`",
        "- `training_stability_sanity.json`",
        "- `formal_analysis_packet.md`",
        "",
        "No method ordering, score table, or traffic-control claim is produced in this packet.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_no_ranking_analysis(
    *,
    root: str | Path,
    guard_audit_json: str | Path,
    analysis_plan: str | Path,
    out_dir: str | Path,
    approval_phrase: str,
) -> dict[str, Any]:
    root_path = Path(root)
    audit_path = Path(guard_audit_json)
    plan_path = Path(analysis_plan)
    out_path = Path(out_dir)
    _validate_approval_and_plan(
        root=root_path,
        guard_audit_json=audit_path,
        analysis_plan=plan_path,
        approval_phrase=approval_phrase,
    )
    _assert_allowed_out_dir(out_path)

    audit = _read_json(audit_path)
    guard_summary = _guard_audit_summary(audit_path)
    training_sanity = _run_training_sanity(root_path, audit.get("runs") or [])

    failures = []
    if guard_summary.get("guard_pass_fail") != "FORMAL_JINAN_3SEED_GUARD_PASS":
        failures.append("guard audit did not pass")
    if not guard_summary.get("budget_consistency"):
        failures.append("budget consistency failed")
    failures.extend(training_sanity.get("failures") or [])
    status = NO_RANKING_ANALYSIS_FAIL if failures else NO_RANKING_ANALYSIS_PASS

    report = {
        "report_status": status,
        "scope": "guard_and_training_stability_only_no_method_ordering_no_score_table",
        "allowed_metrics_only": [
            "guard_pass_fail",
            "budget_consistency",
            "finite_training_logs",
            "checkpoint_load_status",
            "env_reward_source_nonzero_status",
        ],
        "guard_summary": guard_summary,
        "training_sanity": training_sanity,
        "failures": failures,
    }

    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "guard_audit_summary.json", guard_summary)
    write_json(out_path / "training_stability_sanity.json", training_sanity)
    _write_packet(out_path / "formal_analysis_packet.md", guard_summary=guard_summary, training_sanity=training_sanity)
    _assert_allowed_out_dir(out_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--guard_audit_json", required=True)
    parser.add_argument("--analysis_plan", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--approval_phrase", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_no_ranking_analysis(
        root=args.root,
        guard_audit_json=args.guard_audit_json,
        analysis_plan=args.analysis_plan,
        out_dir=args.out_dir,
        approval_phrase=args.approval_phrase,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["report_status"] != NO_RANKING_ANALYSIS_PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
