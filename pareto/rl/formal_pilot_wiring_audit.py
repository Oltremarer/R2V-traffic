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

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS
from pareto.common.io import write_json
from pareto.rl.formal_pilot_runner import (
    FORMAL_PILOT_WIRING_APPROVAL_PHRASE,
    FORMAL_PILOT_WIRING_METHOD,
)


FORMAL_PILOT_WIRING_ALLOWED_OUTPUTS = {
    "metadata.json",
    "status.json",
    "command.txt",
    "ppo_config.json",
    "formal_gate_decision.json",
    "pilot_spec.json",
    "reward_components.jsonl",
    "loss_debug.jsonl",
    "train_metrics.jsonl",
    "action_debug.jsonl",
    "checkpoint_last.pt",
    "training_checkpoint_last.pt",
}
FORMAL_PILOT_WIRING_ALLOWED_DIRS = {"llmlight_work"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scan_outputs(run_dir: Path) -> tuple[list[str], list[str]]:
    forbidden: list[str] = []
    non_allowlisted: list[str] = []
    for path in run_dir.iterdir():
        name = path.name
        if path.is_dir():
            if name not in FORMAL_PILOT_WIRING_ALLOWED_DIRS:
                non_allowlisted.append(name)
            continue
        if name in FORBIDDEN_PREFLIGHT_ARTIFACTS or (path.suffix == ".tex" and "performance" in name):
            forbidden.append(name)
        if name not in FORMAL_PILOT_WIRING_ALLOWED_OUTPUTS:
            non_allowlisted.append(name)
    return sorted(set(forbidden)), sorted(set(non_allowlisted))


def _reward_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row.get("total_reward", row.get("reward")) for row in rows]
    finite = bool(values) and all(_finite_number(value) for value in values)
    nonzero = finite and any(abs(float(value)) > 1e-12 for value in values)
    return {
        "pass": bool(finite and nonzero),
        "row_count": len(rows),
        "finite": bool(finite),
        "nonzero": bool(nonzero),
    }


def _loss_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for row in rows:
        for key, value in row.items():
            if key.endswith("loss") or key in {"approx_kl", "clip_fraction", "ratio_mean", "grad_norm"}:
                checked += 1
                if not _finite_number(value):
                    return {"pass": False, "row_count": len(rows), "finite_value_count": checked}
    return {"pass": bool(rows), "row_count": len(rows), "finite_value_count": checked}


def _scope_guard(metadata: dict[str, Any]) -> dict[str, Any]:
    expected_false = [
        "formal_experiment",
        "performance_claim",
        "traffic_result_value_reading_executed",
        "method_ranking_executed",
        "paper_result_claim",
        "seed_expansion_allowed",
        "city_expansion_allowed",
    ]
    failures = [key for key in expected_false if bool(metadata.get(key, False))]
    requires_new_approval = bool(metadata.get("formal_experiment_requires_new_pro_approval", False))
    return {
        "pass": not failures and requires_new_approval,
        "false_flag_failures": failures,
        "formal_experiment_requires_new_pro_approval": requires_new_approval,
    }


def build_formal_pilot_wiring_guard_audit(
    run_dir: str | Path,
    *,
    guard_commit: str,
    representation_commit: str,
    execution_commit: str,
) -> dict[str, Any]:
    run = Path(run_dir)
    metadata = _read_json(run / "metadata.json")
    status = _read_json(run / "status.json")
    reward_rows = _read_jsonl(run / "reward_components.jsonl")
    loss_rows = _read_jsonl(run / "loss_debug.jsonl")
    forbidden, non_allowlisted = _scan_outputs(run)
    action_guard = metadata.get("action_guard", {})

    closed_loop = {
        "pass": bool(
            metadata.get("formal_pilot_wiring_dry_run")
            and metadata.get("real_env_rollout")
            and metadata.get("cityflow_step_called")
            and metadata.get("real_ppo_update")
            and int(status.get("policy_update_count", metadata.get("policy_update_count", 0))) >= 1
        ),
        "status": status.get("status"),
        "policy_update_count": int(status.get("policy_update_count", metadata.get("policy_update_count", 0))),
    }
    vectorq = {
        "pass": bool(
            metadata.get("reward_adapter") == FORMAL_PILOT_WIRING_METHOD
            and metadata.get("vector_model_loaded")
            and metadata.get("vector_model_hash_verified")
            and metadata.get("representation_run_id")
        ),
        "reward_adapter": metadata.get("reward_adapter"),
        "representation_run_id": metadata.get("representation_run_id"),
        "vector_model_dir": metadata.get("vector_model_dir"),
        "vector_model_hash_verified": bool(metadata.get("vector_model_hash_verified", False)),
        "state_encoder_hash": metadata.get("state_encoder_hash"),
        "obs_dim": metadata.get("obs_dim"),
    }
    action = {
        "pass": bool(
            int(action_guard.get("unique_actions_used", 0)) > 1
            and float(action_guard.get("global_single_action_rate", 1.0))
            <= float(action_guard.get("max_single_action_rate_allowed", 0.95))
        ),
        "unique_actions_used": int(action_guard.get("unique_actions_used", 0)),
        "global_single_action_rate": float(action_guard.get("global_single_action_rate", 1.0)),
        "threshold": float(action_guard.get("max_single_action_rate_allowed", 0.95)),
    }
    checkpoint = {
        "pass": bool(metadata.get("checkpoint_load_verified") and metadata.get("checkpoint_valid")),
        "checkpoint_load_verified": bool(metadata.get("checkpoint_load_verified", False)),
        "checkpoint_valid": bool(metadata.get("checkpoint_valid", False)),
    }
    artifacts = {
        "pass": not forbidden and not non_allowlisted,
        "forbidden": forbidden,
        "non_allowlisted": non_allowlisted,
        "allowed_outputs": sorted(FORMAL_PILOT_WIRING_ALLOWED_OUTPUTS),
        "allowed_dirs": sorted(FORMAL_PILOT_WIRING_ALLOWED_DIRS),
    }
    audit = {
        "approval_phrase": FORMAL_PILOT_WIRING_APPROVAL_PHRASE,
        "run_dir": str(run),
        "provenance": {
            "guard_commit": guard_commit,
            "representation_commit": representation_commit,
            "execution_commit": execution_commit,
        },
        "closed_loop_executed": closed_loop,
        "vectorq_wiring": vectorq,
        "reward_finite_nonzero": _reward_guard(reward_rows),
        "loss_finite": _loss_guard(loss_rows),
        "action_non_collapse": action,
        "checkpoint_roundtrip": checkpoint,
        "artifact_allowlist": artifacts,
        "forbidden_artifact_scan": {"pass": not forbidden, "forbidden": forbidden},
        "scope_flags": _scope_guard(metadata),
        "forbidden_interpretation": [
            "traffic performance values",
            "method comparison",
            "ranking",
            "best method",
            "traffic improvement",
            "formal result",
            "paper result",
        ],
    }
    checks = [
        audit["closed_loop_executed"]["pass"],
        audit["vectorq_wiring"]["pass"],
        audit["reward_finite_nonzero"]["pass"],
        audit["loss_finite"]["pass"],
        audit["action_non_collapse"]["pass"],
        audit["checkpoint_roundtrip"]["pass"],
        audit["artifact_allowlist"]["pass"],
        audit["forbidden_artifact_scan"]["pass"],
        audit["scope_flags"]["pass"],
    ]
    audit["overall_pass"] = bool(all(checks))
    return audit


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Formal-Pilot Wiring Dry-Run Guard Audit",
        "",
        f"- overall_pass: `{audit['overall_pass']}`",
        f"- approval_phrase: `{audit['approval_phrase']}`",
        f"- run_dir: `{audit['run_dir']}`",
        "",
        "## Guard Checks",
    ]
    for key in (
        "closed_loop_executed",
        "vectorq_wiring",
        "reward_finite_nonzero",
        "loss_finite",
        "action_non_collapse",
        "checkpoint_roundtrip",
        "artifact_allowlist",
        "scope_flags",
    ):
        lines.append(f"- {key}: `{'PASS' if audit[key]['pass'] else 'FAIL'}`")
    lines.extend(
        [
            "",
            "No traffic performance values, method ranking, best-method claim, traffic-improvement claim, or paper-result claim are reported in this audit.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--guard_commit", required=True)
    parser.add_argument("--representation_commit", required=True)
    parser.add_argument("--execution_commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_formal_pilot_wiring_guard_audit(
        args.run_dir,
        guard_commit=args.guard_commit,
        representation_commit=args.representation_commit,
        execution_commit=args.execution_commit,
    )
    write_json(args.out_json, audit)
    write_markdown(args.out_md, audit)
    print(json.dumps({"overall_pass": audit["overall_pass"]}, indent=2, sort_keys=True))
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
