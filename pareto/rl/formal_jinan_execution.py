#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig, load_formal_ppo_dryrun_config
from pareto.rl.formal_run_plan import FORMAL_EXECUTION_APPROVAL_PHRASE, REQUIRED_ALLOWED_OUTPUTS, load_formal_run_plan


DEPRECATED_FORMAL_JINAN_EXECUTION_MESSAGE = (
    "formal_jinan_execution.py is deprecated for formal Jinan execution; use "
    "pareto/rl/formal_pilot_runner.py --formal_jinan_3seed_execution with the "
    "current FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE"
)
FORMAL_ROOT_ALLOWED_OUTPUTS = set(REQUIRED_ALLOWED_OUTPUTS) | {"formal_run_plan.json", "formal_run_plan.md"}
FORMAL_JINAN_DONE_STATUS = "FORMAL_JINAN_3SEED_RUN_DONE"
FORMAL_ROOT_TRANSIENT_OUTPUTS = {
    "BOUNDED_JINAN_PILOT_DRY_RUN_DONE",
    "action_debug.jsonl",
    "formal_gate_decision.json",
    "pilot_spec.json",
    "ppo_config.json",
}
FORMAL_ROOT_TRANSIENT_DIRS = {"llmlight_work"}


def validate_formal_jinan_execution_request(
    run_plan_path: str | Path,
    *,
    method: str,
    seed_id: int,
    approval_phrase: str,
) -> dict[str, Any]:
    raise ValueError(DEPRECATED_FORMAL_JINAN_EXECUTION_MESSAGE)
    if approval_phrase != FORMAL_EXECUTION_APPROVAL_PHRASE:
        raise ValueError("formal execution requires the exact Pro approval phrase")
    plan = load_formal_run_plan(run_plan_path).to_dict()
    candidate_seeds = [int(value) for value in plan["seed_binding"]["candidate_seeds"]]
    if int(seed_id) not in candidate_seeds:
        raise ValueError(f"seed_id {seed_id} is not an approved seed: {candidate_seeds}")
    methods = list(plan["methods"]["ppo_methods"])
    if method not in methods:
        raise ValueError(f"method {method} is not an approved method: {methods}")
    return {
        "approval_phrase": approval_phrase,
        "candidate_seeds": candidate_seeds,
        "method": method,
        "plan": plan,
        "run_plan_path": str(run_plan_path),
        "seed_id": int(seed_id),
    }


def prepare_seed_bound_config(config: FormalPPODryRunConfig, *, seed_id: int) -> FormalPPODryRunConfig:
    payload = config.to_dict()
    payload.pop("source_path", None)
    payload["pilot"] = dict(payload["pilot"])
    payload["pilot"]["cityflow_seed"] = int(seed_id)
    payload["pilot"]["policy_seed"] = int(seed_id)
    payload["pilot"]["model_seed"] = int(seed_id)
    return FormalPPODryRunConfig.from_dict(payload, source_path=config.source_path)


def build_formal_execution_context(request: dict[str, Any], *, packet_commit: str | None = None) -> dict[str, Any]:
    seed_id = int(request["seed_id"])
    return {
        "formal_jinan_3seed_execution": True,
        "formal_experiment": True,
        "pilot_only": False,
        "pilot_dry_run_execution": False,
        "bounded_jinan_1seed_pilot_dry_run": False,
        "bounded_pilot_not_formal": False,
        "performance_claim": False,
        "not_for_main_results": True,
        "exclude_from_analysis": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "checkpoint_use": "formal_jinan_3seed_guarded_execution_last_only",
        "result_mode": "formal_jinan_3seed_guarded_execution_raw",
        "status_label": FORMAL_JINAN_DONE_STATUS,
        "pro_approval_phrase_verified": True,
        "pro_approval_phrase": FORMAL_EXECUTION_APPROVAL_PHRASE,
        "formal_run_plan_path": str(request["run_plan_path"]),
        "formal_run_plan_packet_commit": packet_commit,
        "cityflow_seed": seed_id,
        "policy_seed": seed_id,
        "model_seed": seed_id,
    }


def finalize_formal_jinan_seed_outputs(run_dir: str | Path) -> None:
    root = Path(run_dir)
    for name in FORMAL_ROOT_TRANSIENT_OUTPUTS:
        path = root / name
        if path.exists():
            path.unlink()
    for name in FORMAL_ROOT_TRANSIENT_DIRS:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
    assert_no_forbidden_performance_artifacts(root)
    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in FORMAL_ROOT_ALLOWED_OUTPUTS
    )
    if unexpected:
        raise ValueError(f"formal Jinan seed run produced non-allowlisted root artifacts: {unexpected}")


def _copy_plan_artifacts(run_plan_path: str | Path, run_plan_doc: str | Path | None, out_dir: Path) -> None:
    shutil.copyfile(run_plan_path, out_dir / "formal_run_plan.json")
    if run_plan_doc is not None and Path(run_plan_doc).exists():
        shutil.copyfile(run_plan_doc, out_dir / "formal_run_plan.md")
    else:
        (out_dir / "formal_run_plan.md").write_text(
            f"# Formal Jinan 3-Seed Run Plan\n\nSource JSON: `{run_plan_path}`\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_spec", required=True)
    parser.add_argument("--run_plan", required=True)
    parser.add_argument("--run_plan_doc")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed_id", type=int, required=True)
    parser.add_argument("--approval_phrase", required=True)
    parser.add_argument("--packet_commit")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_decision_steps_per_episode", type=int, default=120)
    parser.add_argument("--objective_normalizer", required=True)
    parser.add_argument("--objective_normalizer_hash", required=True)
    parser.add_argument("--film_model_dir")
    parser.add_argument("--film_model_hash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = validate_formal_jinan_execution_request(
        args.run_plan,
        method=args.method,
        seed_id=args.seed_id,
        approval_phrase=args.approval_phrase,
    )
    config = prepare_seed_bound_config(load_formal_ppo_dryrun_config(args.base_spec), seed_id=args.seed_id)
    context = build_formal_execution_context(request, packet_commit=args.packet_commit)
    from pareto.rl.formal_pilot_runner import run_bounded_jinan_pilot_dry_run

    out_dir = Path(args.out_dir)
    payload = run_bounded_jinan_pilot_dry_run(
        config,
        args.method,
        out_dir,
        episodes=args.episodes,
        max_decision_steps_per_episode=args.max_decision_steps_per_episode,
        objective_normalizer=args.objective_normalizer,
        objective_normalizer_hash=args.objective_normalizer_hash,
        film_model_dir=args.film_model_dir,
        film_model_hash=args.film_model_hash,
        device=args.device,
        approved_seed_ids=tuple(request["candidate_seeds"]),
        formal_execution_context=context,
    )
    _copy_plan_artifacts(args.run_plan, args.run_plan_doc, out_dir)
    finalize_formal_jinan_seed_outputs(out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
