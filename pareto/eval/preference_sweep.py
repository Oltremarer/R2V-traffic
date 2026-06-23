#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_experiment_spec import load_formal_experiment_spec


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


PAPER_FINAL_PREFERENCE_SWEEP_APPROVAL_PHRASE = "PPTS PARETO PPO PAPER FINAL PREFERENCE SWEEP EXECUTION GO"


def validate_paper_final_preference_sweep_request(
    request: dict,
    *,
    approval_phrase: str | None,
) -> dict:
    if approval_phrase != PAPER_FINAL_PREFERENCE_SWEEP_APPROVAL_PHRASE:
        raise ValueError("paper-final preference sweep requires exact reviewer approval")
    required = {
        "paper_final_manifest",
        "city",
        "traffic_file",
        "method",
        "seed",
        "preference_id",
        "metric_source_policy",
    }
    missing = sorted(required - set(request))
    if missing:
        raise ValueError(f"preference sweep request missing fields: {missing}")
    if request.get("output_root_empty") is not True:
        raise ValueError("preference sweep output root must be empty before execution")
    if request.get("deterministic_policy_loading") is not True:
        raise ValueError("preference sweep requires deterministic policy loading")
    if request.get("action_diagnostics_enabled") is not True:
        raise ValueError("preference sweep requires action diagnostics")
    return {
        **dict(request),
        "dry_run": False,
        "approval_phrase": approval_phrase,
        "ranking_generated": False,
        "plot_generated": False,
        "paper_table_generated": False,
    }


def run_dry_run(args: argparse.Namespace) -> dict:
    spec = load_formal_experiment_spec(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": True,
        "env_rollout": False,
        "performance_claim": False,
        "preference_sweep": False,
        "scenario": spec.scenario,
        "traffic_file": spec.traffic_file,
        "eval_preferences": spec.eval_preferences,
        "preference_sampling": spec.preference_sampling,
        "spec_hash": spec.spec_hash(),
    }
    write_json(out_dir / "metadata.json", payload)
    write_json(out_dir / "status.json", {"status": "SWEEP_DRY_RUN_DONE"})
    (out_dir / "SWEEP_DRY_RUN_DONE").write_text("preference sweep dry run only; no env rollout\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--execute_guarded", action="store_true")
    parser.add_argument("--paper_final_manifest")
    parser.add_argument("--approval_phrase")
    parser.add_argument("--city")
    parser.add_argument("--traffic_file")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preference_id")
    parser.add_argument("--metric_source_policy")
    parser.add_argument("--output_root_empty", action="store_true")
    parser.add_argument("--deterministic_policy_loading", action="store_true")
    parser.add_argument("--action_diagnostics_enabled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute_guarded:
        request = {
            "paper_final_manifest": args.paper_final_manifest,
            "city": args.city,
            "traffic_file": args.traffic_file,
            "method": args.method,
            "seed": args.seed,
            "preference_id": args.preference_id,
            "output_root_empty": bool(args.output_root_empty),
            "metric_source_policy": args.metric_source_policy,
            "deterministic_policy_loading": bool(args.deterministic_policy_loading),
            "action_diagnostics_enabled": bool(args.action_diagnostics_enabled),
        }
        validate_paper_final_preference_sweep_request(request, approval_phrase=args.approval_phrase)
        raise SystemExit("paper-final preference sweep executor is guarded; execution requires a later runner packet")
    if not args.dry_run:
        raise SystemExit("preference sweep run is not implemented or allowed; pass --dry_run")
    print(json.dumps(run_dry_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
