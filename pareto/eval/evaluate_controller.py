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
from pareto.train_common import write_json


def run_dry_run(args: argparse.Namespace) -> dict:
    spec = load_formal_experiment_spec(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": True,
        "env_rollout": False,
        "performance_claim": False,
        "controller_evaluation": False,
        "scenario": spec.scenario,
        "traffic_file": spec.traffic_file,
        "eval_protocol": spec.eval_protocol,
        "eval_preferences": spec.eval_preferences,
        "spec_hash": spec.spec_hash(),
    }
    write_json(out_dir / "metadata.json", payload)
    write_json(out_dir / "status.json", {"status": "EVAL_DRY_RUN_DONE"})
    (out_dir / "EVAL_DRY_RUN_DONE").write_text("controller eval dry run only; no env rollout\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("controller evaluation run is not implemented or allowed; pass --dry_run")
    print(json.dumps(run_dry_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
