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
from pareto.rl.formal_preflight_checks import run_preflight_checks
from pareto.train_common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--checks_only", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real_records")
    parser.add_argument("--num_sensitivity_records", type=int, default=64)
    parser.add_argument("--min_real_mean_delta", type=float, default=1e-4)
    parser.add_argument("--min_real_max_delta", type=float, default=1e-3)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if not args.checks_only:
        raise ValueError("formal preflight runner currently supports --checks_only only")
    specs = [load_formal_experiment_spec(path) for path in args.specs]
    report = run_preflight_checks(
        specs,
        root=args.root,
        device=args.device,
        real_records_path=args.real_records,
        num_sensitivity_records=args.num_sensitivity_records,
        min_real_mean_delta=args.min_real_mean_delta,
        min_real_max_delta=args.min_real_max_delta,
    )
    report.update(
        {
            "spec_paths": [str(Path(path)) for path in args.specs],
            "root": str(Path(args.root)),
            "runner": "formal_preflight_runner",
        }
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "preflight_checks.json", report)
    write_json(
        out_dir / "metadata.json",
        {
            "checks_only": True,
            "env_rollout": False,
            "ppo_training": False,
            "policy_update": False,
            "performance_claim": False,
            "formal_experiment_allowed": False,
            "passed": bool(report["passed"]),
            "real_records": args.real_records,
        },
    )
    (out_dir / "PREFLIGHT_CHECKS_DONE").write_text("done\n", encoding="utf-8")
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
