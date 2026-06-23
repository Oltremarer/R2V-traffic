#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config
from pareto.rl.formal_ppo_trainer import run_synthetic_ppo_dry_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("formal PPO dry-run requires --dry_run; pilot execution is not allowed")
    config = load_formal_ppo_dryrun_config(args.spec)
    payload = run_synthetic_ppo_dry_run(config, args.method, args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
