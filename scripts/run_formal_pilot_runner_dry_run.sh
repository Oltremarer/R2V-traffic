#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-film_scalar_potential}"
OUT_DIR="${2:-records/formal_pilot_runner_dryrun/${METHOD}}"

python pareto/rl/formal_pilot_runner.py \
  --spec configs/formal/jinan_1seed_film_pilot_dryrun.json \
  --method "${METHOD}" \
  --mock_env \
  --episodes 1 \
  --max_decision_steps_per_episode 1 \
  --out_dir "${OUT_DIR}"
