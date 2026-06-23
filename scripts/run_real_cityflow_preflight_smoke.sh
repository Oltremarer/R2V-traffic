#!/usr/bin/env bash
set -euo pipefail

SPEC="${SPEC:-configs/formal/jinan_1seed_film_pilot_dryrun.json}"
ROOT_OUT="${ROOT_OUT:-records/real_cityflow_preflight}"
STEPS="${STEPS:-3}"

for METHOD in film_scalar_potential weighted_proxy env_reward; do
  python pareto/rl/real_cityflow_preflight_smoke.py \
    --spec "$SPEC" \
    --method "$METHOD" \
    --episodes 1 \
    --max_decision_steps_per_episode "$STEPS" \
    --out_dir "$ROOT_OUT/$METHOD" \
    --real_env_preflight
done
