#!/usr/bin/env bash
set -euo pipefail

python pareto/rl/formal_preflight_rollout.py \
  --spec configs/formal/film_jinan_preflight.json \
  --preflight_only \
  --episodes 2 \
  --max_decision_steps_per_episode 10 \
  --preflight_checks_json records/formal_preflight/checks_real_records/preflight_checks.json \
  --out_dir records/formal_preflight_rollout/film

python pareto/rl/formal_preflight_rollout.py \
  --spec configs/formal/weighted_proxy_jinan_preflight.json \
  --preflight_only \
  --episodes 2 \
  --max_decision_steps_per_episode 10 \
  --preflight_checks_json records/formal_preflight/checks_real_records/preflight_checks.json \
  --out_dir records/formal_preflight_rollout/weighted_proxy

python pareto/rl/formal_preflight_rollout.py \
  --spec configs/formal/env_reward_jinan_preflight.json \
  --preflight_only \
  --episodes 2 \
  --max_decision_steps_per_episode 10 \
  --preflight_checks_json records/formal_preflight/checks_real_records/preflight_checks.json \
  --out_dir records/formal_preflight_rollout/env_reward
