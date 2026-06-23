#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
RECORDS_ROOT="${RECORDS_ROOT:-data/pareto_records_split_norm/jinan/smoke3600}"
FORMAL_GATE_DECISION="${FORMAL_GATE_DECISION:-records/preformal_final/formal_gate_decision.json}"
OUT_DIR="${OUT_DIR:-records/wiring_only_ppo_smoke/jinan/smoke3600}"
VECTOR_MODEL_DIR="${VECTOR_MODEL_DIR:-model_weights/pareto_quality/jinan/preformal_final/lowrank_iso}"
SCALAR_MODEL_DIR="${SCALAR_MODEL_DIR:-model_weights/cond_scalar/jinan/preformal_final/film}"

args=(
  --records_root "$RECORDS_ROOT"
  --split test
  --out_dir "$OUT_DIR"
  --formal_gate_decision "$FORMAL_GATE_DECISION"
  --device "$DEVICE"
  --max_transitions 8
)

if [[ -d "$VECTOR_MODEL_DIR" ]]; then
  args+=(--vector_model_dir "$VECTOR_MODEL_DIR")
fi

if [[ -d "$SCALAR_MODEL_DIR" ]]; then
  args+=(--scalar_model_dir "$SCALAR_MODEL_DIR")
fi

"$PYTHON_BIN" pareto/rl/ppo_wiring_smoke.py "${args[@]}"
