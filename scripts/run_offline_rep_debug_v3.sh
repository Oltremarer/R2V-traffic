#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
RECORDS_ROOT="${RECORDS_ROOT:-data/pareto_records_split_norm/jinan/smoke3600}"
PAIRS_ROOT="${PAIRS_ROOT:-data/pareto_pairs/jinan/smoke3600}"
VECTOR_ROOT="${VECTOR_ROOT:-model_weights/pareto_quality/jinan/debug_v3}"
SCALAR_ROOT="${SCALAR_ROOT:-model_weights/cond_scalar/jinan/debug_v3}"
RECORDS_OUT="${RECORDS_OUT:-records/debug_v3}"

mkdir -p "$VECTOR_ROOT" "$SCALAR_ROOT" "$RECORDS_OUT"

COMMON_VECTOR_ARGS=(
  --records_root "$RECORDS_ROOT"
  --pairs_root "$PAIRS_ROOT"
  --epochs 20
  --batch_size 128
  --seed 0
  --device "$DEVICE"
  --hidden_dim 128
  --num_layers 3
  --dropout 0.1
  --lr 0.001
  --training_schedule joint
  --reversal_loss_weight 1.0
  --dominance_coord_loss_weight 0.3
  --dominance_utility_loss_weight 0.5
  --dominance_margin 0.1
  --calibration_loss_weight 0.01
)

run_vector() {
  local name="$1"
  shift
  "$PYTHON_BIN" pareto/train_vector_quality.py \
    "${COMMON_VECTOR_ARGS[@]}" \
    --output_dir "$VECTOR_ROOT/$name" \
    "$@"
}

run_vector shared_template_balanced \
  --architecture shared_mlp \
  --reversal_sampler template_balanced \
  --reversal_template_min_count 20

run_vector shared_margin \
  --architecture shared_mlp \
  --pref_margin_loss_weight 0.25 \
  --rev_margin_loss_weight 0.25 \
  --margin_clip 2.0

run_vector shared_template_balanced_margin \
  --architecture shared_mlp \
  --reversal_sampler template_balanced \
  --reversal_template_min_count 20 \
  --pref_margin_loss_weight 0.25 \
  --rev_margin_loss_weight 0.25 \
  --margin_clip 2.0

run_vector residual_template_balanced_margin \
  --architecture residual_tower \
  --tower_residual_alpha 0.5 \
  --reversal_sampler template_balanced \
  --reversal_template_min_count 20 \
  --pref_margin_loss_weight 0.25 \
  --rev_margin_loss_weight 0.25 \
  --margin_clip 2.0

"$PYTHON_BIN" pareto/train_conditioned_scalar.py \
  --records_root "$RECORDS_ROOT" \
  --pairs_root "$PAIRS_ROOT" \
  --output_dir "$SCALAR_ROOT/film_template_balanced_margin" \
  --epochs 20 \
  --batch_size 128 \
  --seed 0 \
  --device "$DEVICE" \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.1 \
  --lr 0.001 \
  --architecture film \
  --training_schedule joint \
  --preference_loss_weight 1.0 \
  --reversal_loss_weight 1.0 \
  --dominance_loss_weight 0.2 \
  --dominance_margin 0.1 \
  --reversal_sampler template_balanced \
  --reversal_template_min_count 20 \
  --pref_margin_loss_weight 0.25 \
  --rev_margin_loss_weight 0.25 \
  --margin_clip 2.0

for name in \
  shared_template_balanced \
  shared_margin \
  shared_template_balanced_margin \
  residual_template_balanced_margin
do
  "$PYTHON_BIN" pareto/eval/run_offline_diagnostics.py \
    --records_root "$RECORDS_ROOT" \
    --pairs_root "$PAIRS_ROOT" \
    --vector_model_dir "$VECTOR_ROOT/$name" \
    --scalar_model_dir "$SCALAR_ROOT/film_template_balanced_margin" \
    --out "$RECORDS_OUT/${name}_vs_film_diagnostics.json" \
    --device "$DEVICE"

  "$PYTHON_BIN" pareto/eval/model_fairness_report.py \
    --vector_model_dir "$VECTOR_ROOT/$name" \
    --scalar_model_dir "$SCALAR_ROOT/film_template_balanced_margin" \
    --out "$RECORDS_OUT/${name}_model_fairness.json"
done

"$PYTHON_BIN" pareto/eval/offline_model_selection.py \
  --run_dirs \
    "$VECTOR_ROOT/shared_template_balanced" \
    "$VECTOR_ROOT/shared_margin" \
    "$VECTOR_ROOT/shared_template_balanced_margin" \
    "$VECTOR_ROOT/residual_template_balanced_margin" \
  --out "$RECORDS_OUT/model_selection_report.json"
