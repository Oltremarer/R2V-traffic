#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
RECORDS_ROOT="${RECORDS_ROOT:-data/pareto_records_split_norm/jinan/smoke3600}"
BASE_PAIRS_ROOT="${BASE_PAIRS_ROOT:-data/pareto_pairs/jinan/smoke3600_rich}"
FINAL_PAIRS_ROOT="${FINAL_PAIRS_ROOT:-data/pareto_pairs/jinan/smoke3600_rich_v2}"
VECTOR_ROOT="${VECTOR_ROOT:-model_weights/pareto_quality/jinan/preformal_final}"
SCALAR_ROOT="${SCALAR_ROOT:-model_weights/cond_scalar/jinan/preformal_final}"
RECORDS_OUT="${RECORDS_OUT:-records/preformal_final}"
REPORT_OUT="${REPORT_OUT:-docs/pro_reviews/preformal_vectorq_final_2026-05-30.md}"

mkdir -p "$VECTOR_ROOT" "$SCALAR_ROOT" "$RECORDS_OUT" "$(dirname "$REPORT_OUT")"

build_split_pairs() {
  local split="$1"
  local seed="$2"
  local objective_pairs="$3"
  local preference_pairs="$4"
  local dominance_pairs="$5"
  local reversal_pairs="$6"
  local min_es_conflict="$7"
  local min_es_reversal="$8"

  "$PYTHON_BIN" pareto/data/build_pairs.py \
    --buffers "$RECORDS_ROOT/${split}_raw.jsonl" \
    --out_dir "$FINAL_PAIRS_ROOT/$split" \
    --split "$split" \
    --num_objective_pairs "$objective_pairs" \
    --num_preference_pairs "$preference_pairs" \
    --num_dominance_pairs "$dominance_pairs" \
    --num_reversal_pairs "$reversal_pairs" \
    --min_efficiency_stability_conflict "$min_es_conflict" \
    --reversal_template_quota "efficiency__stability:$min_es_reversal" \
    --seed "$seed"
}

if [[ ! -d "$FINAL_PAIRS_ROOT/train" ]]; then
  build_split_pairs train 0 2400 2400 500 600 80 80
  build_split_pairs val 1 1000 1000 180 200 25 25
  build_split_pairs test 2 1000 1000 180 200 25 25
fi

"$PYTHON_BIN" pareto/data/validate_pairs.py \
  --pairs_dir "$FINAL_PAIRS_ROOT/train" \
  --report "$RECORDS_OUT/pair_report_train.json" \
  --strict \
  --min_objective_per_head 400 \
  --min_preference_pairs 2000 \
  --min_dominance_pairs 400 \
  --min_reversal_pairs 500 \
  --min_eff_controlled_stability 100 \
  --min_efficiency_stability_conflict 80 \
  --min_reversal_template_pair efficiency__stability:60 \
  --positive_ratio_low 0.30 \
  --positive_ratio_high 0.70 \
  --positive_ratio_by_objective_low 0.25 \
  --positive_ratio_by_objective_high 0.75 \
  --require_no_ties

for split in val test; do
  "$PYTHON_BIN" pareto/data/validate_pairs.py \
    --pairs_dir "$FINAL_PAIRS_ROOT/$split" \
    --report "$RECORDS_OUT/pair_report_${split}.json" \
    --strict \
    --min_objective_per_head 150 \
    --min_preference_pairs 700 \
    --min_dominance_pairs 120 \
    --min_reversal_pairs 150 \
    --min_eff_controlled_stability 30 \
    --min_efficiency_stability_conflict 20 \
    --min_reversal_template_pair efficiency__stability:20 \
    --positive_ratio_low 0.25 \
    --positive_ratio_high 0.75 \
    --positive_ratio_by_objective_low 0.20 \
    --positive_ratio_by_objective_high 0.80 \
    --require_no_ties
done

COMMON_VECTOR_ARGS=(
  --records_root "$RECORDS_ROOT"
  --pairs_root "$FINAL_PAIRS_ROOT"
  --epochs 20
  --batch_size 128
  --seed 0
  --device "$DEVICE"
  --hidden_dim 128
  --num_layers 3
  --dropout 0.1
  --lr 0.001
  --training_schedule joint
  --architecture residual_tower
  --tower_residual_alpha 0.5
  --reversal_sampler template_balanced
  --reversal_template_min_count 20
  --reversal_loss_weight 1.0
  --dominance_coord_loss_weight 0.3
  --dominance_utility_loss_weight 0.5
  --dominance_margin 0.1
  --calibration_loss_weight 0.01
  --pref_margin_loss_weight 0.25
  --rev_margin_loss_weight 0.25
  --pref_hinge_loss_weight 0.25
  --rev_hinge_loss_weight 0.25
  --classification_margin 0.5
  --margin_clip 2.0
  --score_mode low_rank_interaction
  --interaction_rank 4
  --interaction_beta 0.3
)

run_vector() {
  local name="$1"
  shift
  "$PYTHON_BIN" pareto/train_vector_quality.py \
    "${COMMON_VECTOR_ARGS[@]}" \
    --output_dir "$VECTOR_ROOT/$name" \
    "$@"
}

run_vector lowrank
run_vector lowrank_iso \
  --isotonic_dominance_weight 0.25 \
  --isotonic_margin_floor 0.05 \
  --use_objective_margins_for_dominance

if [[ -d "$BASE_PAIRS_ROOT/train" ]]; then
  "$PYTHON_BIN" pareto/train_vector_quality.py \
    "${COMMON_VECTOR_ARGS[@]}" \
    --pairs_root "$BASE_PAIRS_ROOT" \
    --output_dir "$VECTOR_ROOT/lowrank_base_pairs"
fi

"$PYTHON_BIN" pareto/train_conditioned_scalar.py \
  --records_root "$RECORDS_ROOT" \
  --pairs_root "$FINAL_PAIRS_ROOT" \
  --output_dir "$SCALAR_ROOT/film_rich_v2" \
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
  --pref_hinge_loss_weight 0.25 \
  --rev_hinge_loss_weight 0.25 \
  --classification_margin 0.5 \
  --margin_clip 2.0

for name in lowrank lowrank_iso lowrank_base_pairs; do
  if [[ ! -d "$VECTOR_ROOT/$name" ]]; then
    continue
  fi
  "$PYTHON_BIN" pareto/eval/run_offline_diagnostics.py \
    --records_root "$RECORDS_ROOT" \
    --pairs_root "$FINAL_PAIRS_ROOT" \
    --vector_model_dir "$VECTOR_ROOT/$name" \
    --scalar_model_dir "$SCALAR_ROOT/film_rich_v2" \
    --out "$RECORDS_OUT/${name}_vs_film_diagnostics.json" \
    --device "$DEVICE"

  "$PYTHON_BIN" pareto/eval/offline_pair_bootstrap.py \
    --records_root "$RECORDS_ROOT" \
    --pairs_root "$FINAL_PAIRS_ROOT" \
    --split test \
    --vector_model_dir "$VECTOR_ROOT/$name" \
    --scalar_model_dir "$SCALAR_ROOT/film_rich_v2" \
    --out "$RECORDS_OUT/${name}_vs_film_pair_bootstrap_test.json" \
    --device "$DEVICE" \
    --n_boot 1000 \
    --seed 0
done

"$PYTHON_BIN" pareto/eval/offline_model_selection.py \
  --run_dirs "$VECTOR_ROOT/lowrank" "$VECTOR_ROOT/lowrank_iso" \
  --out "$RECORDS_OUT/model_selection_report.json"

BEST_VECTOR_DIR="$VECTOR_ROOT/lowrank"
if [[ -f "$RECORDS_OUT/model_selection_report.json" ]]; then
  BEST_VECTOR_DIR="$("$PYTHON_BIN" - <<'PY' "$RECORDS_OUT/model_selection_report.json" "$VECTOR_ROOT"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
root = sys.argv[2]
best = report.get("best_run_dir") or report.get("best_run") or f"{root}/lowrank"
print(best)
PY
)"
fi

"$PYTHON_BIN" scripts/check_formal_gate.py \
  --vector_metrics "$BEST_VECTOR_DIR/diagnostics_test.json" \
  --film_metrics "$SCALAR_ROOT/film_rich_v2/diagnostics_test.json" \
  --out "$RECORDS_OUT/formal_gate_decision.json"

cat > "$REPORT_OUT" <<EOF
# Final Preformal VectorQ Review Packet

Generated by \`scripts/run_preformal_vectorq_final.sh\`.

- records_root: \`$RECORDS_ROOT\`
- pairs_root: \`$FINAL_PAIRS_ROOT\`
- vector_root: \`$VECTOR_ROOT\`
- scalar_root: \`$SCALAR_ROOT\`
- records_out: \`$RECORDS_OUT\`

Review files:

- \`$RECORDS_OUT/model_selection_report.json\`
- \`$RECORDS_OUT/formal_gate_decision.json\`
- \`$RECORDS_OUT/*_vs_film_diagnostics.json\`
- \`$RECORDS_OUT/*_vs_film_pair_bootstrap_test.json\`
- \`$RECORDS_OUT/pair_report_train.json\`
- \`$RECORDS_OUT/pair_report_val.json\`
- \`$RECORDS_OUT/pair_report_test.json\`

Formal PPO remains blocked unless \`formal_gate_decision.json\` reports
\`ppo_formal_allowed=true\`.
EOF

echo "Final preformal packet written to $REPORT_OUT"
