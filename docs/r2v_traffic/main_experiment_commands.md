# Main Experiment Commands

Main comparison:

- `baseline_uniform`
- `r2v_diffusion_not_rare_to_val_full`

Seeds: `0 1 2`.

Generate the complete ordered main plan as a shell script:

```bash
python3 -m pareto.r2v.traffic_experiment_plan \
  --plan main_pipeline \
  --format shell \
  --output records/r2v_traffic_runs/main_jinan_3seed/main_pipeline.sh \
  --python_bin "${PYTHON_BIN:-python3}" \
  --transition_glob 'records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed{seed}/transitions_raw.jsonl' \
  --diffusion_artifact_template 'records/r2v_traffic/diffusion_seed{seed}_scores.jsonl' \
  --output_root records/r2v_traffic_runs/main_jinan_3seed \
  --main_seeds 0,1,2
```

The generated script orders strict paper readiness, baseline/R2V main runs, paper-table readiness, final aggregation, and paper artifact manifesting. Use `--format json` for a machine-readable manifest with the same commands.
Add `--include_validation` to the JSON export when you want a machine-readable structural check of command order, baseline/R2V flags, and artifact-manifest coverage.

For final paper runs, prefer the generated `main_pipeline` script. Its strict readiness commands and R2V runner commands both require `repair_metadata_policy=require_metadata`, so proxy repair metadata cannot slip into the main evidence chain. The manual loop below can still be used for integration smoke when `R2V_REPAIR_METADATA_POLICY=metadata_or_proxy`.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export DEVICE="${DEVICE:-cuda}"
export MAIN_ROOT="records/r2v_traffic_runs/main_jinan_3seed"
export TRANSITION_ROOT="records/paper_final_data_buffers/paper_final_20260602_v1/jinan"
export SEEDS="0 1 2"
export WANDB_PROJECT="R2V-Traffic"
export R2V_REPAIR_METADATA_POLICY="${R2V_REPAIR_METADATA_POLICY:-require_metadata}"
```

## Baseline

Run readiness for each seed before launching training:

```bash
for seed in $SEEDS; do
  "${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
    --scenario jinan \
    --traffic_file anon_3_4_jinan_real.json \
    --transition_glob "${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl" \
    --seed "${seed}" \
    --output "${MAIN_ROOT}/seed${seed}/baseline_readiness.json"
done
```

```bash
for seed in $SEEDS; do
  "${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
    --python_bin "${PYTHON_BIN}" \
    --transition_glob "${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl" \
    --output_root "${MAIN_ROOT}/seed${seed}/baseline_uniform" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --r2v off \
    --r2v_sampling_mode off \
    --repair_story not_rare_to_val \
    --repair_metadata_policy metadata_or_proxy \
    --gate_variant full
done
```

## R2V full

```bash
for seed in $SEEDS; do
  score_var="R2V_DIFFUSION_SCORE_ARTIFACT_SEED${seed}"
  score_artifact="${!score_var:-}"
  if [ -z "$score_artifact" ]; then
    if [ "$R2V_REPAIR_METADATA_POLICY" = "require_metadata" ]; then
      echo "Set ${score_var} to a real paper-eligible diffusion artifact, or use R2V_REPAIR_METADATA_POLICY=metadata_or_proxy for integration smoke." >&2
      exit 2
    fi
    score_artifact="${MAIN_ROOT}/seed${seed}/artifacts/diffusion_seed${seed}_proxy_scores.jsonl"
    "${PYTHON_BIN}" -m pareto.r2v.build_generative_score_artifact \
      --transitions ${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl \
      --output "${score_artifact}" \
      --summary_output "${MAIN_ROOT}/seed${seed}/artifacts/diffusion_seed${seed}_proxy_scores_summary.json" \
      --backend diffusion \
      --adapter traffic_feature_density_proxy
  fi

  "${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
    --scenario jinan \
    --traffic_file anon_3_4_jinan_real.json \
    --transition_glob "${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl" \
    --seed "${seed}" \
    --require_diffusion_artifacts \
    --diffusion_artifact "${seed}:${score_artifact}" \
    --output "${MAIN_ROOT}/seed${seed}/r2v_readiness.json"

  "${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
    --python_bin "${PYTHON_BIN}" \
    --transition_glob "${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl" \
    --output_root "${MAIN_ROOT}/seed${seed}/r2v_diffusion_not_rare_to_val_full" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --r2v on \
    --r2v_mode traffic \
    --generative_backend diffusion \
    --r2v_artifact_path "${score_artifact}" \
    --repair_story not_rare_to_val \
    --repair_metadata_policy "${R2V_REPAIR_METADATA_POLICY}" \
    --gate_variant full \
    --r2v_sampling_mode full_r2v \
    --rare_fraction 0.2
done
```

If `R2V_DIFFUSION_SCORE_ARTIFACT_SEED{seed}` is not set and `R2V_REPAIR_METADATA_POLICY=metadata_or_proxy`, the script builds a proxy score artifact with `paper_claim_eligible=false`; label that run as proxy-backed R2V, not final diffusion-backed R2V. With the default `require_metadata` policy, missing real diffusion artifacts fail fast.
For diffusion-backed runs, readiness checks that the artifact has finite rarity/support scores and covers the transition IDs for that seed. If the artifact contains true repaired source/final gates, set `R2V_REPAIR_METADATA_POLICY=require_metadata` for strict paper runs.

For strict paper diffusion readiness, run this after setting real diffusion artifacts:

```bash
for seed in $SEEDS; do
  score_var="R2V_DIFFUSION_SCORE_ARTIFACT_SEED${seed}"
  score_artifact="${!score_var:?set real diffusion score artifact path}"
  "${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
    --scenario jinan \
    --traffic_file anon_3_4_jinan_real.json \
    --transition_glob "${TRANSITION_ROOT}/*/seed${seed}/transitions_raw.jsonl" \
    --seed "${seed}" \
    --require_diffusion_artifacts \
    --require_paper_claim_eligible_diffusion \
    --repair_metadata_policy require_metadata \
    --require_strict_repair_metadata_policy \
    --diffusion_artifact "${seed}:${score_artifact}" \
    --output "${MAIN_ROOT}/seed${seed}/r2v_paper_readiness.json"
done
```

Strict paper readiness requires `paper_claim_eligible=true`, non-empty `model_checkpoint`, `config_hash`, and `normalization_id`, plus a non-proxy adapter for every score row. It also requires `repair_metadata_policy=require_metadata`, so proxy repair metadata cannot be used for final paper diffusion claims.

## W&B

Use project `R2V-Traffic`, group `jinan-main`, and names:

- `main__jinan__baseline_uniform__seed{seed}`;
- `main__jinan__r2v_diffusion_not_rare_to_val_full__seed{seed}`.

## Paper Table Readiness

Before aggregating the main table:

```bash
"${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
  --no-require_cityflow_data \
  --performance_path "${MAIN_ROOT}/aggregation/r2v_performance_rows.jsonl" \
  --require_performance_metrics \
  --expected_performance_method baseline_uniform \
  --expected_performance_method r2v_diffusion_not_rare_to_val_full \
  --expected_performance_seed 0 \
  --expected_performance_seed 1 \
  --expected_performance_seed 2 \
  --require_completed_performance_status \
  --output "${MAIN_ROOT}/aggregation/performance_readiness.json"
```

This check requires exactly the baseline/R2V method coverage over seeds 0, 1, and 2, all five traffic metrics, and completed evaluation status. It still keeps status separate from performance metrics.

After readiness passes, aggregate performance and integrity artifacts separately:

```bash
"${PYTHON_BIN}" -m pareto.r2v.result_aggregation \
  --performance_path "${MAIN_ROOT}/aggregation/r2v_performance_rows.jsonl" \
  --integrity_path "${MAIN_ROOT}/seed0/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --integrity_path "${MAIN_ROOT}/seed1/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --integrity_path "${MAIN_ROOT}/seed2/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --output "${MAIN_ROOT}/aggregation/r2v_result_aggregation.json"
```

Only the rows in `r2v_performance_rows.jsonl` feed performance means. The `r2v_summary.json` files are integrity/status summaries.

Finally, freeze the paper evidence bundle with hashes:

```bash
"${PYTHON_BIN}" -m pareto.r2v.paper_artifact_manifest \
  --artifact performance:main_performance_rows:"${MAIN_ROOT}/aggregation/r2v_performance_rows.jsonl" \
  --artifact aggregation:main_result_aggregation:"${MAIN_ROOT}/aggregation/r2v_result_aggregation.json" \
  --artifact readiness:main_performance_readiness:"${MAIN_ROOT}/aggregation/performance_readiness.json" \
  --artifact diffusion_score:seed0_diffusion_scores:"${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0:?set real diffusion score artifact path}" \
  --artifact diffusion_score:seed1_diffusion_scores:"${R2V_DIFFUSION_SCORE_ARTIFACT_SEED1:?set real diffusion score artifact path}" \
  --artifact diffusion_score:seed2_diffusion_scores:"${R2V_DIFFUSION_SCORE_ARTIFACT_SEED2:?set real diffusion score artifact path}" \
  --artifact integrity:seed0_r2v_summary:"${MAIN_ROOT}/seed0/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --artifact integrity:seed1_r2v_summary:"${MAIN_ROOT}/seed1/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --artifact integrity:seed2_r2v_summary:"${MAIN_ROOT}/seed2/r2v_diffusion_not_rare_to_val_full/artifacts/r2v_summary.json" \
  --output "${MAIN_ROOT}/aggregation/paper_artifact_manifest.json"
```

This manifest is an evidence index, not a metric table. It records file hashes and keeps performance artifacts separate from integrity/status artifacts. A file labeled `performance` must contain recognized traffic metrics; status-only rows are blocked. A file labeled `diffusion_score` must be real paper-eligible diffusion evidence: `paper_claim_eligible=true`, non-proxy adapter, and non-empty `model_checkpoint`, `config_hash`, and `normalization_id` on every row. Proxy score artifacts are blocked here.
