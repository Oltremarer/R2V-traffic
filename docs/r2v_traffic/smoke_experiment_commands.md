# Smoke Experiment Commands

Use the experiment machine with CityFlow data and transition buffers.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export DEVICE="${DEVICE:-cuda}"
export SMOKE_ROOT="records/r2v_traffic_runs/smoke_jinan_seed0"
export TRANSITIONS="records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl"
export R2V_DIFFUSION_SCORE_ARTIFACT_SEED0="${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0:-${SMOKE_ROOT}/artifacts/diffusion_seed0_proxy_scores.jsonl}"
```

## Baseline dry-run

First run readiness:

```bash
"${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
  --scenario jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --transition_glob "${TRANSITIONS}" \
  --seed 0 \
  --output "${SMOKE_ROOT}/readiness_baseline.json"
```

```bash
"${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
  --dry_run \
  --python_bin "${PYTHON_BIN}" \
  --transition_glob "${TRANSITIONS}" \
  --output_root "${SMOKE_ROOT}/baseline" \
  --r2v off \
  --r2v_mode traffic \
  --generative_backend diffusion \
  --repair_story not_rare_to_val \
  --repair_metadata_policy metadata_or_proxy \
  --gate_variant full \
  --r2v_sampling_mode off \
  --rare_fraction 0.2 \
  --force
```

Pass: status is `DRY_RUN_READY`; command plan has no `build_r2v_weighted_transitions`.

## R2V dry-run

For a true diffusion-backed smoke, set `R2V_DIFFUSION_SCORE_ARTIFACT_SEED0` to an artifact produced by the diffusion scorer. If you only need integration smoke, build a proxy artifact first:

```bash
"${PYTHON_BIN}" -m pareto.r2v.build_generative_score_artifact \
  --transitions ${TRANSITIONS} \
  --output "${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0}" \
  --summary_output "${SMOKE_ROOT}/artifacts/diffusion_seed0_proxy_scores_summary.json" \
  --backend diffusion \
  --adapter traffic_feature_density_proxy
```

Proxy artifacts have `paper_claim_eligible=false`; use them for readiness and command wiring only, not for paper diffusion claims.

Then check readiness:

```bash
"${PYTHON_BIN}" -m pareto.r2v.experiment_readiness \
  --scenario jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --transition_glob "${TRANSITIONS}" \
  --seed 0 \
  --require_diffusion_artifacts \
  --diffusion_artifact 0:"${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0}" \
  --output "${SMOKE_ROOT}/readiness_r2v_diffusion.json"
```

```bash
"${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
  --dry_run \
  --python_bin "${PYTHON_BIN}" \
  --transition_glob "${TRANSITIONS}" \
  --output_root "${SMOKE_ROOT}/r2v_diffusion_not_rare_to_val_full" \
  --r2v on \
  --r2v_mode traffic \
  --generative_backend diffusion \
  --r2v_artifact_path "${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0}" \
  --repair_story not_rare_to_val \
  --repair_metadata_policy metadata_or_proxy \
  --gate_variant full \
  --r2v_sampling_mode full_r2v \
  --rare_fraction 0.2 \
  --force
```

Pass: status is `DRY_RUN_READY`; command plan begins with `build_r2v_weighted_transitions`.
Readiness also verifies the diffusion score artifact has finite scores and covers the transition IDs matched by `TRANSITIONS`.

## Explicit external diffusion score artifact

```bash
export R2V_DIFFUSION_SCORE_ARTIFACT_SEED0="${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0:?set diffusion score artifact path}"

"${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
  --dry_run \
  --python_bin "${PYTHON_BIN}" \
  --transition_glob "${TRANSITIONS}" \
  --output_root "${SMOKE_ROOT}/r2v_diffusion_artifact" \
  --r2v on \
  --generative_backend diffusion \
  --r2v_artifact_path "${R2V_DIFFUSION_SCORE_ARTIFACT_SEED0}" \
  --repair_story not_rare_to_val \
  --repair_metadata_policy require_metadata \
  --gate_variant full \
  --r2v_sampling_mode full_r2v \
  --force
```

Without a real diffusion-produced artifact, the run is proxy-backed smoke and should not be reported as a true diffusion result.
