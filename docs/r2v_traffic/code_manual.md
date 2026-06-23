# R2V-Traffic Code Manual

## Artifact schema

`pareto.r2v.build_r2v_candidates --weighted_output ...` now writes rows with both the legacy v1 weighted-transition metadata and the v2 traffic artifact metadata. The v1 fields keep existing pair sampling compatible; the v2 fields make traffic gates and provenance explicit. When R2V sampling is enabled, `pareto.data.build_pairs` validates the v2 traffic artifact before constructing pairs, so legacy v1-only weighted files are rejected at runtime instead of silently entering training.

Use `pareto.r2v.traffic_artifact_schema.validate_r2v_traffic_artifact(rows)` to validate v2 traffic artifacts. It first validates the existing v1 weighted-transition contract, then checks v2 fields:

- `metadata.r2v_traffic_schema_version`;
- `metadata.r2v_traffic_transition_id`;
- `metadata.r2v_traffic_sample_id`;
- `metadata.r2v_gate_variant`;
- `metadata.r2v_generative_backend`;
- `metadata.r2v_admission_mode`;
- `metadata.r2v_row_role`, one of `source`, `repaired`, or `repair_rejected`;
- `metadata.r2v_repair_rejected`, which must be true only for `repair_rejected` rows;
- `metadata.r2v_repaired_from_transition_id`, required for `repaired` and `repair_rejected` rows;
- `metadata.r2v_traffic_gates` with `rare`, `ood`, `support`, `dynamics`.

The validator requires `metadata.r2v_traffic_transition_id` and `metadata.r2v_traffic_sample_id` to match the row's `transition_id` and `sample_id`; these metadata fields are traceability echoes, not independent IDs.

Use `upgrade_weighted_row_to_v2_metadata(row)` only when wrapping older v1-only rows produced before this change.

## Config

`R2VTrafficConfig` supports:

- `r2v`: `on` or `off`;
- `r2v_mode`: `traffic`;
- `generative_backend`: `diffusion`;
- `repair_story`: `not_rare_to_val` or `not_val_to_val`;
- `repair_metadata_policy`: `require_metadata` or `metadata_or_proxy`;
- `gate_variant`: `full`, `no_support`, `no_ood`, `no_dynamics`;
- `r2v_admission_mode`: `weights_only` or `weights_plus_repaired`;
- `r2v_sampling_mode`: `full_r2v`, `admitted_only`, `rare_only`, `value_only`, `random_same_count`, `same_candidates_random_weights`, `shuffled_value`, `inverted_rarity`, or `off` for baseline.

When `r2v` is off, CLI flags always emit `--r2v_sampling_mode off`.

## Admission Modes

The conservative default is `weights_only`. It preserves the original transition set and only changes R2V sample weights and metadata.

`weights_plus_repaired` is an explicit opt-in interface for repaired-transition replay. It appends admitted repaired rows only when an admitted candidate has a repaired transition object. It may also append explicitly repaired but gate-rejected proposals as `r2v_row_role=repair_rejected`, with `r2v_admitted=false` and `r2v_sample_weight=--repair_rejected_weight`. The payload can come from the external score/repair artifact field `repaired_transition`, or from the source transition at `metadata.r2v_repaired_transition` or the configured `--repaired_transition_key`. If an admitted candidate lacks that repaired payload, candidate building fails closed. This prevents proxy score artifacts from silently pretending to be enhanced repaired datasets.

## Aggregation

`aggregate_r2v_results(performance_paths, integrity_paths)` aggregates only known performance metrics from performance files. Status/integrity data is summarized separately and cannot become a ranking metric.

The output also records `input_artifacts.performance` and `input_artifacts.integrity` entries with each input file's path, SHA256 hash, byte size, and line count. This keeps the final aggregation table traceable back to the exact performance rows and R2V integrity summaries that produced it.

The same behavior is available as a CLI:

```bash
python3 -m pareto.r2v.result_aggregation \
  --performance_path records/r2v_traffic_runs/aggregation/r2v_performance_rows.jsonl \
  --integrity_path records/r2v_traffic_runs/main_jinan_3seed/seed0/r2v/artifacts/r2v_summary.json \
  --output records/r2v_traffic_runs/aggregation/r2v_result_aggregation.json
```

## Paper artifact manifest

Use `pareto.r2v.paper_artifact_manifest` after readiness and aggregation to freeze the paper evidence bundle:

```bash
python3 -m pareto.r2v.paper_artifact_manifest \
  --artifact performance:main_performance_rows:records/r2v_traffic_runs/aggregation/r2v_performance_rows.jsonl \
  --artifact integrity:seed0_r2v_summary:records/r2v_traffic_runs/main/seed0/r2v/artifacts/r2v_summary.json \
  --artifact aggregation:main_aggregation:records/r2v_traffic_runs/aggregation/r2v_result_aggregation.json \
  --output records/r2v_traffic_runs/aggregation/paper_artifact_manifest.json
```

The manifest records `sha256`, size, line count, JSON format, and schema version when available. Missing artifacts make the manifest `BLOCKED`. Artifact types are explicit, so performance rows and integrity/status summaries remain separate in the paper evidence bundle. A file labeled `performance` must contain all five required traffic metrics: `average_travel_time`, `queue_length`, `delay`, `throughput`, and `reward`. Status-only or partial-metric rows are blocked.

A file labeled `aggregation` must be a `r2v-traffic-result-aggregation-v1` JSON produced by `pareto.r2v.result_aggregation`, with non-empty performance rows, all five required traffic metrics, and valid input artifact hashes for both performance and integrity inputs. The manifest also cross-checks those hashes against the performance and integrity artifacts bundled in the same manifest. Missing or unmatched aggregation provenance blocks the paper manifest.

A file labeled `integrity` must be an R2V candidate summary with `schema_version=r2v-tsc-candidate-summary-v1`, non-negative `candidate_count` and `admitted_count`, and `gate_counts` for `rare`, `value`, `support`, and `safety`. A status-only JSON is not a valid R2V integrity summary.

A file labeled `diffusion_score` must also be paper-eligible: every score row needs `paper_claim_eligible=true`, a non-proxy adapter, and non-empty `model_checkpoint`, `config_hash`, and `normalization_id`. Proxy artifacts from `traffic_feature_density_proxy` are blocked from the final paper evidence bundle even though they remain valid smoke/integration artifacts.

A file labeled `weighted_transitions` must satisfy the v2 traffic artifact schema via `validate_r2v_traffic_artifact`. Legacy v1-only weighted rows, duplicate transition IDs, mismatched metadata IDs, missing traffic gate masks, unsupported admission modes, invalid row-role/admission combinations, or invalid sample weights make the manifest `BLOCKED`. The runtime pair builder enforces the same schema when `--r2v_sampling_mode` is not `off`. The manifest records weighted-row count, admitted count, gate variants, generative backends, admission modes, row roles, and score-artifact source paths. If a weighted artifact reports `r2v_generative_backend=diffusion`, the paper manifest requires its `r2v_score_artifact_path` values to match bundled `diffusion_score` artifacts.

A file labeled `readiness` must be a JSON readiness report with `status=READY` and `failed_count=0`. A preflight report that is itself `BLOCKED` cannot be included in a final paper evidence bundle.

## Score artifact builder

Use `pareto.r2v.build_generative_score_artifact` to create a loader-compatible score artifact from transition buffers:

```bash
python3 -m pareto.r2v.build_generative_score_artifact \
  --transitions records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl \
  --output records/r2v_traffic/diffusion_seed0_scores.jsonl \
  --summary_output records/r2v_traffic/diffusion_seed0_scores_summary.json \
  --backend diffusion \
  --adapter traffic_feature_density_proxy
```

This adapter is deterministic and proxy-backed. It is useful for smoke and integration readiness because it writes valid `transition_id`, `rarity_score`, and `support_score` rows, but its rows include `paper_claim_eligible=false`. Do not report this as a final diffusion-model result. For paper diffusion runs, pass an artifact produced by the actual diffusion scoring/repair pipeline through `--r2v_artifact_path`.

Weighted-transition provenance follows the same rule. If `build_r2v_candidates` is run without `--score_artifact`, the v2 weighted artifact records `metadata.r2v_generative_backend=<candidate_model>` such as `feature_density_proxy`, even if a higher-level runner has `generative_backend=diffusion` configured for later real-artifact runs. Only a real `--score_artifact` path may propagate `--score_artifact_backend diffusion` into `metadata.r2v_generative_backend`. This prevents proxy smoke artifacts from being mislabeled as diffusion evidence.

When a real score artifact is used, weighted rows also record `metadata.r2v_score_artifact_path` and `metadata.r2v_score_artifact_backend`. Final paper manifests cross-check those paths against bundled diffusion score artifacts.

An external diffusion score/repair artifact may also include `repaired_transition` on a score row. When `--r2v_admission_mode weights_plus_repaired` is enabled, admitted candidates can append that repaired transition as a repaired replay row. The repaired object must contain its own non-empty `transition_id` and `sample_id`; otherwise the loader, readiness checker, manifest builder, or candidate builder rejects it.

## Experiment planning

Use:

- `build_smoke_commands(spec)`;
- `build_main_commands(spec)`;
- `build_main_pipeline_commands(spec)`;
- `build_strict_paper_readiness_commands(spec)`;
- `build_performance_readiness_command(spec)`;
- `build_result_aggregation_command(spec)`;
- `build_ablation_commands(spec)`.

The returned dicts include `name`, `argv`, `shell`, `output_root`, `wandb`, and `metadata`.

Smoke, main, and ablation commands target `pareto.r2v.jinan_pair_ablation_runner`, which now accepts:

- `--r2v on/off/paired`;
- `--r2v_mode traffic`;
- `--generative_backend diffusion`;
- `--repair_story not_rare_to_val/not_val_to_val`;
- `--repair_metadata_policy require_metadata/metadata_or_proxy`;
- `--gate_variant full/no_support/no_ood/no_dynamics`;
- `--r2v_admission_mode weights_only/weights_plus_repaired`;
- `--r2v_admitted_weight`, the exact sample weight assigned to admitted candidates by the runner;
- `--r2v_repair_rejected_weight`, the sample weight assigned to explicit repaired proposals that remain rejected;
- `--r2v_artifact_path` for a diffusion score artifact;
- `--rare_fraction` as a convenience wrapper over `rare_quantile`.

Default `--r2v paired` preserves the old baseline+R2V paired runner behavior.

The lower-level candidate builder still supports `--admitted_weight_bonus` for score-proportional exploratory weighting, but the paper-facing runner maps `--r2v_admitted_weight` to exact `--admitted_weight` to match the original R2V admission semantics.

`metadata_or_proxy` is the runner default for smoke/integration command plans: if repaired source/final gate metadata exists, it is used; if both are absent, computed traffic gates are converted into clearly marked proxy repair metadata. Use `require_metadata` for strict paper diffusion runs when repaired artifacts must already contain source/final gates.

`build_strict_paper_readiness_commands(spec)` generates one `pareto.r2v.experiment_readiness` command per main seed. These commands require real diffusion artifacts, `paper_claim_eligible=true`, non-proxy diffusion adapters, and `repair_metadata_policy=require_metadata`.

`build_performance_readiness_command(spec)` generates the paper-table readiness command. It requires baseline/R2V method coverage over the main seeds, all five traffic metrics, and completed evaluation status before aggregation.

`build_result_aggregation_command(spec)` generates the final `pareto.r2v.result_aggregation` command for the main run, using the expected performance rows JSONL plus one integrity summary per main seed.

`build_paper_artifact_manifest_command(spec)` generates the final evidence-bundle manifest command for the main run.

`build_main_pipeline_commands(spec)` returns the complete ordered main pipeline: strict paper readiness for each main seed, baseline/R2V main runner commands, paper-table performance readiness, result aggregation, then paper artifact manifesting. In this final-paper pipeline, R2V runner commands use `--repair_metadata_policy require_metadata`, conservative `--r2v_admission_mode weights_only`, and each seed's `--r2v_artifact_path` from `diffusion_artifact_template`, matching the strict readiness contract. The looser `metadata_or_proxy` path remains for smoke and integration plans only.

The same planner is available as a CLI:

```bash
python3 -m pareto.r2v.traffic_experiment_plan \
  --plan main_pipeline \
  --format shell \
  --output records/r2v_traffic_runs/main_pipeline.sh \
  --output_root records/r2v_traffic_runs \
  --main_seeds 0,1,2
```

Use `--format json` when another script should consume the plan, and `--format shell` when you want a reviewable bash script. Supported plans are `smoke`, `main`, `main_pipeline`, `strict_paper_readiness`, `performance_readiness`, `result_aggregation`, `paper_artifact_manifest`, and `ablation`.

Add `--include_validation` with `--format json` to include a structural self-check. For `main_pipeline`, validation checks command order, baseline-off flags, R2V diffusion/not_rare_to_val/full flags, per-seed diffusion artifact wiring, and paper-manifest artifact-type coverage.

## Readiness

Use `pareto.r2v.experiment_readiness` before smoke/main runs:

```bash
python3 -m pareto.r2v.experiment_readiness \
  --scenario jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --transition_glob 'records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed{seed}/transitions_raw.jsonl' \
  --seed 0 \
  --require_diffusion_artifacts \
  --diffusion_artifact 0:records/r2v_traffic/diffusion_seed0_scores.jsonl \
  --output records/r2v_traffic_runs/readiness_seed0.json
```

The command exits with code `2` when blocked and writes a JSON report listing missing files, malformed diffusion score rows, missing score coverage for transition IDs, or missing metric columns.

For final paper diffusion runs, add `--require_paper_claim_eligible_diffusion`, `--repair_metadata_policy require_metadata`, and `--require_strict_repair_metadata_policy`. This rejects proxy score artifacts whose rows have `paper_claim_eligible=false` or omit the field. It also requires every row to carry non-empty `model_checkpoint`, `config_hash`, and `normalization_id`, rejects proxy adapters such as `traffic_feature_density_proxy`, and blocks `metadata_or_proxy` repair metadata in strict paper readiness.

For final paper result tables, add `--expected_performance_method`, `--expected_performance_seed`, and `--require_completed_performance_status` when checking performance rows. This verifies the baseline/R2V x seed grid without treating status as a performance metric.
