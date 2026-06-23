# Test Report

## Environment

The default `python3` environment did not have `pytest`, so I created a temporary local venv at `/tmp/r2v-traffic-pytest-venv` and installed `pytest` and `numpy`.

## Commands run

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pytest -q tests/pareto/test_r2v_traffic_artifact_schema.py tests/pareto/test_r2v_result_aggregation.py tests/pareto/test_r2v_traffic_experiment_plan.py
```

Initial result before integration fixes: `14 passed`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pytest -q tests/pareto/test_r2v_artifact_validation.py tests/pareto/test_r2v_traffic_candidate_selector.py tests/pareto/test_pair_builder.py tests/pareto/test_r2v_jinan_pair_ablation_runner.py
```

Initial result before integration fixes: `62 passed`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pytest -q tests/pareto/test_r2v_traffic_artifact_schema.py tests/pareto/test_r2v_result_aggregation.py tests/pareto/test_r2v_traffic_experiment_plan.py tests/pareto/test_r2v_artifact_validation.py tests/pareto/test_r2v_traffic_candidate_selector.py tests/pareto/test_pair_builder.py tests/pareto/test_r2v_jinan_pair_ablation_runner.py
```

Intermediate targeted result after adding the zero-reward boundary test: `80 passed`.

After adding the readiness checker, diffusion score coverage validation, score-artifact builder, proxy repair-metadata path, strict paper diffusion readiness, strict repair-metadata readiness, strict paper-readiness command generation, generated performance-readiness commands, generated result-aggregation commands, generated paper-artifact-manifest commands, generated main-pipeline commands, strict main-pipeline repair-metadata enforcement, experiment-plan validation, experiment-plan JSON/shell CLI export, paper artifact manifesting, paper-manifest performance-content validation, paper-manifest aggregation-content validation, paper-manifest aggregation-provenance validation, paper-manifest aggregation/bundle hash matching, paper-manifest integrity-content validation, paper-manifest diffusion-score validation, paper-manifest weighted-transition schema validation, paper-manifest readiness status validation, result-aggregation CLI, result-aggregation row-count regression, result-aggregation input-artifact hashing, explicit R2V admission-mode support, external score-artifact repaired-transition payloads, repaired-transition payload validation, `repair_rejected_weight` wiring, row-role/admission consistency validation, exact `admitted_weight` runner semantics, and runtime v2 weighted-artifact validation in pair sampling, the targeted suite is:

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pytest -q tests/pareto/test_r2v_build_score_artifact.py tests/pareto/test_r2v_experiment_readiness.py tests/pareto/test_r2v_paper_artifact_manifest.py tests/pareto/test_r2v_traffic_artifact_schema.py tests/pareto/test_r2v_result_aggregation.py tests/pareto/test_r2v_traffic_experiment_plan.py tests/pareto/test_r2v_artifact_validation.py tests/pareto/test_r2v_traffic_candidate_selector.py tests/pareto/test_pair_builder.py tests/pareto/test_r2v_jinan_pair_ablation_runner.py
```

Result after requiring v2 R2V-Traffic weighted artifacts at runtime pair sampling: `157 passed`.

Latest result after fixing proxy-vs-diffusion backend provenance and adding runner command-plan coverage:

```bash
/Users/azure/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q tests/pareto/test_r2v_build_score_artifact.py tests/pareto/test_r2v_experiment_readiness.py tests/pareto/test_r2v_paper_artifact_manifest.py tests/pareto/test_r2v_traffic_artifact_schema.py tests/pareto/test_r2v_result_aggregation.py tests/pareto/test_r2v_traffic_experiment_plan.py tests/pareto/test_r2v_artifact_validation.py tests/pareto/test_r2v_traffic_candidate_selector.py tests/pareto/test_pair_builder.py tests/pareto/test_r2v_jinan_pair_ablation_runner.py
```

Result: `159 passed`.

Latest result after fixing final main-pipeline diffusion artifact wiring, weighted score-artifact provenance, traffic metadata ID validation, and rare-as-final-gate semantics:

```bash
/Users/azure/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q tests/pareto/test_r2v_build_score_artifact.py tests/pareto/test_r2v_experiment_readiness.py tests/pareto/test_r2v_paper_artifact_manifest.py tests/pareto/test_r2v_traffic_artifact_schema.py tests/pareto/test_r2v_result_aggregation.py tests/pareto/test_r2v_traffic_experiment_plan.py tests/pareto/test_r2v_artifact_validation.py tests/pareto/test_r2v_traffic_candidate_selector.py tests/pareto/test_pair_builder.py tests/pareto/test_r2v_jinan_pair_ablation_runner.py
```

Result: `163 passed`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m py_compile pareto/r2v/__init__.py pareto/r2v/build_generative_score_artifact.py pareto/r2v/build_r2v_candidates.py pareto/r2v/experiment_readiness.py pareto/r2v/generative_scorer.py pareto/r2v/paper_artifact_manifest.py pareto/r2v/traffic_artifact_schema.py pareto/r2v/result_aggregation.py pareto/r2v/traffic_experiment_plan.py pareto/r2v/traffic_candidate_selector.py pareto/r2v/jinan_pair_ablation_runner.py
```

Result: passed with no output.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact performance:perf:/tmp/perf.jsonl --artifact integrity:summary:/tmp/summary.json --output /tmp/manifest.json
```

Result: wrote a `READY` manifest with one complete performance artifact, one R2V candidate-summary integrity artifact, and SHA256 hashes for both files.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact integrity:bad:/tmp/status_only_summary.json --output /tmp/status_only_integrity_manifest.json
```

Result for a status-only JSON labeled as `integrity`: manifest `status=BLOCKED`, failed entry `status=invalid_content`, and message says the R2V candidate summary schema is missing or unsupported.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact performance:bad:/tmp/status_only.jsonl --output /tmp/status_only_manifest.json
```

Result for a status-only JSONL labeled as `performance`: exit code `2`, manifest `status=BLOCKED`, failed entry `status=invalid_content`, and `performance_metric_count=0`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact performance:bad:/tmp/incomplete_performance.jsonl --output /tmp/incomplete_performance_manifest.json
```

Result for a partial-metric JSONL labeled as `performance`: manifest `status=BLOCKED`, failed entry `status=invalid_content`, and `performance_missing_metrics` lists the absent required traffic metrics.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact aggregation:bad:/tmp/status_only_aggregation.json --output /tmp/status_only_aggregation_manifest.json
```

Result for a status-only JSON labeled as `aggregation`: manifest `status=BLOCKED`, failed entry `status=invalid_content`, and message says the result aggregation schema is missing or unsupported.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact diffusion_score:bad:/tmp/proxy_scores.jsonl --output /tmp/proxy_diffusion_manifest.json
```

Result for a proxy diffusion-score JSONL labeled as `diffusion_score`: exit code `2`, manifest `status=BLOCKED`, failed entry `status=invalid_content`, and `paper_claim_proxy_adapter_count=1`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact weighted_transitions:bad:/tmp/legacy_weighted_transitions.jsonl --output /tmp/legacy_weighted_manifest.json
```

Result for a legacy v1-only weighted-transition JSONL labeled as `weighted_transitions`: manifest `status=BLOCKED`, failed entry `status=invalid_content`, and message mentions missing `r2v_traffic_schema_version`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.paper_artifact_manifest --artifact readiness:bad:/tmp/blocked_readiness.json --output /tmp/blocked_readiness_manifest.json
```

Result for a `BLOCKED` readiness JSON labeled as `readiness`: manifest `status=BLOCKED`, failed entry `status=invalid_content`, and message says the readiness artifact is not `READY`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.traffic_experiment_plan --plan main_pipeline --format json --output /tmp/main_pipeline.json --python_bin python3 --transition_glob 'records/jinan/seed{seed}/transitions_raw.jsonl' --diffusion_artifact_template 'records/r2v_traffic/diffusion_seed{seed}_scores.jsonl' --output_root runs/main --main_seeds 0,1,2
```

Result: wrote a JSON plan with `plan=main_pipeline`, 12 commands, first command `paper_readiness_seed0_r2v_diffusion_not_rare_to_val_full`, and last command `build_main_paper_artifact_manifest`.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.traffic_experiment_plan --plan main_pipeline --format json --include_validation --output /tmp/main_pipeline_validated.json --python_bin python3 --transition_glob 'records/jinan/seed{seed}/transitions_raw.jsonl' --diffusion_artifact_template 'records/r2v_traffic/diffusion_seed{seed}_scores.jsonl' --output_root runs/main --main_seeds 0,1,2
```

Result: wrote a JSON plan whose `validation.status` is `READY` and whose checks cover command order, baseline-off flags, R2V main config, and paper-manifest coverage.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.traffic_experiment_plan --plan paper_artifact_manifest --format json --output /tmp/paper_manifest_plan.json --python_bin python3 --diffusion_artifact_template 'records/r2v_traffic/diffusion_seed{seed}_scores.jsonl' --output_root runs/main --main_seeds 0,1,2
```

Result: wrote a JSON plan with one command, `build_main_paper_artifact_manifest`, including 1 aggregation artifact, 3 diffusion-score artifacts, 3 integrity artifacts, 1 performance artifact, 4 readiness artifacts, and 3 weighted-transition artifacts.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.traffic_experiment_plan --plan smoke --format shell --output /tmp/smoke.sh --python_bin python3 --transition_glob 'records/jinan/seed{seed}/transitions_raw.jsonl' --output_root runs/smoke --smoke_seeds 0
```

Result: wrote a bash script beginning with `set -euo pipefail`, then the baseline smoke command and the R2V smoke command.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.jinan_pair_ablation_runner --dry_run --python_bin /tmp/r2v-traffic-pytest-venv/bin/python --transition_input records/jinan/seed0/transitions_raw.jsonl --output_root /tmp/r2v-traffic-dryrun-off --r2v off --r2v_mode traffic --generative_backend diffusion --repair_story not_rare_to_val --repair_metadata_policy metadata_or_proxy --gate_variant full --r2v_sampling_mode off --rare_fraction 0.2 --support_gate on --ood_gate on --dynamics_gate on --force
```

Result: `DRY_RUN_READY`, 7 commands, no R2V artifact build step.

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.jinan_pair_ablation_runner --dry_run --python_bin /tmp/r2v-traffic-pytest-venv/bin/python --transition_input records/jinan/seed0/transitions_raw.jsonl --output_root /tmp/r2v-traffic-dryrun-on --r2v on --r2v_mode traffic --generative_backend diffusion --repair_story not_rare_to_val --repair_metadata_policy metadata_or_proxy --gate_variant no_dynamics --r2v_sampling_mode full_r2v --rare_fraction 0.2 --support_gate on --ood_gate on --dynamics_gate off --force
```

Result: `DRY_RUN_READY`, 8 commands, R2V candidate build step includes `--gate_variant no_dynamics`.

## Coverage

Covered:

- v2 artifact schema;
- runtime weighted outputs include v2 traffic artifact metadata and validate with `validate_r2v_traffic_artifact`;
- transition/sample ID preservation;
- gate mask keys and booleans;
- admission weights;
- runner `--r2v_admitted_weight` maps to exact candidate-builder `--admitted_weight`, not a bonus over `base_weight`;
- default `weights_only` admission mode preserves source transition rows and records the mode in metadata;
- opt-in `weights_plus_repaired` appends explicit repaired transition payloads for admitted candidates, and appends gate-rejected repair proposals only as `r2v_row_role=repair_rejected` with `r2v_admitted=false` and the configured rejected weight;
- v2 weighted-transition validation rejects unknown row roles, repaired rows without a source link, and `repair_rejected` rows that are also marked admitted;
- runtime R2V pair sampling rejects legacy v1-only weighted artifacts before pair construction;
- proxy candidate runs without a real score artifact record `feature_density_proxy` or the configured candidate model as `r2v_generative_backend`, not `diffusion`;
- runner command plans only pass `--score_artifact_backend diffusion` when an actual score artifact path is supplied;
- main-pipeline R2V commands consume each seed's real diffusion score artifact through `--r2v_artifact_path`;
- main-pipeline validation checks per-seed diffusion artifact wiring;
- paper artifact manifests block diffusion weighted-transition artifacts whose recorded score artifact paths do not match bundled diffusion-score artifacts;
- v2 traffic artifact validation rejects metadata transition/sample IDs that do not match row IDs;
- full admission does not require the rare gate; rare remains a detector/source/sampling signal while value/support/dynamics decide admission;
- external diffusion score/repair artifacts can provide `repaired_transition` payloads for the opt-in `weights_plus_repaired` append path;
- readiness and paper-manifest validation reject malformed score-artifact `repaired_transition` payloads;
- fail-closed validation;
- baseline-off command behavior;
- R2V-on command behavior;
- `not_rare_to_val` + full gate;
- `not_val_to_val` interface;
- proxy score-artifact builder output can be loaded and accepted by readiness;
- strict paper readiness rejects proxy diffusion artifacts with `paper_claim_eligible=false`, missing diffusion provenance, and proxy adapters;
- strict paper readiness rejects `repair_metadata_policy=metadata_or_proxy` when `--require_strict_repair_metadata_policy` is set, while accepting `require_metadata`;
- strict paper-readiness command generation emits one readiness preflight per main seed with real-diffusion and strict repair-metadata requirements;
- performance-readiness command generation emits the baseline/R2V x main-seed coverage check before aggregation;
- result-aggregation command generation emits the final aggregation CLI command with performance rows and per-seed integrity summaries;
- complete main-pipeline command generation orders strict paper readiness, baseline/R2V main runs, performance readiness, result aggregation, and paper artifact manifesting;
- main-pipeline R2V runner commands use `repair_metadata_policy=require_metadata`, matching strict paper readiness;
- main-pipeline R2V runner commands use conservative `r2v_admission_mode=weights_only`;
- experiment-plan validation checks main-pipeline order, baseline/R2V flags, and paper artifact manifest coverage;
- experiment-plan CLI export writes machine-readable JSON plans and reviewable shell scripts;
- paper artifact manifests record file hashes, explicit artifact types, JSON/JSONL shape, and fail-closed missing-artifact status;
- paper artifact manifests block status-only or partial-metric files labeled as performance artifacts;
- paper artifact manifests validate files labeled as aggregation artifacts against the result aggregation schema and required five metrics;
- paper artifact manifests validate files labeled as aggregation artifacts include performance and integrity input-artifact hashes;
- paper artifact manifests validate aggregation input hashes match the performance and integrity artifacts bundled in the same manifest;
- paper artifact manifests validate files labeled as integrity artifacts against the R2V candidate-summary schema and gate counts;
- paper artifact manifests block proxy or missing-provenance files labeled as diffusion-score artifacts;
- paper artifact manifests validate files labeled as weighted-transition artifacts with the v2 R2V-Traffic schema;
- paper artifact manifests record weighted-transition admission modes;
- paper artifact manifests require files labeled as readiness artifacts to report `status=READY` and `failed_count=0`;
- paper performance readiness rejects incomplete baseline/R2V x seed grids and unfinished rows;
- result aggregation reports `row_count` as performance rows, `metric_value_count` as consumed metric observations, and `by_method_row_count` per method;
- result aggregation records SHA256 provenance for performance and integrity input files;
- result aggregation has a `python -m pareto.r2v.result_aggregation` CLI that writes the JSON aggregation artifact;
- `repair_metadata_policy=metadata_or_proxy` lets ordinary transition buffers build `not_rare_to_val` candidates while marking `gate_source=computed_proxy_repair_metadata`;
- zero-reward transitions are not treated as corrupted when gates pass;
- gate ablation config;
- sampling ablation command plan;
- performance/status aggregation separation.

## Not run

Full CityFlow smoke/main experiments were not run locally in this turn. I attempted:

```bash
/tmp/r2v-traffic-pytest-venv/bin/python scripts/smoke_env.py --scenario jinan --policy maxpressure --steps 30 --seed 0 --out_dir /tmp/r2v-traffic-cityflow-smoke
```

It failed before simulator startup because the local workspace is missing `data/Jinan/3_4/anon_3_4_jinan_real.json`.

The new readiness checker makes this blocker explicit:

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.experiment_readiness --root /Users/azure/Documents/R2V-traffic --scenario jinan --traffic_file anon_3_4_jinan_real.json --transition_glob 'records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed{seed}/transitions_raw.jsonl' --seed 0 --require_diffusion_artifacts --require_paper_claim_eligible_diffusion --diffusion_artifact 0:records/r2v_traffic/diffusion_seed0_scores.jsonl --output /tmp/r2v-traffic-readiness.json
```

Result: process exit code `2`, report status `BLOCKED` with 4 failed checks: missing traffic file, roadnet file, transition inputs, and diffusion score artifact. When artifact files exist, tests verify malformed score rows and missing transition-ID coverage also fail.

Strict paper repair-metadata policy was also probed directly:

```bash
/tmp/r2v-traffic-pytest-venv/bin/python -m pareto.r2v.experiment_readiness --root /Users/azure/Documents/R2V-traffic --no-require_cityflow_data --repair_metadata_policy metadata_or_proxy --require_strict_repair_metadata_policy --output /tmp/r2v-traffic-strict-policy-readiness.json
```

Result: process exit code `2`, report status `BLOCKED` with `repair_metadata_policy` failure.

I also tried collecting the broader `tests/pareto` directory in a separate temporary pytest-only venv:

```bash
/tmp/r2v-traffic-pytest/bin/python -m pytest tests/pareto
```

That was not a valid full-suite result for this checkout because collection failed on missing heavy test dependencies such as `numpy` and `torch`. The R2V target suite above was rerun in `/tmp/r2v-traffic-pytest-venv`, which includes the dependencies needed for those tests.
