# Bug Fix Log

## BF-001: baseline command emitted R2V sampling

Symptom: new experiment-plan tests failed because baseline commands emitted `--r2v_sampling_mode full_r2v`.

Root cause: `R2VTrafficConfig` defaulted to `full_r2v` even when `r2v="off"`.

Fix: added sampling mode `off`; `to_cli_flags()` now emits `off` whenever R2V is disabled, and config rejects `r2v="on"` with `r2v_sampling_mode="off"`.

Verification: experiment-plan tests now pass.

## BF-002: experiment-plan commands did not parse against runner

Symptom: review agent found generated smoke commands exited with argparse code 2.

Root cause: `traffic_experiment_plan.py` emitted the requested high-level R2V flags, but `jinan_pair_ablation_runner.py` only accepted its older paired-runner flags.

Fix: added compatible parser fields for `--r2v`, `--r2v_mode`, `--generative_backend`, `--gate_variant`, `--r2v_output_dir`, `--r2v_artifact_path`, weights, `--rare_fraction`, and gate on/off flags. `--r2v off` now builds only baseline methods, `--r2v on` builds only R2V methods, and default `--r2v paired` preserves old behavior.

Verification: parser compatibility test and two runner dry-runs passed.

## BF-003: gate ablations were declarative only

Symptom: review agent found runtime admission still used all gates regardless of `gate_variant`.

Root cause: gate variants existed in the new config wrapper but not in `R2VTrafficSelectorConfig` admission.

Fix: added `gate_variant` to the selector, active-gate resolution, CLI forwarding through `build_r2v_candidates`, and runner forwarding. Original gate maps are still recorded for diagnostics.

Verification: `no_dynamics` test admits a rare/value/support candidate that fails safety, while full-gate tests still block it.

## BF-004: repair-story path failed on ordinary transition buffers

Symptom: `repair_story=not_rare_to_val` failed before candidate construction when transition rows did not already contain source/final repair gate metadata.

Root cause: the selector only supported strict repair metadata mode, but smoke/integration runs need a conservative proxy path until real diffusion repair artifacts exist.

Fix: added `repair_metadata_policy`. Low-level selection defaults to `require_metadata`; runner and smoke commands can use `metadata_or_proxy`, which derives clearly marked proxy source/final gates from computed gates.

Verification: ordinary transition rows plus a proxy score artifact now build `not_rare_to_val` weighted artifacts, and tests assert `gate_source=computed_proxy_repair_metadata`.

## BF-005: proxy diffusion artifacts could pass paper readiness

Symptom: proxy score artifacts were marked `paper_claim_eligible=false`, but readiness did not have a strict mode to reject them before paper-result aggregation.

Root cause: the score loader dropped the `paper_claim_eligible` provenance and readiness only checked score shape and transition coverage.

Fix: preserved `paper_claim_eligible` and `adapter` in score-row provenance, and added `--require_paper_claim_eligible_diffusion` to `experiment_readiness.py`. Strict readiness also requires non-empty `model_checkpoint`, `config_hash`, and `normalization_id`, and rejects proxy adapters.

Verification: strict readiness rejects proxy artifacts, rejects rows with missing paper diffusion provenance, rejects proxy adapters even if the boolean flag is true, and accepts artifacts whose rows explicitly set `paper_claim_eligible=true` with required provenance.

## BF-006: performance readiness did not require a complete main table

Symptom: readiness checked metric columns but did not require baseline/R2V x seed coverage or completed evaluation status.

Root cause: performance checks were row-local and did not validate the expected method/seed grid.

Fix: added `--expected_performance_method`, `--expected_performance_seed`, and `--require_completed_performance_status` to `experiment_readiness.py`.

Verification: readiness now rejects missing method/seed pairs, rejects unfinished rows, and accepts a complete baseline/R2V x 3-seed table.

## BF-007: result aggregation row_count counted metric values

Symptom: `performance.row_count` reported the total number of metric observations, so 2 performance rows with 5 metrics each became `row_count=10`.

Root cause: `_aggregate_performance()` summed per-metric value-list lengths instead of counting input rows that contain performance metrics.

Fix: `row_count` now counts performance rows, `metric_value_count` records the old metric-observation total, and `by_method_row_count` records rows per method.

Verification: added a regression test that expects 2 rows and 10 metric values for a two-row/five-metric table.

## BF-008: strict paper readiness did not block proxy repair metadata

Symptom: strict diffusion readiness could reject proxy score artifacts, but it did not machine-check whether the R2V run was still using `repair_metadata_policy=metadata_or_proxy`.

Root cause: `experiment_readiness.py` only inspected CityFlow files, transition buffers, diffusion score artifacts, and performance rows.

Fix: added `repair_metadata_policy` and `require_strict_repair_metadata_policy` readiness inputs. Strict paper readiness now requires `repair_metadata_policy=require_metadata`; `metadata_or_proxy` remains available for smoke/integration.

Verification: readiness tests now reject `metadata_or_proxy` when strict repair metadata policy is required and accept `require_metadata`.

## BF-009: repair rejected weight was not wired into admission

Symptom: `r2v_repair_rejected_weight` existed in high-level configs, but `build_r2v_candidates` did not accept it, the Jinan runner did not forward it, and `weights_plus_repaired` skipped repaired proposals that failed gates.

Root cause: the traffic migration had implemented admitted repaired rows but not the original R2V repair-rejected channel.

Fix: added `repair_rejected_weight` to `R2VTrafficSelectorConfig`, `--repair_rejected_weight` to the candidate-builder CLI, and runner forwarding from `--r2v_repair_rejected_weight`. `weights_plus_repaired` now appends explicit gate-rejected repaired proposals as `r2v_row_role=repair_rejected`, `r2v_admitted=false`, and the configured rejected weight.

Verification: red-green tests cover rejected repaired proposal appending, candidate-builder CLI parsing, runner command forwarding, and the 152-test R2V target suite.

## BF-010: weighted artifact schema did not check row-role semantics

Symptom: a v2 weighted-transition artifact could claim `r2v_row_role=repair_rejected` while also setting `r2v_admitted=true`, or use an unknown row role, and `validate_r2v_traffic_artifact` would still accept it.

Root cause: the v2 validator checked IDs, gates, weights, backend, and admission mode, but not the proposal/admission role contract.

Fix: added `SUPPORTED_ROW_ROLES` and validator checks for `source`, `repaired`, and `repair_rejected` rows. Repaired rows must be admitted and link to `r2v_repaired_from_transition_id`; repair-rejected rows must be non-admitted, set `r2v_repair_rejected=true`, and link to the source transition. The summary now records `row_roles`.

Verification: red-green schema tests cover unknown roles, repair-rejected/admitted contradictions, missing repaired source links, and the 155-test R2V target suite.

## BF-011: high-level admitted weight was treated as a bonus

Symptom: `--r2v_admitted_weight 2.0` in the Jinan runner was forwarded to `build_r2v_candidates --admitted_weight_bonus 2.0`, so an admitted transition with `base_weight=1.0` could receive weight about `3.0`.

Root cause: the traffic runner reused the selector's exploratory score-proportional bonus knob for the paper-facing admitted-weight flag, while the original R2V admission API treats `admitted_weight` as the exact sample weight.

Fix: added exact `admitted_weight` support to `R2VTrafficSelectorConfig` and `build_r2v_candidates --admitted_weight`. The Jinan runner now forwards `--r2v_admitted_weight` to exact `--admitted_weight`; the older `--admitted_weight_bonus` remains available only as a lower-level compatibility/exploration option.

Verification: red-green tests cover exact admitted-weight override, candidate-builder CLI parsing, and runner command forwarding.

## BF-012: pair builder accepted legacy R2V weighted artifacts

Symptom: `pareto.data.build_pairs` could use a legacy v1-only weighted-transition file when `r2v_sampling_mode` was enabled, even though final paper artifacts and runtime weighted outputs had moved to the v2 R2V-Traffic schema.

Root cause: pair construction validated only the generic v1 weight map contract and did not call `validate_r2v_traffic_artifact`.

Fix: R2V sampling now validates the full v2 traffic artifact before constructing pairs and records the traffic artifact summary in the candidate-sampling report. Existing tests were updated to use v2 weighted rows by default, with a dedicated legacy-artifact rejection test.

Verification: red-green pair-builder test rejects v1-only weighted artifacts; the full pair-builder test file passes with all R2V sampling modes.

## BF-013: proxy candidate artifacts were mislabeled as diffusion

Symptom: a runner configured with `generative_backend=diffusion` could build weighted transitions without a real score artifact and still write `metadata.r2v_generative_backend=diffusion`.

Root cause: `_traffic_artifact_backend()` used `score_artifact_backend` or defaulted to `diffusion` even when `score_artifact_path` was absent, and the Jinan runner always passed `--score_artifact_backend diffusion`.

Fix: the candidate builder now records `candidate_model` as the backend whenever no score artifact is supplied. The Jinan runner only forwards `--score_artifact_backend diffusion` together with an actual `--score_artifact` path.

Verification: red-green tests cover proxy weighted outputs recording `feature_density_proxy`, default runner plans omitting score-artifact backend without an artifact path, and real-artifact runner plans preserving `--score_artifact_backend diffusion`.

## BF-014: final main pipeline preflighted diffusion artifacts but did not use them

Symptom: `main_pipeline` generated strict readiness checks and a final manifest containing per-seed diffusion score artifacts, but the generated R2V runner commands lacked `--r2v_artifact_path` and could fall back to proxy scoring.

Root cause: `build_main_commands()` constructed the R2V config without copying `spec.diffusion_artifact_template.format(seed=seed)` into `r2v_artifact_path`, and plan validation checked only high-level labels.

Fix: main R2V commands now pass each seed's diffusion artifact path to the runner, metadata records the path, and `validate_experiment_plan(..., plan='main_pipeline')` checks the per-seed wiring.

Verification: red-green experiment-plan tests assert `--r2v_artifact_path records/r2v_traffic/diffusion_seed{seed}_scores.jsonl` appears in every main R2V command and that validation includes `r2v_commands_use_seed_diffusion_artifacts`.

## BF-015: paper manifest did not link weighted transitions to score artifacts

Symptom: a paper manifest could bundle diffusion-score artifacts and weighted-transition artifacts side by side without proving that the weighted rows came from those diffusion scores.

Root cause: weighted-transition validation reported backend names but did not record or cross-check score-artifact provenance.

Fix: weighted rows now record `r2v_score_artifact_path` and `r2v_score_artifact_backend`. The paper manifest blocks diffusion-labeled weighted artifacts with missing or unmatched score-artifact paths.

Verification: manifest tests accept matching diffusion score/weighted artifacts, accept standalone proxy weighted artifacts, and block diffusion weighted artifacts whose source score path does not match the bundled diffusion score artifact.

## BF-016: traffic artifact validator did not compare metadata IDs to row IDs

Symptom: a v2 weighted row with `metadata.r2v_traffic_transition_id` or `metadata.r2v_traffic_sample_id` inconsistent with the row-level IDs still passed validation.

Root cause: the validator checked that v2 ID fields existed but did not compare them against `transition_id` and `sample_id`.

Fix: `validate_r2v_traffic_artifact()` now requires metadata IDs to match the row IDs exactly.

Verification: red-green schema test rejects mismatched metadata transition/sample IDs.

## BF-017: rarity was treated as a final admission gate

Symptom: full admission required `rare=true`, so an ordinary transition repaired/proposed into a valuable, supported, dynamics-consistent pattern could still be rejected solely because it was not rare.

Root cause: `_active_gate_keys('full')` included the detector gate `rare` alongside value/support/safety.

Fix: rarity remains available for candidate discovery, source-story checks, scoring, and ablations such as `rare_only`, but final admission gates are now value/OOD, support, and dynamics/safety. Gate ablations remove one of those three admission gates.

Verification: red-green selector test admits a `not_rare_to_val` candidate whose final gates are value/support/safety true and rare false, while rare-but-not-valuable samples remain rejected.
