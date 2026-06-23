# Final R2V-Traffic Migration Review

## Verdict

Accepted as a conservative scaffold and runnable command/interface migration. Not accepted as a finished paper-result package because full CityFlow smoke/main performance experiments were not run in this local turn.

## Fixed review issues

The review agent found two fatal issues:

1. generated experiment commands did not parse against `jinan_pair_ablation_runner`;
2. gate ablations were declarative but not wired into admission.

Both were fixed. The runner now supports `--r2v on/off/paired` and the requested traffic R2V flags, and candidate admission now respects `gate_variant`.

A follow-up runnable-path check found another integration gap: `repair_story=not_rare_to_val` failed on ordinary transition buffers without source/final repair gate metadata. This is now fixed through an explicit `repair_metadata_policy` switch: low-level APIs default to `require_metadata`, while runner/smoke commands can use `metadata_or_proxy` and mark proxy gates as `computed_proxy_repair_metadata`.

A later final review found two paper-evidence fatal issues: generated main R2V commands did not consume the diffusion artifacts that readiness and manifest steps referenced, and weighted artifacts did not prove they came from bundled diffusion score artifacts. Both are now fixed. Main R2V commands include per-seed `--r2v_artifact_path`, validation checks that wiring, weighted rows record score-artifact provenance, and the paper manifest blocks diffusion weighted artifacts whose score paths do not match bundled diffusion-score artifacts.

The original-method audit also found that rarity was accidentally used as a final admission gate. That is now fixed: rarity remains a detector/source/sampling signal, while final admission uses value/OOD, support, and dynamics/safety gates.

## Evidence

- Targeted pytest: `163 passed`.
- Py compile: touched R2V modules passed.
- Runner dry-run baseline-off: `DRY_RUN_READY`, 7 commands, no R2V artifact step.
- Runner dry-run R2V-on/no_dynamics: `DRY_RUN_READY`, 8 commands, R2V artifact step with `--gate_variant no_dynamics`.
- Score artifact builder: proxy rows are loader-compatible and marked `paper_claim_eligible=false`; readiness accepts them when they cover transition IDs.
- Strict paper readiness: `--require_paper_claim_eligible_diffusion` rejects proxy diffusion artifacts, missing diffusion provenance, and proxy adapters before paper-result aggregation.
- Strict repair-metadata readiness: `--require_strict_repair_metadata_policy` rejects `metadata_or_proxy` and accepts `require_metadata`, so proxy repair metadata cannot slip into final paper diffusion checks.
- Strict paper-readiness command generation: one preflight command is generated per main seed, with real diffusion artifact and strict repair-metadata requirements.
- Main R2V artifact wiring: each generated main R2V runner command consumes the matching per-seed diffusion score artifact through `--r2v_artifact_path`.
- Performance-readiness command generation: one paper-table preflight command is generated to require baseline/R2V x seed coverage, all five traffic metrics, and completed status.
- Result-aggregation command generation: one final aggregation command is generated from the expected performance rows and per-seed integrity summaries.
- Main-pipeline command generation: `build_main_pipeline_commands` returns the ordered 3-seed main plan: strict paper readiness, baseline/R2V runs, performance readiness, aggregation, and paper artifact manifesting.
- Main-pipeline strictness: generated R2V main runner commands now use `--repair_metadata_policy require_metadata`, matching the strict paper-readiness checks.
- Main-pipeline validation: `--include_validation` checks command order, baseline-off flags, R2V diffusion/not_rare_to_val/full flags, per-seed diffusion artifact wiring, and paper-manifest artifact coverage.
- Experiment-plan CLI export: `python -m pareto.r2v.traffic_experiment_plan` writes JSON manifests and reviewable shell scripts for smoke/main/main-pipeline/ablation plans.
- Paper artifact manifesting: `python -m pareto.r2v.paper_artifact_manifest` records SHA256 hashes and explicit artifact types, blocks missing artifacts, and is now generated as the last main-pipeline command.
- Paper artifact content validation: files labeled `performance` must contain all five required traffic metrics, so status-only or partial-metric rows cannot enter the evidence bundle as performance.
- Paper aggregation content validation: files labeled `aggregation` must use the result aggregation schema, include non-empty performance rows, and contain all five required traffic metrics.
- Paper aggregation provenance validation: aggregation JSON must include SHA256 hashes for both performance and integrity input artifacts, and those hashes must match artifacts bundled in the same paper manifest.
- Paper integrity content validation: files labeled `integrity` must use the R2V candidate summary schema and include candidate/admitted counts plus rare/value/support/safety gate counts.
- Paper diffusion-score content validation: files labeled `diffusion_score` must be paper-eligible, non-proxy, and carry `model_checkpoint`, `config_hash`, and `normalization_id`; proxy or missing-provenance score rows block the evidence bundle.
- Paper weighted-transition content validation: files labeled `weighted_transitions` must satisfy the v2 R2V-Traffic artifact schema; legacy v1-only, malformed weighted rows, invalid row roles, or repair-rejected/admitted contradictions block the evidence bundle.
- Paper weighted/diffusion provenance validation: diffusion-labeled weighted artifacts must record score-artifact paths that match bundled `diffusion_score` artifacts.
- Paper readiness content validation: files labeled `readiness` must report `status=READY` and `failed_count=0`; blocked preflight reports block the evidence bundle.
- Performance table readiness: expected baseline/R2V methods, seeds, all five metrics, and completed status are machine-checked before aggregation.
- Result aggregation: `row_count` counts performance rows, `metric_value_count` counts metric observations, integrity/status artifacts remain outside performance counts, and the CLI writes the JSON aggregation artifact.
- Candidate build reproduction: ordinary transitions plus proxy score artifact can build `not_rare_to_val` weighted artifacts with `--repair_metadata_policy metadata_or_proxy`.
- Runtime weighted-output test: `build_candidates_from_files(... weighted_output=...)` emits v2 traffic metadata and passes `validate_r2v_traffic_artifact`.
- Runtime pair-sampling validation: R2V sampling now validates v2 traffic weighted artifacts and rejects legacy v1-only weighted files before pair construction.
- Proxy/diffusion provenance validation: weighted artifacts built without a real score artifact record proxy backend provenance, and runner plans only pass `--score_artifact_backend diffusion` alongside an actual score artifact path.
- Metadata ID validation: v2 weighted artifacts reject mismatches between row IDs and `metadata.r2v_traffic_transition_id` / `metadata.r2v_traffic_sample_id`.
- Rare-as-detector validation: selector tests show full admission can pass when rare is false but value/support/dynamics pass, while rare-but-not-valuable samples remain rejected.
- Admission-mode contract: weighted artifacts record `r2v_admission_mode`; default `weights_only` preserves the source transition set, while opt-in `weights_plus_repaired` appends explicit admitted repaired payloads and records gate-rejected repair proposals only as `r2v_row_role=repair_rejected`, `r2v_admitted=false`.
- Admitted-weight contract: runner `--r2v_admitted_weight` maps to exact candidate-builder `--admitted_weight`, matching original R2V admission semantics.
- External repair artifact path: diffusion score/repair artifacts may provide row-level `repaired_transition` payloads; `weights_plus_repaired` appends them only after independent admission gates pass.
- Repair-payload fail-closed validation: `repaired_transition` payloads must be objects with non-empty `transition_id` and `sample_id`; readiness and paper manifest reject malformed repair proposals.
- Readiness checker: current checkout exits with code `2` and reports `BLOCKED` with missing CityFlow data, transition inputs, and diffusion score artifact instead of silently proceeding. Tests also cover malformed diffusion score rows and incomplete transition-ID coverage.

## Boundary checks

- `rare != valuable`: preserved in selector and tests; rarity is not a final admission gate.
- Zero reward is not treated as corruption: no zero-reward filter was added.
- Generator/proposal is not final admission: gates decide admission.
- Bellman/PPO target code was not modified.
- Baseline default behavior is preserved through `--r2v_sampling_mode off`.
- Weighted outputs retain legacy pair-builder fields and add v2 traffic schema fields.
- `weights_plus_repaired` is an explicit interface, not the default paper path.
- Metrics/status separation is enforced by `result_aggregation.py`.
- Smoke/main preconditions are machine-checkable through `experiment_readiness.py`.
- Proxy artifacts are machine-blocked from strict paper diffusion readiness.
- Proxy smoke artifacts are not labeled as diffusion weighted-transition evidence unless a real score artifact is supplied.
- Diffusion weighted-transition evidence must link back to bundled diffusion score artifacts.
- Partial or unfinished main performance tables are machine-blocked from paper aggregation.

## Remaining risks

Full diffusion repair is represented through score-artifact/backend interfaces, proxy score artifacts, and conservative weighting; true repaired traffic-state append mode remains future work. Proxy artifacts and proxy repair metadata are integration aids, not paper diffusion evidence. Real paper claims require completed traffic metric rows.

Local CityFlow smoke was attempted but blocked before simulator startup because `data/Jinan/3_4/anon_3_4_jinan_real.json` is absent in this checkout.
