# Implementation Log

## Repository state

Target repo:

- path: `/Users/azure/Documents/R2V-traffic`
- branch: `main`
- base commit: `1468e67b1257d79532de8dc42529d1ab43192446`
- remote diff at start after checkout: `0 ahead / 0 behind origin/main`

Original R2V repo:

- path: `/Users/azure/Documents/R2V`
- branch: `main`
- commit: `77a97317f9f2b4f11c80f6c7a90fa770a0cc93ee`
- remote diff: `0 ahead / 0 behind origin/main`

## Work completed

1. Created the goal and launched parallel subagents for literature, traffic code, original R2V method, migration matrix, math/method, motivation, implementation review, testing, experiments, final review, and macro audit.
2. Added focused tests first for v2 artifact schema, result aggregation, and experiment plans.
3. Implemented `traffic_artifact_schema.py`.
4. Implemented `result_aggregation.py`.
5. Implemented `traffic_experiment_plan.py`.
6. Wired gate variants into runtime candidate admission and `build_r2v_candidates`.
7. Added compatible `--r2v on/off/paired`, gate, backend, and artifact flags to `jinan_pair_ablation_runner`.
8. Wired v2 traffic artifact metadata into runtime weighted outputs while preserving the v1 fields used by pair sampling.
9. Added `experiment_readiness.py` to make missing CityFlow data, transition inputs, diffusion artifacts, and performance metrics explicit machine-readable blockers.
10. Added `build_generative_score_artifact.py` so transition buffers can produce loader-compatible proxy score artifacts for smoke/integration runs.
11. Exported new schema helpers from `pareto/r2v/__init__.py`.
12. Generated the requested `docs/r2v_traffic/` documents.

## Important bug fixed

The first test run found that baseline command plans still emitted `--r2v_sampling_mode full_r2v`. I fixed this by allowing `off` in the config layer, forcing `r2v=off` to emit `--r2v_sampling_mode off`, and rejecting `r2v=on` with sampling mode `off`.

The review agent then found two fatal integration gaps: experiment-plan commands did not parse against the runner, and gate variants were declarative only. I fixed both by adding runner-compatible flags, parser/dry-run tests, and active-gate admission logic.

Follow-up hardening found that v2 artifact validation existed as a helper but was not emitted by the actual weighted artifact path. I fixed this by upgrading `apply_candidate_weights()` output to include `metadata.r2v_traffic_schema_version`, traffic gate aliases, gate variant, and generative backend, then validating the rows before returning.

The local CityFlow smoke blocker is now represented by an executable readiness report instead of prose only. In this checkout, the checker reports missing Jinan traffic file, roadnet file, transition inputs, and diffusion score artifact. When artifacts exist, it also validates score schema, finite rarity/support scores, and transition-ID coverage.

The score artifact builder intentionally marks generated proxy rows with `paper_claim_eligible=false`. These artifacts are valid for smoke and integration wiring, but true paper diffusion claims still require actual diffusion-produced score/repair artifacts.

Follow-up runnable-path testing found that `repair_story=not_rare_to_val` could still fail on ordinary transition buffers because source/final repair gate metadata was required before candidate construction. I kept the low-level selector default as `require_metadata` but added an explicit `repair_metadata_policy=metadata_or_proxy` path for runner/smoke use. Proxy repair metadata is marked with `gate_source=computed_proxy_repair_metadata`, so integration smoke is runnable without confusing it with true repaired diffusion artifacts.

I also added strict paper-readiness enforcement for diffusion score artifacts. `experiment_readiness.py --require_paper_claim_eligible_diffusion` rejects proxy artifacts marked `paper_claim_eligible=false`, artifacts that omit the field, artifacts missing `model_checkpoint`/`config_hash`/`normalization_id`, and artifacts that use proxy adapters.

I added paper-table performance readiness checks as well. The readiness CLI can now require expected methods, expected seeds, and completed status, so a partial baseline/R2V x 3-seed table cannot be mistaken for final paper evidence.

I tightened strict paper readiness again by adding `--repair_metadata_policy` plus `--require_strict_repair_metadata_policy`. Final paper diffusion readiness now requires `repair_metadata_policy=require_metadata`; the `metadata_or_proxy` path remains available for smoke/integration only.

I then mirrored that strict readiness contract in `traffic_experiment_plan.py` with `build_strict_paper_readiness_commands(spec)`, so the generated experiment plan can emit one paper preflight command per main seed instead of relying only on hand-written docs.

I also added a `python -m pareto.r2v.result_aggregation` CLI so the final performance-vs-integrity aggregation can be run as a normal experiment command instead of a Python heredoc.

Finally, I added `build_result_aggregation_command(spec)` to `traffic_experiment_plan.py`, so the generated experiment plan includes the final aggregation command and does not require hand-assembling performance/integrity paths.

I also added `build_performance_readiness_command(spec)` so the generated main experiment plan includes the paper-table readiness check before aggregation. This checks baseline/R2V method coverage, the main seeds, all traffic metrics, and completed evaluation status.

Finally for the command planner, I added `build_main_pipeline_commands(spec)`. It returns the complete ordered main pipeline: strict paper readiness, baseline/R2V main runs, performance readiness, result aggregation, and paper artifact manifesting.

I then exposed the command planner as `python -m pareto.r2v.traffic_experiment_plan`, with JSON and shell output. This makes the smoke/main/main-pipeline/ablation plans exportable without writing ad hoc Python snippets.

I added `paper_artifact_manifest.py` as the paper evidence-bundle index. It records file hashes and explicit artifact types for performance rows, readiness reports, aggregation outputs, diffusion scores, weighted transitions, and integrity/status artifacts. Missing files make the manifest `BLOCKED` instead of silently producing a paper bundle.

I then wired the paper artifact manifest command into `traffic_experiment_plan.py`, so `main_pipeline` ends by generating the evidence manifest and `--plan paper_artifact_manifest` can be exported independently.

I added `validate_experiment_plan(...)` and `--include_validation` for JSON command-plan export. The main-pipeline validation checks command order, baseline-off flags, R2V diffusion/not_rare_to_val/full flags, and paper artifact manifest coverage.

I tightened `paper_artifact_manifest.py` so artifacts labeled `performance` must contain all five required traffic metrics. Status-only rows and partial-metric rows now make the manifest `BLOCKED`, preserving the performance-vs-integrity boundary at the final evidence-bundle layer.

I also tightened `paper_artifact_manifest.py` for artifacts labeled `diffusion_score`. The manifest now blocks proxy score artifacts, rows with `paper_claim_eligible=false` or a missing eligibility flag, rows with proxy adapters such as `traffic_feature_density_proxy`, and rows missing `model_checkpoint`, `config_hash`, or `normalization_id`.

I then tightened `build_main_pipeline_commands(spec)` so strict paper readiness and the final R2V runner commands agree on repair metadata. The generated main pipeline now runs R2V with `--repair_metadata_policy require_metadata`; `metadata_or_proxy` remains available for smoke/integration plans but is no longer the generated final-paper main path.

I also tightened `paper_artifact_manifest.py` for artifacts labeled `weighted_transitions`. The manifest now runs `validate_r2v_traffic_artifact` and records weighted-row, admitted-count, gate-count, weight, gate-variant, backend, admission-mode, and row-role summaries. Legacy v1-only or malformed weighted transition files block the paper evidence bundle.

I also tightened `paper_artifact_manifest.py` for artifacts labeled `readiness`. The manifest now requires readiness JSON reports to have `status=READY` and `failed_count=0`; a `BLOCKED` preflight report can no longer be bundled just because the file exists.

I also tightened `paper_artifact_manifest.py` for artifacts labeled `aggregation`. The manifest now requires the `r2v-traffic-result-aggregation-v1` schema, non-empty performance rows, at least one method, and all five required traffic metrics. Status-only or wrong-schema aggregation files block the paper evidence bundle.

I also tightened `paper_artifact_manifest.py` for artifacts labeled `integrity`. The manifest now requires the R2V candidate summary schema, candidate/admitted counts, and rare/value/support/safety gate counts. Status-only integrity files block the paper evidence bundle.

I then added input-artifact provenance to `result_aggregation.py`. Aggregation JSON now records the performance and integrity input paths, SHA256 hashes, sizes, and line counts. The paper manifest rejects aggregation artifacts that omit those input hashes, so the final result table stays traceable to the exact evidence files.

I further tightened the paper manifest by cross-checking aggregation input hashes against the performance and integrity artifacts bundled in the same manifest. If the aggregation table was built from different files than the bundle contains, the manifest is `BLOCKED`.

I added an explicit R2V admission-mode contract. The default `weights_only` path preserves the original transition set and records `r2v_admission_mode=weights_only` in weighted artifacts. The opt-in `weights_plus_repaired` path appends admitted repaired transition rows only when an explicit repaired transition payload is present; admitted rows without payloads fail closed. Runner and experiment-plan commands now expose `--r2v_admission_mode`, and the paper manifest records weighted-transition admission modes.

I then connected external score/repair artifacts to the repaired append path. `load_generative_score_artifact()` now preserves an optional row-level `repaired_transition` object. Candidate selection propagates that payload, and `weights_plus_repaired` can append it as an admitted repaired row after gates admit the candidate. Source transition metadata remains a fallback for explicit payloads, but admitted candidates without repaired payloads still fail closed.

I tightened that repair payload contract again so malformed repair proposals are blocked before they reach paper evidence or replay construction. A score-artifact `repaired_transition` must be an object with non-empty `transition_id` and `sample_id`; both readiness and paper-manifest validation now reject artifacts that violate this.

I wired `repair_rejected_weight` into the traffic selector, candidate-builder CLI, and Jinan runner command plan. In `weights_plus_repaired`, an explicit repaired proposal that still fails admission gates is appended only as `r2v_row_role=repair_rejected`, with `r2v_admitted=false` and the configured rejected weight. This mirrors the original R2V repair-rejected channel while keeping generator proposals separate from final admission.

I then tightened the v2 weighted-transition artifact validator so row role semantics are machine-checked. Unknown `r2v_row_role` values are rejected, `repaired` rows must be admitted and link back to `r2v_repaired_from_transition_id`, and `repair_rejected` rows must be non-admitted with `r2v_repair_rejected=true` and a source link.

I also corrected admitted-weight semantics at the runner boundary. The high-level `--r2v_admitted_weight` flag now forwards to exact `--admitted_weight`, matching the original R2V admission behavior, instead of being treated as a bonus on top of `base_weight`. The older low-level `--admitted_weight_bonus` remains available for score-proportional exploratory weighting.

I then moved v2 weighted-artifact validation into the runtime pair builder. When `r2v_sampling_mode` is enabled, `pareto.data.build_pairs` now calls `validate_r2v_traffic_artifact` before building pairs and records the traffic artifact summary in the candidate-sampling report. Legacy v1-only weighted files are rejected before pair construction, not only later by the paper manifest.

I then fixed a provenance boundary in the candidate builder and Jinan runner. Proxy candidate selection without a real `--score_artifact` now records the candidate model, for example `feature_density_proxy`, as `metadata.r2v_generative_backend`; it does not inherit the runner's paper-facing `generative_backend=diffusion` label. The runner now only passes `--score_artifact_backend diffusion` when an actual `--score_artifact` path is provided.

The final review then found that main-pipeline preflight and manifest commands referenced diffusion artifacts, but the generated R2V runner commands did not actually consume those artifacts. I fixed the main command plan so each seed's R2V run includes `--r2v_artifact_path <diffusion_seed{seed}_scores.jsonl>`, and main-pipeline validation now checks that per-seed wiring explicitly.

I also added weighted-artifact score provenance. Weighted rows produced from score artifacts now record `metadata.r2v_score_artifact_path` and `metadata.r2v_score_artifact_backend`. The paper manifest blocks diffusion-labeled weighted artifacts unless their recorded score artifact paths match bundled `diffusion_score` artifacts. This prevents proxy-derived weighted transitions from sitting beside diffusion scores in the same paper evidence bundle.

I tightened traffic artifact validation again so `metadata.r2v_traffic_transition_id` and `metadata.r2v_traffic_sample_id` must match the row's own `transition_id` and `sample_id`. Mismatched metadata IDs now fail closed instead of only being reported in docs.

Finally, I aligned the traffic full-gate implementation with the method story: rarity remains a detector/candidate/source signal, but it is not a final admission gate. The active admission gates are value/OOD, support, and dynamics/safety; ablations remove one of those three. This keeps `rare != valuable` operational rather than merely textual.

## Boundary

No full CityFlow training/evaluation was run in this local macOS session. Verification is unit, schema, command-plan, runner dry-run, and py_compile level.
