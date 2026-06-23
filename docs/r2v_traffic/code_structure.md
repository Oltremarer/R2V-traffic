# Code Structure

## New modules

- `pareto/r2v/traffic_artifact_schema.py`: v2 traffic artifact metadata, config flags, gate variant config, v1-to-v2 upgrade, validation summary.
- `pareto/r2v/result_aggregation.py`: generic R2V performance-vs-integrity aggregation plus CLI JSON writer.
- `pareto/r2v/paper_artifact_manifest.py`: paper evidence-bundle manifest builder with file hashes, explicit artifact types, and fail-closed missing-artifact status.
- `pareto/r2v/traffic_experiment_plan.py`: declarative smoke/main/ablation command plans plus strict paper-readiness, performance-readiness, result-aggregation, complete main-pipeline commands, and JSON/shell CLI export.
- `pareto/r2v/experiment_readiness.py`: fail-closed smoke/main readiness checks for CityFlow data, transition inputs, diffusion score artifacts, and performance metric rows.
- `pareto/r2v/build_generative_score_artifact.py`: deterministic proxy score-artifact builder for smoke/integration runs; generated rows are explicitly not paper-claim eligible.

## Existing modules reused

- `pareto/r2v/artifact_validation.py`: v1 fail-closed weighted transition validation.
- `pareto/r2v/generative_scorer.py`: score-artifact loader; now preserves adapter and `paper_claim_eligible` provenance for strict readiness checks.
- `pareto/r2v/traffic_candidate_selector.py`: traffic R2V candidate discovery, repair-story matching, gate calculation, sample weights.
- `pareto/data/build_pairs.py`: opt-in R2V weighted pair sampling.
- `pareto/r2v/jinan_pair_ablation_runner.py`: existing runner for candidate/pair/model ablation.

## Public imports

`pareto/r2v/__init__.py` now exports `R2VTrafficConfig`, `build_gate_variant_config`, and `validate_r2v_traffic_artifact`.
