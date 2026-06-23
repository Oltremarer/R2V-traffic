# Files Changed

## Code

- `pareto/r2v/__init__.py`
- `pareto/r2v/build_generative_score_artifact.py`
- `pareto/r2v/build_r2v_candidates.py`
- `pareto/r2v/experiment_readiness.py`
- `pareto/r2v/generative_scorer.py`
- `pareto/r2v/jinan_pair_ablation_runner.py`
- `pareto/r2v/paper_artifact_manifest.py`
- `pareto/r2v/traffic_candidate_selector.py`
- `pareto/r2v/traffic_artifact_schema.py`
- `pareto/r2v/result_aggregation.py`
- `pareto/r2v/traffic_experiment_plan.py`
- `pareto/data/build_pairs.py`

## Tests

- `tests/pareto/test_r2v_traffic_artifact_schema.py`
- `tests/pareto/test_r2v_build_score_artifact.py`
- `tests/pareto/test_r2v_experiment_readiness.py`
- `tests/pareto/test_r2v_result_aggregation.py`
- `tests/pareto/test_r2v_jinan_pair_ablation_runner.py`
- `tests/pareto/test_r2v_paper_artifact_manifest.py`
- `tests/pareto/test_r2v_traffic_experiment_plan.py`
- `tests/pareto/test_r2v_traffic_candidate_selector.py`
- `tests/pareto/test_pair_builder.py`

## Docs

- Full directory: `docs/r2v_traffic/`

## Existing behavior touched

No baseline learner or Bellman/PPO target code was modified. Existing R2V candidate selector and pair builder were reused and extended only at the R2V admission/runner boundary. Admission mode is now explicit, with conservative `weights_only` as default and `weights_plus_repaired` opt-in only when repaired transition payloads exist. Gate-rejected repaired proposals are marked as `repair_rejected`, not admitted. Proxy candidate runs without a real score artifact are labeled with the proxy candidate model instead of diffusion backend provenance.
