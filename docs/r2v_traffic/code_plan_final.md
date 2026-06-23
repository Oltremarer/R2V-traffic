# Code Plan Final

## Stage 0: artifact/schema validation

Implemented `traffic_artifact_schema.py` with:

- `R2VTrafficConfig`;
- `build_gate_variant_config`;
- `upgrade_weighted_row_to_v2_metadata`;
- `validate_r2v_traffic_artifact`.

Runtime weighted outputs now include v2 traffic metadata in addition to the v1 pair-builder fields.

## Stage 1: opt-in weighted replay

Existing pair builder remains the runtime hook. New config makes baseline-off explicit and tested.

## Stage 2: detector/candidate discovery

Existing `traffic_candidate_selector.py` remains the candidate path. Diffusion is represented as a backend/config and score-artifact interface. Full denoise-and-append traffic repair remains future work.

## Stage 3: repair/proposal

`not_rare_to_val` and `not_val_to_val` remain supported through selector repair-story metadata and tests.

## Stage 4: traffic gates

Full and ablation gate variants are represented in `R2VTrafficConfig`, wired into candidate admission, forwarded by the runner, and tested.

Admission mode is explicit. The default `weights_only` mode preserves the original transition set and only changes replay weights. `weights_plus_repaired` is opt-in and requires explicit repaired transition payloads before appending repaired rows; repaired proposals rejected by gates must stay marked as non-admitted `repair_rejected` rows.

## Stage 5: runner and scripts

Implemented `traffic_experiment_plan.py` to generate smoke/main/ablation commands with stable names, output paths, and W&B metadata. It also generates strict paper-readiness commands, paper-table performance readiness, the final result-aggregation command, the paper artifact manifest command, and a complete ordered main pipeline for the main seeds. The same plans can be exported as JSON or shell through `python -m pareto.r2v.traffic_experiment_plan`. `jinan_pair_ablation_runner` now accepts the R2V runner flags for baseline-off, R2V-on, and paired runs.

## Stage 6: summary / aggregation

Implemented `result_aggregation.py` so performance metrics and integrity/status artifacts stay separate.
