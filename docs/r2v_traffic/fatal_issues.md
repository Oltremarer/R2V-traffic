# Fatal Issues

## Current fatal issues

No fatal code/interface issues remain from the final review pass.

## Resolved

### FATAL-001: generated commands did not parse

Resolved by adding compatible flags to `jinan_pair_ablation_runner` and parser/dry-run tests.

### FATAL-002: gate ablation was not runtime-effective

Resolved by adding `gate_variant` to `R2VTrafficSelectorConfig`, forwarding it through `build_r2v_candidates` and the runner, and testing `no_dynamics`.

### FATAL-003: repair-story candidate build required unavailable repair metadata

Resolved by adding explicit `repair_metadata_policy` support. The strict low-level default remains `require_metadata`; runner/smoke paths can use `metadata_or_proxy`, which builds from computed gates and marks the source as `computed_proxy_repair_metadata`.

### FATAL-004: final main pipeline did not pass diffusion artifacts into R2V runner

Resolved by wiring `spec.diffusion_artifact_template.format(seed=seed)` into each main R2V command as `--r2v_artifact_path`, recording it in command metadata, and adding a validation check named `r2v_commands_use_seed_diffusion_artifacts`.

### FATAL-005: paper manifest did not prove weighted transitions came from bundled diffusion scores

Resolved by recording `metadata.r2v_score_artifact_path` in weighted rows and requiring diffusion-labeled weighted-transition artifacts to match bundled `diffusion_score` artifacts in `paper_artifact_manifest.py`.

### FATAL-006: rarity was an admission gate instead of a detector/source signal

Resolved by removing `rare` from active final-admission gates. The detector still computes rarity, and rare/source-story diagnostics remain available, but `full` admission now means value/OOD + support + dynamics/safety.

## Still blocked for paper performance claims

Full CityFlow smoke/main runs were not executed in this local session. A local smoke attempt failed because `data/Jinan/3_4/anon_3_4_jinan_real.json` is absent. Therefore performance improvement claims remain blocked until completed traffic metrics exist on a machine with the required data.
