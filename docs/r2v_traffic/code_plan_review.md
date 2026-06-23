# Code Plan Review

## Review findings

The plan is intentionally conservative and matches the current repo shape. The target already had candidate selection, v1 weighted metadata, and pair sampling. The missing pieces were a traffic-specific v2 schema wrapper, explicit CLI/config semantics, result aggregation boundaries, and command plans.

## Key corrections from review

- `r2v=off` must force `--r2v_sampling_mode off` even if the config default sampling mode is `full_r2v`.
- `r2v=on` with sampling mode `off` should fail fast.
- The legacy gate name `value` maps to v2 `ood`; docs must explain this alias.
- Experiment command builders are declarative, and the target Jinan runner now accepts the generated R2V flags for dry-run and command planning.
- Full traffic performance still requires actual CityFlow evaluation artifacts, not only pair/model diagnostics.

## Decision

Proceed with v2 schema/config, runtime gate variants, result aggregation, experiment command planning, and focused tests.
