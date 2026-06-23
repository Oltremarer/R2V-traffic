# Code Plan Draft

1. Add a v2 traffic R2V artifact/schema wrapper on top of existing v1 weighted-transition validation.
2. Add explicit `R2VTrafficConfig` flags for opt-in behavior, repair story, gate variant, diffusion backend, weights, and sampling mode.
3. Add a result aggregator that refuses to treat status/integrity artifacts as performance metrics.
4. Add experiment command builders for smoke, main, and ablation plans.
5. Add tests before implementation for config validation, gate variants, fail-closed artifacts, baseline-off commands, and performance/status separation.

Baseline behavior must remain unchanged when R2V is off.
