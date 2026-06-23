# Replay Buffer Analysis

## What was checked

I inspected the legacy replay construction, Pareto transition schema, R2V candidate selector, artifact validation, and pair builder.

## Legacy replay

Legacy replay is generated from `inter_*.pkl` traces. Each sample contains current state, action, next state, average reward, instant reward, time, and a folder/round identifier. The classic Q-target code in learned agents bootstraps from `next_state`; terminal handling is mostly through finite rollout slicing.

## Pareto replay

`TransitionRecord` includes:

- `transition_id`
- `sample_id`
- `next_sample_id`
- `obs_features`
- `next_obs_features`
- `action`
- `env_reward`
- objective values and masks
- `done`
- metadata

This is the correct home for R2V because it gives stable join keys and enough metadata for fail-closed validation.

## R2V weighted replay

Existing `pareto/r2v/traffic_candidate_selector.py` outputs candidate rows and weighted transition metadata. `pareto/r2v/artifact_validation.py` requires:

- unique transition IDs;
- unique join keys;
- schema version;
- `metadata.r2v_admitted`;
- legacy diagnostic gates: rare, value, support, safety;
- finite positive `metadata.r2v_sample_weight`.

`pareto/data/build_pairs.py` supports opt-in sampling modes:

- `full_r2v`
- `admitted_only`
- `rare_only`
- `value_only`
- `random_same_count`
- `same_candidates_random_weights`
- `shuffled_value`
- `inverted_rarity`

The same pair builder also supports non-R2V ER baseline modes, including recency, pressure-priority, reward-priority, TD-error-priority, diversity, and balancing variants. These modes can create non-uniform baselines even when `r2v_sampling_mode=off`; therefore paper comparisons should name the base replay policy explicitly, such as `baseline_uniform` or `baseline_recent`, and compare it to the matching R2V overlay.

The R2V weighted artifact intentionally carries both compatibility and traffic-specific schemas. The legacy metadata version, `r2v-tsc-weighted-transition-v1`, is what older pair-sampling code expects for weights and admission flags. The v2 traffic artifact version, `r2v-traffic-artifact-v2`, adds explicit traffic gate names, row roles, admission modes, and backend provenance. Pair construction now requires the v2 layer whenever R2V sampling is enabled.

Gate names differ slightly across layers. The selector computes legacy `rare/value/support/safety` gates. The traffic artifact exposes `rare/ood/support/dynamics`, where `value` becomes the OOD/value gate and `safety` becomes the dynamics/consistency gate for traffic reporting.

R2V weights affect offline replay/pair sampling. They do not modify the PPO rollout buffer or GAE/return computation in `pareto/rl/ppo_buffer.py`, and they do not change Bellman target formulas in the learned Q-style code. This is the implementation-level version of the method claim: R2V changes the empirical replay distribution or admitted transition set, not the downstream target definition.

## Conclusion

The current code already implements the key interface needed for conservative R2V-Traffic: it can build weighted transition artifacts and use them for pair sampling without changing learner Bellman targets.
