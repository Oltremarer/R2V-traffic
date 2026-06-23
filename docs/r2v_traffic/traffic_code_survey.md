# Traffic ER / RL Code Survey

## State

`Intersection._update_feature` in `utils/cityflow_env.py` builds traffic features such as phase, time in phase, lane vehicle counts, waiting counts, pressure, movement pressure, adjacency, and segment attendance. `Intersection.get_state` filters those fields using `LIST_STATE_FEATURE`.

In the Pareto stack, `pareto/rl/state_encoder.py` converts snapshots into `obs_features` and `next_obs_features`.

## Action / phase

`Intersection.set_signal` maps controller actions to CityFlow phases. With the default set-action pattern, action `a` maps to phase `a + 1`; phase `0` is the yellow transition phase. MaxPressure-style agents compute pressure-based phase choices in `models/maxpressure_agent.py` and related files.

## Reward

Traffic rewards are weighted sums over queue and pressure components from `DIC_REWARD_INFO`. Pareto collection can use queue-penalty environment rewards through `pareto/rl/env_reward_source.py`. Zero or low immediate reward is not corruption; it must be interpreted in context.

One code nuance matters for experiments: LLMLight default reward weights in config can be zeroed for some components, while the Pareto env-reward path can explicitly switch to a queue-penalty proxy. Reward columns should therefore be interpreted with the exact runner/config that produced them, not assumed to be comparable across every baseline by name alone.

## Next state / done

`CityFlowEnv.step` advances the simulator second-by-second inside a macro-action. The wrapper often does not terminate internally; rollout loops impose horizon bounds. `TransitionRecord.done` exists in the Pareto schema and marks collection horizon or episode end.

## Replay / pair construction

Legacy samples are `[state, action, next_state, reward_average, reward_instant, time, folder-round_id]`.

Pareto transitions include IDs and metadata. Pair construction is centralized in `pareto/data/build_pairs.py`. With `--r2v_sampling_mode off`, weighted-transition input is rejected and internal R2V fields are stripped. With R2V enabled, weighted JSONL artifacts must satisfy the v2 R2V-Traffic artifact schema before sampling; legacy v1-only weighted files are rejected before pair construction.

The weighted artifact has two layers on purpose: legacy `r2v-tsc-weighted-transition-v1` metadata keeps the existing pair sampler compatible, while traffic `r2v-traffic-artifact-v2` metadata exposes traffic gates, row roles, admission mode, and backend provenance. The v2 gate names are `rare`, `ood`, `support`, and `dynamics`; they map from the legacy selector names `rare`, `value`, `support`, and `safety`.

`build_pairs.py` also contains ER baseline modes such as recent, pressure-priority, reward-priority, TD-error-priority, diversity, and balancing modes. These are baseline replay policies, not R2V. When a paper compares an ER baseline plus an R2V overlay, the baseline and overlay must use the same base ER mode so the only intended difference is the R2V weighted/admitted candidate layer.

R2V weights currently affect offline pair candidate sampling. They do not rewrite PPO rollout storage: `pareto/rl/ppo_buffer.py` still stores online trajectories and computes returns/GAE in the PPO path.

`collect_pareto_buffer.py` writes schema-first transition JSONL with feature-hash metadata, records `done` at the collection horizon, and emits run artifacts such as `metrics.csv` and `status.json`. These status artifacts are useful for integrity but are not performance rows by themselves.

## Evaluation and logging

Legacy evaluation logs reward, average queue length, waiting time, and travel time. Paper-style evaluation code uses explicit metric keys and should be kept separate from R2V artifact status.

Learned evaluation fills unfinished vehicle leave times using the configured run horizon, so travel-time numbers must be compared only under the same horizon/map protocol. `result_aggregation.py` refuses to treat integrity/status rows as performance rows.

## Remaining risks

The code supports replay weighting and pair sampling, but real end-to-end CityFlow experiments depend on local simulator availability and map data. On this macOS workspace I verified unit-level behavior, not full CityFlow runtime parity.
