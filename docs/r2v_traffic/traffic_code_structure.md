# Traffic Code Structure

## How this was completed

The traffic-code-survey agent inspected the repo read-only, then I merged its findings with local test and code checks.

## Legacy LLMLight / CityFlow stack

`utils/pipeline.py` drives legacy closed-loop training. It calls rollout generation in `utils/generator.py`, writes `inter_*.pkl`, constructs samples in `utils/construct_sample.py`, trains with `utils/updater.py`, and evaluates with `utils/model_test.py`.

`utils/cityflow_env.py` owns the CityFlow wrapper. `CityFlowEnv.reset` creates engine state and returns initial observations. `CityFlowEnv.step` applies macro-actions for `MIN_ACTION_TIME`, advances the simulator, logs state/action, and returns `(next_state, reward, done, average_reward_action_list)`.

## Pareto / R2V stack

The newer offline stack uses schema-first data:

- `pareto/common/scenario.py`: scenario config.
- `pareto/data/schema.py`: `TrajectoryRecord` and `TransitionRecord`.
- `pareto/data/collect_pareto_buffer.py`: transition collection.
- `pareto/rl/state_encoder.py`: state vector encoding.
- `pareto/data/objectives.py`: objective and metric scoring.
- `pareto/r2v/traffic_candidate_selector.py`: R2V candidate selection and weighted transition metadata.
- `pareto/r2v/artifact_validation.py`: fail-closed weighted artifact validation.
- `pareto/data/build_pairs.py`: pair construction with opt-in R2V sampling modes.
- `pareto/train_conditioned_scalar.py` and `pareto/train_vector_quality.py`: offline model training.
- `pareto/rl/*eval*`: learned/reference evaluation and aggregation.

## Conclusion

R2V-Traffic should attach to the Pareto/R2V stack first because it already has transition IDs, JSONL artifacts, sampling weights, tests, and explicit metric separation. The legacy CityFlow stack remains the source of environment behavior and baseline controllers.
