# Traffic R2V Setting

## Transition

Traffic R2V uses

\[
x_i = (s_i, a_i, r_i, s'_i, done_i, intersection_id, time_id, episode_id).
\]

In code, this maps to `TransitionRecord` fields:

- `obs_features` as \(s_i\);
- `action` as \(a_i\);
- `env_reward` and objectives as reward/utility signals;
- `next_obs_features` as \(s'_i\);
- `done` as horizon or terminal marker;
- `intersection_id`, `step`, and `episode` as trace metadata;
- `transition_id` and `sample_id` as stable join keys.

## State

State can include queue length, waiting time, current phase, time in phase, lane vehicle count, pressure, movement pressure, adjacency, and features already encoded by the repo's state encoder.

## Action

The action is the controller-selected traffic signal phase or phase switch. Existing code maps set-style actions to CityFlow phases, with yellow handled by the environment.

## Reward

Reward may be negative queue, negative delay, pressure reward, travel-time proxy, or Pareto objective utilities. Low or zero reward is not corruption.

## Rare and valuable

Rare means uncommon under the behavior data or detector score: unusual queue patterns, phase-pressure mismatch, spillback-like congestion, or multi-intersection coordination patterns.

Valuable means useful training signal for improving travel time, queue, delay, throughput, or reward under evaluation. Rare does not imply valuable. A rare gridlock transition can be harmful; an ordinary-looking transition near a decision boundary can be valuable.

## Conclusion

R2V-Traffic should be described as replay distribution shaping with explicit candidate discovery and independent admission, not as a claim that rare traffic states are inherently good.
