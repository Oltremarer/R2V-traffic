from __future__ import annotations

from pareto.rl.preference_sampler import EpisodeFixedPreferenceSampler


def test_episode_fixed_preference_sampler_holds_w_within_episode():
    sampler = EpisodeFixedPreferenceSampler()

    name_0, w_0 = sampler.preference_for_episode(0)
    assert sampler.preference_for_step(0, 0) == (name_0, w_0)
    assert sampler.preference_for_step(0, 9) == (name_0, w_0)

    name_1, w_1 = sampler.preference_for_episode(1)
    assert (name_1, w_1) != (name_0, w_0)
    assert abs(sum(w_1) - 1.0) < 1e-6
