from __future__ import annotations

from dataclasses import dataclass


PREFERENCE_TEMPLATES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("efficiency", (1.0, 0.0, 0.0, 0.0)),
    ("safety", (0.0, 1.0, 0.0, 0.0)),
    ("fairness", (0.0, 0.0, 1.0, 0.0)),
    ("stability", (0.0, 0.0, 0.0, 1.0)),
    ("balanced", (0.25, 0.25, 0.25, 0.25)),
)


@dataclass(frozen=True)
class EpisodeFixedPreferenceSampler:
    templates: tuple[tuple[str, tuple[float, float, float, float]], ...] = PREFERENCE_TEMPLATES

    def preference_for_episode(self, episode: int) -> tuple[str, tuple[float, float, float, float]]:
        if not self.templates:
            raise ValueError("at least one preference template is required")
        return self.templates[int(episode) % len(self.templates)]

    def preference_for_step(self, episode: int, step: int) -> tuple[str, tuple[float, float, float, float]]:
        del step
        return self.preference_for_episode(episode)
