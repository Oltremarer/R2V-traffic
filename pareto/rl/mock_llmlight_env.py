from __future__ import annotations

from dataclasses import dataclass

import torch

from pareto.constants import OBJECTIVE_NAMES


@dataclass
class MockLLMLightState:
    obs: torch.Tensor
    objectives_norm: list[dict[str, float]]
    sim_time: int
    step: int


class MockLLMLightEnv:
    def __init__(
        self,
        num_intersections: int = 12,
        obs_dim: int = 193,
        min_action_time: int = 30,
        max_steps: int = 10,
        seed: int = 0,
    ) -> None:
        self.num_intersections = int(num_intersections)
        self.obs_dim = int(obs_dim)
        self.min_action_time = int(min_action_time)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.current_state: MockLLMLightState | None = None

    def _make_obs(self, step: int) -> torch.Tensor:
        base = torch.arange(self.num_intersections * self.obs_dim, dtype=torch.float32).reshape(
            self.num_intersections, self.obs_dim
        )
        return torch.sin(base * 0.001 + float(step) * 0.1)

    def _make_objectives(self, step: int, actions: list[int] | None = None) -> list[dict[str, float]]:
        values: list[dict[str, float]] = []
        for idx in range(self.num_intersections):
            action_term = 0.0 if actions is None else float(actions[idx]) * 0.01
            base = 0.05 * float(step) + 0.01 * float(idx) + action_term
            values.append({name: base + 0.1 * objective_idx for objective_idx, name in enumerate(OBJECTIVE_NAMES)})
        return values

    def reset(self) -> MockLLMLightState:
        self.current_state = MockLLMLightState(
            obs=self._make_obs(0),
            objectives_norm=self._make_objectives(0),
            sim_time=0,
            step=0,
        )
        return self.current_state

    def step(self, action_list: list[int]) -> tuple[MockLLMLightState, list[float], bool, dict]:
        if self.current_state is None:
            raise RuntimeError("reset must be called before step")
        if len(action_list) != self.num_intersections:
            raise ValueError(f"action_list length must be {self.num_intersections}, got {len(action_list)}")
        step = self.current_state.step + 1
        next_state = MockLLMLightState(
            obs=self._make_obs(step),
            objectives_norm=self._make_objectives(step, action_list),
            sim_time=step * self.min_action_time,
            step=step,
        )
        rewards = [
            float(next_state.objectives_norm[idx]["efficiency"] - 0.05 * float(action_list[idx]))
            for idx in range(self.num_intersections)
        ]
        done = step >= self.max_steps
        self.current_state = next_state
        return next_state, rewards, done, {
            "sim_time": next_state.sim_time,
            "min_action_time": self.min_action_time,
            "num_intersections": self.num_intersections,
            "mock_env": True,
        }
