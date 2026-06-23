from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from pareto.constants import OBJECTIVE_NAMES
from pareto.models.vector_quality import score_with_preference


def _preference_tensor(w: Sequence[float] | torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(w, dtype=torch.float32, device=device)
    total = tensor.sum().clamp_min(1e-8)
    return tensor / total


def _objective_vector(objectives: Mapping[str, float], device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([float(objectives.get(name, 0.0)) for name in OBJECTIVE_NAMES], dtype=torch.float32, device=device)


@dataclass
class RewardResult:
    reward: float
    debug: dict


class WeightedProxyRewardAdapter:
    name = "weighted_proxy"

    def __init__(self, w: Sequence[float] | torch.Tensor) -> None:
        self.w = _preference_tensor(w)

    def compute(
        self,
        obs_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        objectives_t: Mapping[str, float],
        objectives_tp1: Mapping[str, float],
        done: bool,
    ) -> tuple[float, dict]:
        del obs_t, obs_tp1, objectives_t, done
        reward = float(torch.dot(self.w, _objective_vector(objectives_tp1)))
        return reward, {
            "adapter": self.name,
            "w": [float(value) for value in self.w.tolist()],
            "proxy_objectives_norm_tp1": {name: float(objectives_tp1.get(name, 0.0)) for name in OBJECTIVE_NAMES},
            "weighted_proxy_reward": reward,
            "total_reward": reward,
        }


class VectorQRewardAdapter:
    name = "vector_quality_potential"

    def __init__(
        self,
        model: torch.nn.Module,
        w: Sequence[float] | torch.Tensor,
        scorer: torch.nn.Module | None = None,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.scorer = scorer.to(device).eval() if scorer is not None else None
        self.w = _preference_tensor(w, torch.device(device)).reshape(1, -1)
        self.gamma = float(gamma)
        self.device = torch.device(device)

    def _score(self, obs: torch.Tensor) -> float:
        with torch.no_grad():
            q = self.model(obs.to(self.device).float().reshape(1, -1))
            score_fn = self.scorer if self.scorer is not None else score_with_preference
            return float(score_fn(q, self.w).detach().cpu().item())

    def compute(
        self,
        obs_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        objectives_t: Mapping[str, float],
        objectives_tp1: Mapping[str, float],
        done: bool,
    ) -> tuple[float, dict]:
        del objectives_t, objectives_tp1
        score_t = self._score(obs_t)
        score_tp1 = self._score(obs_tp1)
        reward = (0.0 if done else self.gamma * score_tp1) - score_t
        return float(reward), {
            "adapter": self.name,
            "w": [float(value) for value in self.w.reshape(-1).detach().cpu().tolist()],
            "vector_quality_score_t": score_t,
            "vector_quality_score_tp1": score_tp1,
            "potential_reward": float(reward),
            "total_reward": float(reward),
        }


class ScalarQualityRewardAdapter:
    name = "film_scalar_potential"

    def __init__(
        self,
        model: torch.nn.Module,
        w: Sequence[float] | torch.Tensor,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.w = _preference_tensor(w, torch.device(device)).reshape(1, -1)
        self.gamma = float(gamma)
        self.device = torch.device(device)

    def _score(self, obs: torch.Tensor) -> float:
        with torch.no_grad():
            score = self.model(obs.to(self.device).float().reshape(1, -1), self.w)
            return float(score.detach().cpu().reshape(-1)[0].item())

    def compute(
        self,
        obs_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        objectives_t: Mapping[str, float],
        objectives_tp1: Mapping[str, float],
        done: bool,
    ) -> tuple[float, dict]:
        del objectives_t, objectives_tp1
        score_t = self._score(obs_t)
        score_tp1 = self._score(obs_tp1)
        reward = (0.0 if done else self.gamma * score_tp1) - score_t
        return float(reward), {
            "adapter": self.name,
            "w": [float(value) for value in self.w.reshape(-1).detach().cpu().tolist()],
            "scalar_quality_score_t": score_t,
            "scalar_quality_score_tp1": score_tp1,
            "potential_reward": float(reward),
            "total_reward": float(reward),
        }
