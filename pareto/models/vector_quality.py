from __future__ import annotations

import torch
from torch import nn

from pareto.constants import OBJECTIVE_NAMES


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers = []
    current_dim = input_dim
    for _ in range(max(0, num_layers - 1)):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class VectorQualityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_objectives: int = len(OBJECTIVE_NAMES),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_objectives = int(num_objectives)
        self.net = _mlp(self.input_dim, self.num_objectives, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class VectorQualityTowerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        trunk_layers: int = 2,
        head_layers: int = 2,
        dropout: float = 0.1,
        num_objectives: int = len(OBJECTIVE_NAMES),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_objectives = int(num_objectives)
        self.trunk = _mlp(self.input_dim, hidden_dim, hidden_dim, trunk_layers, dropout)
        self.heads = nn.ModuleList(
            _mlp(hidden_dim, 1, hidden_dim, head_layers, dropout)
            for _ in range(self.num_objectives)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x.float())
        return torch.cat([head(h) for head in self.heads], dim=-1)


class VectorQualityResidualTowerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        trunk_layers: int = 2,
        head_layers: int = 2,
        dropout: float = 0.1,
        num_objectives: int = len(OBJECTIVE_NAMES),
        tower_residual_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_objectives = int(num_objectives)
        self.tower_residual_alpha = float(tower_residual_alpha)
        self.shared = _mlp(self.input_dim, self.num_objectives, hidden_dim, num_layers, dropout)
        self.trunk = _mlp(self.input_dim, hidden_dim, hidden_dim, trunk_layers, dropout)
        self.heads = nn.ModuleList(
            _mlp(hidden_dim, 1, hidden_dim, head_layers, dropout)
            for _ in range(self.num_objectives)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        q_shared = self.shared(x)
        h = self.trunk(x)
        q_tower = torch.cat([head(h) for head in self.heads], dim=-1)
        return q_shared + self.tower_residual_alpha * q_tower


def build_vector_quality_model(
    architecture: str,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    trunk_layers: int = 2,
    head_layers: int = 2,
    tower_residual_alpha: float = 0.5,
) -> nn.Module:
    if architecture == "shared_mlp":
        return VectorQualityNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if architecture == "per_head_tower":
        return VectorQualityTowerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            trunk_layers=trunk_layers,
            head_layers=head_layers,
            dropout=dropout,
        )
    if architecture == "residual_tower":
        return VectorQualityResidualTowerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            trunk_layers=trunk_layers,
            head_layers=head_layers,
            dropout=dropout,
            tower_residual_alpha=tower_residual_alpha,
        )
    raise ValueError(f"unknown vector quality architecture: {architecture}")


def normalize_preference(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = w.float()
    denom = w.sum(dim=-1, keepdim=True).clamp_min(eps)
    return w / denom


def score_with_preference(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = normalize_preference(w)
    return (q * w).sum(dim=-1)


class LinearPreferenceScorer(nn.Module):
    score_mode = "linear"

    def forward(self, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return score_with_preference(q, w)


class LowRankPreferenceScorer(nn.Module):
    score_mode = "low_rank_interaction"

    def __init__(
        self,
        num_objectives: int = len(OBJECTIVE_NAMES),
        rank: int = 4,
        beta: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_objectives = int(num_objectives)
        self.rank = int(rank)
        self.beta = float(beta)
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        self.q_proj = nn.Linear(self.num_objectives, self.rank, bias=False)
        self.w_proj = nn.Linear(self.num_objectives, self.rank, bias=False)

    def forward(self, q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w_norm = normalize_preference(w)
        base = score_with_preference(q, w_norm)
        interaction = (self.q_proj(q.float()) * self.w_proj(w_norm.float())).sum(dim=-1)
        return base + self.beta * interaction


def build_preference_scorer(
    score_mode: str,
    rank: int = 4,
    beta: float = 0.3,
    num_objectives: int = len(OBJECTIVE_NAMES),
) -> nn.Module:
    if score_mode == "linear":
        return LinearPreferenceScorer()
    if score_mode == "low_rank_interaction":
        return LowRankPreferenceScorer(num_objectives=num_objectives, rank=rank, beta=beta)
    raise ValueError(f"unknown preference score mode: {score_mode}")
