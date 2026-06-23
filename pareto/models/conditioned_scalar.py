from __future__ import annotations

import torch
from torch import nn

from pareto.constants import OBJECTIVE_NAMES
from pareto.models.vector_quality import _mlp, normalize_preference


class ConditionedScalarQualityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        preference_dim: int = len(OBJECTIVE_NAMES),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.preference_dim = int(preference_dim)
        self.net = _mlp(self.input_dim + self.preference_dim, 1, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = normalize_preference(w)
        inputs = torch.cat([x.float(), w], dim=-1)
        return self.net(inputs).squeeze(-1)


class FiLMConditionedScalarQualityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        preference_dim: int = len(OBJECTIVE_NAMES),
        film_layers: int = 2,
        head_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.preference_dim = int(preference_dim)
        self.trunk = _mlp(self.input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.film = _mlp(self.preference_dim, 2 * hidden_dim, hidden_dim, film_layers, dropout)
        self.head = _mlp(hidden_dim, 1, hidden_dim, head_layers, dropout)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = normalize_preference(w)
        h = self.trunk(x.float())
        gamma, beta = self.film(w).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return self.head(h).squeeze(-1)


def build_conditioned_scalar_model(
    architecture: str,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    preference_dim: int = len(OBJECTIVE_NAMES),
    film_layers: int = 2,
    head_layers: int = 2,
) -> nn.Module:
    if architecture == "concat":
        return ConditionedScalarQualityNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            preference_dim=preference_dim,
        )
    if architecture == "film":
        return FiLMConditionedScalarQualityNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            preference_dim=preference_dim,
            film_layers=film_layers,
            head_layers=head_layers,
        )
    raise ValueError(f"unknown conditioned scalar architecture: {architecture}")
