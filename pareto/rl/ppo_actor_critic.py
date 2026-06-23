from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class PreferenceConditionedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, preference_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.preference_dim = int(preference_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor_input_dim = self.obs_dim + self.preference_dim
        self.critic_input_dim = self.obs_dim + self.preference_dim
        self.trunk = nn.Sequential(
            nn.Linear(self.actor_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def _features(self, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if w.ndim == 1:
            w = w.unsqueeze(0).expand(obs.shape[0], -1)
        return torch.cat([obs.float(), w.float()], dim=-1)

    def forward(self, obs: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(self._features(obs, w))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, obs: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, w)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        w: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, w)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values


def save_actor_critic_checkpoint(path, model: PreferenceConditionedActorCritic, metadata: dict) -> None:
    import torch
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "obs_dim": model.obs_dim,
                "preference_dim": model.preference_dim,
                "action_dim": model.action_dim,
                "hidden_dim": model.hidden_dim,
            },
            "metadata": metadata,
        },
        path,
    )


def load_actor_critic_checkpoint(path) -> tuple[PreferenceConditionedActorCritic, dict]:
    from pathlib import Path

    payload = torch.load(Path(path), map_location=torch.device("cpu"))
    config = payload["config"]
    model = PreferenceConditionedActorCritic(
        obs_dim=config["obs_dim"],
        preference_dim=config["preference_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config.get("hidden_dim", 64),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def save_training_checkpoint(
    path,
    model: PreferenceConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
    step: int,
    episode: int = 0,
    global_update: int = 0,
) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update({"step": int(step), "episode": int(episode), "global_update": int(global_update)})
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_class": optimizer.__class__.__name__,
            "config": {
                "obs_dim": model.obs_dim,
                "preference_dim": model.preference_dim,
                "action_dim": model.action_dim,
                "hidden_dim": model.hidden_dim,
            },
            "metadata": checkpoint_metadata,
        },
        path,
    )


def load_training_checkpoint(path) -> tuple[PreferenceConditionedActorCritic, torch.optim.Optimizer, dict]:
    from pathlib import Path

    payload = torch.load(Path(path), map_location=torch.device("cpu"))
    config = payload["config"]
    model = PreferenceConditionedActorCritic(
        obs_dim=config["obs_dim"],
        preference_dim=config["preference_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config.get("hidden_dim", 64),
    )
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, optimizer, payload
