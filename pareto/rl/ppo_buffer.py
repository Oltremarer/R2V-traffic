from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PPORolloutBuffer:
    obs: list[torch.Tensor] = field(default_factory=list)
    w: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    def add(
        self,
        obs: torch.Tensor,
        w: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs.append(obs.detach().float().cpu())
        self.w.append(w.detach().float().cpu())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        normalize_advantages: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not self.rewards:
            raise ValueError("cannot compute returns for empty PPO buffer")
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        values = torch.tensor(self.values + [float(last_value)], dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
        for idx in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[idx]
            delta = rewards[idx] + float(gamma) * values[idx + 1] * nonterminal - values[idx]
            gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
            advantages[idx] = gae
        returns = advantages + values[:-1]
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        return {
            "obs": torch.stack(self.obs, dim=0),
            "w": torch.stack(self.w, dim=0),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "rewards": rewards,
            "dones": dones,
            "values": values[:-1],
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "returns": returns,
            "advantages": advantages,
        }
