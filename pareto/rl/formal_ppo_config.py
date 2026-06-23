from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_PPO_FIELDS = {
    "algorithm_label",
    "requires_clipped_objective",
    "rollout_steps",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "update_epochs",
    "minibatch_size",
    "lr",
    "entropy_coef",
    "value_loss_coef",
    "max_grad_norm",
    "normalize_advantages",
}


@dataclass(frozen=True)
class FormalPPODryRunConfig:
    pilot: dict[str, Any]
    ppo: dict[str, Any]
    model: dict[str, Any]
    source_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source_path: str | None = None) -> "FormalPPODryRunConfig":
        config = cls(
            pilot=dict(payload["pilot"]),
            ppo=dict(payload["ppo"]),
            model=dict(payload["model"]),
            source_path=source_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        missing = sorted(REQUIRED_PPO_FIELDS - set(self.ppo))
        if missing:
            raise ValueError(f"missing PPO hyperparameters: {missing}")
        if self.ppo["algorithm_label"] != "PPO":
            raise ValueError("algorithm_label must be PPO")
        if not self.ppo["requires_clipped_objective"]:
            raise ValueError("algorithm_label=PPO requires clipped objective")
        if float(self.ppo["clip_eps"]) <= 0.0:
            raise ValueError("clip_eps must be positive")
        if int(self.ppo["rollout_steps"]) <= 0:
            raise ValueError("rollout_steps must be positive")
        if int(self.ppo["minibatch_size"]) <= 0:
            raise ValueError("minibatch_size must be positive")
        for key in ("obs_dim", "preference_dim", "action_dim", "hidden_dim"):
            if int(self.model[key]) <= 0:
                raise ValueError(f"model.{key} must be positive")

    def ppo_config_hash(self) -> str:
        payload = json.dumps(self.ppo, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"pilot": self.pilot, "ppo": self.ppo, "model": self.model, "source_path": self.source_path}


def load_formal_ppo_dryrun_config(path: str | Path) -> FormalPPODryRunConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalPPODryRunConfig.from_dict(payload, source_path=str(Path(path)))
