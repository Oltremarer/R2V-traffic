from __future__ import annotations

from pathlib import Path

import torch

from pareto.rl.formal_preflight_rollout import (
    PreferenceActorCritic,
    load_preflight_policy_checkpoint,
    save_preflight_policy_checkpoint,
)


def test_preflight_policy_checkpoint_roundtrip_produces_finite_logits(tmp_path: Path):
    model = PreferenceActorCritic(obs_dim=8, preference_dim=4, action_dim=4, hidden_dim=16)
    checkpoint = tmp_path / "checkpoint_last.pt"

    save_preflight_policy_checkpoint(checkpoint, model, {"method": "film_scalar_potential"})
    loaded, payload = load_preflight_policy_checkpoint(checkpoint)

    obs = torch.zeros(3, 8)
    w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    logits, values = loaded(obs, w)

    assert payload["metadata"]["method"] == "film_scalar_potential"
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert logits.shape == (3, 4)
