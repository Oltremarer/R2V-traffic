from __future__ import annotations

import torch

from pareto.rl.ppo_actor_critic import (
    PreferenceConditionedActorCritic,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_training_checkpoint_roundtrips_optimizer_state_and_step(tmp_path):
    model = PreferenceConditionedActorCritic(obs_dim=5, preference_dim=4, action_dim=3, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    obs = torch.randn(4, 5)
    w = torch.ones(4, 4) / 4.0
    logits, values = model(obs, w)
    loss = logits.mean() + values.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model, optimizer, {"dry_run": True}, step=7, episode=2)

    loaded_model, loaded_optimizer, payload = load_training_checkpoint(path)

    assert payload["metadata"]["dry_run"] is True
    assert payload["metadata"]["step"] == 7
    assert payload["metadata"]["episode"] == 2
    assert loaded_optimizer.state_dict()["state"]

    logits_2, values_2 = loaded_model(obs, w)
    next_loss = logits_2.mean() + values_2.mean()
    loaded_optimizer.zero_grad()
    next_loss.backward()
    loaded_optimizer.step()
