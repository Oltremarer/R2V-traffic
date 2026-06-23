import pytest

torch = pytest.importorskip("torch")

from pareto.models.conditioned_scalar import (
    ConditionedScalarQualityNet,
    FiLMConditionedScalarQualityNet,
    build_conditioned_scalar_model,
)


def test_conditioned_scalar_uses_state_and_preference_inputs():
    model = ConditionedScalarQualityNet(input_dim=3, hidden_dim=8, num_layers=2)
    x = torch.randn(5, 3)
    w = torch.tensor([[0.7, 0.1, 0.1, 0.1]]).repeat(5, 1)

    score = model(x, w)

    assert score.shape == (5,)


def test_film_conditioned_scalar_outputs_scores():
    model = FiLMConditionedScalarQualityNet(input_dim=3, hidden_dim=8, num_layers=2)
    x = torch.randn(5, 3)
    w_eff = torch.tensor([[0.7, 0.1, 0.1, 0.1]]).repeat(5, 1)
    w_safety = torch.tensor([[0.1, 0.7, 0.1, 0.1]]).repeat(5, 1)

    score_eff = model(x, w_eff)
    score_safety = model(x, w_safety)

    assert score_eff.shape == (5,)
    assert score_safety.shape == (5,)
    assert not torch.allclose(score_eff, score_safety)


def test_conditioned_scalar_builder_selects_architecture():
    concat = build_conditioned_scalar_model("concat", input_dim=3, hidden_dim=8, num_layers=2)
    film = build_conditioned_scalar_model("film", input_dim=3, hidden_dim=8, num_layers=2)

    assert isinstance(concat, ConditionedScalarQualityNet)
    assert isinstance(film, FiLMConditionedScalarQualityNet)
