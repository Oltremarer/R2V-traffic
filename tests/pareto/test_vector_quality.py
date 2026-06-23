import pytest

torch = pytest.importorskip("torch")

from pareto.models.vector_quality import (
    LinearPreferenceScorer,
    LowRankPreferenceScorer,
    VectorQualityNet,
    VectorQualityResidualTowerNet,
    VectorQualityTowerNet,
    build_preference_scorer,
    build_vector_quality_model,
    score_with_preference,
)


def test_vector_quality_outputs_four_heads_and_scores_preferences():
    model = VectorQualityNet(input_dim=3, hidden_dim=8, num_layers=2)
    x = torch.randn(5, 3)
    w = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).repeat(5, 1)

    q = model(x)
    score = score_with_preference(q, w)

    assert q.shape == (5, 4)
    assert score.shape == (5,)


def test_vector_quality_tower_outputs_four_heads():
    model = VectorQualityTowerNet(input_dim=3, hidden_dim=8, trunk_layers=2, head_layers=2)
    x = torch.randn(5, 3)

    q = model(x)

    assert q.shape == (5, 4)
    assert len(model.heads) == 4


def test_vector_quality_builder_selects_architecture():
    shared = build_vector_quality_model("shared_mlp", input_dim=3, hidden_dim=8, num_layers=2)
    tower = build_vector_quality_model(
        "per_head_tower",
        input_dim=3,
        hidden_dim=8,
        trunk_layers=2,
        head_layers=2,
    )
    residual = build_vector_quality_model(
        "residual_tower",
        input_dim=3,
        hidden_dim=8,
        num_layers=2,
        trunk_layers=2,
        head_layers=2,
        tower_residual_alpha=0.5,
    )

    assert isinstance(shared, VectorQualityNet)
    assert isinstance(tower, VectorQualityTowerNet)
    assert isinstance(residual, VectorQualityResidualTowerNet)
    assert residual(torch.randn(4, 3)).shape == (4, 4)


def test_low_rank_preference_scorer_preserves_linear_base_and_adds_interaction():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    w = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    linear = LinearPreferenceScorer()
    low_rank = LowRankPreferenceScorer(num_objectives=4, rank=2, beta=0.3)

    with torch.no_grad():
        low_rank.q_proj.weight.fill_(1.0)
        low_rank.w_proj.weight.fill_(0.5)

    linear_score = linear(q, w)
    interaction_score = low_rank(q, w)

    assert linear_score.shape == (1,)
    assert interaction_score.shape == (1,)
    assert torch.allclose(linear_score, score_with_preference(q, w))
    assert not torch.allclose(interaction_score, linear_score)


def test_preference_scorer_builder_selects_low_rank_mode():
    scorer = build_preference_scorer("low_rank_interaction", rank=3, beta=0.2)

    assert isinstance(scorer, LowRankPreferenceScorer)
    assert scorer.q_proj.out_features == 3
    assert scorer.beta == 0.2
