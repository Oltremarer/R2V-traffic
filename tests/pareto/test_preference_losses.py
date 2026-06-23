import pytest

torch = pytest.importorskip("torch")

from pareto.losses.preference_losses import (
    calibration_loss,
    dominance_loss,
    isotonic_dominance_loss,
    dominance_utility_loss,
    objective_pair_loss,
    preference_hinge_margin_loss,
    preference_margin_regression_loss,
    preference_pair_loss,
    reversal_hinge_margin_loss,
    reversal_margin_regression_loss,
    reversal_pair_loss,
)


def test_preference_losses_are_finite():
    q_a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    q_b = torch.zeros_like(q_a)
    labels = torch.tensor([1.0, 1.0])
    w = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])

    losses = [
        objective_pair_loss(q_a, q_b, torch.tensor([0, 1]), labels, torch.tensor([0.5, 2.0, 1.0, 1.0])),
        preference_pair_loss(q_a, q_b, w, labels),
        reversal_pair_loss(q_a, q_b, w, torch.flip(w, dims=[0]), labels, torch.tensor([0.0, 0.0])),
        dominance_loss(q_a, q_b, margin=0.1),
        dominance_utility_loss(q_a, q_b, w, margin=0.1),
        isotonic_dominance_loss(q_a, q_b, torch.ones_like(q_a), margin_floor=0.1),
        calibration_loss(q_a),
        preference_margin_regression_loss(q_a, q_b, w, torch.tensor([1.0, 1.0])),
        preference_hinge_margin_loss(q_a, q_b, w, labels, margin=0.5),
        reversal_margin_regression_loss(
            q_a,
            q_b,
            w,
            torch.flip(w, dims=[0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([-1.0, -1.0]),
        ),
        reversal_hinge_margin_loss(
            q_a,
            q_b,
            w,
            torch.flip(w, dims=[0]),
            labels,
            torch.tensor([0.0, 0.0]),
            margin=0.5,
        ),
    ]

    assert all(torch.isfinite(loss).item() for loss in losses)


def test_isotonic_dominance_loss_uses_objective_specific_margins():
    q_dom = torch.tensor([[0.30, 0.30, 0.30, 0.30]])
    q_sub = torch.zeros_like(q_dom)
    small_margins = torch.tensor([[0.05, 0.05, 0.05, 0.05]])
    large_margins = torch.tensor([[0.60, 0.05, 0.05, 0.05]])

    small_loss = isotonic_dominance_loss(q_dom, q_sub, small_margins, margin_floor=0.05)
    large_loss = isotonic_dominance_loss(q_dom, q_sub, large_margins, margin_floor=0.05)

    assert small_loss == 0
    assert large_loss > small_loss
