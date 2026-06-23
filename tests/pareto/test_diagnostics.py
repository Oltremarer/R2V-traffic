import pytest

torch = pytest.importorskip("torch")

from pareto.eval.diagnostics import (
    binary_accuracy_from_logits,
    dominance_preservation_rate,
    dominance_utility_preservation_rate,
    head_target_correlation_matrix,
    output_correlation_matrix,
    pairwise_head_leakage_matrix,
    reversal_accuracy,
)
import pareto.eval.diagnostics as diagnostics


def test_diagnostics_compute_core_metrics():
    logits = torch.tensor([2.0, -2.0, 0.5])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert binary_accuracy_from_logits(logits, labels) == 1.0

    q_dom = torch.ones(3, 4)
    q_sub = torch.zeros(3, 4)
    assert dominance_preservation_rate(q_dom, q_sub) == 1.0
    w_eval = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])
    assert dominance_utility_preservation_rate(q_dom, q_sub, w_eval) == 1.0

    rev = reversal_accuracy(
        torch.tensor([2.0, -2.0]),
        torch.tensor([-2.0, 2.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert rev == 1.0

    q = torch.randn(6, 4)
    objective_targets = torch.randn(6, 4)
    assert head_target_correlation_matrix(q, objective_targets).shape == (4, 4)
    assert output_correlation_matrix(q).shape == (4, 4)


def test_pairwise_head_leakage_matrix_uses_pair_labels():
    q_by_id = {
        "a": torch.tensor([2.0, 0.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 2.0, 0.0, 0.0]),
    }
    pairs = [
        {"a_id": "a", "b_id": "b", "objective": "efficiency", "label": 1},
        {"a_id": "a", "b_id": "b", "objective": "safety", "label": 0},
    ]

    matrix = pairwise_head_leakage_matrix(q_by_id, pairs)

    assert matrix.shape == (4, 4)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0


def test_head_leakage_alias_is_not_exposed():
    assert not hasattr(diagnostics, "head_leakage_matrix")
