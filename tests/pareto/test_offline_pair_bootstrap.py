import pytest

torch = pytest.importorskip("torch")

from pareto.eval.offline_pair_bootstrap import bootstrap_correctness_report


def test_pair_level_bootstrap_correctness_report_is_seeded():
    correctness = {
        "pref_acc": [True, True, False, True],
        "rev_acc": [False, True, False, True],
        "obj_acc": {
            "efficiency": [True, True],
            "safety": [False, True],
        },
    }

    a = bootstrap_correctness_report(correctness, n_boot=50, seed=3)
    b = bootstrap_correctness_report(correctness, n_boot=50, seed=3)

    assert a == b
    assert a["metrics"]["pref_acc"]["n"] == 4
    assert a["metrics"]["obj_acc"]["safety"]["n"] == 2
