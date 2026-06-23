from pareto.eval.bootstrap_metrics import bootstrap_mean_ci


def test_bootstrap_ci_returns_ordered_bounds():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    ci = bootstrap_mean_ci(values, n_boot=100, seed=0)

    assert ci["low"] <= ci["mean"] <= ci["high"]
    assert ci["n"] == len(values)


def test_bootstrap_ci_is_seed_deterministic():
    a = bootstrap_mean_ci([0, 1, 1, 1], n_boot=50, seed=7)
    b = bootstrap_mean_ci([0, 1, 1, 1], n_boot=50, seed=7)

    assert a == b
