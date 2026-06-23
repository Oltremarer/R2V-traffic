from pareto.data.offline_dataset import build_reversal_training_pairs


def test_template_balanced_sampler_oversamples_underrepresented_templates():
    pairs = [
        {"pair_id": "a0", "w_1_name": "efficiency", "w_2_name": "safety"},
        {"pair_id": "a1", "w_1_name": "efficiency", "w_2_name": "safety"},
        {"pair_id": "b0", "w_1_name": "efficiency", "w_2_name": "fairness"},
    ]

    sampled, report = build_reversal_training_pairs(
        pairs,
        sampler="template_balanced",
        min_count=1,
        seed=0,
    )

    assert report["template_counts"]["efficiency__safety"] == 2
    assert report["template_counts"]["efficiency__fairness"] == 1
    assert report["sampled_template_counts"]["efficiency__safety"] == 2
    assert report["sampled_template_counts"]["efficiency__fairness"] == 2
    assert len(sampled) == 4
    assert "oversampling_ratio_by_template" in report


def test_template_balanced_sampler_reports_underpowered_templates():
    pairs = [{"pair_id": "b0", "w_1_name": "efficiency", "w_2_name": "fairness"}]

    sampled, report = build_reversal_training_pairs(
        pairs,
        sampler="template_balanced",
        min_count=2,
        seed=0,
    )

    assert sampled == []
    assert report["underpowered_templates"] == ["efficiency__fairness"]
