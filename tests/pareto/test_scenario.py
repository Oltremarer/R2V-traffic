from pareto.common.scenario import build_llmlight_env_config, resolve_scenario


def test_resolve_scenario_returns_registered_jinan_metadata():
    meta = resolve_scenario("jinan")
    assert meta["template"] == "Jinan"
    assert meta["roadnet"] == "3_4"


def test_build_config_sets_cityflow_seed_and_paths():
    _, env_conf, dic_path = build_llmlight_env_config(
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed=7,
        run_counts=200,
        model_name="MaxPressure",
        work_dir="records/test_run",
    )
    assert env_conf["CITYFLOW_SEED"] == 7
    assert env_conf["NUM_INTERSECTIONS"] == 12
    assert env_conf["RUN_COUNTS"] == 200
    assert dic_path["PATH_TO_DATA"].endswith("data/Jinan/3_4")


def test_unknown_traffic_file_is_rejected():
    try:
        build_llmlight_env_config("jinan", "missing.json", 0, 200)
    except ValueError as exc:
        assert "traffic_file" in str(exc)
    else:
        raise AssertionError("expected invalid traffic file to fail")


def test_advanced_maxpressure_config_keeps_agent_required_phase_features():
    _, env_conf, _ = build_llmlight_env_config(
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed=0,
        run_counts=200,
        model_name="AdvancedMaxPressure",
        work_dir="records/test_run",
    )

    assert env_conf["LIST_STATE_FEATURE"][:2] == [
        "traffic_movement_pressure_queue_efficient",
        "lane_enter_running_part",
    ]
    assert "cur_phase" in env_conf["LIST_STATE_FEATURE"]
