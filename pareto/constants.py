OBJECTIVE_NAMES = ("efficiency", "safety", "fairness", "stability")
OBJECTIVE_INDEX = {name: idx for idx, name in enumerate(OBJECTIVE_NAMES)}

PHASE_NAMES_4 = ("ETWT", "NTST", "ELWL", "NLSL")

SCENARIOS = {
    "jinan": {
        "template": "Jinan",
        "roadnet": "3_4",
        "traffic_files": (
            "anon_3_4_jinan_real.json",
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json",
            "anon_3_4_jinan_synthetic_24000_60min.json",
            "anon_3_4_jinan_synthetic_24h_6000.json",
        ),
        "default_traffic_file": "anon_3_4_jinan_real.json",
        "default_run_counts": 3600,
    },
    "hangzhou": {
        "template": "Hangzhou",
        "roadnet": "4_4",
        "traffic_files": (
            "anon_4_4_hangzhou_real.json",
            "anon_4_4_hangzhou_real_5816.json",
            "anon_4_4_hangzhou_synthetic_24000_60min.json",
        ),
        "default_traffic_file": "anon_4_4_hangzhou_real.json",
        "default_run_counts": 3600,
    },
    "newyork_28x7": {
        "template": "NewYork",
        "roadnet": "28_7",
        "traffic_files": (
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json",
        ),
        "default_traffic_file": "anon_28_7_newyork_real_double.json",
        "default_run_counts": 3600,
    },
}
