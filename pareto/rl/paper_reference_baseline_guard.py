from __future__ import annotations

from pareto.rl.paper_baseline_commands import build_baseline_command_preview
from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


CONVENTIONAL_BASELINES = (
    "Random",
    "FixedTime",
    "MaxPressure",
    "PressLight",
    "MPLight",
    "CoLight",
    "Advanced-Co",
)


def build_reference_baseline_smoke_request(
    *,
    baseline: str,
    city: str,
    traffic_file: str,
    seed: int,
) -> dict:
    if baseline not in CONVENTIONAL_BASELINES:
        raise ValueError(f"baseline {baseline} is not a conventional paper baseline")
    expected_traffic = REQUIRED_CITY_TRAFFIC.get(city)
    if expected_traffic != traffic_file:
        raise ValueError(f"traffic file {traffic_file} is not registered for {city} paper-final smoke")
    preview = build_baseline_command_preview(baseline, city=city, traffic_file=traffic_file, seed=seed)
    return {
        **preview,
        "seed_binding": "cityflow_seed=policy_seed=model_seed=seed_id",
        "paper_final_smoke": True,
        "result_values_read": False,
    }


def validate_reference_baseline_smoke_matrix() -> list[dict]:
    rows = []
    for city, traffic_file in REQUIRED_CITY_TRAFFIC.items():
        for baseline in CONVENTIONAL_BASELINES:
            rows.append(
                build_reference_baseline_smoke_request(
                    baseline=baseline,
                    city=city,
                    traffic_file=traffic_file,
                    seed=0,
                )
            )
    return rows
