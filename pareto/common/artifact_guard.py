from __future__ import annotations

from pathlib import Path


FORBIDDEN_PREFLIGHT_ARTIFACTS = {
    "best_method.json",
    "best_method.txt",
    "leaderboard.csv",
    "main_results.csv",
    "method_ranking.csv",
    "performance_table.csv",
    "performance_table.json",
    "performance_table.md",
    "performance_table.tex",
    "ranking.csv",
    "paper_results.csv",
    "preference_response_plot.pdf",
    "preference_response_plot.png",
    "preference_sweep.csv",
    "traffic_metrics.csv",
}


def assert_no_forbidden_performance_artifacts(run_dir: str | Path) -> None:
    root = Path(run_dir)
    if not root.exists():
        return
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_PREFLIGHT_ARTIFACTS:
            forbidden.append(str(path.relative_to(root)))
        if path.suffix == ".tex" and "performance" in path.name:
            forbidden.append(str(path.relative_to(root)))
    if forbidden:
        raise ValueError(f"forbidden preflight performance artifacts: {sorted(set(forbidden))}")
