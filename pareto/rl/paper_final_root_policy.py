from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pareto.constants import SCENARIOS


PAPER_FINAL_VERSION = "20260602_v1"
PAPER_FINAL_ROOT = Path("records/paper_final")


@dataclass(frozen=True)
class PaperFinalRoots:
    train: Path
    eval: Path
    diagnostics: Path
    preflight: Path


def traffic_slug(traffic_file: str) -> str:
    path = Path(traffic_file)
    return path.stem if path.suffix == ".json" else path.name


def validate_not_stage_a_root(path: str | Path) -> None:
    value = str(path)
    if "records/formal_jinan_3seed_" in value or "stageA" in value:
        raise ValueError(f"Stage-A root cannot be used for paper-final execution: {value}")


def ensure_paper_final_root_empty(path: str | Path) -> None:
    root = Path(path)
    validate_not_stage_a_root(root)
    if not root.exists():
        return
    if any(root.iterdir()):
        raise ValueError(f"paper-final root must be empty before execution: {root}")


def _validate_city_traffic(city: str, traffic_file: str) -> None:
    if city not in SCENARIOS:
        raise ValueError(f"unknown paper-final city: {city}")
    if traffic_file not in SCENARIOS[city]["traffic_files"]:
        raise ValueError(f"traffic file {traffic_file} is not registered for {city}")


def build_paper_final_roots(
    *,
    city: str,
    traffic_file: str,
    method: str,
    seed: int,
    preference_id: str = "balanced",
    version: str = PAPER_FINAL_VERSION,
) -> PaperFinalRoots:
    _validate_city_traffic(city, traffic_file)
    slug = traffic_slug(traffic_file)
    seed_part = f"seed{int(seed)}"
    train_leaf = Path(seed_part) if preference_id == "balanced" else Path(seed_part) / preference_id
    base = PAPER_FINAL_ROOT
    return PaperFinalRoots(
        train=base / f"train_{version}" / city / slug / method / train_leaf,
        eval=base / f"eval_{version}" / city / slug / method / seed_part / preference_id,
        diagnostics=base / f"diagnostics_{version}" / city / slug / method / train_leaf,
        preflight=base / f"preflight_{version}",
    )
