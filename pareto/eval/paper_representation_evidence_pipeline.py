from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from pareto.eval.formal_gate import evaluate_formal_gate
from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


DEFAULT_LEARNED_ARTIFACT_RUN_ID = "paper_final_20260603_v1"
VECTOR_EVIDENCE_ID = "pareto_quality_paper_final_20260603_v1"
SCALAR_EVIDENCE_ID = "cond_scalar_paper_final_20260603_v1"
EVIDENCE_ROOT_PREFIX = "docs/pro_reviews"
LEARNED_ARTIFACT_FILES = (
    "model.pt",
    "metadata.json",
    "diagnostics_val.json",
    "diagnostics_test.json",
    "objective_normalizer.json",
)
PAIR_FILES = (
    "objective_pairs.jsonl",
    "preference_pairs.jsonl",
    "dominance_pairs.jsonl",
    "reversal_pairs.jsonl",
    "pair_report.json",
)
SPLITS = ("train", "val", "test")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def evidence_dir_for(city: str, *, evidence_dir_suffix: str = "paper_final_evidence") -> str:
    return f"{EVIDENCE_ROOT_PREFIX}/pareto_offline_representation_{city}_{evidence_dir_suffix}"


def expected_evidence_filenames(
    *,
    vector_evidence_id: str = VECTOR_EVIDENCE_ID,
    scalar_evidence_id: str = SCALAR_EVIDENCE_ID,
) -> list[str]:
    return [
        f"{vector_evidence_id}_diagnostics_val.json",
        f"{vector_evidence_id}_diagnostics_test.json",
        f"{vector_evidence_id}_metadata.json",
        f"{scalar_evidence_id}_diagnostics_val.json",
        f"{scalar_evidence_id}_diagnostics_test.json",
        f"{scalar_evidence_id}_metadata.json",
        f"{vector_evidence_id}_formal_gate_decision.json",
        f"{vector_evidence_id}_pair_bootstrap_test.json",
        f"{vector_evidence_id}_dominance_error_audit.json",
        "split_records_report.json",
        "objective_norm_paper_final.json",
        "objective_sanity_v4_train.json",
        "pair_report_train.json",
        "pair_report_val.json",
        "pair_report_test.json",
    ]


def _artifact_dir(root: Path, family: str, city: str, run_id: str) -> Path:
    return root / "model_weights" / family / city / "paper_final" / run_id


def _required_files(root: Path, city: str, run_id: str) -> list[Path]:
    files: list[Path] = []
    raw_root = root / "data" / "pareto_records_split" / city / "paper_final"
    records_root = root / "data" / "pareto_records_split_norm" / city / "paper_final"
    pairs_root = root / "data" / "pareto_pairs" / city / "paper_final"
    files.append(raw_root / "train_raw.jsonl")
    files.append(raw_root / "split_records_report.json")
    files.append(root / "data" / "normalizers" / city / "objective_norm_paper_final.json")
    for split in SPLITS:
        files.append(records_root / f"{split}_raw.jsonl")
        for filename in PAIR_FILES:
            files.append(pairs_root / split / filename)
    for family in ("pareto_quality", "cond_scalar"):
        artifact = _artifact_dir(root, family, city, run_id)
        files.extend(artifact / filename for filename in LEARNED_ARTIFACT_FILES)
    return files


def _city_row(
    root: Path,
    city: str,
    run_id: str,
    *,
    vector_evidence_id: str = VECTOR_EVIDENCE_ID,
    scalar_evidence_id: str = SCALAR_EVIDENCE_ID,
    evidence_dir_suffix: str = "paper_final_evidence",
) -> dict[str, Any]:
    missing = [_rel(path, root) for path in _required_files(root, city, run_id) if not path.is_file()]
    final_root = root / "records" / "paper_final"
    if final_root.exists() and any(path.is_file() for path in final_root.rglob("*")):
        missing.append("records/paper_final must contain zero files")
    evidence_dir = root / evidence_dir_for(city, evidence_dir_suffix=evidence_dir_suffix)
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        missing.append(_rel(evidence_dir, root) + " must be absent or empty")
    status = "missing_blocker" if missing else "ready"
    row = {
        "city": city,
        "status": status,
        "evidence_dir": _rel(evidence_dir, root),
        "missing_files": missing,
        "learned_artifact_run_id": run_id,
        "vector_evidence_id": vector_evidence_id,
        "scalar_evidence_id": scalar_evidence_id,
        "evidence_dir_suffix": evidence_dir_suffix,
        "executes_generation_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }
    if missing:
        row["blocker"] = "paper-final representation evidence prerequisites incomplete"
    return row


def build_representation_evidence_plan(
    root: str | Path = ".",
    *,
    learned_artifact_run_id: str = DEFAULT_LEARNED_ARTIFACT_RUN_ID,
    vector_evidence_id: str = VECTOR_EVIDENCE_ID,
    scalar_evidence_id: str = SCALAR_EVIDENCE_ID,
    evidence_dir_suffix: str = "paper_final_evidence",
    cities: tuple[str, ...] = tuple(REQUIRED_CITY_TRAFFIC),
) -> dict[str, Any]:
    root_path = Path(root)
    rows = [
        _city_row(
            root_path,
            city,
            learned_artifact_run_id,
            vector_evidence_id=vector_evidence_id,
            scalar_evidence_id=scalar_evidence_id,
            evidence_dir_suffix=evidence_dir_suffix,
        )
        for city in cities
    ]
    status = "missing_blocker" if any(row["status"] == "missing_blocker" for row in rows) else "ready_request"
    return {
        "packet_type": "paper_representation_evidence_pipeline_plan",
        "status": status,
        "learned_artifact_run_id": learned_artifact_run_id,
        "vector_evidence_id": vector_evidence_id,
        "scalar_evidence_id": scalar_evidence_id,
        "evidence_dir_suffix": evidence_dir_suffix,
        "rows": rows,
        "executes_generation_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _command_path(root: Path, relative: str) -> str:
    return (root / relative).as_posix()


def _external_commands(
    root: Path,
    city: str,
    run_id: str,
    evidence_dir: Path,
    python_executable: str,
    *,
    vector_evidence_id: str = VECTOR_EVIDENCE_ID,
) -> list[list[str]]:
    records_root = _command_path(root, f"data/pareto_records_split_norm/{city}/paper_final")
    pairs_root = _command_path(root, f"data/pareto_pairs/{city}/paper_final")
    raw_train = _command_path(root, f"data/pareto_records_split/{city}/paper_final/train_raw.jsonl")
    vector_dir = _command_path(root, f"model_weights/pareto_quality/{city}/paper_final/{run_id}")
    scalar_dir = _command_path(root, f"model_weights/cond_scalar/{city}/paper_final/{run_id}")
    return [
        [
            python_executable,
            "pareto/data/objective_sanity.py",
            "--buffer",
            raw_train,
            "--out",
            (evidence_dir / "objective_sanity_v4_train.json").as_posix(),
            "--strict",
        ],
        [
            python_executable,
            "pareto/eval/offline_pair_bootstrap.py",
            "--records_root",
            records_root,
            "--pairs_root",
            pairs_root,
            "--split",
            "test",
            "--out",
            (evidence_dir / f"{vector_evidence_id}_pair_bootstrap_test.json").as_posix(),
            "--vector_model_dir",
            vector_dir,
            "--scalar_model_dir",
            scalar_dir,
            "--device",
            "cuda",
            "--n_boot",
            "1000",
            "--seed",
            "20260603",
        ],
        [
            python_executable,
            "pareto/eval/dominance_error_audit.py",
            "--records_root",
            records_root,
            "--pairs_root",
            pairs_root,
            "--model_dir",
            vector_dir,
            "--out",
            (evidence_dir / f"{vector_evidence_id}_dominance_error_audit.json").as_posix(),
            "--split",
            "test",
            "--device",
            "cuda",
        ],
    ]


def materialize_city_evidence(
    root: str | Path,
    city: str,
    *,
    learned_artifact_run_id: str = DEFAULT_LEARNED_ARTIFACT_RUN_ID,
    vector_evidence_id: str = VECTOR_EVIDENCE_ID,
    scalar_evidence_id: str = SCALAR_EVIDENCE_ID,
    evidence_dir_suffix: str = "paper_final_evidence",
    command_runner: Callable[[list[str]], None] = _run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    root_path = Path(root)
    row = _city_row(
        root_path,
        city,
        learned_artifact_run_id,
        vector_evidence_id=vector_evidence_id,
        scalar_evidence_id=scalar_evidence_id,
        evidence_dir_suffix=evidence_dir_suffix,
    )
    if row["status"] != "ready":
        raise ValueError(f"representation evidence prerequisites incomplete: {row['missing_files']}")
    evidence_dir = root_path / evidence_dir_for(city, evidence_dir_suffix=evidence_dir_suffix)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = _artifact_dir(root_path, "pareto_quality", city, learned_artifact_run_id)
    scalar_dir = _artifact_dir(root_path, "cond_scalar", city, learned_artifact_run_id)

    for split in ("val", "test"):
        _copy(vector_dir / f"diagnostics_{split}.json", evidence_dir / f"{vector_evidence_id}_diagnostics_{split}.json")
        _copy(scalar_dir / f"diagnostics_{split}.json", evidence_dir / f"{scalar_evidence_id}_diagnostics_{split}.json")
    _copy(vector_dir / "metadata.json", evidence_dir / f"{vector_evidence_id}_metadata.json")
    _copy(scalar_dir / "metadata.json", evidence_dir / f"{scalar_evidence_id}_metadata.json")
    _copy(
        root_path / "data" / "pareto_records_split" / city / "paper_final" / "split_records_report.json",
        evidence_dir / "split_records_report.json",
    )
    _copy(
        root_path / "data" / "normalizers" / city / "objective_norm_paper_final.json",
        evidence_dir / "objective_norm_paper_final.json",
    )
    for split in SPLITS:
        _copy(
            root_path / "data" / "pareto_pairs" / city / "paper_final" / split / "pair_report.json",
            evidence_dir / f"pair_report_{split}.json",
        )

    vector_test = _load_json(evidence_dir / f"{vector_evidence_id}_diagnostics_test.json")
    scalar_test = _load_json(evidence_dir / f"{scalar_evidence_id}_diagnostics_test.json")
    _write_json(evidence_dir / f"{vector_evidence_id}_formal_gate_decision.json", evaluate_formal_gate(vector_test, scalar_test))

    for command in _external_commands(
        root_path,
        city,
        learned_artifact_run_id,
        evidence_dir,
        python_executable,
        vector_evidence_id=vector_evidence_id,
    ):
        command_runner(command)

    expected = expected_evidence_filenames(
        vector_evidence_id=vector_evidence_id,
        scalar_evidence_id=scalar_evidence_id,
    )
    missing_outputs = [name for name in expected if not (evidence_dir / name).is_file()]
    if missing_outputs:
        raise ValueError(f"representation evidence outputs missing: {missing_outputs}")
    return {
        "city": city,
        "status": "complete",
        "evidence_dir": evidence_dir.as_posix(),
        "outputs": expected,
        "vector_evidence_id": vector_evidence_id,
        "scalar_evidence_id": scalar_evidence_id,
        "evidence_dir_suffix": evidence_dir_suffix,
        "executes_generation_now": True,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--city", action="append", choices=list(REQUIRED_CITY_TRAFFIC))
    parser.add_argument("--learned_artifact_run_id", default=DEFAULT_LEARNED_ARTIFACT_RUN_ID)
    parser.add_argument("--vector_evidence_id", default=VECTOR_EVIDENCE_ID)
    parser.add_argument("--scalar_evidence_id", default=SCALAR_EVIDENCE_ID)
    parser.add_argument("--evidence_dir_suffix", default="paper_final_evidence")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cities = tuple(args.city) if args.city else tuple(REQUIRED_CITY_TRAFFIC)
    if not args.execute:
        print(json.dumps(
            build_representation_evidence_plan(
                args.root,
                learned_artifact_run_id=args.learned_artifact_run_id,
                vector_evidence_id=args.vector_evidence_id,
                scalar_evidence_id=args.scalar_evidence_id,
                evidence_dir_suffix=args.evidence_dir_suffix,
                cities=cities,
            ),
            indent=2,
            sort_keys=True,
        ))
        return
    results = [
        materialize_city_evidence(
            args.root,
            city,
            learned_artifact_run_id=args.learned_artifact_run_id,
            vector_evidence_id=args.vector_evidence_id,
            scalar_evidence_id=args.scalar_evidence_id,
            evidence_dir_suffix=args.evidence_dir_suffix,
            python_executable=args.python_bin,
        )
        for city in cities
    ]
    print(json.dumps({"status": "complete", "rows": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
