from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_formal_pilot_runner_cli_requires_mock_env(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_pilot_runner.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--method",
            "film_scalar_potential",
            "--out_dir",
            str(tmp_path / "blocked"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "requires --mock_env" in (result.stderr + result.stdout)
