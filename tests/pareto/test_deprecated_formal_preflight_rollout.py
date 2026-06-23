from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_deprecated_formal_preflight_rollout_cli_rejects_direct_use(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pareto/rl/formal_preflight_rollout.py"),
            "--spec",
            str(ROOT / "configs/formal/jinan_1seed_film_pilot_dryrun.json"),
            "--out_dir",
            str(tmp_path / "blocked"),
            "--preflight_only",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "formal_preflight_rollout.py is deprecated" in (result.stderr + result.stdout)
    assert not (tmp_path / "blocked" / "metadata.json").exists()
