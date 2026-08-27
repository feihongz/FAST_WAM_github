from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
JIHE_DIR = REPO_ROOT / "scripts" / "jihe"


@pytest.mark.parametrize(
    ("launcher", "benchmark", "fresh_port", "resume_port"),
    (
        ("run_libero_stage3_smoke_8xh100.sh", "LIBERO", "29531", "29533"),
        (
            "run_robotwin_stage3_smoke_8xh100.sh",
            "RoboTwin-2.0",
            "29532",
            "29534",
        ),
    ),
)
def test_one_click_smoke_dry_run_expands_locked_workflow(
    launcher: str,
    benchmark: str,
    fresh_port: str,
    resume_port: str,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "RUN_ID": "pytest",
            # The outer workflow must replace inherited topology settings.
            "NPROC_PER_NODE": "1",
            "NNODES": "9",
            "NODE_RANK": "8",
        }
    )
    result = subprocess.run(
        ["bash", str(JIHE_DIR / launcher)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert f"benchmark={benchmark}" in output
    assert output.count("training.max_steps=2") == 2
    assert output.count("--num_processes 8") == 2
    assert f"--main_process_port {fresh_port}" in output
    assert f"--main_process_port {resume_port}" in output
    assert "phase=fresh" in output
    assert "phase=resume" in output
    assert "step_000001" in output
    assert "fresh, resume, and exact verification are fully planned" in output


def test_public_smoke_launcher_rejects_hydra_arguments() -> None:
    result = subprocess.run(
        [
            "bash",
            str(JIHE_DIR / "run_libero_stage3_smoke_8xh100.sh"),
            "training.max_steps=999",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "takes no arguments" in result.stderr


def test_smoke_launcher_refuses_existing_campaign_root(tmp_path: Path) -> None:
    storage_root = tmp_path / "persistent"
    smoke_root = storage_root / "FastWAM" / "smoke-already-exists"
    smoke_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "0",
            "FASTWAM_STORAGE_ROOT": str(storage_root),
            "SMOKE_ROOT": str(smoke_root),
            "RUN_ID": "pytest-existing",
        }
    )
    result = subprocess.run(
        ["bash", str(JIHE_DIR / "run_libero_stage3_smoke_8xh100.sh")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SMOKE_ROOT already exists" in result.stderr
    assert list(smoke_root.iterdir()) == []
