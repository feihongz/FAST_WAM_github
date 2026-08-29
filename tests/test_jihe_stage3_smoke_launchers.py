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


@pytest.mark.parametrize(
    "launcher",
    (
        "train_libero_stage3_alignment_8xh100.sh",
        "train_robotwin_stage3_alignment_8xh100.sh",
    ),
)
@pytest.mark.parametrize(
    "auto_variable",
    ("NPROC_PER_NODE", "SENSECORE_ACCELERATE_DEVICE_COUNT"),
)
def test_stage3_launcher_resolves_jihe_auto_device_count(
    launcher: str,
    auto_variable: str,
) -> None:
    environment = os.environ.copy()
    environment.pop("NPROC_PER_NODE", None)
    environment.pop("SENSECORE_ACCELERATE_DEVICE_COUNT", None)
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "RUN_ID": "pytest-auto-device-count",
            auto_variable: "auto",
        }
    )
    result = subprocess.run(
        ["bash", str(JIHE_DIR / launcher), "training.max_steps=200"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "world_size=8" in result.stdout
    assert "--num_processes 8" in result.stdout


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


def _fake_missing_ffmpeg_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_environment = tmp_path / "venv"
    fake_bin = tmp_path / "bin"
    marker = tmp_path / "ffmpeg-installed"
    apt_log = tmp_path / "apt.log"
    (fake_environment / "bin").mkdir(parents=True)
    fake_bin.mkdir()

    fake_python = fake_environment / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "[[ -f \"${FAKE_FFMPEG_MARKER}\" ]] || exit 1\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_apt = fake_bin / "apt-get"
    fake_apt.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"${FAKE_APT_LOG}\"\n"
        "for argument in \"$@\"; do\n"
        "  if [[ \"${argument}\" == install ]]; then\n"
        "    touch \"${FAKE_FFMPEG_MARKER}\"\n"
        "  fi\n"
        "done\n",
        encoding="utf-8",
    )
    fake_apt.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_ENV": str(fake_environment),
            "FAKE_FFMPEG_MARKER": str(marker),
            "FAKE_APT_LOG": str(apt_log),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )
    return environment, apt_log


def test_torchcodec_runtime_bootstraps_missing_ffmpeg(tmp_path: Path) -> None:
    environment, apt_log = _fake_missing_ffmpeg_environment(tmp_path)
    helper = JIHE_DIR / "ensure_torchcodec_runtime.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(helper)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = apt_log.read_text(encoding="utf-8")
    assert "Acquire::Retries=3 update" in calls
    assert "Acquire::Retries=3 install -y --no-install-recommends ffmpeg" in calls
    assert "installing the Ubuntu ffmpeg runtime" in result.stdout


def test_torchcodec_runtime_can_disable_automatic_install(tmp_path: Path) -> None:
    environment, apt_log = _fake_missing_ffmpeg_environment(tmp_path)
    environment["FASTWAM_AUTO_INSTALL_FFMPEG"] = "0"
    helper = JIHE_DIR / "ensure_torchcodec_runtime.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(helper)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "automatic installation is disabled" in result.stderr
    assert not apt_log.exists()
