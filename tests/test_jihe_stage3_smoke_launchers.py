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
    (
        "launcher",
        "benchmark",
        "task",
        "max_steps",
        "steps_per_epoch",
        "save_every",
        "keep_last",
        "data_exposure",
        "expected_wall_time",
        "nnodes",
        "world_size",
        "global_batch",
    ),
    (
        (
            "run_libero_stage3_full_8xh100.sh",
            "LIBERO",
            "libero_stage3_alignment_2cam224_1e-4",
            "30000",
            "5697",
            "1000",
            "31",
            "5.266 epochs / 1,440,000 windows",
            "19-23 hours",
            "1",
            "8",
            "48",
        ),
        (
            "run_robotwin_stage3_full_8xh100.sh",
            "RoboTwin-2.0",
            "robotwin_stage3_alignment_3cam384_1e-4",
            "20000",
            "62620",
            "250",
            "41",
            "0.3194 epoch / 1,920,000 windows",
            "90-110 hours",
            "2",
            "16",
            "96",
        ),
    ),
)
def test_full_stage3_launcher_dry_run_locks_formal_contract(
    launcher: str,
    benchmark: str,
    task: str,
    max_steps: str,
    steps_per_epoch: str,
    save_every: str,
    keep_last: str,
    data_exposure: str,
    expected_wall_time: str,
    nnodes: str,
    world_size: str,
    global_batch: str,
) -> None:
    environment = os.environ.copy()
    environment.pop("RESUME_STATE", None)
    environment.pop("MACHINE_RANK", None)
    environment.pop("GROUP_RANK", None)
    environment.pop("MASTER_IP", None)
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": "/tmp/fastwam-persistent",
            "RUN_ID": "pytest-full",
            "NPROC_PER_NODE": "auto",
            # The public wrapper must replace inherited ad-hoc destinations.
            "FASTWAM_OUTPUT_BASE": "/tmp/poisoned-output-base",
            "OUTPUT_DIR": "/tmp/poisoned-output-dir",
            "LOG_FILE": "/tmp/poisoned-log",
            "NNODES": nnodes,
            "NODE_RANK": "0",
        }
    )
    if nnodes == "2":
        environment["MASTER_ADDR"] = "10.0.0.1"
    else:
        environment.pop("MASTER_ADDR", None)
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
    expected_output = (
        f"/tmp/fastwam-persistent/FastWAM/formal_runs/stage3/full/"
        f"{task}/pytest-full"
    )
    assert f"benchmark={benchmark}" in output
    assert f"max_steps={max_steps}" in output
    assert f"steps_per_epoch={steps_per_epoch}" in output
    assert f"data_exposure={data_exposure}" in output
    assert f"expected_wall_time={expected_wall_time}" in output
    assert f"topology={nnodes}x8" in output
    assert f"world_size={world_size}" in output
    assert f"global_batch={global_batch}" in output
    assert f"output_dir={expected_output}" in output
    assert f"training.max_steps={max_steps}" in output
    assert "training.num_epochs=10" in output
    assert "training.learning_rate=1.0e-4" in output
    assert "training.weight_decay=1.0e-4" in output
    assert "training.betas=\\[0.9\\,0.95\\]" in output
    assert "training.max_grad_norm=1.0" in output
    assert "training.lr_scheduler_type=cosine" in output
    assert "training.warmup_ratio=0.05" in output
    assert "training.seed=42" in output
    assert "stage3.num_solver_steps=10" in output
    assert "stage3.lambda_action=1.0" in output
    assert "stage3.lambda_align=1.0" in output
    assert "stage3.lambda_safe=0.5" in output
    assert f"checkpoint.save_every={save_every}" in output
    assert f"checkpoint.keep_last={keep_last}" in output
    assert "checkpoint.save_final=true" in output
    assert "runtime.log_every=100" in output
    assert f"--num_machines {nnodes}" in output
    assert "--machine_rank 0" in output
    assert f"--num_processes {world_size}" in output
    if nnodes == "2":
        assert "--main_process_ip 10.0.0.1" in output
        assert "--deepspeed_multinode_launcher standard" in output
        assert f"log_file={expected_output}/launch.node0.log" in output
    else:
        assert "--deepspeed_multinode_launcher" not in output
        assert f"log_file={expected_output}/launch.log" in output
    assert "/tmp/poisoned" not in output


def test_robotwin_full_stage3_rank1_dry_run_uses_shared_output_and_node_log() -> None:
    environment = os.environ.copy()
    environment.pop("MACHINE_RANK", None)
    environment.pop("GROUP_RANK", None)
    environment.pop("MASTER_IP", None)
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": "/tmp/fastwam-persistent",
            "RUN_ID": "pytest-full-rank1",
            "NPROC_PER_NODE": "8",
            "NNODES": "2",
            "NODE_RANK": "1",
            "MASTER_ADDR": "10.0.0.1",
        }
    )
    result = subprocess.run(
        ["bash", str(JIHE_DIR / "run_robotwin_stage3_full_8xh100.sh")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected_output = (
        "/tmp/fastwam-persistent/FastWAM/formal_runs/stage3/full/"
        "robotwin_stage3_alignment_3cam384_1e-4/pytest-full-rank1"
    )
    assert "node_rank=1" in result.stdout
    assert "--machine_rank 1" in result.stdout
    assert "--num_processes 16" in result.stdout
    assert "--deepspeed_multinode_launcher standard" in result.stdout
    assert f"output_dir={expected_output}" in result.stdout
    assert f"log_file={expected_output}/launch.node1.log" in result.stdout


@pytest.mark.parametrize(
    "launcher",
    (
        "run_libero_stage3_full_8xh100.sh",
        "run_robotwin_stage3_full_8xh100.sh",
    ),
)
def test_public_full_launcher_rejects_hydra_arguments(launcher: str) -> None:
    result = subprocess.run(
        ["bash", str(JIHE_DIR / launcher), "training.max_steps=1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "takes no arguments" in result.stderr


def test_full_launcher_rejects_pilot_resume_state() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "RUN_ID": "pytest-pilot-resume",
            "RESUME_STATE": "/formal_runs/pilots/stage3/run/states/LATEST",
        }
    )
    result = subprocess.run(
        ["bash", str(JIHE_DIR / "run_libero_stage3_full_8xh100.sh")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pilot state cannot resume" in result.stderr


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



def test_torchcodec_runtime_installs_requested_pinned_ffmpeg(tmp_path: Path) -> None:
    environment, apt_log = _fake_missing_ffmpeg_environment(tmp_path)
    environment.update(
        {
            "FASTWAM_FFMPEG_APT_VERSION": "7:4.4.2-0ubuntu0.22.04.1",
            "FASTWAM_FFMPEG_RUNTIME_VERSION": "4.4.2-0ubuntu0.22.04.1",
        }
    )
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
    assert "--allow-downgrades ffmpeg=7:4.4.2-0ubuntu0.22.04.1" in calls
    assert "installing pinned Ubuntu ffmpeg=7:4.4.2-0ubuntu0.22.04.1" in result.stdout

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
