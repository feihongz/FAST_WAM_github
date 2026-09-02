from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_smoke_1xh100.sh"
)
SHORT_LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_smoke"
)
LABEL_MANIFEST_SHA256 = (
    "d6dc98a6a36c30150db30000c86d07c7a1e7d90b1dc5d1a5a60e02126c22b3e0"
)


def _run(
    launcher: Path = LAUNCHER,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(launcher), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _normalized(output: str) -> str:
    return (
        output.replace(r"\[", "[")
        .replace(r"\]", "]")
        .replace(r"\:", ":")
    )


def test_dry_run_is_single_process_full_epoch_and_makes_no_writes(tmp_path):
    storage_root = tmp_path / "persistent"
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(storage_root),
            "RUN_ID": "pytest-libero-gate-smoke",
            "NPROC_PER_NODE": "auto",
            "WORLD_SIZE": "16",
            "RANK": "7",
            "LOCAL_RANK": "7",
        }
    )

    result = _run(environment=environment)

    assert result.returncode == 0, result.stderr
    assert not storage_root.exists()
    assert list(tmp_path.iterdir()) == []
    output = _normalized(result.stdout + result.stderr)
    assert "profile=smoke" in output
    assert "benchmark=LIBERO" in output
    assert "topology=1x1" in output
    assert "process_mode=single_python_no_torchrun" in output
    assert "scripts/train_video_gate.py" in output
    assert "task=libero_stage2_gate_2cam224" in output
    assert "torchrun" not in output.split("[launch]", 1)[1]
    assert "training.batch_size=64" in output
    assert "training.num_workers=0" in output
    assert "training.num_epochs=1" in output
    assert "checkpoint.resume=null" in output
    assert "cublas_workspace_config=:4096:8" in output
    assert "train_samples=48768" in output
    assert "validation_samples=5408" in output
    assert "updates_per_epoch=762" in output
    assert "maximum_updates=762" in output
    assert "run_root=" in output
    assert "/formal_runs/smokes/stage2/gate/libero_1xh100_" in output
    assert "pytest-libero-gate-smoke" in output
    assert "gate_parameters=658977" in output
    assert LABEL_MANIFEST_SHA256 in output
    assert "verify_libero_stage2_gate_smoke.py" in output
    assert "no files, GPUs, packages, or output directories were touched" in output


def test_short_launcher_forces_smoke_over_an_ambient_formal_profile(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "RUN_ID": "pytest-short-gate-smoke",
            "FASTWAM_GATE_PROFILE": "formal",
        }
    )

    result = _run(SHORT_LAUNCHER, environment=environment)

    assert result.returncode == 0, result.stderr
    assert "profile=smoke" in result.stdout
    assert "task=libero_stage2_gate_2cam224" in result.stdout
    assert "training.num_epochs=1" in result.stdout
    assert "maximum_updates=762" in result.stdout
    assert "verify_libero_stage2_gate_smoke.py" in result.stdout
    assert "verify_libero_stage2_gate_formal.py" not in result.stdout
    assert "cublas_workspace_config=:4096:8" in result.stdout
    assert not (tmp_path / "persistent").exists()


def test_launcher_rejects_arguments_and_multi_gpu_visibility(tmp_path):
    positional = _run(LAUNCHER, "training.num_epochs=2")
    assert positional.returncode != 0
    assert "takes no arguments" in positional.stderr

    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "FASTWAM_CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    multi_gpu = _run(environment=environment)
    assert multi_gpu.returncode != 0
    assert "exactly one device" in multi_gpu.stderr
    assert not (tmp_path / "persistent").exists()


def test_launcher_forwards_termination_signals_to_the_active_process():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "trap cleanup_heartbeat EXIT" in source
    assert "trap 'terminate_on_signal INT 130' INT" in source
    assert "trap 'terminate_on_signal TERM 143' TERM" in source
    assert "trap 'terminate_on_signal HUP 129' HUP" in source
    assert 'kill -s "${signal_name}" "${ACTIVE_PID}"' in source
    assert "trap cleanup_heartbeat EXIT INT TERM HUP" not in source


def test_gate_smoke_launchers_have_valid_bash_syntax():
    for launcher in (LAUNCHER, SHORT_LAUNCHER):
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
