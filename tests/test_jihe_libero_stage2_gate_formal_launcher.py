from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_smoke_1xh100.sh"
)
FORMAL_LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_full_1xh100.sh"
)
SHORT_LAUNCHER = REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate"


def _run(
    launcher: Path = FORMAL_LAUNCHER,
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


def test_formal_dry_run_is_single_process_fresh_and_makes_no_writes(tmp_path):
    storage_root = tmp_path / "persistent"
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(storage_root),
            "RUN_ID": "pytest-libero-gate-formal",
            "FASTWAM_GATE_PROFILE": "smoke",
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
    assert "profile=formal" in output
    assert "benchmark=LIBERO" in output
    assert "topology=1x1" in output
    assert "process_mode=single_python_no_torchrun" in output
    assert "scripts/train_video_gate.py" in output
    training_launch = output.split("[launch]", 1)[1].split("[launch]", 1)[0]
    assert "torchrun" not in training_launch
    assert "accelerate" not in training_launch
    assert "training.batch_size=64" in output
    assert "training.num_workers=0" in output
    assert "training.num_epochs=20" in output
    assert "training.early_stop_patience=3" in output
    assert "training.min_delta=1.0e-4" in output
    assert "checkpoint.resume=null" in output
    assert "cublas_workspace_config=:4096:8" in output
    assert "updates_per_epoch=762" in output
    assert "maximum_updates=15240" in output
    assert "epochs=20" in output
    assert "early_stop_patience=3" in output
    assert "min_delta=1.0e-4" in output
    assert "/formal_runs/stage2/gate/libero_stage2_gate_2cam224_20ep/" in output
    assert "pytest-libero-gate-formal" in output
    assert "verify_libero_stage2_gate_formal.py" in output
    assert "verify_libero_stage2_gate_smoke.py" not in output
    assert "no files, GPUs, packages, or output directories were touched" in output


def test_short_formal_launcher_overrides_an_ambient_smoke_profile(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "RUN_ID": "pytest-short-gate-formal",
            "FASTWAM_GATE_PROFILE": "smoke",
        }
    )

    result = _run(SHORT_LAUNCHER, environment=environment)

    assert result.returncode == 0, result.stderr
    assert "profile=formal" in result.stdout
    assert "training.num_epochs=20" in result.stdout
    assert "maximum_updates=15240" in result.stdout
    assert "verify_libero_stage2_gate_formal.py" in result.stdout
    assert not (tmp_path / "persistent").exists()


def test_formal_launcher_rejects_arguments_and_multi_gpu_visibility(tmp_path):
    positional = _run(SHORT_LAUNCHER, "training.num_epochs=2")
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
    multi_gpu = _run(SHORT_LAUNCHER, environment=environment)
    assert multi_gpu.returncode != 0
    assert "exactly one device" in multi_gpu.stderr
    assert not (tmp_path / "persistent").exists()


def test_engine_rejects_an_unknown_profile_without_writes(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "FASTWAM_GATE_PROFILE": "typo",
        }
    )

    result = _run(ENGINE, environment=environment)

    assert result.returncode != 0
    assert "FASTWAM_GATE_PROFILE must be smoke or formal" in result.stderr
    assert not (tmp_path / "persistent").exists()


def test_engine_forwards_signals_and_checks_repository_after_both_phases():
    source = ENGINE.read_text(encoding="utf-8")

    assert "trap cleanup_heartbeat EXIT" in source
    assert "trap 'terminate_on_signal INT 130' INT" in source
    assert "trap 'terminate_on_signal TERM 143' TERM" in source
    assert "trap 'terminate_on_signal HUP 129' HUP" in source
    assert 'kill -s "${signal_name}" "${ACTIVE_PID}"' in source
    assert "trap cleanup_heartbeat EXIT INT TERM HUP" not in source

    assert '[[ "$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)" == "${GIT_COMMIT}" ]]' in source
    assert 'git -C "${FASTWAM_REPO_DIR}" diff --quiet' in source
    assert 'git -C "${FASTWAM_REPO_DIR}" diff --cached --quiet' in source
    assert "ls-files --others --exclude-standard -- src configs scripts tests" in source
    assert 'mkdir -p -- "${RUN_ROOT%/*}"' in source
    assert 'mkdir -- "${RUN_ROOT}"' in source
    assert 'mkdir -p -- "${RUN_ROOT}"' not in source
    training_guard = source.index('verify_repository_immutability "training"')
    verification_guard = source.index('verify_repository_immutability "verification"')
    train_wait = source.index('fail "LIBERO Gate ${PROFILE} training failed')
    verify_wait = source.index('fail "LIBERO Gate ${PROFILE} verification failed')
    assert train_wait < training_guard < verify_wait < verification_guard


def test_gate_formal_launchers_have_valid_bash_syntax():
    for launcher in (ENGINE, FORMAL_LAUNCHER, SHORT_LAUNCHER):
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
