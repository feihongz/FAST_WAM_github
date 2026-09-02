from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "scripts" / "jihe" / "_run_libero_stage2_gate_4xh100.sh"
FORMAL_LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_full_4xh100.sh"
)
SHORT_LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_gate_4xh100"
)


def _environment(tmp_path: Path, **updates: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "FASTWAM_CUDA_VISIBLE_DEVICES",
        "FASTWAM_GATE_PROFILE",
        "FASTWAM_GATE_RESUME",
        "FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT",
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MACHINE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NNODES",
        "NODE_RANK",
        "NPROC_PER_NODE",
        "RANK",
        "ROLE_RANK",
        "SENSECORE_ACCELERATE_DEVICE_COUNT",
        "WORLD_SIZE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "FASTWAM_CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "RUN_ID": "pytest-libero-gate-ddp4",
            **updates,
        }
    )
    return environment


def _run(
    launcher: Path = SHORT_LAUNCHER,
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


def test_auto_dry_run_is_fresh_ddp4_and_makes_no_writes(tmp_path):
    storage_root = tmp_path / "persistent"
    environment = _environment(
        tmp_path,
        NPROC_PER_NODE="auto",
        SENSECORE_ACCELERATE_DEVICE_COUNT="4",
        # JiHe may inject stale allocation-wide rank metadata.  The nested,
        # single-node torchrun must replace it rather than inherit it.
        WORLD_SIZE="16",
        RANK="7",
        LOCAL_RANK="3",
    )

    result = _run(environment=environment)

    assert result.returncode == 0, result.stderr
    assert not storage_root.exists()
    assert list(tmp_path.iterdir()) == []
    output = _normalized(result.stdout + result.stderr)
    assert "profile=formal" in output
    assert "benchmark=LIBERO" in output
    assert "topology=1x4" in output
    assert "world_size=4" in output
    assert "process_mode=torchrun_native_ddp" in output
    assert "per_rank_batch=16" in output
    assert "global_batch=64" in output
    assert "updates_per_epoch=762" in output
    assert "maximum_updates=15240" in output
    # The persisted training contract keeps batch_size as the logical global
    # batch. The distributed loader derives 16 samples per rank at world 4.
    assert "training.batch_size=64" in output
    assert "training.num_epochs=20" in output
    assert "training.early_stop_patience=3" in output
    assert "training.min_delta=1.0e-4" in output
    assert "checkpoint.resume=null" in output
    assert "/formal_runs/stage2/gate/libero_stage2_gate_2cam224_20ep_4xh100/" in output
    assert "pytest-libero-gate-ddp4" in output

    launches = output.split("[launch]")
    assert len(launches) >= 3
    training_launch = launches[1]
    assert "torchrun" in training_launch
    assert "--standalone" in training_launch
    assert "--nnodes=1" in training_launch
    assert "--nproc_per_node=4" in training_launch
    assert "--max_restarts=0" in training_launch
    assert "accelerate" not in training_launch
    assert "verify_libero_stage2_gate_formal_4xh100.py" in output
    assert "no files, GPUs, packages, or output directories were touched" in output


def test_explicit_four_rank_dry_run_is_accepted(tmp_path):
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="4",
            SENSECORE_ACCELERATE_DEVICE_COUNT="4",
        )
    )

    assert result.returncode == 0, result.stderr
    output = _normalized(result.stdout + result.stderr)
    assert "topology=1x4" in output
    assert "--nproc_per_node=4" in output
    assert not (tmp_path / "persistent").exists()


def test_literal_auto_from_both_jihe_variables_is_normalized(tmp_path):
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="auto",
            SENSECORE_ACCELERATE_DEVICE_COUNT="auto",
        )
    )

    assert result.returncode == 0, result.stderr
    output = _normalized(result.stdout + result.stderr)
    assert "topology=1x4" in output
    assert "--nproc_per_node=4" in output
    assert not (tmp_path / "persistent").exists()


def test_resume_dry_run_uses_explicit_existing_run_contract_without_writes(
    tmp_path,
):
    run_root = tmp_path / "persistent" / "FastWAM" / "resume-run"
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="4",
            FASTWAM_GATE_RESUME="1",
            FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT=str(run_root),
        )
    )

    assert result.returncode == 0, result.stderr
    output = _normalized(result.stdout + result.stderr)
    expected_state = run_root / "gate_run" / "training_state.pt"
    assert "resume_mode=1" in output
    assert f"resume_state={expected_state}" in output
    assert f"checkpoint.resume={expected_state}" in output
    assert "checkpoint.resume=null" not in output
    assert not run_root.exists()


def test_resume_requires_explicit_run_root_without_writes(tmp_path):
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="4",
            FASTWAM_GATE_RESUME="1",
        )
    )

    assert result.returncode != 0
    assert (
        "FASTWAM_GATE_RESUME=1 requires "
        "FASTWAM_LIBERO_GATE_FORMAL_4XH100_ROOT"
    ) in result.stderr
    assert not (tmp_path / "persistent").exists()


@pytest.mark.parametrize("resume_mode", ["-1", "2", "true", "garbage"])
def test_launcher_rejects_invalid_resume_mode_without_writes(
    tmp_path,
    resume_mode: str,
):
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="4",
            FASTWAM_GATE_RESUME=resume_mode,
        )
    )

    assert result.returncode != 0
    assert "FASTWAM_GATE_RESUME must be 0 or 1" in result.stderr
    assert not (tmp_path / "persistent").exists()


@pytest.mark.parametrize("requested", ["0", "1", "3", "8", "garbage"])
def test_launcher_rejects_non_four_nproc_without_writes(
    tmp_path,
    requested: str,
):
    result = _run(
        environment=_environment(tmp_path, NPROC_PER_NODE=requested)
    )

    assert result.returncode != 0
    assert "exactly 4" in result.stderr
    assert not (tmp_path / "persistent").exists()


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"NNODES": "2", "NODE_RANK": "0"}, "NNODES must be exactly 1"),
        ({"NNODES": "1", "NODE_RANK": "1"}, "NODE_RANK must be 0"),
        ({"NNODES": "1", "MACHINE_RANK": "1"}, "NODE_RANK must be 0"),
        ({"NNODES": "1", "GROUP_RANK": "1"}, "NODE_RANK must be 0"),
    ],
)
def test_launcher_rejects_multinode_metadata_without_writes(
    tmp_path,
    updates: dict[str, str],
    expected_error: str,
):
    result = _run(
        environment=_environment(tmp_path, NPROC_PER_NODE="4", **updates)
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (tmp_path / "persistent").exists()


@pytest.mark.parametrize(
    "visible_devices",
    [
        "0",
        "0,1,2",
        "0,1,2,3,4",
        "0,1,2,2",
        "0,1,2,",
        ",0,1,2",
        "0,,2,3",
        "0, 1,2,3",
        "0,1,2,3 ",
    ],
)
def test_launcher_rejects_invalid_four_gpu_lists_without_writes(
    tmp_path,
    visible_devices: str,
):
    result = _run(
        environment=_environment(
            tmp_path,
            NPROC_PER_NODE="4",
            FASTWAM_CUDA_VISIBLE_DEVICES=visible_devices,
        )
    )

    assert result.returncode != 0
    assert "FASTWAM_CUDA_VISIBLE_DEVICES" in result.stderr
    assert "four" in result.stderr.lower()
    assert not (tmp_path / "persistent").exists()


def test_short_launcher_forces_formal_profile_and_rejects_arguments(tmp_path):
    environment = _environment(
        tmp_path,
        NPROC_PER_NODE="4",
        FASTWAM_GATE_PROFILE="smoke",
    )

    result = _run(environment=environment)
    positional = _run(
        SHORT_LAUNCHER,
        "training.num_epochs=2",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "profile=formal" in result.stdout
    assert "training.num_epochs=20" in result.stdout
    assert "maximum_updates=15240" in result.stdout
    assert positional.returncode != 0
    assert "takes no arguments" in positional.stderr
    assert not (tmp_path / "persistent").exists()


def test_engine_owns_torchrun_process_group_and_repository_guards():
    source = ENGINE.read_text(encoding="utf-8")

    assert "trap cleanup_heartbeat EXIT" in source
    assert "trap 'terminate_on_signal INT 130' INT" in source
    assert "trap 'terminate_on_signal TERM 143' TERM" in source
    assert "trap 'terminate_on_signal HUP 129' HUP" in source
    assert re.search(r"\bsetsid\b.*\"\$\{COMMAND\[@\]\}\"", source)
    assert re.search(
        r"kill\s+-s\s+\"\$\{signal_name\}\"\s+--\s+\"-\$\{ACTIVE_PID\}\"",
        source,
    )

    assert (
        '[[ "$(git -C "${FASTWAM_REPO_DIR}" rev-parse HEAD)" == "${GIT_COMMIT}" ]]'
        in source
    )
    assert 'git -C "${FASTWAM_REPO_DIR}" diff --quiet' in source
    assert 'git -C "${FASTWAM_REPO_DIR}" diff --cached --quiet' in source
    assert "ls-files --others --exclude-standard -- src configs scripts tests" in source
    assert 'verify_repository_immutability "preflight"' in source
    assert 'verify_repository_immutability "training"' in source
    assert 'verify_repository_immutability "verification"' in source
    assert 'mkdir -p -- "${RUN_ROOT%/*}"' in source
    assert 'mkdir -- "${RUN_ROOT}"' in source
    assert 'mkdir -p -- "${RUN_ROOT}"' not in source
    assert 'if [[ "${RESUME_MODE}" == "0" ]]; then' in source
    assert '[[ -d "${RAW_RUN_ROOT}" && ! -L "${RAW_RUN_ROOT}" ]]' in source
    assert '[[ -f "${RESUME_STATE}" && ! -L "${RESUME_STATE}" ]]' in source
    assert '[[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]]' in source
    assert 'tee -a "${TRAIN_LOG}"' in source

    training_guard = source.index('verify_repository_immutability "training"')
    verification_guard = source.index(
        'verify_repository_immutability "verification"'
    )
    train_wait = source.index("training failed with exit code")
    verify_wait = source.index("verification failed with exit code")
    assert train_wait < training_guard < verify_wait < verification_guard


def test_engine_has_four_h100_runtime_preflight():
    source = ENGINE.read_text(encoding="utf-8")

    assert "torch.cuda.device_count() != 4" in source
    assert "expected exactly four visible CUDA devices" in source
    assert '"H100" not in name' in source


def test_ddp4_launchers_have_valid_bash_syntax():
    for launcher in (ENGINE, FORMAL_LAUNCHER, SHORT_LAUNCHER):
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
