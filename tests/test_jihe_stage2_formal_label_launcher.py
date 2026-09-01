from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_labels_8xh100.sh"
LEGACY_LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_labels_4xh100.sh"
)

FINAL_ADAPTER = (
    "/root/feihong/FastWAM/formal_runs/stage3/full/"
    "libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/"
    "checkpoints/exports/step_030000.pt"
)
FINAL_ADAPTER_SHA256 = (
    "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
)
SELECTION_SHA256 = (
    "426b635d637a0f3e5d31dd13612ff5ad786fd5cfe9ce27b0e8689854d9aa9e9b"
)
FORMAL_COVERAGE_SHA256 = (
    "d114ac25b61ab30f18185c9ea69a33d537b5196b145a8c5c3d6f6fd9d884708f"
)
FORMAL_SAMPLE_IDS_SHA256 = (
    "e1122e20a0f48fd988baad3b70eea5258f4091918460bb78a7b50c4b30924aac"
)


def _run_launcher(
    *arguments: str,
    environment: dict[str, str] | None = None,
    launcher: Path = LAUNCHER,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(launcher), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _unescape_printed_shell_arguments(output: str) -> str:
    return output.replace(r"\[", "[").replace(r"\]", "]")


def test_dry_run_plans_exact_formal_eight_gpu_job_without_writes(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "persistent"
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(storage_root),
            "RUN_ID": "pytest-libero-stage2-labels",
            # The public launcher must replace inherited ad-hoc topology values.
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NPROC_PER_NODE": "auto",
            "SENSECORE_ACCELERATE_DEVICE_COUNT": "auto",
        }
    )

    result = _run_launcher(environment=environment)

    assert result.returncode == 0, result.stderr
    assert not storage_root.exists()
    assert list(tmp_path.iterdir()) == []

    output = _unescape_printed_shell_arguments(result.stdout + result.stderr)
    expected_output = (
        f"{storage_root}/FastWAM/formal_runs/stage2/labels/"
        "libero_stage2_gate_labels_2cam224/pytest-libero-stage2-labels"
    )
    assert "benchmark=LIBERO" in output
    assert "topology=1x8" in output
    assert "world_size=8" in output
    assert "rank_shards=rank_r_handles_r_r+8_..._r+56" in output
    assert "torchrun" in output
    assert "--standalone" in output
    assert "--nproc_per_node=8" in output
    assert "--nproc_per_node=auto" not in output
    assert "--max_restarts=0" in output
    assert "scripts/generate_gate_labels.py" in output
    assert "scripts/prepare_gate_label_selection.py" in output
    assert "task=libero_stage2_gate_labels_2cam224" in output
    assert f"label_selection.expected_sha256={SELECTION_SHA256}" in output
    assert "label_coverage.tier=formal" in output
    assert f"label_coverage.expected_sha256={FORMAL_COVERAGE_SHA256}" in output
    assert "labeling.num_shards=64" in output
    assert "labeling.chunk_size=64" in output
    assert "labeling.shard_indices=null" in output
    assert "runtime.device=cuda" in output
    assert "runtime.require_clean_git=true" in output
    assert f"output_dir={expected_output}" in output
    assert str(storage_root) in output
    assert "generation_success-formal-d114ac25.json" in output
    assert "planned_sample_count=54176" in output
    assert "train_sample_count=48768" in output
    assert "validation_sample_count=5408" in output
    assert "planned_chunk_count=977" in output
    assert "active_cohort_indices=0,1,2,4" in output
    assert "cohort_chunk_counts=0:218,1:219,2:413,4:127" in output
    assert "nonempty_cohort_shards=256" in output
    assert FORMAL_SAMPLE_IDS_SHA256 in output
    assert "merge_completed=false" in output
    assert "merge_gate_labels.py" in output
    assert "ffmpeg_apt_version=7:4.4.2-0ubuntu0.22.04.1" in output
    assert "ffmpeg_runtime_version=4.4.2-0ubuntu0.22.04.1" in output


def test_default_run_id_is_stable_for_same_commit_resume(tmp_path: Path) -> None:
    storage_root = tmp_path / "persistent"
    environment = os.environ.copy()
    environment.pop("RUN_ID", None)
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(storage_root),
            "NPROC_PER_NODE": "auto",
        }
    )
    git_short = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_job_dir = (
        f"{storage_root}/FastWAM/formal_runs/stage2/labels/"
        f"libero_stage2_gate_labels_2cam224/selection_426b635d_{git_short}"
    )

    first = _run_launcher(environment=environment)
    second = _run_launcher(environment=environment)

    for result in (first, second):
        assert result.returncode == 0, result.stderr
        output = _unescape_printed_shell_arguments(result.stdout + result.stderr)
        assert f"output_dir={expected_job_dir}" in output
        assert f"run_id=selection_426b635d_{git_short}" in output
        assert "--nproc_per_node=8" in output
    # Attempt-log names may vary, but the immutable/resumable job directory may not.
    assert not storage_root.exists()
    assert list(tmp_path.iterdir()) == []


def test_launcher_accepts_only_exact_eight_rank_topology(tmp_path: Path) -> None:
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "RUN_ID": "pytest-eight-rank-topology",
            "FASTWAM_CUDA_VISIBLE_DEVICES": (
                "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7"
            ),
        }
    )

    accepted_environment = dict(base_environment, NPROC_PER_NODE="8")
    accepted = _run_launcher(environment=accepted_environment)
    assert accepted.returncode == 0, accepted.stderr
    assert "--nproc_per_node=8" in accepted.stdout

    for invalid_nproc in ("4", "7", "9", "16"):
        rejected_environment = dict(
            base_environment,
            NPROC_PER_NODE=invalid_nproc,
        )
        rejected = _run_launcher(environment=rejected_environment)
        assert rejected.returncode != 0
        assert "require exactly 8 H100s" in rejected.stderr

    for invalid_visible_devices in (
        "0,1,2,3,4,5,6",
        "0,1,2,3,4,5,6,7,8",
        "0,1,2,3,4,5,6,6",
        "0,1,2,3,4,5,6,7,",
        ",0,1,2,3,4,5,6,7",
        "0,1,2,3,4,5,,6,7",
        "0,1,2,3,4,5,6, 7",
    ):
        rejected_environment = dict(
            base_environment,
            NPROC_PER_NODE="8",
            FASTWAM_CUDA_VISIBLE_DEVICES=invalid_visible_devices,
        )
        rejected = _run_launcher(environment=rejected_environment)
        assert rejected.returncode != 0
        assert "FASTWAM_CUDA_VISIBLE_DEVICES" in rejected.stderr

    assert not (tmp_path / "persistent").exists()


def test_retired_four_gpu_entrypoint_forwards_to_eight_gpu_launcher(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "RUN_ID": "pytest-legacy-forward",
            "NPROC_PER_NODE": "auto",
        }
    )

    result = _run_launcher(
        environment=environment,
        launcher=LEGACY_LAUNCHER,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "[compat] forwarding the retired 4xH100 command" in output
    assert "topology=1x8" in output
    assert "world_size=8" in output
    assert "--nproc_per_node=8" in output
    assert not (tmp_path / "persistent").exists()

    rejected = _run_launcher("labeling.num_shards=1", launcher=LEGACY_LAUNCHER)
    assert rejected.returncode != 0
    assert "takes no arguments" in rejected.stderr


def test_dry_run_locks_final_adapter_and_formal_contract(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_STORAGE_ROOT": str(tmp_path / "persistent"),
            "RUN_ID": "pytest-locked-formal-labels",
        }
    )

    result = _run_launcher(environment=environment)

    assert result.returncode == 0, result.stderr
    output = _unescape_printed_shell_arguments(result.stdout + result.stderr)
    assert FINAL_ADAPTER in output
    assert FINAL_ADAPTER_SHA256 in output
    assert "step_030000" in output
    assert "labeling.num_seed_pairs=2" in output
    assert "labeling.num_inference_steps=10" in output
    assert "labeling.relative_margin=0.05" in output
    assert "runtime.mixed_precision=bf16" in output


def test_launcher_declares_strict_generation_receipt_before_formal_merge() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "generation_success-${COVERAGE_TIER}-${COVERAGE_SHA256:0:8}.json" in script
    assert "merge_completed" in script
    assert "merge_completed=false" in script
    assert "planned_sample_count" in script
    assert 'readonly EXPECTED_SAMPLE_COUNT="54176"' in script
    assert 'readonly EXPECTED_TRAIN_SAMPLE_COUNT="48768"' in script
    assert 'readonly EXPECTED_VALIDATION_SAMPLE_COUNT="5408"' in script
    assert 'readonly EXPECTED_SAMPLE_COUNT="273465"' not in script
    assert "planned_chunk_count" in script
    assert 'readonly EXPECTED_CHUNK_COUNT="977"' in script
    assert 'readonly EXPECTED_CHUNK_COUNT="4307"' not in script
    assert "sample_id_sorted_fixed_chunks_per_cohort_shard_v1" in script
    assert 'job_dir.rglob("chunk-*.json")' in script
    assert "known inactive cohorts" in script
    assert "selection_sha256" in script
    assert "coverage_sha256" in script
    assert '"schema_version": 2' in script
    assert "merge_gate_labels.py" in script


def test_launcher_prepares_selection_before_full_preflight_and_model_load() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    prepare_call = '\n"${PREPARE_COMMAND[@]}"\n'
    preflight_call = "\npreflight\n"
    model_launch = 'setsid "${COMMAND[@]}"'
    assert prepare_call in script
    assert script.index(prepare_call) < script.index(preflight_call)
    assert script.index(preflight_call) < script.index(model_launch)
    assert "load_selection_artifacts" in script
    assert SELECTION_SHA256 in script
    assert FORMAL_COVERAGE_SHA256 in script
    assert FORMAL_SAMPLE_IDS_SHA256 in script


def test_launcher_guards_verifier_signals_and_receipt_symlinks() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "set +m" in script
    assert 'setsid "${PYTHON_BIN}" - \\' in script
    assert 'VERIFY_PID="$!"' in script
    assert "terminate_verify" in script
    assert "success_path = Path(success_path_raw).resolve()" not in script
    assert "O_NOFOLLOW" in script
    assert "st_nlink != 1" in script
    assert "metadata.st_nlink != 2" in script
    assert "residue_candidates" in script
    assert "crash-residue recovery" in script
    assert "FASTWAM_FFMPEG_APT_VERSION" in script
    assert "len(names) != 8" in script
    assert "expected exactly eight visible H100 GPUs" in script
    for forbidden in (
        "topology=1x4",
        "--nproc_per_node=4",
        "r+4",
        "16 of the 64 shards",
    ):
        assert forbidden not in script


def test_public_launcher_rejects_positional_or_hydra_arguments() -> None:
    result = _run_launcher("labeling.num_shards=1")

    assert result.returncode != 0
    assert "takes no arguments" in result.stderr


def test_launcher_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
