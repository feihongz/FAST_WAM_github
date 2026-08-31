from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "jihe" / "run_libero_stage2_label_smoke_1xh100.sh"
)

FINAL_ADAPTER = (
    "/root/feihong/FastWAM/formal_runs/stage3/full/"
    "libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/"
    "checkpoints/exports/step_030000.pt"
)
FINAL_ADAPTER_SHA256 = (
    "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
)


def _run_launcher(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _unescape_printed_shell_arguments(output: str) -> str:
    """Normalize Bash ``printf %q`` output for contract assertions."""

    return output.replace(r"\[", "[").replace(r"\]", "]")


def test_dry_run_plans_exact_two_phase_single_sample_workflow_without_writes(
    tmp_path: Path,
) -> None:
    smoke_root = tmp_path / "libero-stage2-label-smoke"
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_SMOKE_ROOT": str(smoke_root),
            "RUN_ID": "pytest-libero-stage2-label-smoke",
        }
    )

    result = _run_launcher(environment=environment)

    assert result.returncode == 0, result.stderr
    assert not smoke_root.exists()
    assert list(tmp_path.iterdir()) == []

    output = _unescape_printed_shell_arguments(result.stdout + result.stderr)
    label_entrypoint = "scripts/generate_gate_labels.py"
    assert output.count(label_entrypoint) == 2
    assert output.count("task=libero_stage2_gate_labels_2cam224") == 2
    assert output.count("labeling.num_shards=1048576") == 2
    assert output.count("labeling.shard_indices=[780575]") == 2
    assert output.count("labeling.chunk_size=1") == 2

    assert "phase=fresh" in output
    assert "phase=resume" in output
    assert "verification" in output.lower()
    assert str(smoke_root) in output


def test_dry_run_locks_final_adapter_and_marks_artifacts_non_mergeable(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_DRY_RUN": "1",
            "FASTWAM_SMOKE_ROOT": str(tmp_path / "smoke"),
            "RUN_ID": "pytest-locked-identity",
        }
    )

    result = _run_launcher(environment=environment)

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert FINAL_ADAPTER in output
    assert FINAL_ADAPTER_SHA256 in output
    assert "step_030000" in output
    assert "formal_merge_allowed=false" in output
    assert "NON-MERGEABLE" in output.upper()


def test_public_launcher_rejects_positional_or_hydra_arguments() -> None:
    result = _run_launcher("labeling.num_seed_pairs=4")

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
