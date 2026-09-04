#!/usr/bin/env python3
"""Strict 8-GPU orchestration for the formal LIBERO Phase-A evaluation.

The seven conditions are evaluated in increasing expected video compute.  A
condition owns a dynamic queue of 40 independent single-GPU task evaluators;
there is never more than one evaluator on a GPU.  Evaluator output is first
written below ``attempts/`` and is atomically promoted only after strict local
validation.

Fresh runs never reuse an existing directory.  Interrupted runs must be
continued explicitly with ``--resume`` and the original immutable run root.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.gating.calibration_artifact import (  # noqa: E402
    load_gate_calibration_selection,
)
from fastwam.alignment.checkpointing import read_git_identity as _read_git_identity  # noqa: E402
from fastwam.alignment.data_identity import (  # noqa: E402
    canonical_data_manifest_sha256,
)

from fastwam.alignment.libero_simulator_identity import (  # noqa: E402
    capture_libero_simulator_runtime_identity,
    verify_libero_simulator_runtime_identity,
)

SCHEMA_VERSION = 1
NUM_GPUS = 8
NUM_TRIALS = 50
MAX_ATTEMPTS = 3
SEED = 42
VIDEO_STEPS = 10
REPLAN_STEPS = 32
ACTION_HORIZON = 32
TIMING_WARMUP_QUERIES = 3
TASK_SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
TARGET_RATES = (0.10, 0.25, 0.50, 0.75, 0.90)

# JiHe may launch this orchestration process through torchrun, Accelerate, or
# an MPI-compatible scheduler.  Every evaluator is intentionally a standalone
# one-GPU process, so none of the parent's distributed topology may leak into
# ``PartialState`` or torch.distributed initialization in the child.
_DISTRIBUTED_ENV_EXACT = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NPROC_PER_NODE",
        "RDZV_ENDPOINT",
        "RDZV_ID",
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NODEID",
        "SLURM_NNODES",
        "SLURM_TASKS_PER_NODE",
    }
)
_DISTRIBUTED_ENV_PREFIXES = (
    "TORCHELASTIC_",
    "ACCELERATE_",
    "OMPI_",
    "PMI_",
    "PMIX_",
    "MV2_",
    "I_MPI_",
    "MPI_",
    "MPIR_",
    "PET_",
)


@dataclass(frozen=True, slots=True)
class FrozenAssets:
    task_config: str = "libero_stage3_alignment_2cam224_1e-4"
    base_checkpoint: str = (
        "/root/feihong/FastWAM/formal_runs/FAST_WAM_github/"
        "libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/"
        "checkpoints/weights/latest.pt"
    )
    base_sha256: str = (
        "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
    )
    adapter_checkpoint: str = (
        "/root/feihong/FastWAM/formal_runs/stage3/full/"
        "libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/"
        "checkpoints/exports/step_030000.pt"
    )
    adapter_sha256: str = (
        "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
    )
    data_manifest: str = (
        "/root/feihong/FastWAM/formal_runs/contracts/stage3/"
        "libero_current_273465f_1693e/libero_stage3_data_manifest.json"
    )
    data_manifest_sha256: str = (
        "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
    )
    stage3_contract_sha256: str = (
        "84ee86f32912ca96fa058b02ce7997362b8350e73f4e0f4377bc8728af3e6d98"
    )
    stage3_global_step: int = 30_000
    normalization_stats: str = (
        "/root/feihong/FastWAM/formal_runs/FAST_WAM_github/"
        "libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/"
        "dataset_stats.json"
    )
    normalization_stats_sha256: str = (
        "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
    )
    vae: str = (
        "/root/feihong/FastWAM/checkpoints/Wan-AI/"
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )
    vae_sha256: str = (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    )
    calibration_complete: str = (
        "/root/feihong/FastWAM/formal_runs/stage2/calibration/"
        "libero_stage2_gate_2cam224/9cd8106_2026-09-04_05-11-41/"
        "calibration/COMPLETE"
    )
    calibration_complete_sha256: str = (
        "a4f1acec144fcbfaaab830fe70dac181f1f5a42ddcce5d8dc222e7531d609955"
    )
    libero_root: str = "/root/feihong/FastWAM/third_party/LIBERO"
    libero_datasets: str = (
        "/root/feihong/FastWAM/datasets/libero_mujoco3.3.2"
    )


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    routing_mode: str
    inference_mode: str | None
    target_with_rate: float | None
    resolved_threshold: float | None
    expected_nfe_per_query: float
    gate_checkpoint: dict[str, Any] | None
    calibration_receipt: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Task:
    suite: str
    task_id: int

    @property
    def key(self) -> str:
        return f"{self.suite}_task{self.task_id:02d}"


@dataclass(frozen=True, slots=True)
class TaskSource:
    """Immutable pre-run identity for one official LIBERO task."""

    suite: str
    task_id: int
    task_name: str
    source_initial_state_count: int
    task_bddl_path: str
    task_bddl_sha256: str
    task_bddl_size_bytes: int
    initial_states_path: str
    initial_states_sha256: str
    initial_states_size_bytes: int

    def expected_task(self) -> dict[str, Any]:
        return {
            "task_suite_name": self.suite,
            "task_id": self.task_id,
            "source_initial_state_count": self.source_initial_state_count,
            "environment_assets": {
                "task_bddl": {
                    "path": self.task_bddl_path,
                    "sha256": self.task_bddl_sha256,
                    "size_bytes": self.task_bddl_size_bytes,
                },
                "initial_states": {
                    "path": self.initial_states_path,
                    "sha256": self.initial_states_sha256,
                    "size_bytes": self.initial_states_size_bytes,
                },
            },
        }


@dataclass(frozen=True, slots=True)
class TaskSourceLedger:
    task_map_path: str
    task_map_sha256: str
    task_map_size_bytes: int
    tasks: tuple[TaskSource, ...]

    def expected_tasks(self) -> list[dict[str, Any]]:
        expected = [source.expected_task() for source in self.tasks]
        task_keys = [
            (entry["task_suite_name"], entry["task_id"]) for entry in expected
        ]
        if task_keys != [(task.suite, task.task_id) for task in all_tasks()]:
            raise ValueError("task source ledger does not match the formal 40-task order")
        return expected

    def manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "kind": "libero_phase_a_task_source_ledger",
            "task_map": {
                "path": self.task_map_path,
                "sha256": self.task_map_sha256,
                "size_bytes": self.task_map_size_bytes,
            },
            "tasks": [
                {**source.expected_task(), "task_name": source.task_name}
                for source in self.tasks
            ],
        }
        return {**payload, "ledger_sha256": _canonical_sha256(payload)}


@dataclass(frozen=True, slots=True)
class RunIdentity:
    commit: str
    tracked_dirty: bool
    untracked_source_files: tuple[str, ...]


class StopRequested(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    before = resolved.stat()
    sha256 = _sha256_file(resolved)
    after = resolved.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while it was hashed: {resolved}")
    if after.st_size <= 0:
        raise ValueError(f"{label} must not be empty: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256,
        "size_bytes": int(after.st_size),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def read_run_identity() -> RunIdentity:
    # Use the exact helper used by ``eval_libero_single.py`` so the launcher
    # and every result agree byte-for-byte on the expected Git identity.
    identity = _read_git_identity(REPO_ROOT)
    return RunIdentity(
        identity.commit,
        identity.tracked_dirty,
        tuple(identity.untracked_source_files),
    )


def require_clean_run_identity(identity: RunIdentity) -> None:
    if identity.tracked_dirty or identity.untracked_source_files:
        raise RuntimeError(
            "formal Phase-A evaluation requires a clean, committed source tree: "
            f"tracked_dirty={identity.tracked_dirty}, "
            f"untracked_source_files={list(identity.untracked_source_files)}"
        )


def require_formal_sources_tracked() -> None:
    required = (
        "experiments/libero/eval_libero_single.py",
        "experiments/libero/formal_eval_validation.py",
        "scripts/run_libero_phase_a.py",
        "scripts/jihe/eval_libero_phase_a_8xh100.sh",
        "src/fastwam/gating/calibration_artifact.py",
        "src/fastwam/alignment/libero_simulator_identity.py",
    )
    for relative in required:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", relative],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"formal Phase-A source must be committed before execution: {relative}"
            )


def all_tasks() -> tuple[Task, ...]:
    tasks = tuple(Task(suite, task_id) for suite in TASK_SUITES for task_id in range(10))
    if len(tasks) != 40 or len({task.key for task in tasks}) != 40:
        raise AssertionError("formal LIBERO Phase-A task set must contain 40 unique tasks")
    return tasks


def _literal_assignment(path: Path, name: str) -> Any:
    """Read one literal module assignment without executing third-party code."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ValueError(f"cannot parse trusted LIBERO task map: {path}") from error
    matches: list[ast.AST] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one literal {name} assignment in {path}")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"{name} must be a Python literal in {path}") from error


def _trusted_initial_state_count(path: Path) -> int:
    """Load an official local init-state file and return its exact cardinality."""

    import torch

    try:
        states = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        states = torch.load(path, map_location="cpu")
    try:
        count = len(states)
    except TypeError as error:
        raise ValueError(f"LIBERO initial states have no finite length: {path}") from error
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"LIBERO initial states must be non-empty: {path}")
    return int(count)


def build_task_source_ledger(assets: FrozenAssets) -> TaskSourceLedger:
    """Freeze the official 40-task environment inputs before any rollout starts."""

    libero_root = Path(assets.libero_root).expanduser().resolve(strict=True)
    benchmark_root = (libero_root / "libero" / "libero").resolve(strict=True)
    try:
        benchmark_root.relative_to(libero_root)
    except ValueError as error:
        raise ValueError("configured LIBERO benchmark root escapes libero_root") from error
    task_map_path = benchmark_root / "benchmark" / "libero_suite_task_map.py"
    task_map_identity_before = _stable_file_identity(
        task_map_path, label="official LIBERO task map"
    )
    task_map = _literal_assignment(task_map_path, "libero_task_map")
    task_map_identity_after = _stable_file_identity(
        task_map_path, label="official LIBERO task map"
    )
    if task_map_identity_after != task_map_identity_before:
        raise RuntimeError("official LIBERO task map changed while it was parsed")
    if not isinstance(task_map, dict):
        raise ValueError("official LIBERO task map must be a dictionary")

    bddl_root = (benchmark_root / "bddl_files").resolve(strict=True)
    init_root = (benchmark_root / "init_files").resolve(strict=True)
    sources: list[TaskSource] = []
    for task in all_tasks():
        names = task_map.get(task.suite)
        if (
            not isinstance(names, list)
            or len(names) != 10
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 10
        ):
            raise ValueError(
                f"official LIBERO suite {task.suite!r} must contain 10 unique names"
            )
        task_name = names[task.task_id]
        if Path(task_name).name != task_name:
            raise ValueError(f"unsafe LIBERO task name: {task_name!r}")
        bddl_path = (bddl_root / task.suite / f"{task_name}.bddl").resolve(
            strict=True
        )
        init_path = (init_root / task.suite / f"{task_name}.pruned_init").resolve(
            strict=True
        )
        try:
            bddl_path.relative_to(bddl_root)
            init_path.relative_to(init_root)
        except ValueError as error:
            raise ValueError(f"LIBERO task asset escapes its trusted root: {task.key}") from error

        bddl_identity = _stable_file_identity(
            bddl_path, label=f"{task.key} BDDL"
        )
        init_identity_before = _stable_file_identity(
            init_path, label=f"{task.key} initial states"
        )
        source_count = _trusted_initial_state_count(init_path)
        init_identity_after = _stable_file_identity(
            init_path, label=f"{task.key} initial states"
        )
        if init_identity_after != init_identity_before:
            raise RuntimeError(
                f"{task.key} initial states changed while their count was loaded"
            )
        sources.append(
            TaskSource(
                suite=task.suite,
                task_id=task.task_id,
                task_name=task_name,
                source_initial_state_count=source_count,
                task_bddl_path=bddl_identity["path"],
                task_bddl_sha256=bddl_identity["sha256"],
                task_bddl_size_bytes=bddl_identity["size_bytes"],
                initial_states_path=init_identity_before["path"],
                initial_states_sha256=init_identity_before["sha256"],
                initial_states_size_bytes=init_identity_before["size_bytes"],
            )
        )
    ledger = TaskSourceLedger(
        task_map_path=task_map_identity_before["path"],
        task_map_sha256=task_map_identity_before["sha256"],
        task_map_size_bytes=task_map_identity_before["size_bytes"],
        tasks=tuple(sources),
    )
    ledger.expected_tasks()
    return ledger


def resolve_conditions(assets: FrozenAssets) -> tuple[Condition, ...]:
    conditions: list[Condition] = [
        Condition("static_wo", "static", "wo", None, None, 0.0, None, None)
    ]
    for target in TARGET_RATES:
        loaded = load_gate_calibration_selection(
            assets.calibration_complete,
            expected_complete_sha256=assets.calibration_complete_sha256,
            target_with_rate=target,
            configured_video_steps=VIDEO_STEPS,
        )
        expected_gate_sources = {
            "adapter_checkpoint_sha256": assets.adapter_sha256,
            "base_checkpoint_sha256": assets.base_sha256,
            "data_manifest_sha256": assets.data_manifest_sha256,
        }
        mismatches = {
            key: (loaded.manifest["gate_checkpoint"].get(key), expected)
            for key, expected in expected_gate_sources.items()
            if loaded.manifest["gate_checkpoint"].get(key) != expected
        }
        observed_stats_sha = loaded.receipt["source_identities"].get(
            "normalization_stats_sha256"
        )
        if observed_stats_sha != assets.normalization_stats_sha256:
            mismatches["normalization_stats_sha256"] = (
                observed_stats_sha,
                assets.normalization_stats_sha256,
            )
        if mismatches:
            raise ValueError(
                "Gate calibration sources do not match frozen Phase-A assets: "
                f"{mismatches}"
            )
        point = loaded.receipt["selected_point"]
        expected_nfe = float(point["expected_video_steps_per_query"])
        conditions.append(
            Condition(
                f"gate_r{round(target * 100):03d}",
                "gate",
                None,
                target,
                loaded.threshold,
                expected_nfe,
                dict(loaded.manifest["gate_checkpoint"]),
                dict(loaded.receipt),
            )
        )
    conditions.append(
        Condition(
            "static_w",
            "static",
            "w",
            None,
            None,
            float(VIDEO_STEPS),
            None,
            None,
        )
    )
    expected_names = (
        "static_wo",
        "gate_r010",
        "gate_r025",
        "gate_r050",
        "gate_r075",
        "gate_r090",
        "static_w",
    )
    if tuple(condition.name for condition in conditions) != expected_names:
        raise AssertionError("formal condition ordering changed")
    if [condition.expected_nfe_per_query for condition in conditions] != sorted(
        condition.expected_nfe_per_query for condition in conditions
    ):
        raise AssertionError("formal conditions must be ordered by expected compute")
    return tuple(conditions)


def _common_overrides(
    *,
    assets: FrozenAssets,
    task: Task,
    attempt_dir: Path,
    simulator_runtime_identity: Mapping[str, Any],
) -> list[str]:
    return [
        f"task={assets.task_config}",
        f"ckpt={assets.base_checkpoint}",
        "gpu_id=0",
        f"seed={SEED}",
        "mixed_precision=bf16",
        "model.load_text_encoder=false",
        f"EVALUATION.task_suite_name={task.suite}",
        f"EVALUATION.task_id={task.task_id}",
        f"EVALUATION.num_trials={NUM_TRIALS}",
        f"EVALUATION.output_dir={attempt_dir}",
        f"EVALUATION.dataset_stats_path={assets.normalization_stats}",
        f"EVALUATION.stage3_adapter_path={assets.adapter_checkpoint}",
        f"EVALUATION.stage3_adapter_sha256={assets.adapter_sha256}",
        f"EVALUATION.stage3_base_sha256={assets.base_sha256}",
        f"EVALUATION.stage3_data_manifest_sha256={assets.data_manifest_sha256}",
        f"EVALUATION.stage3_training_contract_sha256={assets.stage3_contract_sha256}",
        f"EVALUATION.stage3_global_step={assets.stage3_global_step}",
        "EVALUATION.use_manifest_text_cache=true",
        f"EVALUATION.replan_steps={REPLAN_STEPS}",
        f"EVALUATION.action_horizon={ACTION_HORIZON}",
        f"EVALUATION.num_inference_steps={VIDEO_STEPS}",
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.save_videos=false",
        "EVALUATION.timing_enabled=true",
        f"EVALUATION.timing_warmup_queries={TIMING_WARMUP_QUERIES}",
        "EVALUATION.save_query_metrics=true",
        "EVALUATION.retry_invalid_episodes=false",
        "EVALUATION.env_num=1",
        "EVALUATION.device=cuda:0",
        "EVALUATION.simulator_runtime_identity_sha256="
        f"{simulator_runtime_identity['identity_sha256']}",
    ]


def build_evaluator_command(
    *,
    python_bin: str,
    assets: FrozenAssets,
    condition: Condition,
    task: Task,
    attempt_dir: Path,
    simulator_runtime_identity: Mapping[str, Any],
) -> list[str]:
    command = [
        python_bin,
        "experiments/libero/eval_libero_single.py",
        *_common_overrides(
            assets=assets,
            task=task,
            attempt_dir=attempt_dir,
            simulator_runtime_identity=simulator_runtime_identity,
        ),
        f"EVALUATION.routing_mode={condition.routing_mode}",
    ]
    if condition.routing_mode == "static":
        if condition.inference_mode not in {"wo", "w"}:
            raise ValueError("static condition requires wo or w")
        command.append(f"EVALUATION.inference_mode={condition.inference_mode}")
    elif condition.routing_mode == "gate":
        if condition.target_with_rate is None:
            raise ValueError("Gate condition requires a target_with_rate")
        command.extend(
            [
                f"EVALUATION.gate_calibration_complete={assets.calibration_complete}",
                "EVALUATION.gate_calibration_complete_sha256="
                f"{assets.calibration_complete_sha256}",
                "EVALUATION.gate_calibration_target_with_rate="
                f"{condition.target_with_rate}",
            ]
        )
    else:
        raise ValueError(f"unsupported Phase-A routing mode: {condition.routing_mode}")
    return command


def _expected_routing_runtime(condition: Condition) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "binary_video_routing_runtime",
        "routing_mode": condition.routing_mode,
        "inference_mode": condition.inference_mode,
        "configured_video_steps": VIDEO_STEPS,
        "gate_threshold": condition.resolved_threshold,
        "gate_decision_rule": (
            "sigmoid(logit) >= gate_threshold -> w"
            if condition.routing_mode == "gate"
            else None
        ),
        "random_video_probability": None,
        "random_seed": None,
        "use_manifest_text_cache": True,
        "prompt_template": (
            "A video recorded from a robot's point of view executing the "
            "following instruction: {task}"
        ),
        "gate_input_preprocessing": (
            "processor.val_transforms_per_camera_then_concat_then_2x_minus_1"
            if condition.routing_mode == "gate"
            else None
        ),
        "timing": {
            "enabled": True,
            "warmup_queries_per_task": TIMING_WARMUP_QUERIES,
            "save_query_metrics": True,
        },
        "gate_checkpoint": condition.gate_checkpoint,
        "gate_calibration": condition.calibration_receipt,
    }


def condition_contract(
    *,
    condition: Condition,
    assets: FrozenAssets,
    run_identity: RunIdentity,
    task_sources: TaskSourceLedger,
    simulator_runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    gate_sha = None
    calibration_manifest_sha = None
    if condition.routing_mode == "gate":
        if condition.gate_checkpoint is None or condition.calibration_receipt is None:
            raise ValueError("Gate condition is missing its verified calibration identity")
        gate_sha = condition.gate_checkpoint["sha256"]
        calibration_manifest_sha = condition.calibration_receipt["manifest_file"][
            "semantic_sha256"
        ]
    runtime_identity = _expected_routing_runtime(condition)
    return {
        "schema_version": 1,
        "kind": "libero_phase_a_condition_contract",
        "condition_id": condition.name,
        "expected_tasks": task_sources.expected_tasks(),
        "num_trials_per_task": NUM_TRIALS,
        "simulator_runtime_identity": dict(simulator_runtime_identity),
        "routing": {
            "routing_mode": condition.routing_mode,
            "configured_video_steps": VIDEO_STEPS,
            "inference_mode": condition.inference_mode,
            "gate_threshold": condition.resolved_threshold,
            "calibration_complete_sha256": (
                assets.calibration_complete_sha256
                if condition.routing_mode == "gate"
                else None
            ),
        },
        "expected_identities": {
            "evaluation_git_identity": {
                "commit": run_identity.commit,
                "tracked_dirty": False,
                "untracked_source_files": [],
            },
            "base_checkpoint_sha256": assets.base_sha256,
            "alignment_export_sha256": assets.adapter_sha256,
            "data_manifest_sha256": assets.data_manifest_sha256,
            "normalization_stats_sha256": assets.normalization_stats_sha256,
            "vae_sha256": assets.vae_sha256,
            "gate_checkpoint_sha256": gate_sha,
            "calibration_manifest_sha256": calibration_manifest_sha,
        },
        "protocol_shared": {
            "schema_version": 1,
            "kind": "libero_closed_loop_evaluation_protocol",
            "benchmark": "LIBERO",
            "seed": SEED,
            "mixed_precision": "bf16",
            "num_trials": NUM_TRIALS,
            "initial_state_order": (
                "official_task_order_repeat_prefix_only_if_num_trials_exceeds_source"
            ),
            "env_num": 1,
            "render_resolution": 256,
            "num_steps_wait": 30,
            "action_horizon": ACTION_HORIZON,
            "replan_steps": REPLAN_STEPS,
            "num_inference_steps": VIDEO_STEPS,
            "binarize_gripper": True,
            "use_action_ensembler": False,
            "visualize_future_video": False,
            "save_videos": False,
            "retry_invalid_episodes": False,
            "max_invalid_episode_retries": 20,
            "black_screen_filter": False,
            "black_screen_thresholds": {
                "mean": 5.0,
                "std": 2.0,
                "minimum_frame_fraction": 0.8,
            },
            "timing": {
                "enabled": True,
                "warmup_queries_per_task": TIMING_WARMUP_QUERIES,
                "save_query_metrics": True,
            },
            "sampling": {
                "sigma_shift": None,
                "text_cfg_scale": 1.0,
                "negative_prompt": "",
                "rand_device": "cpu",
                "tiled": False,
            },
            "model_input": {
                "height": 224,
                "width": 448,
                "num_frames": 33,
                "concat_multi_camera": "horizontal",
                "action_video_freq_ratio": 4,
            },
            "prompt_template": (
                "A video recorded from a robot's point of view executing the "
                "following instruction: {task}"
            ),
            "routing_runtime_identity_sha256": _canonical_sha256(runtime_identity),
            "simulator_runtime_identity_sha256": simulator_runtime_identity[
                "identity_sha256"
            ],
        },
    }


def _formal_validate_task_file(
    contract: Mapping[str, Any], result_path: Path
) -> dict[str, Any]:
    # Delayed import keeps ``--help`` and orchestration inspection lightweight.
    from experiments.libero.formal_eval_validation import validate_task_result_file

    return validate_task_result_file(contract, result_path)


def _formal_aggregate(
    contract: Mapping[str, Any], result_paths: Sequence[Path]
) -> dict[str, Any]:
    from experiments.libero.formal_eval_validation import (
        aggregate_condition_result_files,
    )

    return aggregate_condition_result_files(contract, result_paths)


def _formal_validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.libero.formal_eval_validation import validate_condition_contract

    return validate_condition_contract(contract)


def validate_task_result(
    result_path: Path,
    *,
    task: Task,
    condition: Condition,
    assets: FrozenAssets,
    run_identity: RunIdentity,
    task_sources: TaskSourceLedger,
    simulator_runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Call the shared fail-closed validator before result promotion."""

    contract = condition_contract(
        condition=condition,
        assets=assets,
        run_identity=run_identity,
        task_sources=task_sources,
        simulator_runtime_identity=simulator_runtime_identity,
    )
    formal = _formal_validate_task_file(contract, result_path)
    if (
        formal.get("task_suite_name") != task.suite
        or formal.get("task_id") != task.task_id
    ):
        raise ValueError("strict validator returned the wrong task identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "libero_phase_a_task_validation",
        "validated_at": _utc_now(),
        "task": asdict(task),
        "condition": condition.name,
        "result_sha256": formal["result_file"]["sha256"],
        "successes": formal["successes"],
        "total_episodes": formal["num_episodes"],
        "query_count": formal["query_count"],
        "w_query_count": formal["route_counts"]["w"],
        "actual_total_video_nfe": formal["actual_total_video_nfe"],
        "strict_validator_receipt": formal,
    }


def _result_glob(condition_dir: Path, task: Task) -> list[Path]:
    return sorted(
        (condition_dir / "results" / task.suite).glob(
            f"gpu*_task{task.task_id}_results.json"
        )
    )


def _next_attempt_number(attempt_task_dir: Path) -> int:
    numbers: list[int] = []
    if attempt_task_dir.is_dir():
        for path in attempt_task_dir.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if suffix.isdigit():
                    numbers.append(int(suffix))
    return max(numbers, default=0) + 1


_ATTEMPT_TERMINAL_FAILURES = frozenset({"rejected", "launch_failed"})
_ATTEMPT_STATUSES = frozenset(
    {
        "prepared",
        "running",
        "promoted",
        "rejected",
        "launch_failed",
        "aborted",
        "abandoned",
    }
)


def _recover_attempt_history(
    attempt_task_dir: Path,
    *,
    condition: Condition,
    task: Task,
) -> int:
    """Close stale attempts and return only the terminal failure count.

    Attempt directory numbers are immutable event IDs. They are deliberately
    independent of the scientific retry budget so a scheduler interruption or
    parent crash can never exhaust evaluator retries.
    """

    terminal_failures = 0
    if not attempt_task_dir.is_dir():
        return terminal_failures
    for attempt_dir in sorted(attempt_task_dir.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
            continue
        suffix = attempt_dir.name.removeprefix("attempt_")
        if not suffix.isdigit() or int(suffix) <= 0:
            raise ValueError(f"invalid attempt directory name: {attempt_dir}")
        attempt_number = int(suffix)
        record_path = attempt_dir / "attempt.json"
        if record_path.is_file():
            record = _read_json(record_path, label="Phase-A attempt record")
            if record.get("attempt") != attempt_number:
                raise ValueError(f"attempt record number mismatch: {record_path}")
            if record.get("condition") != condition.name:
                raise ValueError(f"attempt record condition mismatch: {record_path}")
            if record.get("task") != asdict(task):
                raise ValueError(f"attempt record task mismatch: {record_path}")
            status = record.get("status")
            if status is None:
                # Compatibility with attempts created before lifecycle states
                # were explicit.
                status = "abandoned"
            elif status not in _ATTEMPT_STATUSES:
                raise ValueError(f"unknown attempt status {status!r}: {record_path}")
            if status in {"prepared", "running"}:
                status = "abandoned"
            if status == "abandoned" and record.get("status") != "abandoned":
                record["status"] = "abandoned"
                record["abandoned_at"] = _utc_now()
                record["error"] = "parent process ended before a terminal attempt state"
                _atomic_json(record_path, record)
        else:
            status = "abandoned"
            _atomic_json(
                record_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "libero_phase_a_task_attempt",
                    "condition": condition.name,
                    "task": asdict(task),
                    "attempt": attempt_number,
                    "status": status,
                    "abandoned_at": _utc_now(),
                    "error": "attempt directory existed without an attempt record",
                },
            )
        if status in _ATTEMPT_TERMINAL_FAILURES:
            terminal_failures += 1
    return terminal_failures


def _promote_result(
    *,
    source: Path,
    condition_dir: Path,
    task: Task,
    receipt: Mapping[str, Any],
) -> Path:
    final_dir = condition_dir / "results" / task.suite
    final_dir.mkdir(parents=True, exist_ok=True)
    target = final_dir / source.name
    if target.exists() or _result_glob(condition_dir, task):
        raise FileExistsError(f"refusing to overwrite promoted result for {task.key}")
    os.replace(source, target)
    promoted_receipt = dict(receipt)
    promoted_receipt["promoted_at"] = _utc_now()
    promoted_receipt["promoted_result"] = str(target)
    _atomic_json(
        condition_dir / "validation_receipts" / f"{task.key}.json",
        promoted_receipt,
    )
    return target


def _condition_complete_payload(
    *,
    condition: Condition,
    receipts: Sequence[Mapping[str, Any]],
    strict_aggregate: Mapping[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    if not completed_at:
        raise ValueError("condition completion timestamp must be non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "libero_phase_a_condition_complete",
        "completed_at": completed_at,
        "condition": asdict(condition),
        "num_tasks": 40,
        "num_trials_per_task": NUM_TRIALS,
        "total_episodes": 40 * NUM_TRIALS,
        "result_sha256": {
            receipt["task"]["suite"] + f"/task{receipt['task']['task_id']}": receipt[
                "result_sha256"
            ]
            for receipt in receipts
        },
        "aggregate": {
            "successes": sum(int(receipt["successes"]) for receipt in receipts),
            "queries": sum(int(receipt["query_count"]) for receipt in receipts),
            "w_queries": sum(int(receipt["w_query_count"]) for receipt in receipts),
            "actual_total_video_nfe": sum(
                int(receipt["actual_total_video_nfe"]) for receipt in receipts
            ),
        },
        "strict_validator_aggregate": dict(strict_aggregate),
    }


class Orchestrator:
    def __init__(
        self,
        *,
        run_root: Path,
        python_bin: str,
        gpu_devices: tuple[str, ...],
        assets: FrozenAssets,
        conditions: tuple[Condition, ...],
        run_identity: RunIdentity,
        task_sources: TaskSourceLedger,
        simulator_runtime_identity: Mapping[str, Any],
        retry_delay_s: float,
    ) -> None:
        self.run_root = run_root
        self.python_bin = python_bin
        self.gpu_devices = gpu_devices
        self.assets = assets
        self.conditions = conditions
        self.run_identity = run_identity
        self.task_sources = task_sources
        self.simulator_runtime_identity = dict(simulator_runtime_identity)
        self.retry_delay_s = retry_delay_s
        self.stop_event = threading.Event()
        self.active_lock = threading.Lock()
        self.active: dict[int, subprocess.Popen[str]] = {}
        self.signal_number: int | None = None

    def request_stop(self, signum: int) -> None:
        self.signal_number = signum
        self.stop_event.set()
        print(
            f"[signal] received={signal.Signals(signum).name}; "
            "terminating active evaluators",
            flush=True,
        )
        with self.active_lock:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def _spawn_active(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        log_stream: Any,
    ) -> subprocess.Popen[str] | None:
        """Atomically stop-check, spawn, and register one evaluator child."""

        with self.active_lock:
            if self.stop_event.is_set():
                return None
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.active[process.pid] = process
            return process

    def _unregister_active(self, process: subprocess.Popen[str]) -> None:
        with self.active_lock:
            self.active.pop(process.pid, None)

    def _task_worker(
        self,
        *,
        condition: Condition,
        task_queue: queue.Queue[Task],
        gpu_slot: int,
        failures: list[str],
        failures_lock: threading.Lock,
    ) -> None:
        condition_dir = self.run_root / "conditions" / condition.name
        gpu_device = self.gpu_devices[gpu_slot]
        while not self.stop_event.is_set():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                succeeded = self._run_task(
                    condition=condition,
                    condition_dir=condition_dir,
                    task=task,
                    gpu_slot=gpu_slot,
                    gpu_device=gpu_device,
                )
                if not succeeded:
                    with failures_lock:
                        failures.append(task.key)
            finally:
                task_queue.task_done()

    def _run_task(
        self,
        *,
        condition: Condition,
        condition_dir: Path,
        task: Task,
        gpu_slot: int,
        gpu_device: str,
    ) -> bool:
        attempt_task_dir = self.run_root / "attempts" / condition.name / task.key
        terminal_failures = _recover_attempt_history(
            attempt_task_dir,
            condition=condition,
            task=task,
        )
        if terminal_failures >= MAX_ATTEMPTS:
            print(
                f"[failed-final] {condition.name} {task.key}: "
                f"{terminal_failures} terminal failures exhausted the budget",
                flush=True,
            )
            return False

        while terminal_failures < MAX_ATTEMPTS:
            if self.stop_event.is_set():
                return False
            attempt_number = _next_attempt_number(attempt_task_dir)
            attempt_dir = attempt_task_dir / f"attempt_{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = build_evaluator_command(
                python_bin=self.python_bin,
                assets=self.assets,
                condition=condition,
                task=task,
                attempt_dir=attempt_dir,
                simulator_runtime_identity=self.simulator_runtime_identity,
            )
            command_record = {
                "schema_version": SCHEMA_VERSION,
                "kind": "libero_phase_a_task_attempt",
                "started_at": _utc_now(),
                "condition": condition.name,
                "task": asdict(task),
                "attempt": attempt_number,
                "gpu_slot": gpu_slot,
                "cuda_visible_device": gpu_device,
                "argv": command,
                "status": "prepared",
            }
            _atomic_json(attempt_dir / "attempt.json", command_record)
            log_path = attempt_dir / "evaluator.log"
            print(
                f"[start] condition={condition.name} task={task.key} "
                f"gpu_slot={gpu_slot} attempt={attempt_number}",
                flush=True,
            )
            environment = single_gpu_child_environment(
                os.environ, gpu_device=gpu_device
            )
            process: subprocess.Popen[str] | None = None
            try:
                with log_path.open("x", encoding="utf-8") as log_stream:
                    command_record["status"] = "running"
                    _atomic_json(attempt_dir / "attempt.json", command_record)
                    process = self._spawn_active(
                        command,
                        environment=environment,
                        log_stream=log_stream,
                    )
                    if process is None:
                        command_record["status"] = "aborted"
                        command_record["finished_at"] = _utc_now()
                        command_record["error"] = "stop requested before child launch"
                        _atomic_json(attempt_dir / "attempt.json", command_record)
                        return False
                    try:
                        return_code = process.wait()
                    finally:
                        self._unregister_active(process)
            except OSError as error:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                command_record["finished_at"] = _utc_now()
                command_record["error"] = f"{type(error).__name__}: {error}"
                if self.stop_event.is_set():
                    command_record["status"] = "aborted"
                    _atomic_json(attempt_dir / "attempt.json", command_record)
                    return False
                command_record["status"] = "launch_failed"
                terminal_failures += 1
                _atomic_json(attempt_dir / "attempt.json", command_record)
                print(
                    f"[retry] condition={condition.name} task={task.key} "
                    f"attempt_id={attempt_number} terminal_failure="
                    f"{terminal_failures}/{MAX_ATTEMPTS} error={error}",
                    flush=True,
                )
                if terminal_failures < MAX_ATTEMPTS:
                    self.stop_event.wait(self.retry_delay_s)
                continue

            command_record["finished_at"] = _utc_now()
            command_record["return_code"] = return_code
            if self.stop_event.is_set():
                command_record["status"] = "aborted"
                command_record["error"] = (
                    "parent stop requested while evaluator child was active"
                )
                _atomic_json(attempt_dir / "attempt.json", command_record)
                return False
            candidate = (
                attempt_dir
                / task.suite
                / f"gpu0_task{task.task_id}_results.json"
            )
            try:
                if return_code != 0:
                    raise RuntimeError(f"evaluator exited with code {return_code}")
                if not candidate.is_file():
                    raise FileNotFoundError(f"evaluator result is missing: {candidate}")
                receipt = validate_task_result(
                    candidate,
                    task=task,
                    condition=condition,
                    assets=self.assets,
                    run_identity=self.run_identity,
                    task_sources=self.task_sources,
                    simulator_runtime_identity=self.simulator_runtime_identity,
                )
                promoted = _promote_result(
                    source=candidate,
                    condition_dir=condition_dir,
                    task=task,
                    receipt=receipt,
                )
                command_record["status"] = "promoted"
                command_record["promoted_result"] = str(promoted)
                _atomic_json(attempt_dir / "attempt.json", command_record)
                print(
                    f"[ok] condition={condition.name} task={task.key} "
                    f"gpu_slot={gpu_slot} attempt={attempt_number}",
                    flush=True,
                )
                return True
            except Exception as error:
                command_record["status"] = "rejected"
                command_record["error"] = f"{type(error).__name__}: {error}"
                terminal_failures += 1
                _atomic_json(attempt_dir / "attempt.json", command_record)
                print(
                    f"[retry] condition={condition.name} task={task.key} "
                    f"attempt_id={attempt_number} terminal_failure="
                    f"{terminal_failures}/{MAX_ATTEMPTS} error={error}",
                    flush=True,
                )
                if terminal_failures < MAX_ATTEMPTS and not self.stop_event.is_set():
                    self.stop_event.wait(self.retry_delay_s)
        return False

    def _existing_tasks(
        self, condition: Condition
    ) -> tuple[list[Task], list[dict[str, Any]]]:
        condition_dir = self.run_root / "conditions" / condition.name
        pending: list[Task] = []
        receipts: list[dict[str, Any]] = []
        for task in all_tasks():
            existing = _result_glob(condition_dir, task)
            if not existing:
                pending.append(task)
                continue
            if len(existing) != 1:
                raise RuntimeError(f"multiple promoted results found for {task.key}")
            receipt = validate_task_result(
                existing[0],
                task=task,
                condition=condition,
                assets=self.assets,
                run_identity=self.run_identity,
                task_sources=self.task_sources,
                simulator_runtime_identity=self.simulator_runtime_identity,
            )
            receipts.append(receipt)
        return pending, receipts

    def run_condition(self, condition: Condition) -> None:
        verify_libero_simulator_runtime_identity(
            self.simulator_runtime_identity, libero_root=self.assets.libero_root
        )
        condition_dir = self.run_root / "conditions" / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        contract = _formal_validate_contract(
            condition_contract(
                condition=condition,
                assets=self.assets,
                run_identity=self.run_identity,
                task_sources=self.task_sources,
                simulator_runtime_identity=self.simulator_runtime_identity,
            )
        )
        contract_path = condition_dir / "condition_contract.json"
        if contract_path.exists():
            if _read_json(contract_path, label="condition contract") != contract:
                raise ValueError(f"condition contract changed for {condition.name}")
        else:
            _atomic_json(contract_path, contract)
        pending, receipts = self._existing_tasks(condition)
        complete_path = condition_dir / "COMPLETE.json"
        if complete_path.exists():
            if pending or len(receipts) != 40:
                raise ValueError(
                    f"condition COMPLETE conflicts with missing results: "
                    f"{condition.name} pending={len(pending)}"
                )
            result_paths = [
                _result_glob(condition_dir, task)[0] for task in all_tasks()
            ]
            strict_aggregate = _formal_aggregate(contract, result_paths)
            recorded_complete = _read_json(
                complete_path, label="condition COMPLETE marker"
            )
            completed_at = recorded_complete.get("completed_at")
            if not isinstance(completed_at, str) or not completed_at:
                raise ValueError(
                    f"condition COMPLETE has an invalid timestamp: {complete_path}"
                )
            expected_complete = _condition_complete_payload(
                condition=condition,
                receipts=receipts,
                strict_aggregate=strict_aggregate,
                completed_at=completed_at,
            )
            if recorded_complete != expected_complete:
                raise ValueError(
                    f"condition COMPLETE disagrees with validated results: "
                    f"{complete_path}"
                )
            verify_libero_simulator_runtime_identity(
                self.simulator_runtime_identity,
                libero_root=self.assets.libero_root,
            )
            print(f"[condition-resume-complete] {condition.name}", flush=True)
            return
        print(
            f"[condition] name={condition.name} pending={len(pending)} "
            f"completed={40 - len(pending)} expected_nfe/query="
            f"{condition.expected_nfe_per_query:.6f}",
            flush=True,
        )
        if pending:
            task_queue: queue.Queue[Task] = queue.Queue()
            for task in pending:
                task_queue.put(task)
            failures: list[str] = []
            failures_lock = threading.Lock()
            workers = [
                threading.Thread(
                    target=self._task_worker,
                    kwargs={
                        "condition": condition,
                        "task_queue": task_queue,
                        "gpu_slot": slot,
                        "failures": failures,
                        "failures_lock": failures_lock,
                    },
                    name=f"phase-a-gpu-{slot}",
                )
                for slot in range(NUM_GPUS)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            if self.stop_event.is_set():
                raise StopRequested("Phase-A run was interrupted")
            if failures:
                raise RuntimeError(
                    f"condition {condition.name} exhausted retries: {sorted(failures)}"
                )

        pending_after, receipts = self._existing_tasks(condition)
        if pending_after or len(receipts) != 40:
            raise RuntimeError(
                f"condition {condition.name} is incomplete: pending={len(pending_after)}"
            )
        result_paths = [
            _result_glob(condition_dir, task)[0] for task in all_tasks()
        ]
        strict_aggregate = _formal_aggregate(contract, result_paths)
        verify_libero_simulator_runtime_identity(
            self.simulator_runtime_identity, libero_root=self.assets.libero_root
        )
        complete = _condition_complete_payload(
            condition=condition,
            receipts=receipts,
            strict_aggregate=strict_aggregate,
            completed_at=_utc_now(),
        )
        _atomic_json(complete_path, complete)
        print(f"[condition-complete] {condition.name}", flush=True)

    def run(self) -> None:
        for condition in self.conditions:
            if self.stop_event.is_set():
                raise StopRequested("Phase-A run was interrupted")
            self.run_condition(condition)
        verify_libero_simulator_runtime_identity(
            self.simulator_runtime_identity, libero_root=self.assets.libero_root
        )
        _atomic_json(
            self.run_root / "COMPLETE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "libero_phase_a_complete",
                "completed_at": _utc_now(),
                "conditions": [condition.name for condition in self.conditions],
                "total_tasks": len(self.conditions) * 40,
                "total_episodes": len(self.conditions) * 40 * NUM_TRIALS,
            },
        )


def _experiment_spec(
    *,
    assets: FrozenAssets,
    conditions: Sequence[Condition],
    run_identity: RunIdentity,
    task_sources: TaskSourceLedger,
    simulator_runtime_identity: Mapping[str, Any],
    python_bin: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "libero_phase_a_experiment_spec",
        "repo_root": str(REPO_ROOT),
        "python_bin": os.path.abspath(os.path.expanduser(python_bin)),
        "git_identity": {
            "commit": run_identity.commit,
            "tracked_dirty": run_identity.tracked_dirty,
            "untracked_source_files": list(run_identity.untracked_source_files),
        },
        "assets": asdict(assets),
        "task_source_ledger": task_sources.manifest(),
        "simulator_runtime_identity": dict(simulator_runtime_identity),
        "protocol": {
            "condition_order": [asdict(condition) for condition in conditions],
            "task_suites": list(TASK_SUITES),
            "tasks_per_suite": 10,
            "num_tasks": 40,
            "num_trials_per_task": NUM_TRIALS,
            "num_gpus": NUM_GPUS,
            "max_evaluators_per_gpu": 1,
            "max_attempts_per_task": MAX_ATTEMPTS,
            "seed": SEED,
            "video_steps": VIDEO_STEPS,
            "replan_steps": REPLAN_STEPS,
            "action_horizon": ACTION_HORIZON,
            "use_manifest_text_cache": True,
            "save_videos": False,
            "visualize_future_video": False,
            "timing_warmup_queries": TIMING_WARMUP_QUERIES,
        },
    }


def _libero_config_payload(assets: FrozenAssets) -> str:
    benchmark_root = Path(assets.libero_root) / "libero" / "libero"
    return "\n".join(
        [
            f"assets: {benchmark_root / 'assets'}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"benchmark_root: {benchmark_root}",
            f"datasets: {assets.libero_datasets}",
            f"init_states: {benchmark_root / 'init_files'}",
            "",
        ]
    )


def _archive_failed_marker(run_root: Path) -> None:
    failed_path = run_root / "FAILED.json"
    if not failed_path.exists():
        return
    if not failed_path.is_file():
        raise ValueError(f"FAILED marker must be a regular file: {failed_path}")
    history_dir = run_root / "history" / "failures"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = history_dir / f"FAILED.{timestamp}.{os.getpid()}.json"
    os.replace(failed_path, target)
    for directory in (history_dir, run_root):
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def prepare_run_root(
    *,
    run_root: Path,
    resume: bool,
    spec: Mapping[str, Any],
    assets: FrozenAssets,
) -> None:
    manifest_path = run_root / "run_manifest.json"
    spec_sha = _canonical_sha256(spec)
    if resume:
        if not run_root.is_dir():
            raise FileNotFoundError(f"resume root does not exist: {run_root}")
        complete_path = run_root / "COMPLETE.json"
        if complete_path.exists():
            raise ValueError(
                f"refusing to resume an already complete Phase-A run: {complete_path}"
            )
        manifest = _read_json(manifest_path, label="Phase-A run manifest")
        if manifest.get("experiment_spec_sha256") != spec_sha:
            raise ValueError("resume experiment spec does not match the original run")
        if manifest.get("experiment_spec") != spec:
            raise ValueError("resume experiment spec payload changed")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "libero_phase_a_run",
                "created_at": _utc_now(),
                "experiment_spec_sha256": spec_sha,
                "experiment_spec": dict(spec),
            },
        )

    config_dir = run_root / "runtime" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    expected_config = _libero_config_payload(assets)
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != expected_config:
            raise ValueError("resume LIBERO runtime config changed")
    else:
        _atomic_text(config_path, expected_config)
    if resume:
        _archive_failed_marker(run_root)


def _preflight_paths(assets: FrozenAssets, python_bin: str) -> None:
    files = (
        assets.base_checkpoint,
        assets.adapter_checkpoint,
        assets.data_manifest,
        assets.normalization_stats,
        assets.vae,
        assets.calibration_complete,
        python_bin,
    )
    missing_files = [path for path in files if not Path(path).expanduser().is_file()]
    directories = (assets.libero_root, assets.libero_datasets)
    missing_dirs = [path for path in directories if not Path(path).expanduser().is_dir()]
    if missing_files or missing_dirs:
        raise FileNotFoundError(
            f"preflight paths missing: files={missing_files}, dirs={missing_dirs}"
        )
    if not os.access(Path(python_bin).expanduser(), os.X_OK):
        raise PermissionError(f"preflight Python is not executable: {python_bin}")


def _preflight_frozen_identities(assets: FrozenAssets) -> None:
    """Hash every large frozen model asset once before creating a run root."""

    physical = (
        ("base checkpoint", assets.base_checkpoint, assets.base_sha256),
        ("Stage 3 adapter", assets.adapter_checkpoint, assets.adapter_sha256),
        (
            "normalization stats",
            assets.normalization_stats,
            assets.normalization_stats_sha256,
        ),
        ("VAE", assets.vae, assets.vae_sha256),
        (
            "Gate calibration COMPLETE",
            assets.calibration_complete,
            assets.calibration_complete_sha256,
        ),
    )
    for label, path, expected_sha256 in physical:
        print(f"[preflight] hashing {label}: {path}", flush=True)
        observed = _stable_file_identity(path, label=label)
        if observed["sha256"] != expected_sha256:
            raise ValueError(
                f"{label} SHA256 mismatch: expected={expected_sha256}, "
                f"actual={observed['sha256']}"
            )

    manifest_path = Path(assets.data_manifest).expanduser().resolve(strict=True)
    before = _stable_file_identity(manifest_path, label="Stage 3 data manifest")
    manifest = _read_json(manifest_path, label="Stage 3 data manifest")
    recorded = manifest.get("manifest_sha256")
    computed = canonical_data_manifest_sha256(manifest)
    after = _stable_file_identity(manifest_path, label="Stage 3 data manifest")
    if before != after:
        raise RuntimeError("Stage 3 data manifest changed during preflight")
    if recorded != assets.data_manifest_sha256 or computed != assets.data_manifest_sha256:
        raise ValueError(
            "Stage 3 data manifest semantic SHA256 mismatch: "
            f"expected={assets.data_manifest_sha256}, recorded={recorded}, "
            f"computed={computed}"
        )


def _print_dry_run(
    *,
    run_root: Path,
    python_bin: str,
    assets: FrozenAssets,
    conditions: Sequence[Condition],
    gpu_devices: Sequence[str],
    task_sources: TaskSourceLedger,
    simulator_runtime_identity: Mapping[str, Any],
) -> None:
    print("[dry-run] no directories or evaluator processes will be created")
    print(f"[locked] run_root={run_root}")
    print(f"[locked] gpu_devices={','.join(gpu_devices)} workers=8 per_gpu=1")
    print("[locked] tasks=40 trials_per_task=50 total_conditions=7")
    print(f"[locked] task_source_ledger={task_sources.manifest()['ledger_sha256']}")
    print(
        "[locked] simulator_runtime_identity="
        f"{simulator_runtime_identity['identity_sha256']}"
    )
    for index, condition in enumerate(conditions, start=1):
        detail = (
            f"target={condition.target_with_rate} threshold={condition.resolved_threshold}"
            if condition.routing_mode == "gate"
            else f"endpoint={condition.inference_mode}"
        )
        print(
            f"[condition {index}/7] {condition.name} {detail} "
            f"validation_expected_nfe/query={condition.expected_nfe_per_query:.6f}"
        )
        sample_attempt = (
            run_root
            / "attempts"
            / condition.name
            / "libero_10_task00"
            / "attempt_01"
        )
        command = build_evaluator_command(
            python_bin=python_bin,
            assets=assets,
            condition=condition,
            task=Task("libero_10", 0),
            attempt_dir=sample_attempt,
            simulator_runtime_identity=simulator_runtime_identity,
        )
        print(f"[sample-command] CUDA_VISIBLE_DEVICES={gpu_devices[0]} {shlex.join(command)}")


def _parse_gpu_devices(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if (
        len(values) != NUM_GPUS
        or len(set(values)) != NUM_GPUS
        or any(not value.isdigit() for value in values)
    ):
        raise ValueError(
            f"Phase-A requires exactly {NUM_GPUS} distinct GPU devices, got {values}"
        )
    return values


def single_gpu_child_environment(
    parent_environment: Mapping[str, str], *, gpu_device: str
) -> dict[str, str]:
    """Return a standalone evaluator environment stripped of DDP topology."""

    assigned = str(gpu_device).strip()
    if not assigned.isdigit():
        raise ValueError("one child evaluator must receive exactly one GPU device token")
    environment = {
        key: value
        for key, value in parent_environment.items()
        if key not in _DISTRIBUTED_ENV_EXACT
        and not key.startswith(_DISTRIBUTED_ENV_PREFIXES)
    }
    # Set this only after sanitization so a parent allocation/list can never
    # survive.  Inside the child, the assigned physical device is cuda:0.
    environment["CUDA_VISIBLE_DEVICES"] = assigned
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment["MUJOCO_EGL_DEVICE_ID"] = assigned
    return environment


def _prepare_parent_simulator_environment(assets: FrozenAssets) -> None:
    canonical_root = str(Path(assets.libero_root).expanduser().resolve(strict=True))
    configured_root = os.environ.get("FASTWAM_LIBERO_ROOT")
    if configured_root is not None and str(
        Path(configured_root).expanduser().resolve(strict=True)
    ) != canonical_root:
        raise ValueError("FASTWAM_LIBERO_ROOT disagrees with frozen LIBERO root")
    os.environ["FASTWAM_LIBERO_ROOT"] = canonical_root
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        configured = os.environ.get(key)
        if configured is not None and configured.strip().lower() != "egl":
            raise ValueError(f"{key} must be egl for formal LIBERO evaluation")
        os.environ[key] = "egl"
    # The parent owns no renderer. Each standalone child gets its assigned
    # physical EGL device together with the matching CUDA_VISIBLE_DEVICES token.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python-bin", default="/root/.venvs/fastwam/bin/python")
    parser.add_argument("--gpu-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-delay-s", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_absolute():
        raise ValueError("run root must be absolute")
    if args.retry_delay_s < 0:
        raise ValueError("retry delay must be non-negative")
    gpu_devices = _parse_gpu_devices(args.gpu_devices)
    assets = FrozenAssets()
    python_bin = os.path.abspath(os.path.expanduser(args.python_bin))
    if Path(sys.executable).expanduser().resolve(strict=True) != Path(
        python_bin
    ).resolve(strict=True):
        raise ValueError(
            "Phase-A launcher must run under the same Python interpreter "
            "used by evaluator children"
        )
    _preflight_paths(assets, python_bin)
    _prepare_parent_simulator_environment(assets)
    simulator_runtime_identity = capture_libero_simulator_runtime_identity(
        assets.libero_root
    )
    conditions = resolve_conditions(assets)
    task_sources = build_task_source_ledger(assets)
    run_identity = read_run_identity()
    spec = _experiment_spec(
        assets=assets,
        conditions=conditions,
        run_identity=run_identity,
        task_sources=task_sources,
        simulator_runtime_identity=simulator_runtime_identity,
        python_bin=python_bin,
    )

    if args.dry_run:
        _print_dry_run(
            run_root=run_root,
            python_bin=python_bin,
            assets=assets,
            conditions=conditions,
            gpu_devices=gpu_devices,
            task_sources=task_sources,
            simulator_runtime_identity=simulator_runtime_identity,
        )
        if run_identity.tracked_dirty or run_identity.untracked_source_files:
            print(
                "[dry-run-warning] execution would reject the current dirty source tree: "
                f"{asdict(run_identity)}"
            )
        return 0

    require_clean_run_identity(run_identity)
    require_formal_sources_tracked()
    _preflight_frozen_identities(assets)
    verify_libero_simulator_runtime_identity(
        simulator_runtime_identity, libero_root=assets.libero_root
    )
    prepare_run_root(
        run_root=run_root,
        resume=args.resume,
        spec=spec,
        assets=assets,
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(
        run_root / "runtime" / "libero_config"
    )

    orchestrator = Orchestrator(
        run_root=run_root,
        python_bin=python_bin,
        gpu_devices=gpu_devices,
        assets=assets,
        conditions=conditions,
        run_identity=run_identity,
        task_sources=task_sources,
        simulator_runtime_identity=simulator_runtime_identity,
        retry_delay_s=args.retry_delay_s,
    )
    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(
            signum, lambda received, _frame, owner=orchestrator: owner.request_stop(received)
        )
    try:
        orchestrator.run()
    except StopRequested as error:
        print(f"[interrupted] {error}", file=sys.stderr, flush=True)
        return 128 + int(orchestrator.signal_number or signal.SIGTERM)
    except Exception as error:
        _atomic_json(
            run_root / "FAILED.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "libero_phase_a_failed",
                "failed_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    print(f"[complete] LIBERO Phase-A run: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
