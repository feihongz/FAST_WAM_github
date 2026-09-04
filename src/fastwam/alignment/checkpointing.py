"""Identity and integrity helpers for lightweight Stage 3 checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np
import torch


LEGACY_TRAINING_STATE_SCHEMA_VERSION = 1
TRAINING_STATE_SCHEMA_VERSION = 2
SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_TRAINING_STATE_SCHEMA_VERSION, TRAINING_STATE_SCHEMA_VERSION}
)
TRAINING_STATE_KIND = "stage3_alignment_training_state"
STRICT_RESUME_PROVENANCE_SCHEMA_VERSION = 1
STRICT_RESUME_PROVENANCE_KIND = "stage3_strict_resume_provenance"
STRICT_RESUME_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_manifest_sha256",
        "source_complete_sha256",
        "source_global_step",
        "source_epoch",
        "source_batch_in_epoch",
        "source_scheduler_last_epoch",
        "source_training_contract_sha256",
        "source_world_size",
        "source_zero_stage",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BaseCheckpointIdentity:
    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GitIdentity:
    commit: str
    tracked_dirty: bool
    untracked_source_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # ``dataclasses.asdict`` preserves tuples.  Contract payloads are
        # canonical JSON mappings, and the Stage 2 artifact schema requires
        # this collection to be a JSON array/list.
        payload["untracked_source_files"] = list(self.untracked_source_files)
        return payload


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_base_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
) -> BaseCheckpointIdentity:
    source = Path(path).expanduser().resolve()
    expected = str(expected_sha256).strip().lower()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError("expected base checkpoint SHA256 must contain 64 hex chars")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(
            "base checkpoint SHA256 mismatch: "
            f"expected={expected}, actual={actual}, path={source}"
        )
    return BaseCheckpointIdentity(
        path=str(source),
        sha256=actual,
        size_bytes=source.stat().st_size,
    )


def read_git_identity(repo_dir: str | Path) -> GitIdentity:
    root = Path(repo_dir).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "configs",
            "scripts",
            "experiments",
            "tests",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return GitIdentity(
        commit=commit,
        tracked_dirty=bool(status.strip()),
        untracked_source_files=tuple(sorted(untracked)),
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def write_text_atomic(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return output


def hash_state_tree(root: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(root)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in {"manifest.json", "COMPLETE"}:
            continue
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


def validate_rng_state_files(
    accelerator_dir: str | Path,
    *,
    world_size: int,
    require_cuda: bool = False,
    gradient_accumulation_steps: int = 1,
) -> None:
    directory = Path(accelerator_dir)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    paths = sorted(directory.glob("random_states_*.pkl"))
    expected_names = {f"random_states_{rank}.pkl" for rank in range(world_size)}
    if {path.name for path in paths} != expected_names:
        raise ValueError(
            "RNG state files do not match saved world size: "
            f"expected={sorted(expected_names)}, "
            f"actual={sorted(path.name for path in paths)}"
        )
    required = {
        "step",
        "random_state",
        "numpy_random_seed",
        "torch_manual_seed",
    }
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"invalid Accelerate RNG state: {path}")
        if not isinstance(payload["step"], int) or payload["step"] < 0:
            raise ValueError(f"invalid Accelerate RNG step: {path}")
        if payload["step"] % gradient_accumulation_steps != 0:
            raise ValueError(
                f"Accelerate state is inside an accumulation window: {path}"
            )
        try:
            python_probe = random.Random()
            python_probe.setstate(payload["random_state"])
            numpy_probe = np.random.RandomState()
            numpy_probe.set_state(payload["numpy_random_seed"])
            torch_probe = torch.Generator(device="cpu")
            torch_probe.set_state(payload["torch_manual_seed"])
            if require_cuda:
                cuda_states = payload.get("torch_cuda_manual_seed")
                if not isinstance(cuda_states, list) or not cuda_states:
                    raise ValueError("CUDA RNG state is missing")
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG device count does not match")
                for device_index, state in enumerate(cuda_states):
                    cuda_probe = torch.Generator(device=f"cuda:{device_index}")
                    cuda_probe.set_state(state)
        except Exception as error:
            raise ValueError(
                f"Accelerate RNG state is not loadable: {path}"
            ) from error


def validate_strict_resume_provenance(value: Any) -> dict[str, Any] | None:
    """Validate the immediate source of a successfully restored state.

    ``None`` denotes a fresh run.  A non-null value is path-independent and
    binds the saved state to the exact source manifest and COMPLETE marker.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != STRICT_RESUME_PROVENANCE_KEYS:
        raise ValueError("training state strict resume provenance schema is invalid")
    if (
        value.get("schema_version") != STRICT_RESUME_PROVENANCE_SCHEMA_VERSION
        or value.get("kind") != STRICT_RESUME_PROVENANCE_KIND
    ):
        raise ValueError("training state strict resume provenance header is invalid")
    for field in (
        "source_manifest_sha256",
        "source_complete_sha256",
        "source_training_contract_sha256",
    ):
        if not isinstance(value.get(field), str) or not _SHA256_PATTERN.fullmatch(
            value[field]
        ):
            raise ValueError(
                f"training state strict resume provenance {field} is invalid"
            )
    for field in (
        "source_global_step",
        "source_epoch",
        "source_batch_in_epoch",
        "source_scheduler_last_epoch",
        "source_zero_stage",
    ):
        field_value = value.get(field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 0
        ):
            raise ValueError(
                f"training state strict resume provenance {field} is invalid"
            )
    world_size = value.get("source_world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(
            "training state strict resume provenance source_world_size is invalid"
        )
    if value["source_zero_stage"] not in {0, 1, 2}:
        raise ValueError(
            "training state strict resume provenance source_zero_stage is invalid"
        )
    if value["source_scheduler_last_epoch"] != value["source_global_step"]:
        raise ValueError(
            "training state strict resume provenance scheduler/step mismatch"
        )
    return dict(value)


def validate_training_state(
    state_dir: str | Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(state_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"training state directory does not exist: {directory}")
    complete_path = directory / "COMPLETE"
    if not complete_path.is_file():
        raise ValueError(f"training state is incomplete: {directory}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"training state manifest is missing: {manifest_path}")
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("training state COMPLETE marker is invalid") from error
    complete_schema_version = (
        complete.get("schema_version") if isinstance(complete, dict) else None
    )
    if (
        not isinstance(complete, dict)
        or complete_schema_version not in SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS
        or complete.get("kind") != TRAINING_STATE_KIND
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("training state manifest SHA256 does not match COMPLETE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema_version = manifest.get("schema_version")
    if (
        manifest_schema_version != complete_schema_version
        or manifest_schema_version not in SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS
        or manifest.get("kind") != TRAINING_STATE_KIND
        or manifest.get("complete") is not True
    ):
        raise ValueError("unsupported or incomplete Stage 3 training state")
    if manifest_schema_version == TRAINING_STATE_SCHEMA_VERSION:
        if "strict_resume_provenance" not in manifest:
            raise ValueError("training state v2 is missing strict resume provenance")
        validate_strict_resume_provenance(manifest["strict_resume_provenance"])
    elif "strict_resume_provenance" in manifest:
        raise ValueError("training state v1 cannot contain strict resume provenance")
    for key, expected_value in expected_contract.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"training state contract mismatch for {key}: "
                f"expected={expected_value!r}, actual={manifest.get(key)!r}"
            )
    global_step = manifest.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("training state global_step must be non-negative")
    if directory.name != f"step_{global_step:06d}":
        raise ValueError("training state directory does not match global_step")
    for key in ("epoch", "batch_in_epoch"):
        if not isinstance(manifest.get(key), int) or manifest[key] < 0:
            raise ValueError(f"training state {key} must be non-negative")
    micro_batches_per_epoch = manifest.get("micro_batches_per_epoch")
    if (
        not isinstance(micro_batches_per_epoch, int)
        or micro_batches_per_epoch <= 0
    ):
        raise ValueError(
            "training state micro_batches_per_epoch must be positive"
        )
    if manifest["batch_in_epoch"] >= micro_batches_per_epoch:
        raise ValueError("training state batch cursor is outside its epoch")
    if manifest.get("micro_step_in_accumulation") != 0:
        raise ValueError("training state is not on an accumulation boundary")
    if manifest.get("scheduler_last_epoch") != global_step:
        raise ValueError("training state scheduler/global_step mismatch")
    training_contract = manifest.get("training_contract")
    training_contract_sha256 = manifest.get("training_contract_sha256")
    if (
        not isinstance(training_contract, Mapping)
        or not isinstance(training_contract_sha256, str)
        or not _SHA256_PATTERN.fullmatch(training_contract_sha256)
        or canonical_json_sha256(training_contract) != training_contract_sha256
    ):
        raise ValueError("training state training contract SHA256 does not match")
    zero_stage = manifest.get("zero_stage")
    if isinstance(zero_stage, bool) or not isinstance(zero_stage, int):
        raise ValueError("training state zero_stage must be an integer")
    if zero_stage not in {0, 1, 2}:
        raise ValueError("training state zero_stage must be 0, 1, or 2")
    effective_deepspeed_config = training_contract.get(
        "effective_deepspeed_config"
    )
    deepspeed_config_sha256 = manifest.get("deepspeed_config_sha256")
    if effective_deepspeed_config is None:
        if zero_stage != 0 or deepspeed_config_sha256 is not None:
            raise ValueError("non-DeepSpeed state has a DeepSpeed config identity")
    else:
        if (
            not isinstance(effective_deepspeed_config, Mapping)
            or not isinstance(deepspeed_config_sha256, str)
            or not _SHA256_PATTERN.fullmatch(deepspeed_config_sha256)
            or canonical_json_sha256(effective_deepspeed_config)
            != deepspeed_config_sha256
        ):
            raise ValueError("training state DeepSpeed config SHA256 does not match")
        zero_optimization = effective_deepspeed_config.get("zero_optimization")
        if (
            not isinstance(zero_optimization, Mapping)
            or zero_optimization.get("stage") != zero_stage
        ):
            raise ValueError("training state DeepSpeed zero stage does not match")
        batch_size = manifest.get("batch_size_per_rank")
        accumulation = manifest.get("gradient_accumulation_steps")
        world_size = manifest.get("world_size")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (batch_size, accumulation, world_size)
        ):
            raise ValueError("training state distributed batch contract is invalid")
        expected_deepspeed_batches = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": accumulation,
            "train_batch_size": batch_size * accumulation * world_size,
        }
        for key, expected_value in expected_deepspeed_batches.items():
            if effective_deepspeed_config.get(key) != expected_value:
                raise ValueError(
                    f"training state DeepSpeed batch contract mismatch for {key}"
                )
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("training state manifest has no file inventory")
    actual_files = hash_state_tree(directory)
    if actual_files != expected_files:
        raise ValueError("training state file inventory or SHA256 does not match")
    world_size = int(manifest.get("world_size", 0))
    if world_size <= 0:
        raise ValueError("training state world_size must be positive")
    validate_rng_state_files(
        directory / "accelerator",
        world_size=world_size,
        require_cuda=manifest.get("device_type") == "cuda",
        gradient_accumulation_steps=int(
            manifest.get("gradient_accumulation_steps", 1)
        ),
    )
    export_path = directory / "adapter_export.pt"
    if not export_path.is_file():
        raise ValueError("training state is missing its Adapter export")
    export = torch.load(export_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(export, dict)
        or export.get("kind") != "stage3_alignment_export"
        or export.get("global_step") != global_step
        or export.get("base_checkpoint_sha256")
        != manifest.get("base_checkpoint_sha256")
        or export.get("training_contract_sha256")
        != manifest.get("training_contract_sha256")
    ):
        raise ValueError("training state Adapter export metadata does not match")
    return manifest
