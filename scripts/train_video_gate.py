#!/usr/bin/env python3
"""Formal Hydra entrypoint for lightweight Stage 2 Video Gate training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import stat
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    read_git_identity,
    resolve_base_checkpoint,
    write_json_atomic,
)
from fastwam.gating.artifacts import (
    load_validated_merged_label_artifact,
    publish_json_atomic_no_clobber,
)
from fastwam.gating.contracts import require_sha256
from fastwam.gating.dataset import Stage2GateDataset
from fastwam.gating.selection import (
    SelectionArtifacts,
    load_selection_artifacts,
    selected_rows_for_coverage,
)
from fastwam.gating.runtime_identity import (
    collect_numerical_runtime_environment,
)
from fastwam.gating.source_guard import capture_selected_source_snapshot
from fastwam.gating.trainer import GateTrainer
from fastwam.models.video_gate import BinaryVideoGate
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed


register_default_resolvers()

_ROOT_KEYS = {
    "output_dir",
    "data_manifest",
    "episode_split",
    "label_contract",
    "label_manifest",
    "source_identities",
    "assets",
    "data",
    "gate",
    "training",
    "checkpoint",
    "runtime",
}
_SUBSET_ROOT_KEYS = _ROOT_KEYS | {"label_selection", "label_coverage"}
_TRAINING_CONFIG_SCHEMA_VERSION = 1
_TRAINING_CONFIG_KIND = "stage2_binary_video_gate_training_contract"
_WRITER_LOCK_FILE = ".stage2-gate-writer.lock"


def _resolved_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Stage 2 Gate training config must resolve to a mapping")
    root_keys = set(payload)
    has_selection = "label_selection" in root_keys
    has_coverage = "label_coverage" in root_keys
    if has_selection != has_coverage:
        raise ValueError(
            "label_selection and label_coverage must be provided together"
        )
    expected = _SUBSET_ROOT_KEYS if has_selection else _ROOT_KEYS
    if root_keys != expected:
        missing = sorted(expected - root_keys)
        unexpected = sorted(root_keys - expected)
        raise ValueError(
            "Stage 2 Gate training config fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return payload


def _exact_section(
    config: Mapping[str, Any],
    name: str,
    keys: set[str],
) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    if set(payload) != keys:
        missing = sorted(keys - set(payload))
        unexpected = sorted(set(payload) - keys)
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return payload


def _load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _repo_dir(runtime: Mapping[str, Any]) -> Path:
    configured = runtime.get("repo_dir")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _validated_git_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    identity = read_git_identity(_repo_dir(runtime))
    if bool(runtime["require_clean_git"]) and (
        identity.tracked_dirty or identity.untracked_source_files
    ):
        raise RuntimeError(
            "formal Stage 2 Gate training requires clean tracked files and no "
            "untracked source/config/test files"
        )
    return identity.as_dict()


def _canonicalize_data_paths(
    data_config: Mapping[str, Any],
    *,
    repo_dir: Path,
) -> dict[str, Any]:
    payload = dict(data_config)
    if set(payload) - {"train", "val"}:
        raise ValueError("Stage 2 Gate data config may contain only train and val")
    train_value = payload.get("train")
    if not isinstance(train_value, Mapping):
        raise TypeError("data.train must be a mapping")
    if payload.get("val") is not None:
        raise ValueError("Gate episode splits use one data.train; data.val must be null")
    train = dict(train_value)
    dataset_dirs = train.get("dataset_dirs")
    if not isinstance(dataset_dirs, list) or not dataset_dirs:
        raise ValueError("data.train.dataset_dirs must be a non-empty list")

    def resolve(value: Any) -> str:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_dir / candidate
        return str(candidate.resolve())

    train["dataset_dirs"] = [resolve(path) for path in dataset_dirs]
    for key in ("pretrained_norm_stats", "text_embedding_cache_dir"):
        if train.get(key):
            train[key] = resolve(train[key])
    payload["train"] = train
    payload["val"] = None
    return payload


def _basename(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."}:
        raise ValueError(f"{field} must be a non-empty local basename")
    if Path(value).name != value:
        raise ValueError(f"{field} must be a local basename")
    return value


def _validate_output_paths(
    paths: Mapping[str, Path],
) -> tuple[Path, ...]:
    """Reject aliases before any Gate artifact can be replaced."""

    ordered = tuple(paths.values())
    if len(set(ordered)) != len(ordered):
        duplicates = sorted(
            name
            for name, path in paths.items()
            if sum(candidate == path for candidate in ordered) > 1
        )
        raise ValueError(
            "Gate output artifact basenames must be pairwise distinct; "
            f"duplicates={duplicates}"
        )
    return ordered


def _is_same_file_or_path(first: Path, second: Path) -> bool:
    if first == second:
        return True
    if not os.path.lexists(first) or not os.path.lexists(second):
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _validate_resume_destination(
    *,
    output_dir: Path,
    output_paths: tuple[Path, ...],
    state_path: Path,
    resume_path: Path | None,
) -> None:
    """Ensure a resume cannot combine unrelated generations of outputs."""

    existing = tuple(path for path in output_paths if os.path.lexists(path))
    for target in existing:
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(
                f"refusing to use a non-regular Gate output: {target}"
            )
    if resume_path is None:
        if existing:
            raise RuntimeError(
                "refusing to overwrite an existing Gate run without "
                "checkpoint.resume"
            )
        return
    if _is_same_file_or_path(resume_path, state_path):
        return
    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise RuntimeError(
                f"Gate output_dir must be a regular directory: {output_dir}"
            )
        if any(
            entry.name != _WRITER_LOCK_FILE
            for entry in output_dir.iterdir()
        ):
            raise RuntimeError(
                "external Gate resume requires an empty output_dir; use the "
                "configured state_file to resume an existing run"
            )


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _require_single_process_environment() -> None:
    """Reject distributed launch metadata for the mutable Gate training run."""

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise ValueError(
            "WORLD_SIZE, RANK, and LOCAL_RANK must be integers"
        ) from error
    if world_size != 1 or rank != 0 or local_rank != 0:
        raise RuntimeError(
            "formal Stage 2 Gate training is single-process only; "
            "do not launch it with torchrun"
        )


@contextmanager
def _exclusive_output_writer(output_dir: Path) -> Iterator[Path]:
    """Hold a non-blocking advisory lock for every mutable run artifact."""

    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise RuntimeError(
                f"Gate output_dir must be a regular directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    lock_path = output_dir / _WRITER_LOCK_FILE
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot open Gate writer lock: {lock_path}") from error
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"Gate writer lock is not regular: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise RuntimeError(
                f"another Gate writer already owns output_dir: {output_dir}"
            ) from error
        yield lock_path
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _device(runtime: Mapping[str, Any]) -> torch.device:
    device = torch.device(str(runtime["device"]))
    if bool(runtime["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("formal Stage 2 Gate training requires a CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Gate CUDA device was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Gate CUDA device is out of range: {device}")
        if device.index is not None:
            torch.cuda.set_device(device)
    return device


def _validate_formal_label_dataset(
    dataset: Any,
    data_manifest: Mapping[str, Any],
    *,
    normalization_stats_path: str | Path,
    expected_data_manifest_sha256: str,
) -> dict[str, Any]:
    """Delay Wan22-adjacent runtime imports until formal data validation."""

    from fastwam.gating.runtime import (  # pylint: disable=import-outside-toplevel
        validate_stage2_label_dataset,
    )

    return validate_stage2_label_dataset(
        dataset,
        data_manifest,
        normalization_stats_path=normalization_stats_path,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
    )


def build_training_config_contract(
    *,
    data: Mapping[str, Any],
    gate: Mapping[str, Any],
    training: Mapping[str, Any],
    runtime: Mapping[str, Any],
    numerical_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the stable, resume-bound training semantics (no output paths)."""

    if not isinstance(numerical_runtime, Mapping):
        raise TypeError("numerical_runtime must be a mapping")
    payload = {
        "schema_version": _TRAINING_CONFIG_SCHEMA_VERSION,
        "kind": _TRAINING_CONFIG_KIND,
        "data": dict(data),
        "gate": dict(gate),
        "training": dict(training),
        "runtime": {
            "device": runtime["device"],
            "require_cuda": runtime["require_cuda"],
            "deterministic_algorithms": runtime["deterministic_algorithms"],
            "numerical_runtime": dict(numerical_runtime),
        },
        "dataloader_seed_algorithm": "base_seed_plus_zero_based_epoch_v1",
    }
    # Round-trip rejects tensors, NaN, and other non-portable metadata.
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return json.loads(encoded)


def _seed_data_worker(_worker_id: int) -> None:
    """Derive Python/NumPy worker RNGs from DataLoader's fixed torch seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _epoch_loaders(
    *,
    train_dataset: Stage2GateDataset,
    val_dataset: Stage2GateDataset,
    training: Mapping[str, Any],
    epoch_index: int,
) -> tuple[DataLoader, DataLoader]:
    """Build resume-stable loaders for one zero-based epoch."""

    base_seed = int(training["seed"])
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(base_seed + epoch_index)
    val_generator = torch.Generator(device="cpu")
    val_generator.manual_seed(base_seed + 1_000_000_000 + epoch_index)
    common = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "pin_memory": training["pin_memory"],
        "drop_last": False,
        "worker_init_fn": _seed_data_worker,
        "persistent_workers": False,
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=train_generator,
            **common,
        ),
        DataLoader(
            val_dataset,
            shuffle=False,
            generator=val_generator,
            **common,
        ),
    )


def build_training_identity(
    *,
    label_manifest_sha256: str,
    contract: Mapping[str, Any],
    training_config_sha256: str,
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact identity reused by trainer, resume, and both exports."""

    return {
        "label_manifest_sha256": require_sha256(
            label_manifest_sha256, field="label_manifest_sha256"
        ),
        "adapter_checkpoint_sha256": require_sha256(
            contract.get("adapter_checkpoint_sha256"),
            field="adapter_checkpoint_sha256",
        ),
        "base_checkpoint_sha256": require_sha256(
            contract.get("base_checkpoint_sha256"),
            field="base_checkpoint_sha256",
        ),
        "data_manifest_sha256": require_sha256(
            contract.get("data_manifest_sha256"),
            field="data_manifest_sha256",
        ),
        "episode_split_assignment_sha256": require_sha256(
            contract.get("episode_assignment_sha256"),
            field="episode_split_assignment_sha256",
        ),
        "training_config_sha256": require_sha256(
            training_config_sha256, field="training_config_sha256"
        ),
        "git_identity": dict(git_identity),
    }


def _load_prior_epoch_history(
    summary_path: Path,
    *,
    training_identity: Mapping[str, Any],
    resumed_epoch: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Load committed history, trimming a recoverable summary-ahead record."""

    if not os.path.lexists(summary_path):
        return [], resumed_epoch == 0
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError(f"Gate summary is not a regular file: {summary_path}")
    summary = _load_json_mapping(summary_path, label="existing Gate summary")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != "stage2_binary_video_gate_training_summary"
    ):
        raise ValueError("existing Gate summary schema is unsupported")
    if summary.get("training_identity") != dict(training_identity):
        raise ValueError("existing Gate summary training_identity mismatch")
    raw_history = summary.get("epoch_history", summary.get("new_epoch_history"))
    if not isinstance(raw_history, list):
        raise TypeError("existing Gate summary epoch history must be a list")
    history: list[dict[str, Any]] = []
    previous_epoch = 0
    for raw_record in raw_history:
        if not isinstance(raw_record, Mapping):
            raise TypeError("Gate epoch history records must be mappings")
        record = dict(raw_record)
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("Gate epoch history contains an invalid epoch")
        if epoch <= previous_epoch:
            raise ValueError("Gate epoch history must be strictly ordered")
        previous_epoch = epoch
        if epoch <= resumed_epoch:
            history.append(record)
    complete = [record["epoch"] for record in history] == list(
        range(1, resumed_epoch + 1)
    )
    if summary.get("history_complete") is False:
        complete = False
    return history, complete


def _training_summary(
    trainer: GateTrainer,
    *,
    training_identity: Mapping[str, Any],
    initial_epoch: int,
    prior_history: list[dict[str, Any]],
    new_history: list[dict[str, Any]],
    history_complete: bool,
    stopped_early: bool,
    state_path: Path,
    best_path: Path,
    last_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "stage2_binary_video_gate_training_summary",
        "training_identity": dict(training_identity),
        "initial_epoch": initial_epoch,
        "final_epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "stopped_early": stopped_early,
        "best_epoch": trainer.best_epoch,
        "best_val_bce": trainer.best_val_bce,
        "best_metrics": dict(trainer.best_metrics),
        "history_complete": history_complete,
        "epoch_history": [*prior_history, *new_history],
        "new_epoch_history": list(new_history),
        "state_file": state_path.name,
        "best_file": best_path.name,
        "last_file": last_path.name,
    }


def _load_identity_inputs(
    resolved: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_spec = _exact_section(
        resolved, "data_manifest", {"path", "expected_sha256"}
    )
    split_spec = _exact_section(
        resolved,
        "episode_split",
        {"path", "expected_assignment_sha256"},
    )
    contract_spec = _exact_section(
        resolved, "label_contract", {"path", "expected_sha256"}
    )
    data_manifest = _load_json_mapping(
        manifest_spec["path"], label="Stage 2 data manifest"
    )
    episode_split = _load_json_mapping(
        split_spec["path"], label="Stage 2 episode split"
    )
    contract = _load_json_mapping(
        contract_spec["path"], label="Stage 2 label contract"
    )
    expected_data_sha = require_sha256(
        manifest_spec["expected_sha256"],
        field="data manifest expected_sha256",
    )
    if data_manifest.get("manifest_sha256") != expected_data_sha:
        raise ValueError("Stage 2 data manifest SHA256 mismatch")
    expected_assignment = require_sha256(
        split_spec["expected_assignment_sha256"],
        field="episode split expected_assignment_sha256",
    )
    if episode_split.get("assignment_sha256") != expected_assignment:
        raise ValueError("Stage 2 episode split assignment SHA256 mismatch")
    expected_contract = require_sha256(
        contract_spec["expected_sha256"],
        field="label contract expected_sha256",
    )
    if contract.get("contract_sha256") != expected_contract:
        raise ValueError("Stage 2 label contract SHA256 mismatch")
    return data_manifest, episode_split, contract


def _load_optional_selection(
    resolved: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
) -> tuple[
    SelectionArtifacts | None,
    Mapping[str, Any] | None,
    tuple[str, ...] | None,
]:
    """Load the exact sparse-label coverage, or retain the legacy full path."""

    has_selection = "label_selection" in resolved
    has_coverage = "label_coverage" in resolved
    if has_selection != has_coverage:
        raise ValueError(
            "label_selection and label_coverage must be provided together"
        )
    if not has_selection:
        return None, None, None

    selection_spec = _exact_section(
        resolved, "label_selection", {"directory", "expected_sha256"}
    )
    coverage_spec = _exact_section(
        resolved, "label_coverage", {"tier", "expected_sha256"}
    )
    artifacts = load_selection_artifacts(
        selection_spec["directory"], data_manifest=data_manifest
    )
    expected_selection_sha = require_sha256(
        selection_spec["expected_sha256"],
        field="label selection expected_sha256",
    )
    if artifacts.descriptor["selection_sha256"] != expected_selection_sha:
        raise ValueError("Stage 2 label selection SHA256 mismatch")
    if dict(artifacts.episode_split) != dict(episode_split):
        raise ValueError("Stage 2 label selection episode split mismatch")

    tier = coverage_spec["tier"]
    if not isinstance(tier, str) or not tier:
        raise TypeError("label_coverage.tier must be a non-empty string")
    coverage = artifacts.coverages.get(tier)
    if coverage is None:
        raise ValueError(f"unknown Stage 2 label coverage tier: {tier}")
    expected_coverage_sha = require_sha256(
        coverage_spec["expected_sha256"],
        field="label coverage expected_sha256",
    )
    if coverage["coverage_sha256"] != expected_coverage_sha:
        raise ValueError("Stage 2 label coverage SHA256 mismatch")
    selected_rows = selected_rows_for_coverage(artifacts, tier=tier)
    expected_sample_ids = tuple(sorted(row["sample_id"] for row in selected_rows))
    if len(expected_sample_ids) != coverage["sample_count"]:
        raise ValueError("Stage 2 label coverage sample count mismatch")
    return artifacts, coverage, expected_sample_ids


def _run_train_video_gate_resolved(
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate labels/data, then train only the small BinaryVideoGate."""

    resolved = dict(resolved)
    runtime = _exact_section(
        resolved,
        "runtime",
        {
            "repo_dir",
            "require_clean_git",
            "device",
            "require_cuda",
            "deterministic_algorithms",
        },
    )
    git_identity = _validated_git_identity(runtime)
    resolved["data"] = _canonicalize_data_paths(
        resolved["data"], repo_dir=_repo_dir(runtime)
    )
    data_manifest, episode_split, contract = _load_identity_inputs(resolved)
    selection, coverage, expected_sample_ids = _load_optional_selection(
        resolved,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )

    sources = _exact_section(
        resolved,
        "source_identities",
        {"base_checkpoint_sha256", "adapter_checkpoint_sha256"},
    )
    for field in ("base_checkpoint_sha256", "adapter_checkpoint_sha256"):
        expected = require_sha256(sources[field], field=f"source_identities {field}")
        if contract.get(field) != expected:
            raise ValueError(f"label contract {field} mismatch")
    if contract.get("data_config_sha256") != canonical_json_sha256(
        resolved["data"]
    ):
        raise ValueError("Gate data config differs from label-generation config")

    label_manifest_spec = _exact_section(
        resolved, "label_manifest", {"path", "expected_sha256"}
    )
    merged_binding: dict[str, Any] = {}
    if selection is not None:
        assert coverage is not None
        merged_binding = {
            "selection_sha256": selection.descriptor["selection_sha256"],
            "coverage_sha256": coverage["coverage_sha256"],
            "active_cohort_indices": coverage["active_cohort_indices"],
        }
    merged = load_validated_merged_label_artifact(
        label_manifest_spec["path"],
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
        **merged_binding,
    )
    expected_label_manifest_sha = require_sha256(
        label_manifest_spec["expected_sha256"],
        field="label manifest expected_sha256",
    )
    if merged.manifest["manifest_sha256"] != expected_label_manifest_sha:
        raise ValueError("merged label manifest SHA256 mismatch")

    assets = _exact_section(resolved, "assets", {"normalization_stats"})
    stats_spec = assets["normalization_stats"]
    if not isinstance(stats_spec, Mapping) or set(stats_spec) != {
        "path",
        "expected_sha256",
    }:
        raise ValueError(
            "assets.normalization_stats must contain path and expected_sha256"
        )
    stats_identity = resolve_base_checkpoint(
        stats_spec["path"],
        expected_sha256=str(stats_spec["expected_sha256"]),
    )
    if contract.get("normalization_stats_sha256") != stats_identity.sha256:
        raise ValueError("normalization stats differ from the label contract")
    configured_stats = resolved["data"]["train"].get("pretrained_norm_stats")
    if not configured_stats or Path(configured_stats).resolve() != Path(
        stats_identity.path
    ):
        raise ValueError(
            "data.train.pretrained_norm_stats must equal the verified stats asset"
        )

    source_snapshot = capture_selected_source_snapshot(data_manifest)
    raw_dataset = instantiate(OmegaConf.create(resolved["data"]["train"]))
    _validate_formal_label_dataset(
        raw_dataset,
        data_manifest,
        normalization_stats_path=stats_identity.path,
        expected_data_manifest_sha256=data_manifest["manifest_sha256"],
    )
    source_snapshot.check_stats()
    gate_source = raw_dataset.current_only()
    source_snapshot.check_stats()
    del raw_dataset
    dataset_binding: dict[str, Any] = {}
    if expected_sample_ids is not None:
        dataset_binding["expected_sample_ids"] = expected_sample_ids
    train_dataset = Stage2GateDataset(
        gate_source,
        label_rows=merged.rows,
        data_manifest=data_manifest,
        episode_split=episode_split,
        split="train",
        **dataset_binding,
    )
    val_dataset = Stage2GateDataset(
        gate_source,
        label_rows=merged.rows,
        data_manifest=data_manifest,
        episode_split=episode_split,
        split="validation",
        **dataset_binding,
    )

    gate_config = _exact_section(
        resolved,
        "gate",
        {
            "proprio_dim",
            "context_dim",
            "cnn_channels",
            "context_feature_dim",
            "proprio_hidden_dim",
            "proprio_feature_dim",
            "fusion_hidden_dim",
        },
    )
    training = _exact_section(
        resolved,
        "training",
        {
            "seed",
            "batch_size",
            "num_workers",
            "pin_memory",
            "shuffle",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "num_epochs",
            "early_stop_patience",
            "min_delta",
            "threshold",
            "num_calibration_bins",
        },
    )
    _positive_int(training["batch_size"], field="training.batch_size")
    _positive_int(
        training["num_workers"],
        field="training.num_workers",
        allow_zero=True,
    )
    _positive_int(training["num_epochs"], field="training.num_epochs")
    _positive_int(
        training["early_stop_patience"],
        field="training.early_stop_patience",
        allow_zero=True,
    )
    _positive_int(
        training["num_calibration_bins"],
        field="training.num_calibration_bins",
    )
    if not isinstance(training["pin_memory"], bool):
        raise TypeError("training.pin_memory must be bool")
    if not isinstance(training["shuffle"], bool) or not training["shuffle"]:
        raise ValueError("formal Gate training requires training.shuffle=true")

    set_global_seed(int(training["seed"]))
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))
    device = _device(runtime)
    numerical_runtime = collect_numerical_runtime_environment(device)
    training_contract = build_training_config_contract(
        data=resolved["data"],
        gate=gate_config,
        training=training,
        runtime=runtime,
        numerical_runtime=numerical_runtime,
    )
    training_config_sha = canonical_json_sha256(training_contract)
    training_identity = build_training_identity(
        label_manifest_sha256=expected_label_manifest_sha,
        contract=contract,
        training_config_sha256=training_config_sha,
        git_identity=git_identity,
    )

    gate = BinaryVideoGate(**gate_config).to(device=device)
    train_labels = [
        row["label"] for row in merged.rows if row["split"] == "train"
    ]
    trainer = GateTrainer(
        gate,
        train_labels=train_labels,
        training_identity=training_identity,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
    )

    checkpoint = _exact_section(
        resolved,
        "checkpoint",
        {
            "strict_resume",
            "resume",
            "run_identity_file",
            "state_file",
            "best_file",
            "last_file",
            "summary_file",
        },
    )
    if checkpoint["strict_resume"] is not True:
        raise ValueError("formal Gate training requires checkpoint.strict_resume=true")
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    run_identity_path = output_dir / _basename(
        checkpoint["run_identity_file"], field="checkpoint.run_identity_file"
    )
    state_path = output_dir / _basename(
        checkpoint["state_file"], field="checkpoint.state_file"
    )
    best_path = output_dir / _basename(
        checkpoint["best_file"], field="checkpoint.best_file"
    )
    last_path = output_dir / _basename(
        checkpoint["last_file"], field="checkpoint.last_file"
    )
    summary_path = output_dir / _basename(
        checkpoint["summary_file"], field="checkpoint.summary_file"
    )
    output_paths = _validate_output_paths(
        {
            "run_identity_file": run_identity_path,
            "state_file": state_path,
            "best_file": best_path,
            "last_file": last_path,
            "summary_file": summary_path,
        }
    )
    resume = checkpoint["resume"]
    initial_epoch = 0
    run_identity = {
        "schema_version": 1,
        "kind": "stage2_binary_video_gate_run_identity",
        "training_config": training_contract,
        "training_config_sha256": training_config_sha,
        "training_identity": training_identity,
    }
    if canonical_json_sha256(training_contract) != training_config_sha:
        raise RuntimeError("Gate training config SHA256 drifted before publication")
    resume_path: Path | None = None
    if resume not in (None, ""):
        resume_path = Path(str(resume)).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Gate resume state does not exist: {resume_path}")
    _validate_resume_destination(
        output_dir=output_dir,
        output_paths=output_paths,
        state_path=state_path,
        resume_path=resume_path,
    )
    if resume_path is not None:
        trainer.load_training_state(resume_path)
        initial_epoch = trainer.epoch
    prior_history, history_complete = _load_prior_epoch_history(
        summary_path,
        training_identity=training_identity,
        resumed_epoch=initial_epoch,
    )
    target_epochs = int(training["num_epochs"])
    if trainer.epoch > target_epochs:
        raise ValueError("Gate resume epoch exceeds training.num_epochs")
    source_snapshot.check_stats()
    published_identity = publish_json_atomic_no_clobber(
        run_identity_path, run_identity
    )
    if not published_identity:
        existing_identity = _load_json_mapping(
            run_identity_path, label="existing Gate run identity"
        )
        if existing_identity != run_identity:
            raise RuntimeError("Gate run identity differs from this launch")

    # Materialize imported state at the canonical destination even when no new
    # epoch runs. It also leaves a resumable epoch-zero state for a fresh run.
    source_snapshot.check_stats()
    trainer.save_training_state(state_path)

    history: list[dict[str, Any]] = []
    stopped_early = False
    patience = int(training["early_stop_patience"])
    already_stopped = (
        trainer.epochs_without_improvement > 0
        and trainer.epochs_without_improvement >= patience
    )
    for _ in range(0 if already_stopped else target_epochs - trainer.epoch):
        source_snapshot.check_stats()
        train_loader, val_loader = _epoch_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training=training,
            epoch_index=trainer.epoch,
        )
        fit = trainer.fit(
            train_loader,
            val_loader,
            num_epochs=1,
            early_stop_patience=patience,
            min_delta=float(training["min_delta"]),
            threshold=float(training["threshold"]),
            num_calibration_bins=int(training["num_calibration_bins"]),
        )
        source_snapshot.check_stats()
        if len(fit.epochs) != 1 or fit.epochs[0].get("epoch") != trainer.epoch:
            raise RuntimeError("GateTrainer returned an inconsistent epoch record")
        history.extend(dict(record) for record in fit.epochs)
        progress = _training_summary(
            trainer,
            training_identity=training_identity,
            initial_epoch=initial_epoch,
            prior_history=prior_history,
            new_history=history,
            history_complete=history_complete,
            stopped_early=stopped_early or fit.stopped_early,
            state_path=state_path,
            best_path=best_path,
            last_path=last_path,
        )
        # Publishing history first makes an interrupted summary at most one
        # epoch ahead of state; resume trims that record and repeats the epoch.
        source_snapshot.check_stats()
        write_json_atomic(summary_path, progress)
        source_snapshot.check_stats()
        trainer.save_training_state(state_path)
        if fit.stopped_early:
            stopped_early = True
            break

    # Reuse the same validated mapping; GateTrainer rejects any export drift.
    export_kwargs = {
        "label_manifest_sha256": training_identity["label_manifest_sha256"],
        "adapter_checkpoint_sha256": training_identity[
            "adapter_checkpoint_sha256"
        ],
        "data_manifest_sha256": training_identity["data_manifest_sha256"],
        "episode_split_assignment_sha256": training_identity[
            "episode_split_assignment_sha256"
        ],
        "training_config_sha256": training_identity["training_config_sha256"],
        "git_identity": training_identity["git_identity"],
    }
    # One final content pass gates the durable final state and both exports.
    source_snapshot.check_content()
    trainer.save_training_state(state_path)
    source_snapshot.check_stats()
    trainer.export_checkpoint(best_path, selection="best", **export_kwargs)
    source_snapshot.check_stats()
    trainer.export_checkpoint(last_path, selection="last", **export_kwargs)
    summary = _training_summary(
        trainer,
        training_identity=training_identity,
        initial_epoch=initial_epoch,
        prior_history=prior_history,
        new_history=history,
        history_complete=history_complete,
        stopped_early=stopped_early or already_stopped,
        state_path=state_path,
        best_path=best_path,
        last_path=last_path,
    )
    source_snapshot.check_stats()
    write_json_atomic(summary_path, summary)
    return summary


def run_train_video_gate(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Run one exclusive, single-process formal Gate training writer."""

    resolved = _resolved_config(config)
    _require_single_process_environment()
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    with _exclusive_output_writer(output_dir):
        return _run_train_video_gate_resolved(resolved)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_video_gate",
)
def main(config: DictConfig) -> None:
    summary = run_train_video_gate(config)
    print(
        "Stage 2 Video Gate training complete:\n"
        f"  epoch: {summary['final_epoch']}\n"
        f"  global_step: {summary['global_step']}\n"
        f"  best_val_bce: {summary['best_val_bce']}"
    )


if __name__ == "__main__":
    main()
