#!/usr/bin/env python3
"""Formal Hydra entrypoint for offline Stage 2 Gate-label generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    read_git_identity,
    resolve_base_checkpoint,
    sha256_file,
)
from fastwam.gating.artifacts import (
    build_label_artifact_context,
    build_label_contract,
    publish_json_atomic_no_clobber,
)
from fastwam.gating.contracts import build_episode_split, require_sha256
from fastwam.gating.label_job import run_label_job
from fastwam.gating.runtime import (
    inspect_alignment_export,
    load_stage2_label_model,
    validate_stage2_label_dataset,
)
from fastwam.gating.runtime_identity import (
    collect_numerical_runtime_environment,
)
from fastwam.gating.source_guard import (
    capture_selected_source_snapshot,
    make_source_stat_guard,
)
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed


register_default_resolvers()

_ROOT_KEYS = {
    "output_dir",
    "base",
    "adapter",
    "assets",
    "data_manifest",
    "episode_split",
    "data",
    "model",
    "labeling",
    "runtime",
}


def _resolved_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Stage 2 label config must resolve to a mapping")
    if set(payload) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - set(payload))
        unexpected = sorted(set(payload) - _ROOT_KEYS)
        raise ValueError(
            "Stage 2 label config fields do not match schema; "
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
            "formal Stage 2 labels require clean tracked files and no "
            "untracked source/config/test files"
        )
    return identity.as_dict()


def _validate_environment(runtime: Mapping[str, Any]) -> None:
    required = runtime["required_environment"]
    if not isinstance(required, Mapping):
        raise TypeError("runtime.required_environment must be a mapping")
    mismatches = {
        str(name): {"expected": str(expected), "actual": os.environ.get(str(name))}
        for name, expected in required.items()
        if os.environ.get(str(name)) != str(expected)
    }
    if mismatches:
        raise RuntimeError(f"Stage 2 required environment mismatch: {mismatches}")


def _canonicalize_data_paths(
    data_config: Mapping[str, Any],
    *,
    repo_dir: Path,
) -> dict[str, Any]:
    payload = dict(data_config)
    if set(payload) - {"train", "val"}:
        raise ValueError("Stage 2 data config may contain only train and val")
    train_value = payload.get("train")
    if not isinstance(train_value, Mapping):
        raise TypeError("data.train must be a mapping")
    if payload.get("val") is not None:
        raise ValueError("Stage 2 labels use data.train only; data.val must be null")
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


def _model_dtype(value: Any) -> torch.dtype:
    normalized = str(value).strip().lower()
    if normalized in {"fp32", "float32", "no"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError("runtime.mixed_precision must be fp32, fp16, or bf16")


def collect_label_runtime_environment(
    device: Any,
    *,
    package_version_resolver: Any = None,
    torch_runtime: Any = None,
) -> dict[str, Any]:
    """Collect numerical software/hardware identity for label compatibility."""

    return collect_numerical_runtime_environment(
        device,
        package_version_resolver=package_version_resolver,
        torch_runtime=torch_runtime,
    )


def build_label_runtime_config(
    *,
    model: Mapping[str, Any],
    mixed_precision: Any,
    device: Any = None,
    runtime_environment: Mapping[str, Any] | None = None,
    required_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical numerical model/runtime payload bound into every label row."""

    normalized_precision = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }[_model_dtype(mixed_precision)]
    if runtime_environment is None:
        if device is None:
            raise ValueError("device is required for the label runtime contract")
        runtime_environment = collect_label_runtime_environment(device)
    if not isinstance(runtime_environment, Mapping):
        raise TypeError("runtime_environment must be a mapping")
    if required_environment is None:
        required_environment = {}
    if not isinstance(required_environment, Mapping):
        raise TypeError("required_environment must be a mapping")
    payload = {
        "schema_version": 1,
        "kind": "stage2_label_runtime_config",
        "model": dict(model),
        "mixed_precision": normalized_precision,
        "numerical_runtime": dict(runtime_environment),
        "required_environment": dict(required_environment),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 2 label runtime config is not canonical JSON") from error
    return json.loads(encoded)


def _device(runtime: Mapping[str, Any]) -> str:
    value = str(runtime["device"])
    device = torch.device(value)
    if bool(runtime["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("formal Stage 2 label generation requires a CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 2 CUDA device was requested but is unavailable")
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError("WORLD_SIZE must be an integer") from error
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        local_rank = os.environ.get("LOCAL_RANK")
        if world_size > 1 and device.index is not None:
            raise ValueError(
                "torchrun Stage 2 label generation requires runtime.device=cuda; "
                "LOCAL_RANK selects each process GPU"
            )
        if world_size > 1 and local_rank is None:
            raise ValueError(
                "torchrun Stage 2 label generation requires LOCAL_RANK"
            )
        if device.index is None and local_rank is not None:
            try:
                device = torch.device("cuda", int(local_rank))
            except (TypeError, ValueError) as error:
                raise ValueError("LOCAL_RANK must be an integer") from error
            value = str(device)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Stage 2 CUDA device is out of range: {device}")
        if device.index is not None:
            torch.cuda.set_device(device)
    return value


def _shard_indices(value: Any, *, num_shards: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("labeling.shard_indices must be a sequence or null")
    indices = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise TypeError("labeling.shard_indices must contain integers")
    if list(indices) != sorted(set(indices)):
        raise ValueError("labeling.shard_indices must be sorted and unique")
    if any(item < 0 or item >= num_shards for item in indices):
        raise ValueError("labeling.shard_indices contains an out-of-range shard")
    if not indices:
        raise ValueError("labeling.shard_indices must not be empty")
    return indices


def _rank_shard_indices(
    value: Any,
    *,
    num_shards: int,
) -> tuple[int, ...] | None:
    """Partition all shards by torchrun rank when no subset is explicit."""

    explicit = _shard_indices(value, num_shards=num_shards)
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE and RANK must be integers") from error
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("WORLD_SIZE/RANK do not describe a valid process group")
    if explicit is not None and world_size > 1:
        raise ValueError(
            "explicit labeling.shard_indices cannot be combined with torchrun; "
            "use null for rank partitioning or launch independent single-process jobs"
        )
    if explicit is not None or world_size == 1:
        return explicit
    if world_size > num_shards:
        raise ValueError("WORLD_SIZE cannot exceed labeling.num_shards")
    assigned = tuple(range(rank, num_shards, world_size))
    if not assigned:
        raise RuntimeError("torchrun rank received no Stage 2 label shards")
    return assigned


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    published = publish_json_atomic_no_clobber(path, payload)
    if not published:
        existing = _load_json_mapping(path, label=f"existing {label}")
        if existing != dict(payload):
            raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    return path


def _require_asset_unchanged(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    """Close the inspect-to-load window for an already resolved model asset."""

    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} changed while the Stage 2 model was loading")


def _instantiate_label_dataset_under_source_guard(
    train_config: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Bracket LeRobot construction so it cannot retain unbound source bytes."""

    source_snapshot = capture_selected_source_snapshot(data_manifest)
    dataset = instantiate(OmegaConf.create(train_config))
    source_snapshot.check_stats()
    return dataset, source_snapshot


def run_generate_gate_labels(
    config: DictConfig | Mapping[str, Any],
) -> Any:
    """Construct and validate every dependency before running paired rollouts."""

    resolved = _resolved_config(config)
    runtime = _exact_section(
        resolved,
        "runtime",
        {
            "repo_dir",
            "require_clean_git",
            "require_cuda",
            "device",
            "mixed_precision",
            "required_environment",
        },
    )
    _validate_environment(runtime)
    git_identity = _validated_git_identity(runtime)
    repo_dir = _repo_dir(runtime)
    resolved["data"] = _canonicalize_data_paths(
        resolved["data"], repo_dir=repo_dir
    )

    manifest_spec = _exact_section(
        resolved, "data_manifest", {"path", "expected_sha256"}
    )
    split_spec = _exact_section(
        resolved,
        "episode_split",
        {
            "path",
            "validation_fraction",
            "split_seed",
            "expected_assignment_sha256",
        },
    )
    data_manifest = _load_json_mapping(
        manifest_spec["path"], label="Stage 2 data manifest"
    )
    expected_data_sha = require_sha256(
        manifest_spec["expected_sha256"],
        field="data manifest expected_sha256",
    )
    if data_manifest.get("manifest_sha256") != expected_data_sha:
        raise ValueError("Stage 2 data manifest SHA256 mismatch")
    episode_split = build_episode_split(
        data_manifest,
        validation_fraction=split_spec["validation_fraction"],
        split_seed=split_spec["split_seed"],
    )
    expected_assignment = split_spec["expected_assignment_sha256"]
    if expected_assignment not in (None, ""):
        expected_assignment = require_sha256(
            expected_assignment,
            field="episode split expected_assignment_sha256",
        )
        if episode_split["assignment_sha256"] != expected_assignment:
            raise ValueError("Stage 2 episode split assignment SHA256 mismatch")
    _write_immutable_json(
        Path(str(split_spec["path"])).expanduser().resolve(),
        episode_split,
        label="Stage 2 episode split",
    )

    base = _exact_section(resolved, "base", {"checkpoint", "expected_sha256"})
    adapter = _exact_section(
        resolved, "adapter", {"checkpoint", "expected_sha256"}
    )
    assets = _exact_section(resolved, "assets", {"vae", "normalization_stats"})
    vae_spec = assets["vae"]
    stats_spec = assets["normalization_stats"]
    for asset_name, asset_spec in (("vae", vae_spec), ("normalization_stats", stats_spec)):
        if not isinstance(asset_spec, Mapping) or set(asset_spec) != {
            "path",
            "expected_sha256",
        }:
            raise ValueError(
                f"assets.{asset_name} must contain path and expected_sha256"
            )
    base_identity = resolve_base_checkpoint(
        base["checkpoint"], expected_sha256=str(base["expected_sha256"])
    )
    expected_adapter_sha = require_sha256(
        adapter["expected_sha256"], field="adapter expected_sha256"
    )
    adapter_identity = inspect_alignment_export(
        adapter["checkpoint"],
        expected_sha256=expected_adapter_sha,
        expected_base_checkpoint_sha256=base_identity.sha256,
        expected_data_manifest_sha256=expected_data_sha,
    )
    vae_identity = resolve_base_checkpoint(
        vae_spec["path"], expected_sha256=str(vae_spec["expected_sha256"])
    )
    exported_assets = adapter_identity["export_metadata"].get("asset_identities")
    if not isinstance(exported_assets, Mapping):
        raise ValueError("alignment export has no asset identity mapping")
    exported_vae = exported_assets.get("vae")
    if not isinstance(exported_vae, Mapping):
        raise ValueError("alignment export has no VAE identity")
    if exported_vae.get("sha256") != vae_identity.sha256:
        raise ValueError("alignment export VAE SHA256 differs from Stage 2 config")
    exported_vae_path = exported_vae.get("path")
    if not exported_vae_path or Path(str(exported_vae_path)).expanduser().resolve() != Path(
        vae_identity.path
    ):
        raise ValueError("alignment export VAE path differs from Stage 2 config")
    stats_identity = resolve_base_checkpoint(
        stats_spec["path"],
        expected_sha256=str(stats_spec["expected_sha256"]),
    )
    configured_stats = resolved["data"]["train"].get("pretrained_norm_stats")
    if not configured_stats or Path(configured_stats).resolve() != Path(
        stats_identity.path
    ):
        raise ValueError(
            "data.train.pretrained_norm_stats must equal the verified stats asset"
        )

    # LeRobot construction can open parquet and Arrow caches even though video
    # decoding remains lazy. Snapshot first, then close the construction window.
    dataset, source_snapshot = _instantiate_label_dataset_under_source_guard(
        resolved["data"]["train"],
        data_manifest,
    )
    validate_stage2_label_dataset(
        dataset,
        data_manifest,
        normalization_stats_path=stats_identity.path,
        expected_data_manifest_sha256=expected_data_sha,
    )
    source_snapshot.check_stats()
    source_guard = make_source_stat_guard(source_snapshot)

    labeling = _exact_section(
        resolved,
        "labeling",
        {
            "base_seed",
            "num_seed_pairs",
            "relative_margin",
            "relative_gain_epsilon",
            "num_inference_steps",
            "sigma_shift",
            "rand_device",
            "tiled",
            "num_shards",
            "chunk_size",
            "shard_indices",
            "contract_file",
            "runtime_config_file",
        },
    )
    num_shards = labeling["num_shards"]
    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise TypeError("labeling.num_shards must be an integer")
    shard_indices = _rank_shard_indices(
        labeling["shard_indices"], num_shards=num_shards
    )
    chunk_size = labeling["chunk_size"]
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("labeling.chunk_size must be a positive integer")
    set_global_seed(int(labeling["base_seed"]))
    device = _device(runtime)
    if not isinstance(resolved["model"], Mapping):
        raise TypeError("model must be a mapping")
    model = instantiate(
        OmegaConf.create(resolved["model"]),
        model_dtype=_model_dtype(runtime["mixed_precision"]),
        device=device,
    )
    model_vae_path = getattr(model, "model_paths", {}).get("vae")
    if not model_vae_path or Path(str(model_vae_path)).expanduser().resolve() != Path(
        vae_identity.path
    ):
        raise RuntimeError("the Stage 2 model did not load the contract-bound VAE")
    _require_asset_unchanged(
        vae_identity.path,
        expected_sha256=vae_identity.sha256,
        label="the contract-bound VAE",
    )
    model_identity = load_stage2_label_model(
        model,
        base_checkpoint_path=base_identity.path,
        expected_base_checkpoint_sha256=base_identity.sha256,
        alignment_export_path=adapter_identity["path"],
        expected_alignment_export_sha256=adapter_identity["sha256"],
        expected_data_manifest_sha256=expected_data_sha,
    )
    if model_identity["base_checkpoint"]["sha256"] != base_identity.sha256:
        raise RuntimeError("Stage 2 loaded base identity drifted")
    if model_identity["alignment_export"]["sha256"] != expected_adapter_sha:
        raise RuntimeError("Stage 2 loaded Adapter identity drifted")

    label_runtime_config = build_label_runtime_config(
        model=resolved["model"],
        mixed_precision=runtime["mixed_precision"],
        device=device,
        required_environment=runtime["required_environment"],
    )

    contract = build_label_contract(
        data_manifest=data_manifest,
        episode_split=episode_split,
        base_checkpoint_sha256=base_identity.sha256,
        adapter_checkpoint_sha256=expected_adapter_sha,
        normalization_stats_sha256=stats_identity.sha256,
        data_config_sha256=canonical_json_sha256(resolved["data"]),
        vae_sha256=vae_identity.sha256,
        label_runtime_config_sha256=canonical_json_sha256(label_runtime_config),
        git_identity=git_identity,
        base_seed=labeling["base_seed"],
        num_seed_pairs=labeling["num_seed_pairs"],
        relative_margin=labeling["relative_margin"],
        relative_gain_epsilon=labeling["relative_gain_epsilon"],
        num_inference_steps=labeling["num_inference_steps"],
        sigma_shift=labeling["sigma_shift"],
        rand_device=labeling["rand_device"],
        tiled=labeling["tiled"],
        num_shards=num_shards,
        chunk_size=chunk_size,
    )
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    for field in ("contract_file", "runtime_config_file"):
        filename = labeling[field]
        if (
            not isinstance(filename, str)
            or filename in {"", ".", ".."}
            or Path(filename).name != filename
        ):
            raise ValueError(f"labeling.{field} must be a local basename")
    if labeling["contract_file"] == labeling["runtime_config_file"]:
        raise ValueError("labeling artifact filenames must differ")
    _write_immutable_json(
        output_dir / labeling["runtime_config_file"],
        label_runtime_config,
        label="Stage 2 label runtime config",
    )
    _write_immutable_json(
        output_dir / labeling["contract_file"],
        contract,
        label="Stage 2 label contract",
    )

    result = run_label_job(
        model,
        dataset,
        context=context,
        output_dir=output_dir,
        chunk_size=chunk_size,
        shard_indices=shard_indices,
        dependencies=None,
        source_guard=source_guard,
    )
    source_snapshot.check_content()
    return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_gate_labels",
)
def main(config: DictConfig) -> None:
    result = run_generate_gate_labels(config)
    print(f"Stage 2 Gate label job complete: {result}")


if __name__ == "__main__":
    main()
