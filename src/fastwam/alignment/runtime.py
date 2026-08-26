"""Public runtime for the independent Stage 3 alignment training chain."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    build_datasets,
)
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed

from .checkpointing import (
    BaseCheckpointIdentity,
    GitIdentity,
    canonical_json_sha256,
    read_git_identity,
    resolve_base_checkpoint,
    write_json_atomic,
    write_text_atomic,
)
from .data_identity import validate_data_manifest
from .formal_trainer import Stage3AlignmentTrainer
from .text_cache_binding import bind_validated_text_cache_integrity


logger = get_logger(__name__)


def _repo_dir(runtime_config: dict[str, Any]) -> Path:
    configured = runtime_config.get("repo_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _canonicalize_data_paths(
    data_config: dict[str, Any],
    *,
    repo_dir: Path,
) -> dict[str, Any]:
    payload = dict(data_config)
    train = dict(payload["train"])
    dataset_dirs = train.get("dataset_dirs")
    if not isinstance(dataset_dirs, list) or not dataset_dirs:
        raise ValueError("data.train.dataset_dirs must be a non-empty list")

    def resolve(path: str) -> str:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = repo_dir / candidate
        return str(candidate.resolve())

    train["dataset_dirs"] = [resolve(str(path)) for path in dataset_dirs]
    for key in ("pretrained_norm_stats", "text_embedding_cache_dir"):
        if train.get(key):
            train[key] = resolve(str(train[key]))
    payload["train"] = train
    return payload


def _validate_formal_dataset(
    train_dataset,
    *,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    if getattr(train_dataset, "strict_data_mode", False) is not True:
        raise RuntimeError("formal Stage 3 requires strict_data_mode=true")
    if getattr(train_dataset, "skip_padding_as_possible", None) is not False:
        raise RuntimeError(
            "formal Stage 3 requires skip_padding_as_possible=false"
        )
    base = getattr(train_dataset, "lerobot_dataset", None)
    if base is None or getattr(base, "strict_data_mode", False) is not True:
        raise RuntimeError("formal Stage 3 dataset did not propagate strict mode")
    multi = getattr(base, "multi_dataset", None)
    datasets = getattr(multi, "_datasets", None)
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError("formal Stage 3 requires a non-empty LeRobot dataset")
    for dataset in datasets:
        if getattr(dataset, "video_backend", None) != "torchcodec":
            raise RuntimeError("formal Stage 3 requires video_backend=torchcodec")
        if getattr(dataset, "allow_video_backend_fallback", True):
            raise RuntimeError("formal Stage 3 forbids video decoder fallback")
    expected_length = int(runtime_config["expected_dataset_length"])
    expected_episodes = int(runtime_config["expected_dataset_episodes"])
    actual_length = len(train_dataset)
    actual_episodes = int(getattr(multi, "num_episodes"))
    if actual_length != expected_length or actual_episodes != expected_episodes:
        raise RuntimeError(
            "Stage 3 dataset cardinality mismatch: "
            f"expected={expected_length} frames/{expected_episodes} episodes, "
            f"actual={actual_length} frames/{actual_episodes} episodes"
        )
    return {
        "dataset_length": actual_length,
        "dataset_episodes": actual_episodes,
        "dataset_roots": [str(Path(dataset.root).resolve()) for dataset in datasets],
        "video_backend": "torchcodec",
        "strict_data_mode": True,
        "skip_padding_as_possible": False,
    }


def _resolved_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Stage 3 root config must resolve to a mapping")
    required = {
        "output_dir",
        "base",
        "assets",
        "data_manifest",
        "model",
        "data",
        "stage3",
        "training",
        "checkpoint",
        "runtime",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Stage 3 config is missing sections: {sorted(missing)}")
    return payload


def _activate_accelerator_cuda_device(
    device: torch.device | str,
    *,
    num_processes: int = 1,
    local_process_index: int = 0,
) -> torch.device:
    """Select Accelerate's CUDA device, including its single-GPU ``cuda`` form."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("accelerator device must be CUDA")
    device_index = resolved.index
    if num_processes > 1:
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("distributed CUDA runtime has no visible devices")
        expected_index = local_process_index % device_count
        if device_index != expected_index:
            raise RuntimeError(
                "Accelerate CUDA device does not match the rank-local device: "
                f"device={resolved}, local_process_index={local_process_index}, "
                f"visible_device_count={device_count}"
            )
    elif device_index is None:
        device_index = int(torch.cuda.current_device())
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)


def _all_rank_value(
    accelerator: Accelerator,
    operation: Callable[[], dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    """Run identity work on every rank and require byte-identical results."""

    try:
        local = {"ok": True, "value": operation()}
    except Exception as error:
        local = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    gathered = gather_object([local])
    if not isinstance(gathered, list) or len(gathered) != accelerator.num_processes:
        raise RuntimeError(f"Stage 3 {label} all-rank gather returned invalid data")
    failures = [
        item
        for item in gathered
        if not isinstance(item, dict) or item.get("ok") is not True
    ]
    if failures:
        failure = failures[0]
        if isinstance(failure, dict):
            error_type = failure.get("error_type", "UnknownError")
            message = failure.get("message", "missing error message")
        else:
            error_type = type(failure).__name__
            message = "invalid all-rank status payload"
        raise RuntimeError(
            f"Stage 3 {label} identity failed on at least one rank: "
            f"{error_type}: {message}"
        )
    value = gathered[0].get("value")
    if not isinstance(value, dict):
        raise RuntimeError(f"Stage 3 {label} identity is invalid")
    if any(item.get("value") != value for item in gathered[1:]):
        raise RuntimeError(f"Stage 3 {label} identity differs across ranks")
    return value


def _run_all_rank_phase(
    accelerator: Accelerator,
    operation: Callable[[], Any],
    *,
    label: str,
) -> Any:
    """Run local work, then make every rank observe any Python exception."""

    value: Any = None
    try:
        value = operation()
        local = {"ok": True}
    except Exception as error:
        local = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    gathered = gather_object([local])
    if not isinstance(gathered, list) or len(gathered) != accelerator.num_processes:
        raise RuntimeError(f"Stage 3 {label} status gather returned invalid data")
    failures = [
        item
        for item in gathered
        if not isinstance(item, dict) or item.get("ok") is not True
    ]
    if failures:
        failure = failures[0]
        if isinstance(failure, dict):
            error_type = failure.get("error_type", "UnknownError")
            message = failure.get("message", "missing error message")
        else:
            error_type = type(failure).__name__
            message = "invalid all-rank status payload"
        raise RuntimeError(
            f"Stage 3 {label} failed on at least one rank: "
            f"{error_type}: {message}"
        )
    return value


def _main_rank_value(
    accelerator: Accelerator,
    operation: Callable[[], dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    """Run expensive identity work only on rank zero and broadcast its result."""

    status: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            status[0] = {"ok": True, "value": operation()}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    broadcast_object_list(status, from_process=0)

    def require_valid_broadcast() -> dict[str, Any]:
        result = status[0]
        if not isinstance(result, dict) or result.get("ok") is not True:
            if isinstance(result, dict):
                error_type = result.get("error_type", "UnknownError")
                message = result.get("message", "missing error message")
            else:
                error_type = type(result).__name__
                message = "invalid main-process status payload"
            raise RuntimeError(
                f"Stage 3 {label} failed on main process: "
                f"{error_type}: {message}"
            )
        value = result.get("value")
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Stage 3 {label} main-process value is invalid"
            )
        return value

    return _run_all_rank_phase(
        accelerator,
        require_valid_broadcast,
        label=f"{label} broadcast",
    )


def _resolve_base_identity(
    accelerator: Accelerator,
    base_config: dict[str, Any],
) -> BaseCheckpointIdentity:
    checkpoint = base_config.get("checkpoint")
    expected_sha256 = base_config.get("expected_sha256")
    if not checkpoint or not expected_sha256:
        raise ValueError(
            "base.checkpoint and base.expected_sha256 are mandatory for Stage 3"
        )
    payload = _all_rank_value(
        accelerator,
        lambda: resolve_base_checkpoint(
            checkpoint,
            expected_sha256=str(expected_sha256),
        ).as_dict(),
        label="base checkpoint",
    )
    return BaseCheckpointIdentity(**payload)


def _resolve_git_identity(
    accelerator: Accelerator,
    runtime_config: dict[str, Any],
) -> GitIdentity:
    repo_dir = _repo_dir(runtime_config)
    payload = _all_rank_value(
        accelerator,
        lambda: read_git_identity(repo_dir).as_dict(),
        label="Git source",
    )
    payload["untracked_source_files"] = tuple(
        payload.get("untracked_source_files", ())
    )
    identity = GitIdentity(**payload)
    if bool(runtime_config.get("require_clean_git", True)) and (
        identity.tracked_dirty or identity.untracked_source_files
    ):
        raise RuntimeError(
            "formal Stage 3 training requires clean tracked files and no "
            "untracked source/config/test files"
        )
    return identity


def _resolve_asset_identities(
    accelerator: Accelerator,
    assets_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    required = {"vae", "normalization_stats"}
    if set(assets_config) != required:
        raise ValueError(
            "Stage 3 assets must contain exactly vae and normalization_stats"
        )

    def resolve_all() -> dict[str, Any]:
        identities: dict[str, Any] = {}
        for name in sorted(required):
            spec = assets_config[name]
            if not isinstance(spec, dict):
                raise TypeError(f"assets.{name} must be a mapping")
            identities[name] = resolve_base_checkpoint(
                spec.get("path"),
                expected_sha256=str(spec.get("expected_sha256", "")),
            ).as_dict()
        return identities

    return _all_rank_value(
        accelerator,
        resolve_all,
        label="external assets",
    )


def _validate_required_environment(runtime_config: dict[str, Any]) -> dict[str, str]:
    required = runtime_config.get("required_environment", {})
    if not isinstance(required, dict):
        raise TypeError("runtime.required_environment must be a mapping")
    actual: dict[str, str] = {}
    for name, expected in required.items():
        value = os.environ.get(str(name))
        if value != str(expected):
            raise RuntimeError(
                f"Stage 3 requires environment {name}={expected!r}, "
                f"got {value!r}"
            )
        actual[str(name)] = value
    return actual


def _resolve_data_identity(
    accelerator: Accelerator,
    train_dataset,
    *,
    manifest_config: dict[str, Any],
    normalization_stats_path: str,
) -> dict[str, Any]:
    path_value = manifest_config.get("path")
    expected_sha256 = str(
        manifest_config.get("expected_sha256", "")
    ).strip().lower()
    if not path_value or len(expected_sha256) != 64:
        raise RuntimeError(
            "formal Stage 3 requires data_manifest.path and "
            "data_manifest.expected_sha256"
        )
    if manifest_config.get("full_content_verify") is not True:
        raise RuntimeError(
            "formal Stage 3 requires data_manifest.full_content_verify=true"
        )
    manifest_path = Path(path_value).expanduser().resolve()

    def read_local_manifest() -> dict[str, Any]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read Stage 3 data manifest: {manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError("Stage 3 data manifest must be a mapping")
        return manifest

    def validate_on_main() -> dict[str, Any]:
        manifest = read_local_manifest()
        validated = validate_data_manifest(
            train_dataset,
            manifest,
            normalization_stats_path=normalization_stats_path,
            full_content_verify=True,
        )
        actual_sha256 = validated["manifest_sha256"]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Stage 3 data manifest SHA256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        return {
            "path": str(manifest_path),
            "sha256": actual_sha256,
            "num_frames": int(validated["num_frames"]),
            "dataset_roots": [
                row["root"] for row in validated["dataset_roots"]
            ],
            "full_content_verified": True,
        }

    identity = _main_rank_value(
        accelerator,
        validate_on_main,
        label="dataset manifest",
    )

    def bind_local_verified_manifest() -> dict[str, Any]:
        manifest = read_local_manifest()
        validated = validate_data_manifest(
            train_dataset,
            manifest,
            normalization_stats_path=normalization_stats_path,
            full_content_verify=False,
        )
        recorded_sha256 = validated["manifest_sha256"]
        local_identity = {
            "path": str(manifest_path),
            "sha256": recorded_sha256,
            "num_frames": int(validated["num_frames"]),
            "dataset_roots": [
                row["root"] for row in validated["dataset_roots"]
            ],
            "full_content_verified": True,
        }
        if recorded_sha256 != expected_sha256 or local_identity != identity:
            raise ValueError(
                "Stage 3 local data manifest identity differs from the "
                "main-process validation"
            )
        return bind_validated_text_cache_integrity(train_dataset, manifest)

    text_cache_verification = _all_rank_value(
        accelerator,
        bind_local_verified_manifest,
        label="dataset manifest local binding",
    )
    return {
        **identity,
        "text_cache_verification": text_cache_verification,
    }


def run_stage3_alignment_training(
    config: DictConfig | dict[str, Any],
) -> Stage3AlignmentTrainer:
    """Train Stage 3 with a frozen local 5B base and distributed Adapter only."""

    resolved = _resolved_config(config)
    training_config = dict(resolved["training"])
    precision = _normalize_mixed_precision(training_config["mixed_precision"])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        mixed_precision=precision,
        step_scheduler_with_optimizer=False,
    )
    setup_logging(
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    runtime_config = dict(resolved["runtime"])
    resolved["data"] = _canonicalize_data_paths(
        dict(resolved["data"]),
        repo_dir=_repo_dir(runtime_config),
    )

    def setup_local_runtime() -> dict[str, str]:
        environment = _validate_required_environment(runtime_config)
        if (
            bool(runtime_config.get("require_cuda", True))
            and accelerator.device.type != "cuda"
        ):
            raise RuntimeError("formal Stage 3 training requires a CUDA device")
        if accelerator.device.type == "cuda":
            _activate_accelerator_cuda_device(
                accelerator.device,
                num_processes=accelerator.num_processes,
                local_process_index=accelerator.local_process_index,
            )
        return environment

    environment_identity = _run_all_rank_phase(
        accelerator,
        setup_local_runtime,
        label="local runtime setup",
    )

    output_dir = Path(resolved["output_dir"]).expanduser().resolve()
    resolved["output_dir"] = str(output_dir)
    config_identity = _all_rank_value(
        accelerator,
        lambda: {
            "resolved_config_sha256": canonical_json_sha256(resolved),
        },
        label="resolved config",
    )

    def prepare_output() -> None:
        misc.register_work_dir(output_dir)
        if accelerator.is_main_process:
            write_text_atomic(
                output_dir / "config.yaml",
                OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True),
            )

    _run_all_rank_phase(
        accelerator,
        prepare_output,
        label="output setup",
    )

    base_identity = _resolve_base_identity(
        accelerator,
        dict(resolved["base"]),
    )
    asset_identities = _resolve_asset_identities(
        accelerator,
        dict(resolved["assets"]),
    )
    def validate_stats_binding() -> None:
        configured_stats = (
            resolved.get("data", {})
            .get("train", {})
            .get("pretrained_norm_stats")
        )
        if (
            not configured_stats
            or Path(configured_stats).expanduser().resolve()
            != Path(asset_identities["normalization_stats"]["path"])
        ):
            raise RuntimeError(
                "data.train.pretrained_norm_stats must be the contract-bound "
                "normalization_stats asset"
            )

    _run_all_rank_phase(
        accelerator,
        validate_stats_binding,
        label="normalization stats binding",
    )
    git_identity = _resolve_git_identity(accelerator, runtime_config)

    def write_run_identity() -> None:
        if accelerator.is_main_process:
            write_json_atomic(
                output_dir / "run_identity.json",
                {
                    "base": base_identity.as_dict(),
                    "assets": asset_identities,
                    "git": git_identity.as_dict(),
                    "environment": environment_identity,
                    "config": config_identity,
                },
            )

    _run_all_rank_phase(
        accelerator,
        write_run_identity,
        label="run identity write",
    )

    _run_all_rank_phase(
        accelerator,
        lambda: set_global_seed(int(training_config["seed"])),
        label="global RNG setup",
    )
    model_dtype = _mixed_precision_to_model_dtype(precision)

    def build_local_model():
        local_model = instantiate(
            OmegaConf.create(resolved["model"]),
            model_dtype=model_dtype,
            device=str(accelerator.device),
        )
        load_base = getattr(local_model, "load_frozen_base_checkpoint", None)
        if not callable(load_base):
            raise TypeError(
                "Stage 3 model must expose load_frozen_base_checkpoint()"
            )
        model_vae_path = Path(
            str(getattr(local_model, "model_paths", {}).get("vae", ""))
        )
        if model_vae_path.expanduser().resolve() != Path(
            asset_identities["vae"]["path"]
        ):
            raise RuntimeError(
                "the instantiated model did not load the contract-bound VAE"
            )
        metadata = load_base(base_identity.path)
        return local_model, metadata

    model, base_metadata = _run_all_rank_phase(
        accelerator,
        build_local_model,
        label="model construction and base load",
    )
    gc.collect()
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(
        "Loaded strict Stage 3 base: path=%s sha256=%s step=%s",
        base_identity.path,
        base_identity.sha256,
        base_metadata.get("step") if isinstance(base_metadata, dict) else None,
    )

    def build_local_dataset():
        train_dataset, _ = build_datasets(OmegaConf.create(resolved["data"]))
        identity = _validate_formal_dataset(
            train_dataset,
            runtime_config=runtime_config,
        )
        return train_dataset, identity

    train_dataset, dataset_identity = _run_all_rank_phase(
        accelerator,
        build_local_dataset,
        label="dataset construction",
    )
    dataset_identity.update(
        _resolve_data_identity(
            accelerator,
            train_dataset,
            manifest_config=dict(resolved["data_manifest"]),
            normalization_stats_path=asset_identities[
                "normalization_stats"
            ]["path"],
        )
    )

    def write_dataset_identity() -> None:
        if accelerator.is_main_process:
            write_json_atomic(
                output_dir / "dataset_identity.json",
                dataset_identity,
            )

    _run_all_rank_phase(
        accelerator,
        write_dataset_identity,
        label="dataset identity write",
    )
    trainer = Stage3AlignmentTrainer(
        accelerator=accelerator,
        model=model,
        train_dataset=train_dataset,
        config=resolved,
        base_identity=base_identity,
        git_identity=git_identity,
        asset_identities=asset_identities,
        data_identity=dataset_identity,
    )
    trainer.train()
    accelerator.end_training()
    return trainer


__all__ = ["run_stage3_alignment_training"]
