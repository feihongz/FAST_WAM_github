"""Strict artifact inspection and loading for Stage 3 endpoint evaluation.

Stage 3 exports only the small alignment Adapter.  Endpoint evaluation must
therefore combine the exact frozen UnifiedShared base with the exact Adapter
export; treating either file as a standalone model checkpoint is incorrect.
This module keeps that two-file contract benchmark independent and fail
closed.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from fastwam.alignment.checkpointing import resolve_base_checkpoint, sha256_file
from fastwam.models.wan22.video_action_alignment import (
    ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
    load_alignment_checkpoint,
)


ALIGNED_ARTIFACT_IDENTITY_SCHEMA_VERSION = 1
ALIGNED_ARTIFACT_IDENTITY_KIND = "stage3_aligned_artifact_identity"
ALIGNED_MODEL_IDENTITY_SCHEMA_VERSION = 1
ALIGNED_MODEL_IDENTITY_KIND = "stage3_aligned_model_identity"
_ALIGNMENT_EXPORT_KIND = "stage3_alignment_export"


def _json_copy(value: Any, *, field: str) -> Any:
    """Return a detached JSON value or reject non-serializable metadata."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be JSON serializable") from error


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must contain exactly 64 lowercase hex chars")
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"alignment export {field} must be a string or null")
    return value


def _validated_export_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("alignment export must be a mapping")
    if (
        payload.get("schema_version") != ALIGNMENT_CHECKPOINT_SCHEMA_VERSION
        or payload.get("kind") != _ALIGNMENT_EXPORT_KIND
    ):
        raise ValueError("unsupported alignment export")
    if not isinstance(payload.get("adapter"), Mapping):
        raise ValueError("alignment export is missing Adapter state")

    base_checkpoint = payload.get("base_checkpoint")
    if not isinstance(base_checkpoint, str) or not base_checkpoint:
        raise ValueError("alignment export base_checkpoint must be non-empty")
    base_sha256 = _require_sha256(
        payload.get("base_checkpoint_sha256"),
        field="alignment export base_checkpoint_sha256",
    )
    data_sha256 = _require_sha256(
        payload.get("data_manifest_sha256"),
        field="alignment export data_manifest_sha256",
    )
    alignment_config = payload.get("alignment_config")
    if not isinstance(alignment_config, Mapping):
        raise ValueError("alignment export alignment_config must be a mapping")

    global_step = payload.get("global_step")
    if global_step is not None and (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("alignment export global_step must be non-negative or null")
    training_contract_sha256 = payload.get("training_contract_sha256")
    if training_contract_sha256 is not None:
        training_contract_sha256 = _require_sha256(
            training_contract_sha256,
            field="alignment export training_contract_sha256",
        )
    asset_identities = payload.get("asset_identities", {})
    if not isinstance(asset_identities, Mapping):
        raise ValueError("alignment export asset_identities must be a mapping")

    metadata = {
        "schema_version": ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
        "kind": _ALIGNMENT_EXPORT_KIND,
        "base_checkpoint": base_checkpoint,
        "base_checkpoint_sha256": base_sha256,
        "data_manifest_sha256": data_sha256,
        "alignment_config": dict(alignment_config),
        "global_step": global_step,
        "git_commit": _optional_string(payload.get("git_commit"), field="git_commit"),
        "training_contract_sha256": training_contract_sha256,
        "asset_identities": dict(asset_identities),
    }
    return _json_copy(metadata, field="alignment export metadata")


def inspect_alignment_export(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_base_checkpoint_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_training_contract_sha256: str | None = None,
    expected_global_step: int | None = None,
) -> dict[str, Any]:
    """Inspect an Adapter export without constructing or mutating a model."""

    expected_sha256 = _require_sha256(
        expected_sha256,
        field="expected alignment export SHA256",
    )
    if expected_base_checkpoint_sha256 is not None:
        expected_base_checkpoint_sha256 = _require_sha256(
            expected_base_checkpoint_sha256,
            field="expected base checkpoint SHA256",
        )
    if expected_data_manifest_sha256 is not None:
        expected_data_manifest_sha256 = _require_sha256(
            expected_data_manifest_sha256,
            field="expected data manifest SHA256",
        )
    if expected_training_contract_sha256 is not None:
        expected_training_contract_sha256 = _require_sha256(
            expected_training_contract_sha256,
            field="expected training contract SHA256",
        )
    if expected_global_step is not None and (
        isinstance(expected_global_step, bool)
        or not isinstance(expected_global_step, int)
        or expected_global_step < 0
    ):
        raise ValueError("expected global step must be a non-negative integer or null")

    source = Path(path).expanduser().resolve()
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "alignment export SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}, path={source}"
        )
    size_bytes = source.stat().st_size
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = _validated_export_metadata(payload)
    if (
        expected_base_checkpoint_sha256 is not None
        and metadata["base_checkpoint_sha256"]
        != expected_base_checkpoint_sha256
    ):
        raise ValueError("alignment export base checkpoint hash mismatch")
    if (
        expected_data_manifest_sha256 is not None
        and metadata["data_manifest_sha256"] != expected_data_manifest_sha256
    ):
        raise ValueError("alignment export data manifest hash mismatch")
    if (
        expected_training_contract_sha256 is not None
        and metadata["training_contract_sha256"]
        != expected_training_contract_sha256
    ):
        raise ValueError("alignment export training contract hash mismatch")
    if (
        expected_global_step is not None
        and metadata["global_step"] != expected_global_step
    ):
        raise ValueError("alignment export global step mismatch")

    # Detect replacement during inspection rather than reporting stale metadata.
    if source.stat().st_size != size_bytes or sha256_file(source) != actual_sha256:
        raise RuntimeError("alignment export changed while it was being inspected")
    identity = {
        "path": str(source),
        "sha256": actual_sha256,
        "size_bytes": size_bytes,
        "export_metadata": metadata,
    }
    return _json_copy(identity, field="alignment export identity")


def _inspect_runtime_assets(
    export_metadata: Mapping[str, Any],
    *,
    asset_paths: Mapping[str, str | Path] | None,
    expected_asset_sha256: Mapping[str, str] | None,
) -> dict[str, dict[str, Any]]:
    if asset_paths is None and expected_asset_sha256 is None:
        return {}
    if not isinstance(asset_paths, Mapping) or not isinstance(
        expected_asset_sha256, Mapping
    ):
        raise ValueError(
            "asset_paths and expected_asset_sha256 must be provided together"
        )
    if set(asset_paths) != set(expected_asset_sha256):
        raise ValueError("runtime asset path/SHA256 keys do not match")

    export_assets = export_metadata.get("asset_identities")
    if not isinstance(export_assets, Mapping):
        raise ValueError("alignment export asset identities are invalid")
    identities: dict[str, dict[str, Any]] = {}
    for name in sorted(asset_paths):
        if not isinstance(name, str) or not name:
            raise ValueError("runtime asset names must be non-empty strings")
        expected_sha256 = _require_sha256(
            expected_asset_sha256[name],
            field=f"expected {name} SHA256",
        )
        export_identity = export_assets.get(name)
        if not isinstance(export_identity, Mapping):
            raise ValueError(f"alignment export is missing {name} asset identity")
        export_sha256 = _require_sha256(
            export_identity.get("sha256"),
            field=f"alignment export {name} SHA256",
        )
        if export_sha256 != expected_sha256:
            raise ValueError(f"alignment export {name} SHA256 mismatch")

        source = Path(asset_paths[name]).expanduser().resolve()
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{name} SHA256 mismatch: expected={expected_sha256}, "
                f"actual={actual_sha256}, path={source}"
            )
        size_bytes = source.stat().st_size
        export_size = export_identity.get("size_bytes")
        if (
            isinstance(export_size, bool)
            or not isinstance(export_size, int)
            or export_size < 0
            or export_size != size_bytes
        ):
            raise ValueError(f"alignment export {name} size does not match runtime asset")
        identities[name] = {
            "path": str(source),
            "sha256": actual_sha256,
            "size_bytes": size_bytes,
            "export_identity": dict(export_identity),
        }
    return _json_copy(identities, field="runtime asset identities")


def inspect_aligned_model_artifacts(
    *,
    base_checkpoint_path: str | Path,
    expected_base_checkpoint_sha256: str,
    alignment_export_path: str | Path,
    expected_alignment_export_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_contract_sha256: str | None = None,
    expected_global_step: int | None = None,
    asset_paths: Mapping[str, str | Path] | None = None,
    expected_asset_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate every endpoint artifact before allocating the aligned model."""

    expected_data_manifest_sha256 = _require_sha256(
        expected_data_manifest_sha256,
        field="expected data manifest SHA256",
    )
    base_identity = resolve_base_checkpoint(
        base_checkpoint_path,
        expected_sha256=expected_base_checkpoint_sha256,
    )
    export_identity = inspect_alignment_export(
        alignment_export_path,
        expected_sha256=expected_alignment_export_sha256,
        expected_base_checkpoint_sha256=base_identity.sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_training_contract_sha256=expected_training_contract_sha256,
        expected_global_step=expected_global_step,
    )
    runtime_assets = _inspect_runtime_assets(
        export_identity["export_metadata"],
        asset_paths=asset_paths,
        expected_asset_sha256=expected_asset_sha256,
    )
    identity = {
        "schema_version": ALIGNED_ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "kind": ALIGNED_ARTIFACT_IDENTITY_KIND,
        "base_checkpoint": base_identity.as_dict(),
        "alignment_export": export_identity,
        "data_manifest_sha256": expected_data_manifest_sha256,
        "runtime_assets": runtime_assets,
    }
    return _json_copy(identity, field="aligned artifact identity")


def _assert_file_identity_unchanged(
    label: str,
    identity: Any,
    *,
    phase: str,
) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError(f"aligned artifact {label} identity is invalid")
    path = Path(str(identity.get("path")))
    expected_sha256 = _require_sha256(
        identity.get("sha256"),
        field=f"aligned artifact {label} SHA256",
    )
    expected_size = identity.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(f"aligned artifact {label} size is invalid")
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} changed {phase}")


def _assert_identity_files_unchanged(
    artifact_identity: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    candidates = {
        "base checkpoint": artifact_identity.get("base_checkpoint"),
        "alignment export": artifact_identity.get("alignment_export"),
    }
    runtime_assets = artifact_identity.get("runtime_assets", {})
    if not isinstance(runtime_assets, Mapping):
        raise ValueError("aligned artifact runtime_assets must be a mapping")
    candidates.update(
        {f"runtime asset {name}": identity for name, identity in runtime_assets.items()}
    )
    for label, identity in candidates.items():
        _assert_file_identity_unchanged(label, identity, phase=phase)


def verify_aligned_runtime_asset(
    artifact_identity: Mapping[str, Any],
    asset_name: str,
    *,
    phase: str,
) -> dict[str, Any]:
    """Reverify one runtime asset after a consumer has read it."""

    if (
        not isinstance(artifact_identity, Mapping)
        or artifact_identity.get("schema_version")
        != ALIGNED_ARTIFACT_IDENTITY_SCHEMA_VERSION
        or artifact_identity.get("kind") != ALIGNED_ARTIFACT_IDENTITY_KIND
    ):
        raise ValueError("unsupported aligned artifact identity")
    if not isinstance(asset_name, str) or not asset_name:
        raise ValueError("asset_name must be a non-empty string")
    runtime_assets = artifact_identity.get("runtime_assets")
    if not isinstance(runtime_assets, Mapping):
        raise ValueError("aligned artifact runtime_assets must be a mapping")
    identity = runtime_assets.get(asset_name)
    if not isinstance(identity, Mapping):
        raise ValueError(f"aligned artifact has no runtime asset {asset_name}")
    _assert_file_identity_unchanged(
        f"runtime asset {asset_name}",
        identity,
        phase=phase,
    )
    return _json_copy(identity, field=f"runtime asset {asset_name} identity")


def load_prepared_aligned_model(
    model: Any,
    artifact_identity: Mapping[str, Any],
    *,
    _alignment_loader: Any = None,
) -> dict[str, Any]:
    """Load a pre-inspected frozen base followed by its strict Adapter export."""

    from fastwam.models.wan22.fastwam_unified_aligned import FastWAMUnifiedAligned

    if not isinstance(model, FastWAMUnifiedAligned):
        raise TypeError("Stage 3 endpoint eval requires FastWAMUnifiedAligned")
    if (
        not isinstance(artifact_identity, Mapping)
        or artifact_identity.get("schema_version")
        != ALIGNED_ARTIFACT_IDENTITY_SCHEMA_VERSION
        or artifact_identity.get("kind") != ALIGNED_ARTIFACT_IDENTITY_KIND
    ):
        raise ValueError("unsupported aligned artifact identity")
    base_identity = artifact_identity.get("base_checkpoint")
    export_identity = artifact_identity.get("alignment_export")
    if not isinstance(base_identity, Mapping) or not isinstance(
        export_identity, Mapping
    ):
        raise ValueError("aligned artifact identity is incomplete")
    export_metadata = export_identity.get("export_metadata")
    if not isinstance(export_metadata, Mapping):
        raise ValueError("aligned artifact export metadata is invalid")

    adapter = getattr(model, "alignment_adapter", None)
    if adapter is None or not callable(getattr(adapter, "config", None)):
        raise TypeError("aligned model must expose a configured alignment_adapter")
    model_config = _json_copy(adapter.config(), field="model alignment Adapter config")
    if export_metadata.get("alignment_config") != model_config:
        raise ValueError("alignment export config does not match model Adapter")

    # The inspection normally occurs before 5B model construction. Recheck all
    # files now, then again after the two strict loads, to close that time gap.
    _assert_identity_files_unchanged(
        artifact_identity,
        phase="between inspection and model load",
    )
    runtime_assets = artifact_identity.get("runtime_assets", {})
    if not isinstance(runtime_assets, Mapping):
        raise ValueError("aligned artifact runtime_assets must be a mapping")
    vae_identity = runtime_assets.get("vae")
    if vae_identity is not None:
        if not isinstance(vae_identity, Mapping):
            raise ValueError("aligned artifact VAE identity is invalid")
        model_paths = getattr(model, "model_paths", None)
        if not isinstance(model_paths, Mapping) or not model_paths.get("vae"):
            raise RuntimeError(
                "the instantiated model does not report its loaded VAE path"
            )
        model_vae_path = Path(str(model_paths["vae"])).expanduser().resolve()
        expected_vae_path = Path(str(vae_identity.get("path"))).expanduser().resolve()
        if model_vae_path != expected_vae_path:
            raise RuntimeError(
                "the instantiated model did not load the contract-bound VAE"
            )
    base_metadata = model.load_frozen_base_checkpoint(str(base_identity["path"]))
    if not isinstance(base_metadata, Mapping):
        raise TypeError("base checkpoint loader must return metadata mapping")
    alignment_loader = (
        load_alignment_checkpoint
        if _alignment_loader is None
        else _alignment_loader
    )
    alignment_loader(
        str(export_identity["path"]),
        adapter,
        expected_base_checkpoint_sha256=str(base_identity["sha256"]),
        expected_data_manifest_sha256=str(
            artifact_identity["data_manifest_sha256"]
        ),
        map_location="cpu",
    )

    model.eval()
    model.requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise RuntimeError("aligned endpoint model did not enter eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("aligned endpoint model is not fully frozen")
    _assert_identity_files_unchanged(
        artifact_identity,
        phase="while it was being loaded",
    )
    reported_model_paths = getattr(model, "model_paths", {})
    runtime_model_paths = {}
    if isinstance(reported_model_paths, Mapping):
        for name in ("vae", "text_encoder", "tokenizer"):
            value = reported_model_paths.get(name)
            if value is not None and str(value).strip():
                runtime_model_paths[name] = str(
                    Path(str(value)).expanduser().resolve()
                )

    identity = {
        "schema_version": ALIGNED_MODEL_IDENTITY_SCHEMA_VERSION,
        "kind": ALIGNED_MODEL_IDENTITY_KIND,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "base_checkpoint": dict(base_identity),
        "alignment_export": dict(export_identity),
        "data_manifest_sha256": artifact_identity["data_manifest_sha256"],
        "runtime_assets": dict(artifact_identity.get("runtime_assets", {})),
        "runtime_model_paths": runtime_model_paths,
    }
    return _json_copy(identity, field="aligned model identity")


def load_frozen_aligned_model(
    model: Any,
    *,
    base_checkpoint_path: str | Path,
    expected_base_checkpoint_sha256: str,
    alignment_export_path: str | Path,
    expected_alignment_export_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_contract_sha256: str | None = None,
    expected_global_step: int | None = None,
    asset_paths: Mapping[str, str | Path] | None = None,
    expected_asset_sha256: Mapping[str, str] | None = None,
    _alignment_loader: Any = None,
) -> dict[str, Any]:
    """Convenience wrapper for inspect-then-load callers such as Stage 2."""

    artifact_identity = inspect_aligned_model_artifacts(
        base_checkpoint_path=base_checkpoint_path,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        alignment_export_path=alignment_export_path,
        expected_alignment_export_sha256=expected_alignment_export_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_training_contract_sha256=expected_training_contract_sha256,
        expected_global_step=expected_global_step,
        asset_paths=asset_paths,
        expected_asset_sha256=expected_asset_sha256,
    )
    return load_prepared_aligned_model(
        model,
        artifact_identity,
        _alignment_loader=_alignment_loader,
    )


__all__ = [
    "ALIGNED_ARTIFACT_IDENTITY_KIND",
    "ALIGNED_ARTIFACT_IDENTITY_SCHEMA_VERSION",
    "ALIGNED_MODEL_IDENTITY_KIND",
    "ALIGNED_MODEL_IDENTITY_SCHEMA_VERSION",
    "inspect_aligned_model_artifacts",
    "inspect_alignment_export",
    "load_frozen_aligned_model",
    "load_prepared_aligned_model",
    "verify_aligned_runtime_asset",
]
