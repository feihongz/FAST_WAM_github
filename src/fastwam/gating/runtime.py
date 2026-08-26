"""Strict Stage 2 label-model loading from frozen Stage 3 artifacts.

This module deliberately accepts an already-instantiated aligned model.  Model
construction is large and deployment-specific; artifact identity and load
order are not.  The helpers below keep those two concerns separate while
making the formal label-generation path fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from fastwam.alignment.checkpointing import resolve_base_checkpoint, sha256_file
from fastwam.alignment.data_identity import validate_data_manifest
from fastwam.alignment.text_cache_binding import (
    bind_validated_text_cache_integrity,
)
from fastwam.models.wan22.video_action_alignment import (
    ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
    load_alignment_checkpoint,
)

from .contracts import require_sha256


STAGE2_LABEL_MODEL_IDENTITY_SCHEMA_VERSION = 1
STAGE2_LABEL_MODEL_IDENTITY_KIND = "stage2_label_model_identity"
STAGE2_LABEL_DATA_IDENTITY_KIND = "stage2_label_data_identity"
_ALIGNMENT_EXPORT_KIND = "stage3_alignment_export"


def _json_copy(value: Any, *, field: str) -> Any:
    """Return a detached JSON value or reject non-serializable metadata."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be JSON serializable") from error


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
    base_sha256 = require_sha256(
        payload.get("base_checkpoint_sha256"),
        field="alignment export base_checkpoint_sha256",
    )
    data_sha256 = require_sha256(
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
        training_contract_sha256 = require_sha256(
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
) -> dict[str, Any]:
    """Inspect an Adapter export without constructing or mutating a model.

    The file hash is verified before deserialization.  Optional base/data
    expectations let a future factory or CLI reject the wrong export before it
    allocates the 5B model.
    """

    expected_sha256 = require_sha256(
        expected_sha256,
        field="expected alignment export SHA256",
    )
    if expected_base_checkpoint_sha256 is not None:
        expected_base_checkpoint_sha256 = require_sha256(
            expected_base_checkpoint_sha256,
            field="expected base checkpoint SHA256",
        )
    if expected_data_manifest_sha256 is not None:
        expected_data_manifest_sha256 = require_sha256(
            expected_data_manifest_sha256,
            field="expected data manifest SHA256",
        )

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

    # Detect replacement during inspection rather than reporting a stale
    # identity for a different file.
    if source.stat().st_size != size_bytes or sha256_file(source) != actual_sha256:
        raise RuntimeError("alignment export changed while it was being inspected")
    identity = {
        "path": str(source),
        "sha256": actual_sha256,
        "size_bytes": size_bytes,
        "export_metadata": metadata,
    }
    return _json_copy(identity, field="alignment export identity")


def load_stage2_label_model(
    model: Any,
    *,
    base_checkpoint_path: str | Path,
    expected_base_checkpoint_sha256: str,
    alignment_export_path: str | Path,
    expected_alignment_export_sha256: str,
    expected_data_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly load and freeze the aligned model used to make Gate labels.

    Load order is fixed to frozen base first and final alignment Adapter
    second.  The returned identity is emitted only after both strict loads and
    the final inference-only freeze have completed.
    """

    # Keep Gate-only training/imports independent from the 5B WAM module. The
    # aligned implementation is needed only by offline label generation, so
    # import it at the point where that path is actually used.
    from fastwam.models.wan22.fastwam_unified_aligned import (
        FastWAMUnifiedAligned,
    )

    if not isinstance(model, FastWAMUnifiedAligned):
        raise TypeError("Stage 2 labels require FastWAMUnifiedAligned")
    expected_base_checkpoint_sha256 = require_sha256(
        expected_base_checkpoint_sha256,
        field="expected base checkpoint SHA256",
    )
    expected_alignment_export_sha256 = require_sha256(
        expected_alignment_export_sha256,
        field="expected alignment export SHA256",
    )
    expected_data_manifest_sha256 = require_sha256(
        expected_data_manifest_sha256,
        field="expected data manifest SHA256",
    )

    # Finish all artifact/config checks before either loader mutates the model.
    base_identity = resolve_base_checkpoint(
        base_checkpoint_path,
        expected_sha256=expected_base_checkpoint_sha256,
    )
    export_identity = inspect_alignment_export(
        alignment_export_path,
        expected_sha256=expected_alignment_export_sha256,
        expected_base_checkpoint_sha256=base_identity.sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
    )
    adapter = getattr(model, "alignment_adapter", None)
    if adapter is None or not callable(getattr(adapter, "config", None)):
        raise TypeError("aligned model must expose a configured alignment_adapter")
    model_config = _json_copy(
        adapter.config(),
        field="model alignment Adapter config",
    )
    export_metadata = export_identity["export_metadata"]
    if export_metadata["alignment_config"] != model_config:
        raise ValueError("alignment export config does not match model Adapter")

    base_metadata = model.load_frozen_base_checkpoint(base_identity.path)
    if not isinstance(base_metadata, Mapping):
        raise TypeError("base checkpoint loader must return metadata mapping")
    load_alignment_checkpoint(
        export_identity["path"],
        adapter,
        expected_base_checkpoint_sha256=base_identity.sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        map_location="cpu",
    )

    model.eval()
    model.requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise RuntimeError("Stage 2 label model did not enter eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Stage 2 label model is not fully frozen")

    # Recheck both artifacts after their loaders return.  This closes the
    # inspect/load time-of-check gap and prevents a mixed identity report.
    if sha256_file(base_identity.path) != base_identity.sha256:
        raise RuntimeError("base checkpoint changed while it was being loaded")
    if sha256_file(export_identity["path"]) != export_identity["sha256"]:
        raise RuntimeError("alignment export changed while it was being loaded")

    identity = {
        "schema_version": STAGE2_LABEL_MODEL_IDENTITY_SCHEMA_VERSION,
        "kind": STAGE2_LABEL_MODEL_IDENTITY_KIND,
        "model_class": (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}"
        ),
        "base_checkpoint": base_identity.as_dict(),
        "alignment_export": export_identity,
        "data_manifest_sha256": expected_data_manifest_sha256,
    }
    return _json_copy(identity, field="Stage 2 label model identity")


def validate_stage2_label_dataset(
    dataset: Any,
    data_manifest: Mapping[str, Any],
    *,
    normalization_stats_path: str | Path,
    expected_data_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate the no-fallback, normalized dataset used for formal labels."""

    expected_sha256 = require_sha256(
        expected_data_manifest_sha256,
        field="expected data manifest SHA256",
    )
    if data_manifest.get("manifest_sha256") != expected_sha256:
        raise ValueError("Stage 2 label data manifest SHA256 mismatch")
    if getattr(dataset, "strict_data_mode", False) is not True:
        raise RuntimeError("formal Stage 2 labels require strict_data_mode=true")
    if getattr(dataset, "skip_padding_as_possible", None) is not False:
        raise RuntimeError(
            "formal Stage 2 labels require skip_padding_as_possible=false"
        )
    base = getattr(dataset, "lerobot_dataset", None)
    if base is None or getattr(base, "strict_data_mode", False) is not True:
        raise RuntimeError("formal Stage 2 dataset did not propagate strict mode")
    processor = getattr(base, "processor", None)
    if processor is None:
        raise RuntimeError("formal Stage 2 labels require an action normalizer")
    try:
        normalizer = processor.normalizer
    except Exception as error:
        raise RuntimeError(
            "formal Stage 2 labels require a configured action normalizer"
        ) from error
    if normalizer is None:
        raise RuntimeError("formal Stage 2 labels require a configured action normalizer")

    multi = getattr(base, "multi_dataset", None)
    parts = getattr(multi, "_datasets", None)
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("formal Stage 2 labels require non-empty LeRobot data")
    for part in parts:
        if getattr(part, "video_backend", None) != "torchcodec":
            raise RuntimeError("formal Stage 2 labels require video_backend=torchcodec")
        if getattr(part, "allow_video_backend_fallback", True):
            raise RuntimeError("formal Stage 2 labels forbid video decoder fallback")

    stats_path = Path(normalization_stats_path).expanduser().resolve()
    validated = validate_data_manifest(
        dataset,
        data_manifest,
        normalization_stats_path=stats_path,
        full_content_verify=True,
    )
    if validated.get("manifest_sha256") != expected_sha256:
        raise RuntimeError("validated Stage 2 data manifest SHA256 drifted")
    text_cache_verification = bind_validated_text_cache_integrity(
        dataset,
        data_manifest,
    )
    identity = {
        "schema_version": 1,
        "kind": STAGE2_LABEL_DATA_IDENTITY_KIND,
        "data_manifest_sha256": expected_sha256,
        "normalization_stats_path": str(stats_path),
        "dataset_length": len(dataset),
        "dataset_episodes": int(getattr(multi, "num_episodes")),
        "dataset_roots": [str(Path(part.root).resolve()) for part in parts],
        "strict_data_mode": True,
        "skip_padding_as_possible": False,
        "video_backend": "torchcodec",
        "content_verified": True,
        "text_cache_verification": text_cache_verification,
        "normalized_action_space": True,
    }
    return _json_copy(identity, field="Stage 2 label data identity")


__all__ = [
    "STAGE2_LABEL_MODEL_IDENTITY_KIND",
    "STAGE2_LABEL_MODEL_IDENTITY_SCHEMA_VERSION",
    "STAGE2_LABEL_DATA_IDENTITY_KIND",
    "inspect_alignment_export",
    "load_stage2_label_model",
    "validate_stage2_label_dataset",
]
