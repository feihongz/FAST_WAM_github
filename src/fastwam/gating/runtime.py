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

from fastwam.alignment.eval_loading import (
    inspect_alignment_export,
    load_frozen_aligned_model,
)
from fastwam.alignment.data_identity import validate_data_manifest
from fastwam.alignment.text_cache_binding import (
    bind_validated_text_cache_integrity,
)
from fastwam.models.wan22.video_action_alignment import load_alignment_checkpoint

from .contracts import require_sha256
STAGE2_LABEL_MODEL_IDENTITY_SCHEMA_VERSION = 1
STAGE2_LABEL_MODEL_IDENTITY_KIND = "stage2_label_model_identity"
STAGE2_LABEL_DATA_IDENTITY_KIND = "stage2_label_data_identity"


def _json_copy(value: Any, *, field: str) -> Any:
    """Return a detached JSON value or reject non-serializable metadata."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be JSON serializable") from error


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

    aligned_identity = load_frozen_aligned_model(
        model,
        base_checkpoint_path=base_checkpoint_path,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        alignment_export_path=alignment_export_path,
        expected_alignment_export_sha256=expected_alignment_export_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        _alignment_loader=load_alignment_checkpoint,
    )
    identity = {
        "schema_version": STAGE2_LABEL_MODEL_IDENTITY_SCHEMA_VERSION,
        "kind": STAGE2_LABEL_MODEL_IDENTITY_KIND,
        "model_class": aligned_identity["model_class"],
        "base_checkpoint": aligned_identity["base_checkpoint"],
        "alignment_export": aligned_identity["alignment_export"],
        "data_manifest_sha256": aligned_identity["data_manifest_sha256"],
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
