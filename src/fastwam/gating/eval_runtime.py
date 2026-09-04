"""Strict, manifest-bound assets used by Stage 2 Gate evaluation.

The binary Gate was trained with cached text features and the tokenizer's
original padding mask.  Endpoint evaluation must consume those same immutable
bytes; ``FastWAM.encode_prompt`` intentionally replaces that mask with all
ones for the WAM attention path and therefore is not a valid Gate input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.alignment.checkpointing import sha256_file
from fastwam.alignment.data_identity import (
    canonical_data_manifest_sha256,
    resolve_text_cache_index_descriptor_path,
)
from fastwam.alignment.text_cache_index import TextCacheIndex, prompt_sha256
from fastwam.models.video_gate import BinaryVideoGate

from .checkpointing import load_gate_checkpoint
from .contracts import require_sha256


@dataclass(frozen=True, slots=True)
class BoundPromptContext:
    """One verified cached prompt, prepared for Gate and WAM consumers."""

    context: torch.Tensor
    gate_context_mask: torch.Tensor
    model_context_mask: torch.Tensor
    identity: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedGate:
    """Frozen Gate and the JSON-safe identity of its checkpoint bytes."""

    gate: BinaryVideoGate
    identity: dict[str, Any]


def _json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value)


def _safe_cache_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("text cache relative_path must be a non-empty string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("text cache relative_path must stay below its root")
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("text cache payload escapes its root") from error
    if not candidate.is_file():
        raise ValueError("text cache payload must be a regular file")
    return candidate


def _validate_prompt_payload(
    payload: Any,
    *,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(payload, Mapping) or set(payload) != {"context", "mask"}:
        raise ValueError("text cache payload must contain exactly context and mask")
    context = payload.get("context")
    mask = payload.get("mask")
    if not isinstance(context, torch.Tensor) or context.ndim != 2:
        raise ValueError("text cache context must have shape [L,D]")
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
        raise ValueError("text cache mask must have shape [L]")
    if context.shape[0] != context_len or mask.shape[0] != context_len:
        raise ValueError("text cache tensors disagree with context_len")
    if context.shape[1] <= 0 or not context.is_floating_point():
        raise ValueError("text cache context must be non-empty floating point")
    if context.dtype != torch.bfloat16:
        raise ValueError("formal text cache context must have bfloat16 dtype")
    if mask.dtype != torch.bool or not bool(mask.any().item()):
        raise ValueError("text cache mask must be non-empty bool")
    if not torch.isfinite(context).all():
        raise ValueError("text cache context contains non-finite values")
    context = context.detach().clone()
    mask = mask.detach().clone()
    context[~mask] = 0
    return context, mask


class ManifestBoundPromptContextProvider:
    """Read exact cached contexts selected by a Stage 3 data manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str,
        expected_prompt_template: str,
    ) -> None:
        path = Path(manifest_path).expanduser().resolve(strict=True)
        manifest = _json_mapping(path, label="Stage 3 data manifest")
        expected_sha = require_sha256(
            expected_manifest_sha256,
            field="expected Stage 3 data manifest SHA256",
        )
        recorded_sha = require_sha256(
            manifest.get("manifest_sha256"),
            field="Stage 3 data manifest manifest_sha256",
        )
        if recorded_sha != expected_sha:
            raise ValueError("Stage 3 data manifest SHA256 mismatch")
        if canonical_data_manifest_sha256(manifest) != recorded_sha:
            raise ValueError("Stage 3 data manifest self-hash mismatch")

        text_cache = manifest.get("text_embedding_cache")
        if not isinstance(text_cache, Mapping):
            raise ValueError("Stage 3 data manifest has no text embedding cache")
        if text_cache.get("prompt_template") != expected_prompt_template:
            raise ValueError("Stage 3 text cache prompt template mismatch")
        context_len = text_cache.get("context_len")
        if isinstance(context_len, bool) or not isinstance(context_len, int) or context_len <= 0:
            raise ValueError("Stage 3 text cache context_len must be positive")

        schema_version = manifest.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError(f"unsupported Stage 3 data manifest schema: {schema_version!r}")

        self.manifest_path = path
        self.manifest_sha256 = recorded_sha
        self.prompt_template = expected_prompt_template
        self.context_len = int(context_len)
        self.schema_version = int(schema_version)
        self._v1_root: Path | None = None
        self._v1_entries: dict[str, dict[str, Any]] = {}
        self._v2_index: TextCacheIndex | None = None
        self._cache: dict[str, BoundPromptContext] = {}

        if self.schema_version == 1:
            root_value = text_cache.get("root")
            if not isinstance(root_value, str) or not Path(root_value).is_absolute():
                raise ValueError("v1 text cache root must be absolute")
            root = Path(root_value).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("v1 text cache root must be a directory")
            entries = text_cache.get("files")
            if not isinstance(entries, list) or not entries:
                raise ValueError("v1 text cache files must be a non-empty list")
            by_prompt: dict[str, dict[str, Any]] = {}
            for raw_entry in entries:
                if not isinstance(raw_entry, Mapping):
                    raise ValueError("v1 text cache file entries must be mappings")
                entry = dict(raw_entry)
                if set(entry) != {
                    "prompt_sha256",
                    "relative_path",
                    "role",
                    "sha256",
                    "size_bytes",
                }:
                    raise ValueError("v1 text cache file entry schema mismatch")
                digest = require_sha256(
                    entry["prompt_sha256"], field="text cache prompt_sha256"
                )
                require_sha256(entry["sha256"], field="text cache payload sha256")
                if entry["role"] != "text_embedding":
                    raise ValueError("v1 text cache file role mismatch")
                size = entry["size_bytes"]
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError("v1 text cache payload size must be positive")
                if digest in by_prompt:
                    raise ValueError("v1 text cache contains a duplicate prompt")
                by_prompt[digest] = entry
            self._v1_root = root
            self._v1_entries = by_prompt
        else:
            descriptor_path = resolve_text_cache_index_descriptor_path(manifest)
            self._v2_index = TextCacheIndex(
                descriptor_path,
                verify_index_sha256=True,
            )

    def close(self) -> None:
        if self._v2_index is not None:
            self._v2_index.close()

    def __enter__(self) -> "ManifestBoundPromptContextProvider":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def load(self, prompt: str, *, device: str | torch.device) -> BoundPromptContext:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached

        digest = prompt_sha256(prompt)
        if self.schema_version == 1:
            try:
                entry = self._v1_entries[digest]
            except KeyError as error:
                raise KeyError(f"prompt is absent from Stage 3 text cache: {digest}") from error
            assert self._v1_root is not None
            path = _safe_cache_path(self._v1_root, entry["relative_path"])
            payload_bytes = path.read_bytes()
            if len(payload_bytes) != entry["size_bytes"]:
                raise ValueError("text cache payload size mismatch")
            payload_sha = hashlib.sha256(payload_bytes).hexdigest()
            if payload_sha != entry["sha256"]:
                raise ValueError("text cache payload SHA256 mismatch")
            try:
                payload = torch.load(
                    io.BytesIO(payload_bytes),
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as error:
                raise ValueError("verified text cache payload cannot be deserialized") from error
            source_identity = {
                "path": str(path),
                "sha256": payload_sha,
                "size_bytes": len(payload_bytes),
            }
        else:
            assert self._v2_index is not None
            record = self._v2_index.lookup_prompt(prompt)
            payload = self._v2_index.load_verified_payload(prompt, map_location="cpu")
            source_identity = {
                "descriptor_path": str(self._v2_index.descriptor_path),
                "payload_sha256": record.payload_sha256,
                "size_bytes": record.size_bytes,
            }

        context, mask = _validate_prompt_payload(payload, context_len=self.context_len)
        target = torch.device(device)
        context = context.unsqueeze(0).to(device=target)
        gate_mask = mask.unsqueeze(0).to(device=target)
        model_mask = torch.ones_like(gate_mask)
        identity = {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_schema_version": self.schema_version,
            "prompt_sha256": digest,
            "context_shape": list(context.shape),
            "context_dtype": str(context.dtype),
            "valid_token_count": int(gate_mask.sum().item()),
            "source": source_identity,
        }
        result = BoundPromptContext(
            context=context,
            gate_context_mask=gate_mask,
            model_context_mask=model_mask,
            identity=identity,
        )
        self._cache[prompt] = result
        return result


def load_gate_for_evaluation(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_label_manifest_sha256: str,
    expected_adapter_checkpoint_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_data_manifest_sha256: str,
    expected_episode_split_assignment_sha256: str,
    expected_training_config_sha256: str,
    expected_git_identity: Mapping[str, Any],
    device: str | torch.device,
) -> LoadedGate:
    """Verify a Gate export before moving its frozen module to the eval device."""

    source = Path(path).expanduser().resolve(strict=True)
    expected_file_sha = require_sha256(
        expected_checkpoint_sha256,
        field="expected Gate checkpoint SHA256",
    )
    size_bytes = source.stat().st_size
    actual_file_sha = sha256_file(source)
    if actual_file_sha != expected_file_sha:
        raise ValueError(
            "Gate checkpoint SHA256 mismatch: "
            f"expected={expected_file_sha}, actual={actual_file_sha}"
        )
    gate, payload = load_gate_checkpoint(
        source,
        expected_label_manifest_sha256=expected_label_manifest_sha256,
        expected_adapter_checkpoint_sha256=expected_adapter_checkpoint_sha256,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_episode_split_assignment_sha256=(
            expected_episode_split_assignment_sha256
        ),
        expected_training_config_sha256=expected_training_config_sha256,
        expected_git_identity=expected_git_identity,
        map_location="cpu",
    )
    if source.stat().st_size != size_bytes or sha256_file(source) != actual_file_sha:
        raise RuntimeError("Gate checkpoint changed while it was being loaded")
    gate = gate.to(device=device).eval()
    gate.requires_grad_(False)
    identity = {
        "path": str(source),
        "sha256": actual_file_sha,
        "size_bytes": size_bytes,
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "parameter_count": payload["parameter_count"],
        "global_step": payload["global_step"],
        "epoch": payload["epoch"],
        "best_metrics": dict(payload["best_metrics"]),
        "label_manifest_sha256": payload["label_manifest_sha256"],
        "adapter_checkpoint_sha256": payload["adapter_checkpoint_sha256"],
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "episode_split_assignment_sha256": payload[
            "episode_split_assignment_sha256"
        ],
        "training_config_sha256": payload["training_config_sha256"],
        "git_identity": dict(payload["git_identity"]),
    }
    return LoadedGate(gate=gate, identity=identity)


__all__ = [
    "BoundPromptContext",
    "LoadedGate",
    "ManifestBoundPromptContextProvider",
    "load_gate_for_evaluation",
]
