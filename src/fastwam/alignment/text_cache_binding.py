"""Bind a validated data manifest to the dataset text-cache reader."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .data_identity import (
    DATA_MANIFEST_SCHEMA_VERSION,
    DATA_MANIFEST_V2_SCHEMA_VERSION,
    require_supported_data_manifest_header,
    resolve_text_cache_index_descriptor_path,
)
from .text_cache_index import (
    TEXT_CACHE_INDEX_MODE,
    TextCacheIndexIdentity,
    text_cache_contract_sha256,
)


TEXT_CACHE_INLINE_VERIFICATION_MODE = "manifest_inline_sha256_v1"
TEXT_CACHE_PAYLOAD_VERIFICATION_MODE = "sha256_before_deserialization_v1"


def _manifest_text_cache_identity(
    text_cache: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> TextCacheIndexIdentity:
    files = integrity.get("files")
    if not isinstance(files, list):
        raise ValueError("v2 text cache integrity files must be a list")
    by_role = {
        entry.get("role"): entry
        for entry in files
        if isinstance(entry, Mapping)
    }
    if set(by_role) != {
        "text_cache_index_descriptor",
        "text_cache_index",
    }:
        raise ValueError("v2 text cache integrity file roles are invalid")
    descriptor = by_role["text_cache_index_descriptor"]
    index = by_role["text_cache_index"]
    context_len = text_cache.get("context_len")
    prompt_template = text_cache.get("prompt_template")
    filename_suffix = text_cache.get("filename_suffix")
    return TextCacheIndexIdentity(
        descriptor_file_sha256=descriptor.get("sha256"),
        descriptor_size_bytes=descriptor.get("size_bytes"),
        descriptor_sha256=integrity.get("descriptor_sha256"),
        index_sha256=index.get("sha256"),
        index_size_bytes=index.get("size_bytes"),
        record_count=text_cache.get("required_prompt_count"),
        prompt_set_sha256=text_cache.get("prompt_set_sha256"),
        contract_sha256=text_cache_contract_sha256(
            context_len=context_len,
            prompt_template=prompt_template,
            filename_suffix=filename_suffix,
        ),
        cache_root=text_cache.get("root"),
        context_len=context_len,
        prompt_template=prompt_template,
        filename_suffix=filename_suffix,
        index_relative_path=index.get("relative_path"),
    )


def bind_validated_text_cache_integrity(
    dataset: Any,
    data_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Activate the runtime reader required by an already-validated manifest.

    Schema v1 lists every payload in the manifest and keeps the legacy reader.
    Schema v2 binds a compact index and requires each payload's bytes to match
    that index before ``torch.load`` can see them.
    """

    schema_version = require_supported_data_manifest_header(data_manifest)
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        return {
            "manifest_schema_version": DATA_MANIFEST_SCHEMA_VERSION,
            "integrity_mode": TEXT_CACHE_INLINE_VERIFICATION_MODE,
            "payload_verification": "eager_manifest_hash_before_training",
        }
    if schema_version != DATA_MANIFEST_V2_SCHEMA_VERSION:
        raise AssertionError("supported manifest header returned an unknown schema")

    text_cache = data_manifest.get("text_embedding_cache")
    if not isinstance(text_cache, Mapping):
        raise ValueError("v2 text_embedding_cache must be a mapping")
    integrity = text_cache.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("v2 text cache integrity must be a mapping")
    if integrity.get("mode") != TEXT_CACHE_INDEX_MODE:
        raise ValueError("unsupported v2 text cache integrity mode")

    count = text_cache.get("required_prompt_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("v2 required_prompt_count must be a positive integer")
    prompt_set_sha256 = text_cache.get("prompt_set_sha256")
    if (
        not isinstance(prompt_set_sha256, str)
        or len(prompt_set_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prompt_set_sha256)
    ):
        raise ValueError("v2 prompt_set_sha256 must be lowercase SHA256")

    bind = getattr(dataset, "bind_text_cache_index", None)
    if not callable(bind):
        raise TypeError(
            "v2 formal dataset must expose bind_text_cache_index()"
        )
    descriptor_path = resolve_text_cache_index_descriptor_path(data_manifest)
    expected_identity = _manifest_text_cache_identity(text_cache, integrity)
    bind(descriptor_path, expected_identity)

    return {
        "manifest_schema_version": DATA_MANIFEST_V2_SCHEMA_VERSION,
        "integrity_mode": TEXT_CACHE_INDEX_MODE,
        "payload_verification": TEXT_CACHE_PAYLOAD_VERIFICATION_MODE,
        "descriptor_path": str(descriptor_path),
        "required_prompt_count": count,
        "prompt_set_sha256": prompt_set_sha256,
    }


__all__ = [
    "TEXT_CACHE_INLINE_VERIFICATION_MODE",
    "TEXT_CACHE_PAYLOAD_VERIFICATION_MODE",
    "bind_validated_text_cache_integrity",
]
