from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam.alignment import text_cache_binding as binding
from fastwam.alignment.text_cache_index import (
    TextCacheIndexIdentity,
    text_cache_contract_sha256,
)


def _v1_manifest() -> dict:
    return {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
    }


def _v2_manifest(cache_root: str = "/unused/cache") -> dict:
    return {
        "schema_version": 2,
        "kind": "stage3_robot_video_data_manifest",
        "text_embedding_cache": {
            "root": cache_root,
            "context_len": 128,
            "prompt_template": "Instruction: {task}",
            "filename_suffix": ".t5_len128.wan22ti2v5b.pt",
            "required_prompt_count": 17,
            "prompt_set_sha256": "a" * 64,
            "integrity": {
                "mode": "binary_sha256_index_v1",
                "descriptor_sha256": "b" * 64,
                "files": [
                    {
                        "role": "text_cache_index_descriptor",
                        "relative_path": "cache-index.json",
                        "size_bytes": 301,
                        "sha256": "c" * 64,
                    },
                    {
                        "role": "text_cache_index",
                        "relative_path": "cache-index.bin",
                        "size_bytes": 4096,
                        "sha256": "d" * 64,
                    },
                ],
            },
        },
    }


def test_v1_binding_preserves_legacy_eager_manifest_contract():
    result = binding.bind_validated_text_cache_integrity(object(), _v1_manifest())

    assert result == {
        "manifest_schema_version": 1,
        "integrity_mode": "manifest_inline_sha256_v1",
        "payload_verification": "eager_manifest_hash_before_training",
    }


def test_v2_binding_activates_dataset_verifier_and_reports_semantics(
    tmp_path, monkeypatch
):
    descriptor = tmp_path / "cache-index.json"
    descriptor.write_text("{}", encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    calls = []
    dataset = SimpleNamespace(
        bind_text_cache_index=lambda path, identity: calls.append(
            (Path(path), identity)
        )
    )
    monkeypatch.setattr(
        binding,
        "resolve_text_cache_index_descriptor_path",
        lambda manifest: descriptor.resolve(),
    )

    result = binding.bind_validated_text_cache_integrity(
        dataset,
        _v2_manifest(str(cache_root.resolve())),
    )

    assert len(calls) == 1
    assert calls[0][0] == descriptor.resolve()
    expected_identity = calls[0][1]
    assert isinstance(expected_identity, TextCacheIndexIdentity)
    assert expected_identity.descriptor_file_sha256 == "c" * 64
    assert expected_identity.descriptor_sha256 == "b" * 64
    assert expected_identity.index_sha256 == "d" * 64
    assert expected_identity.index_size_bytes == 4096
    assert expected_identity.record_count == 17
    assert expected_identity.prompt_set_sha256 == "a" * 64
    assert expected_identity.contract_sha256 == text_cache_contract_sha256(
        context_len=128,
        prompt_template="Instruction: {task}",
        filename_suffix=".t5_len128.wan22ti2v5b.pt",
    )
    assert result == {
        "manifest_schema_version": 2,
        "integrity_mode": "binary_sha256_index_v1",
        "payload_verification": "sha256_before_deserialization_v1",
        "descriptor_path": str(descriptor.resolve()),
        "required_prompt_count": 17,
        "prompt_set_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mode", "integrity mode"),
        ("count", "required_prompt_count"),
        ("prompt_sha", "prompt_set_sha256"),
        ("binder", "bind_text_cache_index"),
    ],
)
def test_v2_binding_fails_closed_before_loader_use(
    tmp_path, monkeypatch, mutation, message
):
    manifest = _v2_manifest()
    descriptor = tmp_path / "cache-index.json"
    descriptor.write_text("{}", encoding="utf-8")
    calls = []
    dataset = SimpleNamespace(
        bind_text_cache_index=lambda path, identity: calls.append(
            (Path(path), identity)
        )
    )
    if mutation == "mode":
        manifest["text_embedding_cache"]["integrity"]["mode"] = "unknown"
    elif mutation == "count":
        manifest["text_embedding_cache"]["required_prompt_count"] = 0
    elif mutation == "prompt_sha":
        manifest["text_embedding_cache"]["prompt_set_sha256"] = "A" * 64
    else:
        dataset = object()
    monkeypatch.setattr(
        binding,
        "resolve_text_cache_index_descriptor_path",
        lambda payload: descriptor.resolve(),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        binding.bind_validated_text_cache_integrity(dataset, manifest)
    assert not calls
