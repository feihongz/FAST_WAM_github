from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import fastwam.alignment.text_cache_index as index_module
from fastwam.alignment.text_cache_index import (
    TextCacheIndex,
    build_text_cache_index,
    canonical_text_cache_descriptor_sha256,
    load_text_cache_index_descriptor,
    prompt_sha256,
)


PROMPT_TEMPLATE = "Do the following task: {task}"
CONTEXT_LEN = 4
FILENAME_SUFFIX = ".t5_len4.wan22ti2v5b.pt"


def _cache_path(root: Path, prompt: str) -> Path:
    return root / f"{prompt_sha256(prompt)}{FILENAME_SUFFIX}"


def _save_payload(
    root: Path,
    prompt: str,
    *,
    context_dtype: torch.dtype = torch.bfloat16,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(root, prompt)
    torch.save(
        {
            "context": torch.arange(
                CONTEXT_LEN * 3, dtype=torch.float32
            ).reshape(CONTEXT_LEN, 3).to(context_dtype),
            "mask": torch.tensor([True, True, False, False]),
        },
        path,
    )
    return path


def _build(tmp_path: Path, prompts: list[str]):
    cache_root = tmp_path / "cache"
    for prompt in set(prompts):
        _save_payload(cache_root, prompt)
    index_path = tmp_path / "identity" / "cache.index"
    descriptor_path = tmp_path / "identity" / "cache.index.json"
    progress: list[tuple[int, int]] = []
    descriptor = build_text_cache_index(
        cache_root=cache_root,
        prompts=prompts,
        context_len=CONTEXT_LEN,
        prompt_template=PROMPT_TEMPLATE,
        filename_suffix=FILENAME_SUFFIX,
        index_path=index_path,
        descriptor_path=descriptor_path,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    return cache_root, index_path, descriptor_path, descriptor, progress


def _rewrite_records(
    index_path: Path,
    transform,
) -> None:
    payload = bytearray(index_path.read_bytes())
    header_size = index_module._HEADER_SIZE
    record_size = index_module._RECORD_SIZE
    records = [
        bytes(payload[offset : offset + record_size])
        for offset in range(header_size, len(payload), record_size)
    ]
    rewritten = transform(records)
    payload[header_size:] = b"".join(rewritten)
    header = list(index_module._HEADER.unpack_from(payload, 0))
    header[8] = hashlib.sha256(payload[header_size:]).digest()
    payload[:header_size] = index_module._HEADER.pack(*header)
    index_path.write_bytes(payload)


def test_build_lookup_and_verified_load_are_deterministic(tmp_path):
    prompts = ["second prompt", "first prompt", "second prompt"]
    cache_root, index_path, descriptor_path, descriptor, progress = _build(
        tmp_path, prompts
    )

    assert descriptor["record_count"] == 2
    assert descriptor["index"]["size_bytes"] == (
        index_module._HEADER_SIZE + 2 * index_module._RECORD_SIZE
    )
    assert progress == [(1, 2), (2, 2)]
    assert load_text_cache_index_descriptor(descriptor_path) == descriptor
    assert descriptor["descriptor_sha256"] == (
        canonical_text_cache_descriptor_sha256(descriptor)
    )

    with TextCacheIndex(descriptor_path) as cache_index:
        cache_index.require_exact_prompts(reversed(prompts))
        record = cache_index.lookup_prompt("first prompt")
        assert record.size_bytes == _cache_path(
            cache_root, "first prompt"
        ).stat().st_size
        loaded = cache_index.load_verified_payload("first prompt")
    assert loaded["context"].dtype == torch.bfloat16
    assert loaded["context"].shape == (CONTEXT_LEN, 3)
    assert loaded["mask"].dtype == torch.bool


def test_builder_never_enumerates_cache_directory(tmp_path, monkeypatch):
    prompts = ["a", "b"]
    cache_root = tmp_path / "cache"
    for prompt in prompts:
        _save_payload(cache_root, prompt)

    def reject_enumeration(*_args, **_kwargs):
        raise AssertionError("cache directory enumeration is forbidden")

    monkeypatch.setattr(Path, "iterdir", reject_enumeration)
    monkeypatch.setattr(Path, "glob", reject_enumeration)
    monkeypatch.setattr(Path, "rglob", reject_enumeration)
    descriptor = build_text_cache_index(
        cache_root=cache_root,
        prompts=prompts,
        context_len=CONTEXT_LEN,
        prompt_template=PROMPT_TEMPLATE,
        filename_suffix=FILENAME_SUFFIX,
        index_path=tmp_path / "identity" / "cache.index",
    )
    assert descriptor["record_count"] == 2


def test_payload_sha_is_checked_before_torch_load(tmp_path, monkeypatch):
    cache_root, _, descriptor_path, _, _ = _build(tmp_path, ["prompt"])
    payload_path = _cache_path(cache_root, "prompt")
    changed = bytearray(payload_path.read_bytes())
    changed[-1] ^= 1
    payload_path.write_bytes(changed)

    calls = 0

    def forbidden_torch_load(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("torch.load must not see unverified bytes")

    monkeypatch.setattr(index_module.torch, "load", forbidden_torch_load)
    with TextCacheIndex(descriptor_path) as cache_index:
        with pytest.raises(ValueError, match="payload SHA256"):
            cache_index.load_verified_payload("prompt")
    assert calls == 0


def test_index_and_descriptor_same_size_tamper_fail_closed(tmp_path):
    _, index_path, descriptor_path, _, _ = _build(tmp_path, ["a", "b"])
    original = bytearray(index_path.read_bytes())
    original[-1] ^= 1
    index_path.write_bytes(original)
    with pytest.raises(ValueError, match="index SHA256"):
        TextCacheIndex(descriptor_path)

    _, _, second_descriptor_path, _, _ = _build(
        tmp_path / "second", ["a", "b"]
    )
    descriptor = json.loads(second_descriptor_path.read_text(encoding="utf-8"))
    descriptor["record_count"] += 1
    second_descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor SHA256"):
        load_text_cache_index_descriptor(second_descriptor_path)


def test_open_index_is_immune_to_live_same_inode_mutation(tmp_path):
    _, index_path, descriptor_path, _, _ = _build(tmp_path, ["prompt"])
    with TextCacheIndex(descriptor_path) as cache_index:
        original = cache_index.lookup_prompt("prompt")
        with index_path.open("r+b") as handle:
            handle.seek(index_module._HEADER_SIZE + 32)
            first = handle.read(1)
            handle.seek(index_module._HEADER_SIZE + 32)
            handle.write(bytes([first[0] ^ 1]))
            handle.flush()

        # The verified receipt is retained as owned immutable bytes, not as a
        # live mmap that could observe the external write.
        assert isinstance(cache_index._mapping, bytes)
        assert cache_index.lookup_prompt("prompt") == original

    with pytest.raises(ValueError, match="index SHA256"):
        TextCacheIndex(descriptor_path)


@pytest.mark.parametrize(
    "transform",
    [
        lambda records: list(reversed(records)),
        lambda records: [records[0], records[0]],
    ],
    ids=["unsorted", "duplicate"],
)
def test_binary_records_must_be_strictly_sorted_and_unique(tmp_path, transform):
    _, index_path, descriptor_path, _, _ = _build(tmp_path, ["a", "b"])
    _rewrite_records(index_path, transform)
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        TextCacheIndex(descriptor_path, verify_index_sha256=False)


def test_exact_prompt_coverage_and_missing_lookup_fail_closed(tmp_path):
    _, _, descriptor_path, _, _ = _build(tmp_path, ["a", "b"])
    with TextCacheIndex(descriptor_path) as cache_index:
        with pytest.raises(ValueError, match="prompt count"):
            cache_index.require_exact_prompts(["a"])
        with pytest.raises(ValueError, match="prompt set"):
            cache_index.require_exact_prompts(["a", "c"])
        with pytest.raises(KeyError, match="absent"):
            cache_index.lookup_prompt("missing")


def test_verified_payload_tensor_contract_is_strict(tmp_path):
    cache_root = tmp_path / "cache"
    _save_payload(cache_root, "prompt", context_dtype=torch.float32)
    descriptor_path = tmp_path / "identity" / "cache.index.json"
    build_text_cache_index(
        cache_root=cache_root,
        prompts=["prompt"],
        context_len=CONTEXT_LEN,
        prompt_template=PROMPT_TEMPLATE,
        filename_suffix=FILENAME_SUFFIX,
        index_path=tmp_path / "identity" / "cache.index",
        descriptor_path=descriptor_path,
    )
    with TextCacheIndex(descriptor_path) as cache_index:
        with pytest.raises(ValueError, match="bfloat16"):
            cache_index.load_verified_payload("prompt")


def test_existing_output_is_rejected_before_payload_reads(tmp_path, monkeypatch):
    cache_root, index_path, descriptor_path, _, _ = _build(tmp_path, ["prompt"])

    monkeypatch.setattr(
        index_module,
        "_read_stable_regular_file",
        lambda *_args, **_kwargs: pytest.fail("payload must not be reopened"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        build_text_cache_index(
            cache_root=cache_root,
            prompts=["prompt"],
            context_len=CONTEXT_LEN,
            prompt_template=PROMPT_TEMPLATE,
            filename_suffix=FILENAME_SUFFIX,
            index_path=index_path,
            descriptor_path=descriptor_path,
        )


def test_parallel_build_is_byte_deterministic(tmp_path):
    prompts = [f"prompt-{index}" for index in range(12)]
    cache_root = tmp_path / "cache"
    for prompt in prompts:
        _save_payload(cache_root, prompt)

    built = []
    for name, workers in (("single", 1), ("parallel", 4)):
        output_dir = tmp_path / name
        index_path = output_dir / "cache.index"
        descriptor_path = output_dir / "cache.index.json"
        progress = []
        descriptor = build_text_cache_index(
            cache_root=cache_root,
            prompts=reversed(prompts),
            context_len=CONTEXT_LEN,
            prompt_template=PROMPT_TEMPLATE,
            filename_suffix=FILENAME_SUFFIX,
            index_path=index_path,
            descriptor_path=descriptor_path,
            workers=workers,
            progress=lambda completed, total: progress.append((completed, total)),
        )
        built.append((index_path.read_bytes(), descriptor, progress))

    assert built[0][0] == built[1][0]
    assert built[0][1] == built[1][1]
    assert built[0][2] == built[1][2] == [
        (completed, len(prompts)) for completed in range(1, len(prompts) + 1)
    ]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_workers_must_be_a_positive_integer_before_payload_reads(
    tmp_path,
    monkeypatch,
    workers,
):
    cache_root = tmp_path / "cache"
    _save_payload(cache_root, "prompt")
    index_path = tmp_path / "identity" / "cache.index"
    descriptor_path = tmp_path / "identity" / "cache.index.json"
    monkeypatch.setattr(
        index_module,
        "_read_stable_regular_file",
        lambda *_args, **_kwargs: pytest.fail("invalid workers must fail first"),
    )

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        build_text_cache_index(
            cache_root=cache_root,
            prompts=["prompt"],
            context_len=CONTEXT_LEN,
            prompt_template=PROMPT_TEMPLATE,
            filename_suffix=FILENAME_SUFFIX,
            index_path=index_path,
            descriptor_path=descriptor_path,
            workers=workers,
        )
    assert not index_path.exists()
    assert not descriptor_path.exists()


def test_parallel_read_failure_does_not_publish_index_or_descriptor(
    tmp_path,
    monkeypatch,
):
    prompts = ["good-1", "bad", "good-2", "good-3"]
    cache_root = tmp_path / "cache"
    for prompt in prompts:
        _save_payload(cache_root, prompt)
    index_path = tmp_path / "identity" / "cache.index"
    descriptor_path = tmp_path / "identity" / "cache.index.json"
    original_reader = index_module._read_stable_regular_file
    bad_name = f"{prompt_sha256('bad')}{FILENAME_SUFFIX}"

    def fail_one_payload(root, relative_path):
        if relative_path == bad_name:
            raise OSError("injected cache read failure")
        return original_reader(root, relative_path)

    monkeypatch.setattr(
        index_module,
        "_read_stable_regular_file",
        fail_one_payload,
    )
    with pytest.raises(OSError, match="injected cache read failure"):
        build_text_cache_index(
            cache_root=cache_root,
            prompts=prompts,
            context_len=CONTEXT_LEN,
            prompt_template=PROMPT_TEMPLATE,
            filename_suffix=FILENAME_SUFFIX,
            index_path=index_path,
            descriptor_path=descriptor_path,
            workers=4,
        )
    assert not index_path.exists()
    assert not descriptor_path.exists()
    assert not list(index_path.parent.glob(f".{index_path.name}.tmp.*"))


def test_descriptor_schema_rejects_bool(tmp_path):
    _, _, descriptor_path, _, _ = _build(tmp_path, ["prompt"])
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["schema_version"] = True
    descriptor["descriptor_sha256"] = canonical_text_cache_descriptor_sha256(
        descriptor
    )
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported.*schema"):
        load_text_cache_index_descriptor(descriptor_path)


def test_builder_rejects_colliding_output_paths_before_payload_reads(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "cache"
    _save_payload(cache_root, "prompt")
    output_path = tmp_path / "identity" / "cache.index"
    monkeypatch.setattr(
        index_module,
        "_read_stable_regular_file",
        lambda *_args, **_kwargs: pytest.fail("colliding paths must fail first"),
    )

    with pytest.raises(ValueError, match="must be different"):
        build_text_cache_index(
            cache_root=cache_root,
            prompts=["prompt"],
            context_len=CONTEXT_LEN,
            prompt_template=PROMPT_TEMPLATE,
            filename_suffix=FILENAME_SUFFIX,
            index_path=output_path,
            descriptor_path=output_path,
        )
    assert not output_path.exists()


def test_index_sha_verification_flag_must_be_bool(tmp_path):
    _, _, descriptor_path, _, _ = _build(tmp_path, ["prompt"])
    with pytest.raises(TypeError, match="must be bool"):
        TextCacheIndex(descriptor_path, verify_index_sha256=1)
