"""Compact, fail-closed identity for large text-embedding caches.

The binary index contains one fixed-width record per selected prompt.  It is
small enough to memory-map even when the cache contains millions of individual
``.pt`` files.  The index builder derives every cache path from an explicit
prompt; it never walks the cache directory.

The index is an integrity receipt, not an optimization hint.  A payload is
read into bytes, checked against the receipt, and only then deserialized from
those same bytes.  This prevents an unverified pickle/torch payload from being
consumed by a formal job.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any
import uuid

import torch


TEXT_CACHE_INDEX_SCHEMA_VERSION = 1
TEXT_CACHE_INDEX_KIND = "fastwam_text_cache_index_descriptor"
TEXT_CACHE_INDEX_MODE = "binary_sha256_index_v1"

_MAGIC = b"FASTWAMTCIDXv1\0\0"
_HEADER = struct.Struct(">16sIIIIQ32s32s32s24s")
_RECORD = struct.Struct(">32s32sQ")
_HEADER_SIZE = _HEADER.size
_RECORD_SIZE = _RECORD.size
_FLAGS = 0
_RESERVED = b"\0" * 24
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_SET_DOMAIN = b"fastwam:text-cache-prompt-set:v1\0"

_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "cache_root",
        "context_len",
        "prompt_template",
        "filename_suffix",
        "record_count",
        "prompt_set_sha256",
        "contract_sha256",
        "index",
        "descriptor_sha256",
    }
)
_INDEX_FILE_KEYS = frozenset({"relative_path", "size_bytes", "sha256"})


@dataclass(frozen=True, slots=True)
class TextCacheIndexIdentity:
    """Immutable identity that a formal runtime pins across worker reopens.

    The descriptor's semantic self-hash and its exact on-disk file hash are
    both retained.  This intentionally duplicates a little information: the
    former authenticates the descriptor fields, while the latter binds the
    exact file selected by the Stage 3 data manifest.
    """

    descriptor_file_sha256: str
    descriptor_size_bytes: int
    descriptor_sha256: str
    index_sha256: str
    index_size_bytes: int
    record_count: int
    prompt_set_sha256: str
    contract_sha256: str
    cache_root: str
    context_len: int
    prompt_template: str
    filename_suffix: str
    index_relative_path: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.descriptor_file_sha256,
            field="descriptor_file_sha256",
        )
        _positive_integer(
            self.descriptor_size_bytes,
            field="descriptor_size_bytes",
        )
        _require_sha256(self.descriptor_sha256, field="descriptor_sha256")
        _require_sha256(self.index_sha256, field="index_sha256")
        _positive_integer(self.index_size_bytes, field="index_size_bytes")
        _positive_integer(self.record_count, field="record_count")
        _require_sha256(self.prompt_set_sha256, field="prompt_set_sha256")
        _require_sha256(self.contract_sha256, field="contract_sha256")
        cache_root = _absolute_directory(self.cache_root, field="cache_root")
        if str(cache_root) != self.cache_root:
            raise ValueError("cache_root must be a canonical absolute directory")
        contract = _contract_payload(
            context_len=self.context_len,
            prompt_template=self.prompt_template,
            filename_suffix=self.filename_suffix,
        )
        if text_cache_contract_sha256(**contract) != self.contract_sha256:
            raise ValueError("contract_sha256 disagrees with the cache contract")
        _safe_relative_path(
            self.index_relative_path,
            field="index_relative_path",
        )

    @classmethod
    def from_verified_index(
        cls,
        cache_index: "TextCacheIndex",
    ) -> "TextCacheIndexIdentity":
        """Capture identity only from an index whose bytes were SHA-verified."""

        observed_index_sha256 = getattr(
            cache_index, "index_file_sha256", None
        )
        if observed_index_sha256 is None:
            raise RuntimeError(
                "text cache index identity requires verified index bytes"
            )
        descriptor = cache_index.descriptor
        return cls(
            descriptor_file_sha256=cache_index.descriptor_file_sha256,
            descriptor_size_bytes=cache_index.descriptor_file_size_bytes,
            descriptor_sha256=descriptor["descriptor_sha256"],
            index_sha256=observed_index_sha256,
            index_size_bytes=cache_index.index_file_size_bytes,
            record_count=descriptor["record_count"],
            prompt_set_sha256=descriptor["prompt_set_sha256"],
            contract_sha256=descriptor["contract_sha256"],
            cache_root=descriptor["cache_root"],
            context_len=descriptor["context_len"],
            prompt_template=descriptor["prompt_template"],
            filename_suffix=descriptor["filename_suffix"],
            index_relative_path=descriptor["index"]["relative_path"],
        )


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("text cache index metadata must be canonical JSON") from error
    return encoded.encode("utf-8")


def _sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{field} is not a canonical relative path")
    if relative.as_posix() != value:
        raise ValueError(f"{field} is not a canonical POSIX path")
    return value


def _absolute_directory(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{field} must resolve to a directory")
    return resolved


def _contract_payload(
    *, context_len: int, prompt_template: str, filename_suffix: str
) -> dict[str, Any]:
    length = _positive_integer(context_len, field="context_len")
    if not isinstance(prompt_template, str) or "{task}" not in prompt_template:
        raise ValueError("prompt_template must be a string containing {task}")
    if (
        not isinstance(filename_suffix, str)
        or not filename_suffix.startswith(".")
        or "/" in filename_suffix
        or "\\" in filename_suffix
    ):
        raise ValueError("filename_suffix must be a safe filename suffix")
    return {
        "context_len": length,
        "prompt_template": prompt_template,
        "filename_suffix": filename_suffix,
    }


def text_cache_contract_sha256(
    *, context_len: int, prompt_template: str, filename_suffix: str
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            _contract_payload(
                context_len=context_len,
                prompt_template=prompt_template,
                filename_suffix=filename_suffix,
            )
        )
    )


def canonical_text_cache_descriptor_sha256(
    descriptor: Mapping[str, Any],
) -> str:
    if not isinstance(descriptor, Mapping):
        raise TypeError("text cache index descriptor must be a mapping")
    payload = dict(descriptor)
    payload.pop("descriptor_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _prompt_digest(prompt: str) -> bytes:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("selected prompts must be non-empty strings")
    return hashlib.sha256(prompt.encode("utf-8")).digest()


def prompt_sha256(prompt: str) -> str:
    return _prompt_digest(prompt).hex()


def _normalized_prompts(prompts: Iterable[str]) -> list[tuple[bytes, str]]:
    by_digest: dict[bytes, str] = {}
    for prompt in prompts:
        digest = _prompt_digest(prompt)
        previous = by_digest.get(digest)
        if previous is not None and previous != prompt:
            raise ValueError("SHA256 collision between selected text prompts")
        by_digest[digest] = prompt
    if not by_digest:
        raise ValueError("text cache index requires at least one selected prompt")
    return sorted(by_digest.items(), key=lambda row: row[0])


def _prompt_set_digest(digests: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(_PROMPT_SET_DOMAIN)
    count = 0
    previous: bytes | None = None
    for value in digests:
        if not isinstance(value, bytes) or len(value) != 32:
            raise ValueError("prompt digest must contain exactly 32 bytes")
        if previous is not None and value <= previous:
            raise ValueError("prompt digests must be strictly sorted and unique")
        digest.update(value)
        previous = value
        count += 1
    if count == 0:
        raise ValueError("prompt digest set must not be empty")
    return digest.hexdigest()


def prompt_set_sha256(prompts: Iterable[str]) -> str:
    normalized = _normalized_prompts(prompts)
    return _prompt_set_digest(row[0] for row in normalized)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _resolve_anchored_file(anchor: Path, relative_path: str) -> Path:
    candidate = anchor / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(anchor)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"text cache payload is missing or escapes its root: {candidate}"
        ) from error
    return resolved


def _read_stable_regular_file(anchor: Path, relative_path: str) -> bytes:
    resolved = _resolve_anchored_file(anchor, relative_path)
    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"text cache payload is not regular: {resolved}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ValueError(f"cannot read text cache payload: {resolved}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"text cache payload changed while being read: {resolved}")
    if _resolve_anchored_file(anchor, relative_path) != resolved:
        raise ValueError(f"text cache payload path changed while being read: {resolved}")
    current = resolved.stat()
    if _stat_identity(current) != _stat_identity(after):
        raise ValueError(f"text cache payload changed after being read: {resolved}")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _descriptor_json_bytes(descriptor: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(descriptor) + b"\n"


def _validate_descriptor_mapping(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError("text cache index descriptor must be a mapping")
    payload = dict(descriptor)
    if set(payload) != _DESCRIPTOR_KEYS:
        raise ValueError("text cache index descriptor has unexpected fields")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != TEXT_CACHE_INDEX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported text cache index descriptor schema")
    if payload.get("kind") != TEXT_CACHE_INDEX_KIND:
        raise ValueError("unsupported text cache index descriptor kind")

    cache_root = _absolute_directory(payload.get("cache_root"), field="cache_root")
    contract = _contract_payload(
        context_len=payload.get("context_len"),
        prompt_template=payload.get("prompt_template"),
        filename_suffix=payload.get("filename_suffix"),
    )
    expected_contract = text_cache_contract_sha256(**contract)
    if _require_sha256(
        payload.get("contract_sha256"), field="contract_sha256"
    ) != expected_contract:
        raise ValueError("text cache index contract SHA256 mismatch")
    record_count = _positive_integer(payload.get("record_count"), field="record_count")
    prompt_set = _require_sha256(
        payload.get("prompt_set_sha256"), field="prompt_set_sha256"
    )

    index_file = payload.get("index")
    if not isinstance(index_file, Mapping) or set(index_file) != _INDEX_FILE_KEYS:
        raise ValueError("text cache index file descriptor is invalid")
    relative = _safe_relative_path(
        index_file.get("relative_path"), field="index.relative_path"
    )
    size_bytes = _positive_integer(
        index_file.get("size_bytes"), field="index.size_bytes"
    )
    index_sha = _require_sha256(index_file.get("sha256"), field="index.sha256")
    recorded_descriptor_sha = _require_sha256(
        payload.get("descriptor_sha256"), field="descriptor_sha256"
    )
    if canonical_text_cache_descriptor_sha256(payload) != recorded_descriptor_sha:
        raise ValueError("text cache index descriptor SHA256 mismatch")

    payload["cache_root"] = str(cache_root)
    payload["context_len"] = contract["context_len"]
    payload["record_count"] = record_count
    payload["prompt_set_sha256"] = prompt_set
    payload["contract_sha256"] = expected_contract
    payload["index"] = {
        "relative_path": relative,
        "size_bytes": size_bytes,
        "sha256": index_sha,
    }
    return payload


def _load_text_cache_index_descriptor_with_bytes(
    path: str | Path,
) -> tuple[Path, dict[str, Any], bytes]:
    descriptor_path = Path(path).expanduser().resolve(strict=True)
    if not descriptor_path.is_file():
        raise ValueError("text cache index descriptor must be a regular file")
    try:
        raw_payload = _read_stable_regular_file(
            descriptor_path.parent.resolve(strict=True),
            descriptor_path.name,
        )
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read text cache index descriptor") from error
    return descriptor_path, _validate_descriptor_mapping(payload), raw_payload


def load_text_cache_index_descriptor(path: str | Path) -> dict[str, Any]:
    _, descriptor, _ = _load_text_cache_index_descriptor_with_bytes(path)
    return descriptor


@dataclass(frozen=True, slots=True)
class TextCacheIndexRecord:
    prompt_sha256: str
    payload_sha256: str
    size_bytes: int


class TextCacheIndex:
    """Read-only immutable-byte view of a validated text cache index."""

    def __init__(
        self,
        descriptor_path: str | Path,
        *,
        verify_index_sha256: bool = True,
    ) -> None:
        if not isinstance(verify_index_sha256, bool):
            raise TypeError("verify_index_sha256 must be bool")
        (
            self.descriptor_path,
            self.descriptor,
            descriptor_bytes,
        ) = _load_text_cache_index_descriptor_with_bytes(descriptor_path)
        self.descriptor_file_size_bytes = len(descriptor_bytes)
        self.descriptor_file_sha256 = _sha256_bytes(descriptor_bytes)
        descriptor_root = self.descriptor_path.parent.resolve(strict=True)
        relative = self.descriptor["index"]["relative_path"]
        self.index_path = _resolve_anchored_file(descriptor_root, relative)
        self._handle = None
        self._mapping = None
        self.index_file_size_bytes = 0
        self.index_file_sha256: str | None = None
        try:
            # Retain immutable bytes rather than a live mmap.  A read-only mmap
            # can still observe another process mutating the same inode after
            # validation; an owned bytes object cannot.
            index_bytes = _read_stable_regular_file(descriptor_root, relative)
            self.index_file_size_bytes = len(index_bytes)
            if self.index_file_size_bytes != self.descriptor["index"]["size_bytes"]:
                raise ValueError("text cache index size differs from its descriptor")
            self._mapping = index_bytes
            if verify_index_sha256:
                index_sha256 = _sha256_bytes(index_bytes)
                if index_sha256 != self.descriptor["index"]["sha256"]:
                    raise ValueError(
                        "text cache index SHA256 differs from its descriptor"
                    )
                self.index_file_sha256 = index_sha256
            self._validate_binary()
        except Exception:
            self.close()
            raise

    def require_identity(self, expected: TextCacheIndexIdentity) -> None:
        """Reject a reopen whose files differ from the originally bound receipt."""

        if not isinstance(expected, TextCacheIndexIdentity):
            raise TypeError("expected text cache identity has an invalid type")
        observed = TextCacheIndexIdentity.from_verified_index(self)
        if observed == expected:
            return
        for field_name in TextCacheIndexIdentity.__dataclass_fields__:
            if getattr(observed, field_name) != getattr(expected, field_name):
                raise ValueError(
                    "text cache index immutable identity mismatch: "
                    f"{field_name}"
                )
        raise ValueError("text cache index immutable identity mismatch")

    def __enter__(self) -> TextCacheIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._mapping = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def _validate_binary(self) -> None:
        if len(self._mapping) < _HEADER_SIZE:
            raise ValueError("text cache index is shorter than its header")
        (
            magic,
            schema_version,
            header_size,
            record_size,
            flags,
            record_count,
            contract_digest,
            prompt_set_digest,
            records_digest,
            reserved,
        ) = _HEADER.unpack_from(self._mapping, 0)
        if magic != _MAGIC:
            raise ValueError("text cache index magic is invalid")
        if schema_version != TEXT_CACHE_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported text cache binary schema")
        if (
            header_size != _HEADER_SIZE
            or record_size != _RECORD_SIZE
            or flags != _FLAGS
            or reserved != _RESERVED
        ):
            raise ValueError("text cache index header is invalid")
        if record_count != self.descriptor["record_count"]:
            raise ValueError("text cache index record count mismatch")
        expected_size = _HEADER_SIZE + record_count * _RECORD_SIZE
        if len(self._mapping) != expected_size:
            raise ValueError("text cache index byte length is invalid")
        if contract_digest.hex() != self.descriptor["contract_sha256"]:
            raise ValueError("text cache index contract digest mismatch")
        if prompt_set_digest.hex() != self.descriptor["prompt_set_sha256"]:
            raise ValueError("text cache index prompt-set digest mismatch")

        records_view = memoryview(self._mapping)[_HEADER_SIZE:]
        try:
            if hashlib.sha256(records_view).digest() != records_digest:
                raise ValueError("text cache index record digest mismatch")
        finally:
            records_view.release()

        observed_prompt_digest = hashlib.sha256()
        observed_prompt_digest.update(_PROMPT_SET_DOMAIN)
        previous: bytes | None = None
        for index in range(record_count):
            prompt_digest, payload_digest, size_bytes = self._record_tuple(index)
            if previous is not None and prompt_digest <= previous:
                raise ValueError(
                    "text cache index records must be strictly sorted and unique"
                )
            if payload_digest == b"\0" * 32 or size_bytes <= 0:
                raise ValueError("text cache index record is invalid")
            observed_prompt_digest.update(prompt_digest)
            previous = prompt_digest
        if observed_prompt_digest.hexdigest() != self.descriptor[
            "prompt_set_sha256"
        ]:
            raise ValueError("text cache index prompt set is invalid")

    def _record_tuple(self, index: int) -> tuple[bytes, bytes, int]:
        if index < 0 or index >= self.descriptor["record_count"]:
            raise IndexError("text cache index record is out of range")
        offset = _HEADER_SIZE + index * _RECORD_SIZE
        return _RECORD.unpack_from(self._mapping, offset)

    def __len__(self) -> int:
        return int(self.descriptor["record_count"])

    def prompt_digests(self) -> Iterable[bytes]:
        for index in range(len(self)):
            yield self._record_tuple(index)[0]

    def require_exact_prompts(self, prompts: Iterable[str]) -> None:
        expected = _normalized_prompts(prompts)
        if len(expected) != len(self):
            raise ValueError("selected prompt count differs from text cache index")
        if _prompt_set_digest(row[0] for row in expected) != self.descriptor[
            "prompt_set_sha256"
        ]:
            raise ValueError("selected prompt set differs from text cache index")
        for index, (expected_digest, _) in enumerate(expected):
            if self._record_tuple(index)[0] != expected_digest:
                raise ValueError("selected prompts differ from text cache index records")

    def lookup_sha256(self, prompt_digest: str) -> TextCacheIndexRecord:
        wanted_hex = _require_sha256(prompt_digest, field="prompt_sha256")
        wanted = bytes.fromhex(wanted_hex)
        lower = 0
        upper = len(self)
        while lower < upper:
            middle = (lower + upper) // 2
            current = self._record_tuple(middle)[0]
            if current < wanted:
                lower = middle + 1
            else:
                upper = middle
        if lower >= len(self):
            raise KeyError(f"prompt is absent from text cache index: {wanted_hex}")
        current, payload_digest, size_bytes = self._record_tuple(lower)
        if current != wanted:
            raise KeyError(f"prompt is absent from text cache index: {wanted_hex}")
        return TextCacheIndexRecord(
            prompt_sha256=wanted_hex,
            payload_sha256=payload_digest.hex(),
            size_bytes=int(size_bytes),
        )

    def lookup_prompt(self, prompt: str) -> TextCacheIndexRecord:
        return self.lookup_sha256(prompt_sha256(prompt))

    def payload_relative_path(self, record: TextCacheIndexRecord) -> str:
        if not isinstance(record, TextCacheIndexRecord):
            raise TypeError("record must be a TextCacheIndexRecord")
        return record.prompt_sha256 + self.descriptor["filename_suffix"]

    def load_verified_payload(
        self,
        prompt: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Hash bytes before deserializing and validate the tensor contract."""

        record = self.lookup_prompt(prompt)
        cache_root = Path(self.descriptor["cache_root"])
        relative = self.payload_relative_path(record)
        payload_bytes = _read_stable_regular_file(cache_root, relative)
        if len(payload_bytes) != record.size_bytes:
            raise ValueError("text cache payload size differs from its index")
        if _sha256_bytes(payload_bytes) != record.payload_sha256:
            raise ValueError("text cache payload SHA256 differs from its index")

        try:
            payload = torch.load(
                io.BytesIO(payload_bytes),
                map_location=map_location,
                weights_only=True,
            )
        except Exception as error:
            raise ValueError("verified text cache payload cannot be deserialized") from error
        if not isinstance(payload, Mapping) or set(payload) != {"context", "mask"}:
            raise ValueError("text cache payload must contain exactly context and mask")
        context = payload.get("context")
        mask = payload.get("mask")
        if not isinstance(context, torch.Tensor) or context.ndim != 2:
            raise ValueError("text cache context must be a 2D tensor")
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
            raise ValueError("text cache mask must be a 1D tensor")
        context_len = self.descriptor["context_len"]
        if context.shape[0] != context_len or mask.shape[0] != context_len:
            raise ValueError("text cache tensors disagree with context_len")
        if context.shape[1] <= 0 or context.dtype != torch.bfloat16:
            raise ValueError("text cache context must be non-empty bfloat16")
        if mask.dtype != torch.bool:
            raise ValueError("text cache mask must have bool dtype")
        return {"context": context, "mask": mask}


def _packed_cache_record(
    cache_root: Path,
    filename_suffix: str,
    prompt_digest: bytes,
) -> bytes:
    relative = prompt_digest.hex() + filename_suffix
    payload = _read_stable_regular_file(cache_root, relative)
    if not payload:
        raise ValueError(f"text cache payload is empty: {relative}")
    return _RECORD.pack(
        prompt_digest,
        hashlib.sha256(payload).digest(),
        len(payload),
    )


def _ordered_cache_records(
    *,
    cache_root: Path,
    filename_suffix: str,
    prompt_digests: Iterable[bytes],
    workers: int,
) -> Iterable[bytes]:
    """Hash concurrently while yielding records in canonical prompt order."""

    if workers == 1:
        for prompt_digest in prompt_digests:
            yield _packed_cache_record(
                cache_root,
                filename_suffix,
                prompt_digest,
            )
        return

    digest_iterator = iter(prompt_digests)
    maximum_pending = workers * 2
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="fastwam-text-cache-index",
    )
    pending: deque[Future[bytes]] = deque()

    def submit_one() -> bool:
        try:
            prompt_digest = next(digest_iterator)
        except StopIteration:
            return False
        pending.append(
            executor.submit(
                _packed_cache_record,
                cache_root,
                filename_suffix,
                prompt_digest,
            )
        )
        return True

    try:
        while len(pending) < maximum_pending and submit_one():
            pass
        while pending:
            record = pending.popleft().result()
            submit_one()
            yield record
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def build_text_cache_index(
    *,
    cache_root: str | Path,
    prompts: Iterable[str],
    context_len: int,
    prompt_template: str,
    filename_suffix: str,
    index_path: str | Path,
    descriptor_path: str | Path | None = None,
    overwrite: bool = False,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Build an atomic index from exact prompt-derived paths only.

    Existing payloads are read exactly once.  No directory enumeration is
    performed.  The descriptor is published last and acts as the commit marker.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    root = Path(cache_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("cache_root must be a directory")
    normalized = _normalized_prompts(prompts)
    contract = _contract_payload(
        context_len=context_len,
        prompt_template=prompt_template,
        filename_suffix=filename_suffix,
    )
    contract_sha = text_cache_contract_sha256(**contract)
    prompt_set = _prompt_set_digest(row[0] for row in normalized)

    destination = Path(index_path).expanduser().resolve()
    descriptor_destination = (
        Path(descriptor_path).expanduser().resolve()
        if descriptor_path is not None
        else destination.with_suffix(destination.suffix + ".json")
    )
    if destination == descriptor_destination:
        raise ValueError("index_path and descriptor_path must be different files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_index = destination.relative_to(descriptor_destination.parent)
    except ValueError as error:
        raise ValueError("index_path must be inside descriptor_path.parent") from error
    relative_index_text = _safe_relative_path(
        relative_index.as_posix(), field="index relative path"
    )
    if not overwrite and (destination.exists() or descriptor_destination.exists()):
        raise FileExistsError("text cache index output already exists")

    temporary = destination.parent / f".{destination.name}.tmp.{uuid.uuid4().hex}"
    records_digest = hashlib.sha256()
    try:
        with temporary.open("xb+") as handle:
            handle.write(b"\0" * _HEADER_SIZE)
            total = len(normalized)
            records = _ordered_cache_records(
                cache_root=root,
                filename_suffix=contract["filename_suffix"],
                prompt_digests=(row[0] for row in normalized),
                workers=workers,
            )
            for completed, record in enumerate(records, start=1):
                handle.write(record)
                records_digest.update(record)
                if progress is not None:
                    progress(completed, total)
            header = _HEADER.pack(
                _MAGIC,
                TEXT_CACHE_INDEX_SCHEMA_VERSION,
                _HEADER_SIZE,
                _RECORD_SIZE,
                _FLAGS,
                len(normalized),
                bytes.fromhex(contract_sha),
                bytes.fromhex(prompt_set),
                records_digest.digest(),
                _RESERVED,
            )
            handle.seek(0)
            handle.write(header)
            handle.flush()
            os.fsync(handle.fileno())

        index_size = temporary.stat().st_size
        index_sha = _sha256_file(temporary)
        descriptor: dict[str, Any] = {
            "schema_version": TEXT_CACHE_INDEX_SCHEMA_VERSION,
            "kind": TEXT_CACHE_INDEX_KIND,
            "cache_root": str(root),
            **contract,
            "record_count": len(normalized),
            "prompt_set_sha256": prompt_set,
            "contract_sha256": contract_sha,
            "index": {
                "relative_path": relative_index_text,
                "size_bytes": index_size,
                "sha256": index_sha,
            },
        }
        descriptor["descriptor_sha256"] = canonical_text_cache_descriptor_sha256(
            descriptor
        )
        validated_descriptor = _validate_descriptor_mapping(descriptor)

        if not overwrite and (destination.exists() or descriptor_destination.exists()):
            raise FileExistsError("text cache index output appeared during generation")
        os.replace(temporary, destination)
        _atomic_write(
            descriptor_destination,
            _descriptor_json_bytes(validated_descriptor),
        )
        return validated_descriptor
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "TEXT_CACHE_INDEX_KIND",
    "TEXT_CACHE_INDEX_MODE",
    "TEXT_CACHE_INDEX_SCHEMA_VERSION",
    "TextCacheIndex",
    "TextCacheIndexIdentity",
    "TextCacheIndexRecord",
    "build_text_cache_index",
    "canonical_text_cache_descriptor_sha256",
    "load_text_cache_index_descriptor",
    "prompt_set_sha256",
    "prompt_sha256",
    "text_cache_contract_sha256",
]
