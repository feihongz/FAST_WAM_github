"""Fail-closed guards for the selected files in a Stage 3 data manifest.

The canonical manifest binds file contents, but a long-running Stage 2 job
also needs to detect files that change after the initial full validation.  A
snapshot records cheap filesystem identity for every selected file.  Callers
can then use stat-only checks at frequent boundaries and a full SHA256 check
before publishing durable outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any

from fastwam.alignment.data_identity import (
    DATA_MANIFEST_KIND,
    DATA_MANIFEST_SCHEMA_VERSION,
    canonical_data_manifest_sha256,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


class SourceDataDriftError(RuntimeError):
    """A selected source file no longer matches its immutable snapshot."""


@dataclass(frozen=True, slots=True, order=True)
class SourceFileKey:
    """Stable address of one selected file within a data manifest."""

    anchor_kind: str
    anchor_index: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """Immutable content and filesystem identity for one selected file."""

    key: SourceFileKey
    anchor_path: Path
    resolved_path: Path
    expected_sha256: str
    st_dev: int
    st_ino: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class SelectedSourceSnapshot:
    """Immutable snapshot of every selected file in one data manifest."""

    data_manifest_sha256: str
    files: tuple[SourceFileSnapshot, ...]

    @property
    def file_keys(self) -> tuple[SourceFileKey, ...]:
        return tuple(file.key for file in self.files)

    def check_stats(
        self, keys: Iterable[SourceFileKey] | None = None
    ) -> None:
        """Cheaply recheck paths and filesystem identity without hashing."""

        recheck_source_stats(self, keys=keys)

    def check_content(
        self, keys: Iterable[SourceFileKey] | None = None
    ) -> None:
        """Recheck paths, filesystem identity, and full content SHA256."""

        recheck_source_content(self, keys=keys)


def _require_manifest_identity(data_manifest: Mapping[str, Any]) -> str:
    if not isinstance(data_manifest, Mapping):
        raise TypeError("data_manifest must be a mapping")
    if data_manifest.get("schema_version") != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported Stage 3 data manifest schema")
    if data_manifest.get("kind") != DATA_MANIFEST_KIND:
        raise ValueError("unsupported Stage 3 data manifest kind")
    recorded = data_manifest.get("manifest_sha256")
    if (
        not isinstance(recorded, str)
        or not _SHA256_PATTERN.fullmatch(recorded)
        or canonical_data_manifest_sha256(data_manifest) != recorded
    ):
        raise ValueError("Stage 3 data manifest canonical SHA256 mismatch")
    return recorded


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("selected source relative_path must be non-empty")
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"selected source path is not canonical: {value}")
    if relative.as_posix() != value:
        raise ValueError(f"selected source path is not canonical: {value}")
    return value


def _anchor_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise SourceDataDriftError(f"selected source anchor is unavailable: {lexical}") from error
    if not resolved.is_dir():
        raise SourceDataDriftError(
            f"selected source anchor is not a directory: {resolved}"
        )
    return resolved


def _selected_file_entries(
    data_manifest: Mapping[str, Any],
) -> tuple[tuple[str, int, Path, Mapping[str, Any]], ...]:
    result: list[tuple[str, int, Path, Mapping[str, Any]]] = []
    roots = data_manifest.get("dataset_roots")
    if not isinstance(roots, list):
        raise ValueError("data manifest dataset_roots must be a list")
    for dataset_index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise ValueError("data manifest dataset root must be a mapping")
        if root.get("dataset_index") != dataset_index:
            raise ValueError("data manifest dataset roots must be ordered by index")
        anchor = _anchor_path(
            root.get("root"), field=f"dataset_roots[{dataset_index}].root"
        )
        files = root.get("files")
        if not isinstance(files, list):
            raise ValueError("data manifest dataset file inventory must be a list")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("data manifest selected file must be a mapping")
            result.append(("dataset", dataset_index, anchor, entry))

    for anchor_index, field in enumerate(
        ("text_embedding_cache", "normalization_stats"), start=len(roots)
    ):
        container = data_manifest.get(field)
        if not isinstance(container, Mapping):
            raise ValueError(f"data manifest {field} must be a mapping")
        anchor = _anchor_path(container.get("root"), field=f"{field}.root")
        files = container.get("files")
        if not isinstance(files, list):
            raise ValueError(f"data manifest {field}.files must be a list")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("data manifest selected file must be a mapping")
            result.append((field, anchor_index, anchor, entry))
    if not result:
        raise ValueError("data manifest contains no selected source files")
    return tuple(result)


def _resolved_selected_path(anchor: Path, relative_path: str) -> Path:
    candidate = anchor / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(anchor)
    except (OSError, ValueError) as error:
        raise SourceDataDriftError(
            f"selected source path is missing or escapes its anchor: {candidate}"
        ) from error
    return resolved


def _stat_tuple(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _snapshot_stat_tuple(
    source: SourceFileSnapshot,
) -> tuple[int, int, int, int, int]:
    return (
        source.st_dev,
        source.st_ino,
        source.size_bytes,
        source.mtime_ns,
        source.ctime_ns,
    )


def capture_selected_source_snapshot(
    data_manifest: Mapping[str, Any],
) -> SelectedSourceSnapshot:
    """Capture cheap identity for the manifest's selected source files.

    This function deliberately does not hash file contents. Formal label and
    training entrypoints should capture immediately before their existing full
    manifest verification and call ``snapshot.check_stats()`` immediately
    afterward. This brackets the expensive verification without hashing twice.
    A caller without an existing full verification (such as merge) must call
    ``snapshot.check_content()`` before relying on the snapshot.
    """

    manifest_sha256 = _require_manifest_identity(data_manifest)
    snapshots: list[SourceFileSnapshot] = []
    seen_keys: set[SourceFileKey] = set()
    for anchor_kind, anchor_index, anchor, entry in _selected_file_entries(
        data_manifest
    ):
        relative = _safe_relative_path(entry.get("relative_path"))
        key = SourceFileKey(anchor_kind, anchor_index, relative)
        if key in seen_keys:
            raise ValueError(f"duplicate selected source file: {key}")
        seen_keys.add(key)
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
            expected_sha256
        ):
            raise ValueError(f"invalid selected source SHA256: {key}")
        expected_size = entry.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError(f"invalid selected source size: {key}")
        resolved = _resolved_selected_path(anchor, relative)
        try:
            current = resolved.stat()
        except OSError as error:
            raise SourceDataDriftError(
                f"selected source file is unavailable: {resolved}"
            ) from error
        if not stat.S_ISREG(current.st_mode):
            raise SourceDataDriftError(
                f"selected source is not a regular file: {resolved}"
            )
        if current.st_size != expected_size:
            raise SourceDataDriftError(
                f"selected source size drifted from manifest: {resolved}"
            )
        snapshots.append(
            SourceFileSnapshot(
                key=key,
                anchor_path=anchor,
                resolved_path=resolved,
                expected_sha256=expected_sha256,
                st_dev=int(current.st_dev),
                st_ino=int(current.st_ino),
                size_bytes=int(current.st_size),
                mtime_ns=int(current.st_mtime_ns),
                ctime_ns=int(current.st_ctime_ns),
            )
        )
    return SelectedSourceSnapshot(
        data_manifest_sha256=manifest_sha256,
        files=tuple(snapshots),
    )


def _selected_snapshots(
    snapshot: SelectedSourceSnapshot,
    keys: Iterable[SourceFileKey] | None,
) -> tuple[SourceFileSnapshot, ...]:
    if not isinstance(snapshot, SelectedSourceSnapshot):
        raise TypeError("snapshot must be a SelectedSourceSnapshot")
    if keys is None:
        return snapshot.files
    requested = tuple(keys)
    if not requested:
        raise ValueError("source guard keys must not be empty")
    if any(not isinstance(key, SourceFileKey) for key in requested):
        raise TypeError("source guard keys must be SourceFileKey values")
    if len(requested) != len(set(requested)):
        raise ValueError("source guard keys must be unique")
    by_key = {source.key: source for source in snapshot.files}
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"source guard keys are absent from snapshot: {missing}")
    return tuple(by_key[key] for key in requested)


def _resolve_current(source: SourceFileSnapshot) -> Path:
    current = _resolved_selected_path(source.anchor_path, source.key.relative_path)
    if current != source.resolved_path:
        raise SourceDataDriftError(
            "selected source resolved path drifted: "
            f"{source.resolved_path} -> {current}"
        )
    return current


def _require_snapshot_stat(
    source: SourceFileSnapshot, current: os.stat_result
) -> None:
    if not stat.S_ISREG(current.st_mode):
        raise SourceDataDriftError(
            f"selected source is no longer a regular file: {source.resolved_path}"
        )
    if _stat_tuple(current) != _snapshot_stat_tuple(source):
        raise SourceDataDriftError(
            f"selected source filesystem identity drifted: {source.resolved_path}"
        )


def recheck_source_stats(
    snapshot: SelectedSourceSnapshot,
    *,
    keys: Iterable[SourceFileKey] | None = None,
) -> None:
    """Fail if any selected file's path or cheap stat identity changed."""

    for source in _selected_snapshots(snapshot, keys):
        current_path = _resolve_current(source)
        try:
            current = current_path.stat()
        except OSError as error:
            raise SourceDataDriftError(
                f"selected source file is unavailable: {current_path}"
            ) from error
        _require_snapshot_stat(source, current)


def recheck_source_content(
    snapshot: SelectedSourceSnapshot,
    *,
    keys: Iterable[SourceFileKey] | None = None,
) -> None:
    """Fail unless every selected file retains its stat identity and SHA256."""

    selected = _selected_snapshots(snapshot, keys)
    for source in selected:
        current_path = _resolve_current(source)
        digest = hashlib.sha256()
        try:
            with current_path.open("rb") as handle:
                before = os.fstat(handle.fileno())
                _require_snapshot_stat(source, before)
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                after = os.fstat(handle.fileno())
        except SourceDataDriftError:
            raise
        except OSError as error:
            raise SourceDataDriftError(
                f"selected source file cannot be read: {current_path}"
            ) from error
        _require_snapshot_stat(source, after)
        # Re-resolve the lexical anchor/relative path after hashing. This
        # closes the practical symlink/path switch window while the open file
        # descriptor was being read.
        after_path = _resolve_current(source)
        try:
            after_path_stat = after_path.stat()
        except OSError as error:
            raise SourceDataDriftError(
                f"selected source file is unavailable after hashing: {after_path}"
            ) from error
        _require_snapshot_stat(source, after_path_stat)

        if digest.hexdigest() != source.expected_sha256:
            raise SourceDataDriftError(
                f"selected source content SHA256 drifted: {current_path}"
            )

    # Hashing a large inventory is not atomic: a file checked early could
    # drift while a later file is being read. Close that window with one final
    # stat sweep over the complete selected set.
    recheck_source_stats(
        snapshot,
        keys=tuple(source.key for source in selected),
    )


def make_source_stat_guard(
    snapshot: SelectedSourceSnapshot,
    *,
    keys: Iterable[SourceFileKey] | None = None,
) -> Callable[[], None]:
    """Return a fixed no-argument stat guard suitable for long-running jobs."""

    selected_keys = None if keys is None else tuple(keys)
    # Validate the subset eagerly so configuration errors fail before work.
    _selected_snapshots(snapshot, selected_keys)

    def guard() -> None:
        recheck_source_stats(snapshot, keys=selected_keys)

    return guard


__all__ = [
    "SelectedSourceSnapshot",
    "SourceDataDriftError",
    "SourceFileKey",
    "SourceFileSnapshot",
    "capture_selected_source_snapshot",
    "make_source_stat_guard",
    "recheck_source_content",
    "recheck_source_stats",
]
