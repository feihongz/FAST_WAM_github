"""Fail-closed guards for the selected files in a Stage 3 data manifest.

The canonical manifest binds file contents, but a long-running Stage 2 job
also needs to detect files that change after the initial full validation.  A
snapshot records cheap filesystem identity for every selected file.  Callers
can then use stat-only checks at frequent boundaries and a full SHA256 check
before publishing durable outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any

from fastwam.alignment.data_identity import (
    DATA_MANIFEST_SCHEMA_VERSION,
    canonical_data_manifest_sha256,
    require_supported_data_manifest_header,
)
from fastwam.alignment.text_cache_index import TEXT_CACHE_INDEX_MODE


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
    global_file_keys: tuple[SourceFileKey, ...]
    episode_file_keys: Mapping[
        tuple[int, int], tuple[SourceFileKey, ...]
    ] = field(repr=False)
    _files_by_key: Mapping[SourceFileKey, SourceFileSnapshot] = field(
        repr=False,
        compare=False,
    )

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

    def keys_for_sample_identities(
        self,
        identities: Iterable[Mapping[str, Any]],
    ) -> tuple[SourceFileKey, ...]:
        """Select global files and exact episode files touched by samples."""

        return source_keys_for_sample_identities(self, identities)


@dataclass(frozen=True, slots=True)
class SourceStatGuard:
    """Callable full guard with an efficient episode-scoped chunk API."""

    snapshot: SelectedSourceSnapshot
    fixed_keys: tuple[SourceFileKey, ...] | None = None

    def __post_init__(self) -> None:
        selected = None if self.fixed_keys is None else tuple(self.fixed_keys)
        _selected_snapshots(self.snapshot, selected)
        object.__setattr__(self, "fixed_keys", selected)

    def __call__(self) -> None:
        """Preserve the legacy no-argument full/fixed stat guard API."""

        recheck_source_stats(self.snapshot, keys=self.fixed_keys)

    def check_sample_identities(
        self,
        identities: Iterable[Mapping[str, Any]],
    ) -> None:
        """Check only files needed by one label chunk.

        A guard explicitly constructed with ``fixed_keys`` retains that exact
        legacy selection.  The normal formal-job guard instead resolves the
        chunk's touched episodes plus every manifest-global identity file.
        """

        keys = self.fixed_keys
        if keys is None:
            keys = source_keys_for_sample_identities(self.snapshot, identities)
        recheck_source_stats(self.snapshot, keys=keys)


def _require_manifest_identity(data_manifest: Mapping[str, Any]) -> str:
    if not isinstance(data_manifest, Mapping):
        raise TypeError("data_manifest must be a mapping")
    require_supported_data_manifest_header(data_manifest)
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

    schema_version = require_supported_data_manifest_header(data_manifest)
    normalization = data_manifest.get("normalization_stats")
    if not isinstance(normalization, Mapping):
        raise ValueError("data manifest normalization_stats must be a mapping")
    text_cache = data_manifest.get("text_embedding_cache")
    if not isinstance(text_cache, Mapping):
        raise ValueError("data manifest text_embedding_cache must be a mapping")
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        external_inventories = (
            ("text_embedding_cache", text_cache),
            ("normalization_stats", normalization),
        )
    else:
        integrity = text_cache.get("integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("v2 text cache integrity must be a mapping")
        if integrity.get("mode") != TEXT_CACHE_INDEX_MODE:
            raise ValueError("unsupported v2 text cache integrity mode")
        index_files = integrity.get("files")
        if not isinstance(index_files, list) or len(index_files) != 2:
            raise ValueError(
                "v2 text cache integrity must select descriptor and binary index"
            )
        roles = [
            entry.get("role") if isinstance(entry, Mapping) else None
            for entry in index_files
        ]
        if roles != ["text_cache_index_descriptor", "text_cache_index"]:
            raise ValueError("v2 text cache integrity file roles are invalid")
        external_inventories = (
            ("text_cache_index", integrity),
            ("normalization_stats", normalization),
        )

    for anchor_index, (field, container) in enumerate(
        external_inventories, start=len(roots)
    ):
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


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _episode_index_list(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = tuple(
        _nonnegative_integer(item, field=f"{field} entry") for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique episode indices")
    return result


def _build_source_key_inventory(
    data_manifest: Mapping[str, Any],
    keyed_entries: tuple[
        tuple[str, int, Mapping[str, Any], SourceFileKey], ...
    ],
) -> tuple[
    tuple[SourceFileKey, ...],
    Mapping[tuple[int, int], tuple[SourceFileKey, ...]],
]:
    """Bind every dataset episode to its parquet and exact camera files.

    File ownership comes exclusively from canonical manifest fields.  Path
    names are deliberately ignored so a malformed or ambiguous inventory
    cannot silently weaken a chunk-scoped source check.
    """

    roots = data_manifest.get("dataset_roots")
    if not isinstance(roots, list):
        raise ValueError("data manifest dataset_roots must be a list")
    dataset_entries: dict[
        int, list[tuple[Mapping[str, Any], SourceFileKey]]
    ] = {index: [] for index in range(len(roots))}
    global_keys: list[SourceFileKey] = []
    for anchor_kind, anchor_index, entry, key in keyed_entries:
        if anchor_kind == "dataset":
            try:
                dataset_entries[anchor_index].append((entry, key))
            except KeyError as error:
                raise ValueError("selected dataset file has an invalid root") from error
        else:
            global_keys.append(key)

    episode_keys: dict[tuple[int, int], tuple[SourceFileKey, ...]] = {}
    for dataset_index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise ValueError("data manifest dataset root must be a mapping")
        selected_episodes = _episode_index_list(
            root.get("selected_episodes"),
            field=f"dataset_roots[{dataset_index}].selected_episodes",
        )
        boundaries = root.get("episode_boundaries")
        if not isinstance(boundaries, list):
            raise ValueError("data manifest episode_boundaries must be a list")
        boundary_episodes: list[int] = []
        for boundary in boundaries:
            if not isinstance(boundary, Mapping):
                raise ValueError("data manifest episode boundary must be a mapping")
            boundary_episodes.append(
                _nonnegative_integer(
                    boundary.get("episode_index"),
                    field="episode boundary episode_index",
                )
            )
        if tuple(boundary_episodes) != selected_episodes:
            raise ValueError(
                "data manifest selected episodes and episode boundaries disagree"
            )

        video_keys_value = root.get("video_keys")
        if not isinstance(video_keys_value, list) or any(
            not isinstance(value, str) or not value for value in video_keys_value
        ):
            raise ValueError("data manifest video_keys must be non-empty strings")
        video_keys = tuple(video_keys_value)
        if len(video_keys) != len(set(video_keys)):
            raise ValueError("data manifest video_keys must be unique")

        parquet_by_episode: dict[int, list[SourceFileKey]] = {
            episode: [] for episode in selected_episodes
        }
        video_by_episode: dict[int, dict[str, list[SourceFileKey]]] = {
            episode: {video_key: [] for video_key in video_keys}
            for episode in selected_episodes
        }
        for entry, key in dataset_entries[dataset_index]:
            role = entry.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("selected dataset file role must be non-empty")
            if "episode_index" not in entry:
                if role in {"parquet", "video"} or "video_key" in entry:
                    raise ValueError(
                        "episode parquet/video inventory is missing episode_index"
                    )
                global_keys.append(key)
                continue

            episode_index = _nonnegative_integer(
                entry.get("episode_index"),
                field="selected dataset file episode_index",
            )
            if episode_index not in parquet_by_episode:
                raise ValueError(
                    "selected dataset file names an unselected episode"
                )
            if role == "parquet":
                if "video_key" in entry:
                    raise ValueError("parquet episode inventory is ambiguous")
                parquet_by_episode[episode_index].append(key)
            elif role == "video":
                video_key = entry.get("video_key")
                if video_key not in video_by_episode[episode_index]:
                    raise ValueError(
                        "video episode inventory has an unknown video_key"
                    )
                video_by_episode[episode_index][video_key].append(key)
            else:
                raise ValueError(
                    "episode-owned selected file has an ambiguous role"
                )

        for episode_index in selected_episodes:
            parquet_keys = parquet_by_episode[episode_index]
            if len(parquet_keys) != 1:
                raise ValueError(
                    "episode inventory must contain exactly one parquet file"
                )
            selected_keys = list(parquet_keys)
            for video_key in video_keys:
                camera_keys = video_by_episode[episode_index][video_key]
                if len(camera_keys) != 1:
                    raise ValueError(
                        "episode inventory must contain exactly one file per video key"
                    )
                selected_keys.extend(camera_keys)
            episode_keys[(dataset_index, episode_index)] = tuple(selected_keys)

    assigned = set(global_keys)
    for keys in episode_keys.values():
        assigned.update(keys)
    all_keys = {key for _, _, _, key in keyed_entries}
    if assigned != all_keys:
        raise RuntimeError("source guard inventory did not classify every file")
    return tuple(global_keys), MappingProxyType(dict(episode_keys))


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
    keyed_entries: list[
        tuple[str, int, Mapping[str, Any], SourceFileKey]
    ] = []
    seen_keys: set[SourceFileKey] = set()
    selected_entries = _selected_file_entries(data_manifest)
    for anchor_kind, anchor_index, anchor, entry in selected_entries:
        relative = _safe_relative_path(entry.get("relative_path"))
        key = SourceFileKey(anchor_kind, anchor_index, relative)
        if key in seen_keys:
            raise ValueError(f"duplicate selected source file: {key}")
        seen_keys.add(key)
        keyed_entries.append((anchor_kind, anchor_index, entry, key))
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
    global_keys, episode_keys = _build_source_key_inventory(
        data_manifest,
        tuple(keyed_entries),
    )
    files = tuple(snapshots)
    return SelectedSourceSnapshot(
        data_manifest_sha256=manifest_sha256,
        files=files,
        global_file_keys=global_keys,
        episode_file_keys=episode_keys,
        _files_by_key=MappingProxyType({source.key: source for source in files}),
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
    by_key = snapshot._files_by_key
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"source guard keys are absent from snapshot: {missing}")
    return tuple(by_key[key] for key in requested)


def source_keys_for_sample_identities(
    snapshot: SelectedSourceSnapshot,
    identities: Iterable[Mapping[str, Any]],
) -> tuple[SourceFileKey, ...]:
    """Return global plus touched-episode keys without scanning all files."""

    if not isinstance(snapshot, SelectedSourceSnapshot):
        raise TypeError("snapshot must be a SelectedSourceSnapshot")
    touched: set[tuple[int, int]] = set()
    count = 0
    for identity in identities:
        count += 1
        if not isinstance(identity, Mapping):
            raise TypeError("sample identity must be a mapping")
        dataset_index = _nonnegative_integer(
            identity.get("dataset_index"),
            field="sample identity dataset_index",
        )
        episode_index = _nonnegative_integer(
            identity.get("episode_index"),
            field="sample identity episode_index",
        )
        pair = (dataset_index, episode_index)
        if pair not in snapshot.episode_file_keys:
            raise ValueError(
                "sample identity has no unambiguous source episode inventory"
            )
        touched.add(pair)
    if count == 0:
        raise ValueError("sample identities must not be empty")

    keys = list(snapshot.global_file_keys)
    for pair in sorted(touched):
        keys.extend(snapshot.episode_file_keys[pair])
    if len(keys) != len(set(keys)):
        raise RuntimeError("chunk source key selection produced duplicates")
    return tuple(keys)


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
) -> SourceStatGuard:
    """Return a legacy-callable guard with an episode-scoped chunk API."""

    selected_keys = None if keys is None else tuple(keys)
    return SourceStatGuard(snapshot=snapshot, fixed_keys=selected_keys)


__all__ = [
    "SelectedSourceSnapshot",
    "SourceDataDriftError",
    "SourceFileKey",
    "SourceFileSnapshot",
    "SourceStatGuard",
    "capture_selected_source_snapshot",
    "make_source_stat_guard",
    "recheck_source_content",
    "recheck_source_stats",
    "source_keys_for_sample_identities",
]
