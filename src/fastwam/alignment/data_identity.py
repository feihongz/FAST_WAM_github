"""Canonical data identity for formal Stage 3 Alignment training.

The manifest deliberately describes only files selected by an already-created
``RobotVideoDataset``.  It never recursively scans a dataset directory.  This
keeps the identity precise (selected episodes, rather than whatever happens to
be present on disk) and makes validation fail closed when any selected input
drifts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .text_cache_index import (
    TEXT_CACHE_INDEX_MODE,
    TextCacheIndex,
    load_text_cache_index_descriptor,
)


DATA_MANIFEST_SCHEMA_VERSION = 1
DATA_MANIFEST_KIND = "stage3_libero_data_manifest"
DATA_MANIFEST_V2_SCHEMA_VERSION = 2
DATA_MANIFEST_V2_KIND = "stage3_robot_video_data_manifest"
DATA_MANIFEST_SHA256_KEY = "manifest_sha256"

LEROBOT_META_PATHS = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
VIDEO_DECODER_IMPLEMENTATION = (
    "fastwam.datasets.lerobot.lerobot.datasets.video_utils.decode_video_frames"
)
TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE = ".t5_len{context_len}.wan22ti2v5b.pt"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
        raise ValueError("data manifest must be canonical-JSON serializable") from error
    return encoded.encode("utf-8")


def canonical_data_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest, excluding its self-authenticating SHA field."""

    if not isinstance(manifest, Mapping):
        raise TypeError("data manifest must be a mapping")
    payload = dict(manifest)
    payload.pop(DATA_MANIFEST_SHA256_KEY, None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, *, name: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _safe_relative_path(value: str | Path) -> str:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"selected file path must be relative: {value}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"selected file path is not canonical: {value}")
    return relative.as_posix()


def _selected_file(
    *,
    anchor: Path,
    anchor_key: tuple[str, int | None],
    relative_path: str | Path,
    role: str,
    sha_lookup: Mapping[tuple[str, int | None, str], str] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relative = _safe_relative_path(relative_path)
    candidate = anchor / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"selected data file does not exist: {candidate}")

    resolved_anchor = anchor.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    try:
        resolved_candidate.relative_to(resolved_anchor)
    except ValueError as error:
        raise ValueError(
            f"selected data file escapes its manifest root: {candidate}"
        ) from error

    key = (anchor_key[0], anchor_key[1], relative)
    if sha_lookup is None:
        sha256 = _sha256_file(resolved_candidate)
    else:
        try:
            sha256 = sha_lookup[key]
        except KeyError as error:
            raise ValueError(
                f"data manifest does not contain selected file: {key}"
            ) from error
        if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
            raise ValueError(f"invalid selected-file SHA256 for {key}")

    entry: dict[str, Any] = {
        "role": role,
        "relative_path": relative,
        "size_bytes": resolved_candidate.stat().st_size,
        "sha256": sha256,
    }
    if extra:
        entry.update(extra)
    return entry


def _dataset_parts(dataset: Any) -> tuple[Any, Any, list[Any]]:
    try:
        lerobot_dataset = dataset.lerobot_dataset
        multi_dataset = lerobot_dataset.multi_dataset
        datasets = list(multi_dataset._datasets)
        configured_dirs = list(lerobot_dataset.dataset_dirs)
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "expected an instantiated RobotVideoDataset with LeRobot internals"
        ) from error
    if not datasets:
        raise ValueError("RobotVideoDataset contains no underlying datasets")
    if len(configured_dirs) != len(datasets):
        raise ValueError("configured dataset roots disagree with instantiated datasets")
    configured_roots = [str(Path(path).expanduser().resolve()) for path in configured_dirs]
    instantiated_roots = [
        str(Path(part.root).expanduser().resolve()) for part in datasets
    ]
    if configured_roots != instantiated_roots:
        raise ValueError(
            "ordered configured roots disagree with instantiated dataset roots"
        )
    return lerobot_dataset, multi_dataset, datasets


def _selected_episodes(part: Any) -> list[int]:
    try:
        metadata_episodes = part.meta.episodes
    except AttributeError as error:
        raise TypeError("underlying dataset has no episode metadata") from error
    selected = (
        list(part.episodes)
        if part.episodes is not None
        else list(metadata_episodes.keys())
    )
    result = [
        _integer(value, name="selected episode index") for value in selected
    ]
    if len(result) != len(set(result)):
        raise ValueError("selected episode indices must be unique")
    missing = [index for index in result if index not in metadata_episodes]
    if missing:
        raise ValueError(f"selected episodes are absent from metadata: {missing}")
    return result


def _episode_boundaries(part: Any, episodes: list[int]) -> list[dict[str, int]]:
    try:
        starts = list(part.episode_data_index["from"])
        ends = list(part.episode_data_index["to"])
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError("underlying dataset has no episode boundary index") from error
    if len(starts) != len(episodes) or len(ends) != len(episodes):
        raise ValueError("episode boundaries disagree with selected episodes")

    boundaries: list[dict[str, int]] = []
    expected_start = 0
    for position, (episode, raw_start, raw_end) in enumerate(
        zip(episodes, starts, ends, strict=True)
    ):
        start = _integer(raw_start, name=f"episode boundary from[{position}]")
        end = _integer(raw_end, name=f"episode boundary to[{position}]")
        if start != expected_start or end <= start:
            raise ValueError("episode boundaries must be contiguous and non-empty")
        metadata_length = _integer(
            part.meta.episodes[episode].get("length"),
            name=f"metadata episode length[{episode}]",
        )
        if end - start != metadata_length:
            raise ValueError(
                f"episode boundary length disagrees with metadata for episode {episode}"
            )
        boundaries.append(
            {
                "episode_index": episode,
                "from": start,
                "to": end,
                "length": end - start,
            }
        )
        expected_start = end

    num_frames = _integer(part.num_frames, name="underlying dataset num_frames")
    if expected_start != num_frames:
        raise ValueError("episode boundaries disagree with dataset num_frames")
    return boundaries


def _selected_prompt_tasks(part: Any) -> list[str]:
    """Return tasks used by ``LeRobotDataset.__getitem__`` in frame order.

    ``meta/episodes.jsonl`` can list coarse-task and quality labels in addition
    to the actual prompt task.  The frame-level ``task_index`` column is the
    authoritative source used by the loader, so using the broader episode list
    would incorrectly require unrelated text-cache files.
    """

    try:
        hf_dataset = part.hf_dataset
        task_table = part.meta.tasks
    except AttributeError as error:
        raise TypeError(
            "underlying dataset must expose its selected task_index column"
        ) from error

    unique = getattr(hf_dataset, "unique", None)
    if unique is None:
        try:
            raw_indices = hf_dataset["task_index"]
        except (KeyError, TypeError) as error:
            raise TypeError(
                "underlying dataset must expose its selected task_index column"
            ) from error
    else:
        if not callable(unique):
            raise TypeError("hf_dataset.unique must be callable")
        try:
            raw_indices = unique("task_index")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("hf_dataset.unique('task_index') failed") from error
    try:
        iterator = iter(raw_indices)
    except TypeError as error:
        raise TypeError("selected task_index values must be iterable") from error
    indices = [
        _integer(value, name="selected frame task_index") for value in iterator
    ]
    if not indices:
        raise ValueError("selected dataset contains no frame task indices")
    tasks: list[str] = []
    for task_index in dict.fromkeys(indices):
        try:
            task = task_table[task_index]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"selected task_index is absent from metadata: {task_index}"
            ) from error
        if not isinstance(task, str) or not task:
            raise ValueError(f"selected task {task_index} is not a non-empty string")
        tasks.append(task)
    return tasks


def selected_text_cache_prompts(dataset: Any) -> tuple[str, ...]:
    """Return the exact unique prompts selected by a RobotVideoDataset."""

    _, _, parts = _dataset_parts(dataset)
    override = dataset.override_instruction
    if override is not None:
        if not isinstance(override, str) or not override:
            raise ValueError("override_instruction must be a non-empty string")
        tasks = [override]
    else:
        tasks = []
        for part in parts:
            tasks.extend(_selected_prompt_tasks(part))
    unique_tasks = list(dict.fromkeys(tasks))
    if not unique_tasks:
        raise ValueError("selected dataset contains no task instructions")
    return tuple(
        DEFAULT_PROMPT_TEMPLATE.format(task=task) for task in unique_tasks
    )


def _sampling_contract(dataset: Any, lerobot_dataset: Any) -> dict[str, Any]:
    num_frames = _integer(dataset.num_frames, name="RobotVideoDataset num_frames")
    obs_size = _integer(lerobot_dataset.obs_size, name="LeRobot obs_size")
    if num_frames <= 1 or num_frames != obs_size:
        raise ValueError("RobotVideoDataset num_frames disagrees with LeRobot obs_size")
    action_ratio = _integer(
        dataset.action_video_freq_ratio,
        name="action_video_freq_ratio",
    )
    stride = _integer(
        lerobot_dataset.global_sample_stride,
        name="global_sample_stride",
    )
    video_indices = [
        _integer(value, name="video sample index")
        for value in dataset.video_sample_indices
    ]
    if action_ratio <= 0 or stride <= 0:
        raise ValueError("sampling strides must be positive")
    if video_indices != list(range(0, num_frames, action_ratio)):
        raise ValueError("video sample indices disagree with the action/video ratio")
    return {
        "num_frames": num_frames,
        "global_sample_stride": stride,
        "action_video_freq_ratio": action_ratio,
        "video_sample_indices": video_indices,
    }


def require_supported_data_manifest_header(manifest: Mapping[str, Any]) -> int:
    if not isinstance(manifest, Mapping):
        raise TypeError("data manifest must be a mapping")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("data manifest schema_version must be an integer")
    pair = (schema_version, manifest.get("kind"))
    if pair == (DATA_MANIFEST_SCHEMA_VERSION, DATA_MANIFEST_KIND):
        return DATA_MANIFEST_SCHEMA_VERSION
    if pair == (DATA_MANIFEST_V2_SCHEMA_VERSION, DATA_MANIFEST_V2_KIND):
        return DATA_MANIFEST_V2_SCHEMA_VERSION
    raise ValueError(
        "unsupported Stage 3 data manifest schema/kind pair: "
        f"{pair!r}"
    )


def _sha_lookup_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int | None, str], str]:
    lookup: dict[tuple[str, int | None, str], str] = {}

    def add_files(
        anchor_type: str,
        anchor_index: int | None,
        files: Any,
    ) -> None:
        if not isinstance(files, list):
            raise ValueError("data manifest file inventory must be a list")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("data manifest file entry must be a mapping")
            relative = entry.get("relative_path")
            sha256 = entry.get("sha256")
            if not isinstance(relative, str):
                raise ValueError("data manifest file has no relative path")
            key = (anchor_type, anchor_index, relative)
            if key in lookup:
                raise ValueError(f"duplicate data manifest file entry: {key}")
            if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
                raise ValueError(f"invalid data manifest file SHA256: {key}")
            lookup[key] = sha256

    roots = manifest.get("dataset_roots")
    if not isinstance(roots, list):
        raise ValueError("data manifest dataset_roots must be a list")
    for index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise ValueError("data manifest dataset root must be a mapping")
        add_files("dataset", index, root.get("files"))

    text_cache = manifest.get("text_embedding_cache")
    normalization = manifest.get("normalization_stats")
    if not isinstance(text_cache, Mapping) or not isinstance(
        normalization, Mapping
    ):
        raise ValueError("data manifest external file roots are invalid")
    schema_version = require_supported_data_manifest_header(manifest)
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        add_files("text_cache", None, text_cache.get("files"))
    else:
        integrity = text_cache.get("integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("v2 text cache integrity must be a mapping")
        if integrity.get("mode") != TEXT_CACHE_INDEX_MODE:
            raise ValueError("unsupported v2 text cache integrity mode")
        add_files("text_cache_index", None, integrity.get("files"))
    add_files("normalization_stats", None, normalization.get("files"))
    return lookup


def _build_robot_video_dataset_manifest(
    dataset: Any,
    *,
    normalization_stats_path: str | Path,
    sha_lookup: Mapping[tuple[str, int | None, str], str] | None,
    schema_version: int,
    text_cache_index_descriptor_path: str | Path | None,
) -> dict[str, Any]:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("data manifest schema_version must be an integer")
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        manifest_kind = DATA_MANIFEST_KIND
        if text_cache_index_descriptor_path is not None:
            raise ValueError("v1 data manifests do not accept a text cache index")
    elif schema_version == DATA_MANIFEST_V2_SCHEMA_VERSION:
        manifest_kind = DATA_MANIFEST_V2_KIND
        if text_cache_index_descriptor_path is None:
            raise ValueError("v2 data manifests require a text cache index descriptor")
    else:
        raise ValueError("unsupported Stage 3 data manifest schema")
    lerobot_dataset, _, parts = _dataset_parts(dataset)
    sampling = _sampling_contract(dataset, lerobot_dataset)

    dataset_roots: list[dict[str, Any]] = []
    ordered_tasks: list[str] = []
    decoder_datasets: list[dict[str, Any]] = []
    total_frames = 0
    for dataset_index, part in enumerate(parts):
        root = Path(part.root).expanduser().resolve(strict=True)
        episodes = _selected_episodes(part)
        boundaries = _episode_boundaries(part, episodes)
        num_frames = _integer(part.num_frames, name="underlying dataset num_frames")
        total_frames += num_frames

        files: list[dict[str, Any]] = []
        for relative in LEROBOT_META_PATHS:
            files.append(
                _selected_file(
                    anchor=root,
                    anchor_key=("dataset", dataset_index),
                    relative_path=relative,
                    role="metadata",
                    sha_lookup=sha_lookup,
                )
            )
        try:
            video_keys = list(part.meta.video_keys)
        except (AttributeError, TypeError) as error:
            raise TypeError("underlying dataset has no video key metadata") from error
        if not video_keys:
            raise ValueError("Stage 3 Libero data must contain MP4 video features")
        if len(video_keys) != len(set(video_keys)) or not all(
            isinstance(key, str) and key for key in video_keys
        ):
            raise ValueError("underlying dataset video keys are invalid")

        seen_relative_paths = {entry["relative_path"] for entry in files}
        for episode in episodes:
            parquet_path = part.meta.get_data_file_path(episode)
            parquet_relative = _safe_relative_path(parquet_path)
            if parquet_relative in seen_relative_paths:
                raise ValueError(f"duplicate selected file: {parquet_relative}")
            seen_relative_paths.add(parquet_relative)
            files.append(
                _selected_file(
                    anchor=root,
                    anchor_key=("dataset", dataset_index),
                    relative_path=parquet_relative,
                    role="parquet",
                    sha_lookup=sha_lookup,
                    extra={"episode_index": episode},
                )
            )
            for video_key in video_keys:
                video_path = part.meta.get_video_file_path(episode, video_key)
                video_relative = _safe_relative_path(video_path)
                if video_relative in seen_relative_paths:
                    raise ValueError(f"duplicate selected file: {video_relative}")
                seen_relative_paths.add(video_relative)
                files.append(
                    _selected_file(
                        anchor=root,
                        anchor_key=("dataset", dataset_index),
                        relative_path=video_relative,
                        role="video",
                        sha_lookup=sha_lookup,
                        extra={
                            "episode_index": episode,
                            "video_key": video_key,
                        },
                    )
                )

        if dataset.override_instruction is None:
            ordered_tasks.extend(_selected_prompt_tasks(part))
        backend = part.video_backend
        if not isinstance(backend, str) or not backend:
            raise ValueError("underlying dataset video backend must be non-empty")
        decoder_datasets.append(
            {
                "dataset_index": dataset_index,
                "backend": backend,
                "allow_fallback": bool(part.allow_video_backend_fallback),
            }
        )
        dataset_roots.append(
            {
                "dataset_index": dataset_index,
                "root": str(root),
                "selected_episodes": episodes,
                "num_frames": num_frames,
                "episode_boundaries": boundaries,
                "video_keys": video_keys,
                "files": files,
            }
        )

    if total_frames != len(dataset):
        raise ValueError("underlying dataset frames disagree with RobotVideoDataset length")

    override_instruction = dataset.override_instruction
    if override_instruction is not None:
        if not isinstance(override_instruction, str) or not override_instruction:
            raise ValueError("override_instruction must be a non-empty string")
        ordered_tasks = [override_instruction]
    unique_tasks = list(dict.fromkeys(ordered_tasks))
    if not unique_tasks:
        raise ValueError("selected dataset contains no task instructions")

    context_len = _integer(dataset.context_len, name="text context_len")
    if context_len <= 0:
        raise ValueError("text context_len must be positive")
    cache_root_raw = dataset.text_embedding_cache_dir
    if not isinstance(cache_root_raw, (str, Path)):
        raise ValueError("text_embedding_cache_dir must be configured")
    cache_root = Path(cache_root_raw).expanduser().resolve(strict=True)
    if not cache_root.is_dir():
        raise ValueError("text_embedding_cache_dir must be a directory")

    prompts = tuple(
        DEFAULT_PROMPT_TEMPLATE.format(task=task) for task in unique_tasks
    )
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        cache_files: list[dict[str, Any]] = []
        for prompt in prompts:
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            relative = f"{prompt_sha256}.t5_len{context_len}.wan22ti2v5b.pt"
            cache_files.append(
                _selected_file(
                    anchor=cache_root,
                    anchor_key=("text_cache", None),
                    relative_path=relative,
                    role="text_embedding",
                    sha_lookup=sha_lookup,
                    extra={"prompt_sha256": prompt_sha256},
                )
            )
        text_cache_identity: dict[str, Any] = {
            "root": str(cache_root),
            "context_len": context_len,
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "files": cache_files,
        }
    else:
        descriptor_path = (
            Path(text_cache_index_descriptor_path)
            .expanduser()
            .resolve(strict=True)
        )
        descriptor = load_text_cache_index_descriptor(descriptor_path)
        expected_suffix = TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE.format(
            context_len=context_len
        )
        if descriptor["cache_root"] != str(cache_root):
            raise ValueError("text cache index root disagrees with the dataset")
        if (
            descriptor["context_len"] != context_len
            or descriptor["prompt_template"] != DEFAULT_PROMPT_TEMPLATE
            or descriptor["filename_suffix"] != expected_suffix
        ):
            raise ValueError("text cache index contract disagrees with the dataset")

        descriptor_root = descriptor_path.parent.resolve(strict=True)
        index_files = [
            _selected_file(
                anchor=descriptor_root,
                anchor_key=("text_cache_index", None),
                relative_path=descriptor_path.name,
                role="text_cache_index_descriptor",
                sha_lookup=sha_lookup,
            ),
            _selected_file(
                anchor=descriptor_root,
                anchor_key=("text_cache_index", None),
                relative_path=descriptor["index"]["relative_path"],
                role="text_cache_index",
                sha_lookup=sha_lookup,
            ),
        ]
        if (
            index_files[1]["size_bytes"] != descriptor["index"]["size_bytes"]
            or index_files[1]["sha256"] != descriptor["index"]["sha256"]
        ):
            raise ValueError("text cache index file disagrees with its descriptor")
        with TextCacheIndex(
            descriptor_path,
            verify_index_sha256=False,
        ) as cache_index:
            cache_index.require_exact_prompts(prompts)
        text_cache_identity = {
            "root": str(cache_root),
            "context_len": context_len,
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "filename_suffix": expected_suffix,
            "required_prompt_count": descriptor["record_count"],
            "prompt_set_sha256": descriptor["prompt_set_sha256"],
            "integrity": {
                "mode": TEXT_CACHE_INDEX_MODE,
                "root": str(descriptor_root),
                "descriptor_sha256": descriptor["descriptor_sha256"],
                "files": index_files,
            },
        }

    stats_path = Path(normalization_stats_path).expanduser().resolve(strict=True)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"normalization stats file does not exist: {stats_path}"
        )
    stats_root = stats_path.parent
    stats_file = _selected_file(
        anchor=stats_root,
        anchor_key=("normalization_stats", None),
        relative_path=stats_path.name,
        role="normalization_stats",
        sha_lookup=sha_lookup,
    )

    strict_data_mode = bool(dataset.strict_data_mode)
    if strict_data_mode and any(row["allow_fallback"] for row in decoder_datasets):
        raise ValueError("strict data mode cannot allow video backend fallback")
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "kind": manifest_kind,
        "sampling": sampling,
        "num_frames": total_frames,
        "dataset_roots": dataset_roots,
        "text_embedding_cache": text_cache_identity,
        "normalization_stats": {
            "root": str(stats_root),
            "files": [stats_file],
        },
        "decoder": {
            "implementation": VIDEO_DECODER_IMPLEMENTATION,
            "strict_data_mode": strict_data_mode,
            "datasets": decoder_datasets,
        },
    }
    manifest[DATA_MANIFEST_SHA256_KEY] = canonical_data_manifest_sha256(manifest)
    return manifest


def build_robot_video_dataset_manifest(
    dataset: Any,
    *,
    normalization_stats_path: str | Path,
    schema_version: int = DATA_MANIFEST_SCHEMA_VERSION,
    text_cache_index_descriptor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build selected-data identity using the legacy v1 or indexed v2 schema."""

    return _build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=normalization_stats_path,
        sha_lookup=None,
        schema_version=schema_version,
        text_cache_index_descriptor_path=text_cache_index_descriptor_path,
    )


def resolve_text_cache_index_descriptor_path(
    manifest: Mapping[str, Any],
) -> Path:
    if require_supported_data_manifest_header(manifest) != DATA_MANIFEST_V2_SCHEMA_VERSION:
        raise ValueError("text cache index descriptors require a v2 data manifest")
    text_cache = manifest.get("text_embedding_cache")
    if not isinstance(text_cache, Mapping):
        raise ValueError("v2 text_embedding_cache must be a mapping")
    integrity = text_cache.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("v2 text cache integrity must be a mapping")
    if integrity.get("mode") != TEXT_CACHE_INDEX_MODE:
        raise ValueError("unsupported v2 text cache integrity mode")
    bound_descriptor_sha256 = integrity.get("descriptor_sha256")
    if (
        not isinstance(bound_descriptor_sha256, str)
        or not _SHA256_PATTERN.fullmatch(bound_descriptor_sha256)
    ):
        raise ValueError("v2 text cache descriptor SHA256 is invalid")
    root_value = integrity.get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("v2 text cache index root must be absolute")
    root = Path(root_value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("v2 text cache index root must be a directory")
    files = integrity.get("files")
    if (
        not isinstance(files, list)
        or len(files) != 2
        or any(not isinstance(entry, Mapping) for entry in files)
    ):
        raise ValueError("v2 text cache integrity must select exactly two files")
    by_role: dict[str, Mapping[str, Any]] = {}
    for entry in files:
        if set(entry) != {"role", "relative_path", "size_bytes", "sha256"}:
            raise ValueError("v2 text cache file entry fields are invalid")
        role = entry.get("role")
        if not isinstance(role, str):
            raise ValueError("v2 text cache file role must be a string")
        if role in by_role:
            raise ValueError("v2 text cache file roles must be unique")
        by_role[role] = entry
    expected_roles = {"text_cache_index_descriptor", "text_cache_index"}
    if set(by_role) != expected_roles:
        raise ValueError("v2 text cache integrity file roles are invalid")

    descriptor_entry = by_role["text_cache_index_descriptor"]
    relative = _safe_relative_path(descriptor_entry.get("relative_path"))
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("v2 text cache descriptor escapes its root") from error
    if not candidate.is_file():
        raise ValueError("v2 text cache descriptor must be a regular file")
    descriptor_size = descriptor_entry.get("size_bytes")
    descriptor_file_sha = descriptor_entry.get("sha256")
    if (
        isinstance(descriptor_size, bool)
        or not isinstance(descriptor_size, int)
        or descriptor_size <= 0
        or descriptor_size != candidate.stat().st_size
        or not isinstance(descriptor_file_sha, str)
        or not _SHA256_PATTERN.fullmatch(descriptor_file_sha)
        or descriptor_file_sha != _sha256_file(candidate)
    ):
        raise ValueError("v2 text cache descriptor file identity is invalid")

    descriptor = load_text_cache_index_descriptor(candidate)
    if descriptor["descriptor_sha256"] != bound_descriptor_sha256:
        raise ValueError("v2 manifest does not bind the selected cache descriptor")
    bound_fields = {
        "cache_root": "root",
        "context_len": "context_len",
        "prompt_template": "prompt_template",
        "filename_suffix": "filename_suffix",
        "record_count": "required_prompt_count",
        "prompt_set_sha256": "prompt_set_sha256",
    }
    if any(
        descriptor[descriptor_key] != text_cache.get(manifest_key)
        for descriptor_key, manifest_key in bound_fields.items()
    ):
        raise ValueError("v2 text cache contract differs from its descriptor")
    index_entry = by_role["text_cache_index"]
    if any(
        index_entry.get(key) != descriptor["index"].get(key)
        for key in ("relative_path", "size_bytes", "sha256")
    ):
        raise ValueError("v2 text cache index entry differs from its descriptor")
    return candidate


def validate_robot_video_dataset_manifest(
    dataset: Any,
    manifest: Mapping[str, Any],
    *,
    normalization_stats_path: str | Path,
    full_content_verify: bool = True,
) -> dict[str, Any]:
    """Validate an existing manifest against a freshly instantiated dataset.

    Formal training should retain the default ``full_content_verify=True``.
    Setting it to false verifies the self-hash, selected paths, file existence,
    sizes, episode layout, and runtime settings, but trusts recorded file
    content hashes.  It is intended only for fast local diagnostics.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("data manifest must be a mapping")
    schema_version = require_supported_data_manifest_header(manifest)
    recorded_sha256 = manifest.get(DATA_MANIFEST_SHA256_KEY)
    if (
        not isinstance(recorded_sha256, str)
        or not _SHA256_PATTERN.fullmatch(recorded_sha256)
        or recorded_sha256 != canonical_data_manifest_sha256(manifest)
    ):
        raise ValueError("Stage 3 data manifest canonical SHA256 mismatch")

    sha_lookup = None
    if not full_content_verify:
        sha_lookup = _sha_lookup_from_manifest(manifest)
    descriptor_path = (
        resolve_text_cache_index_descriptor_path(manifest)
        if schema_version == DATA_MANIFEST_V2_SCHEMA_VERSION
        else None
    )
    current = _build_robot_video_dataset_manifest(
        dataset,
        normalization_stats_path=normalization_stats_path,
        sha_lookup=sha_lookup,
        schema_version=schema_version,
        text_cache_index_descriptor_path=descriptor_path,
    )
    if current != dict(manifest):
        raise ValueError("Stage 3 selected data identity drifted from manifest")
    return current


# Short aliases for callers that already operate within the Alignment package.
build_data_manifest = build_robot_video_dataset_manifest
validate_data_manifest = validate_robot_video_dataset_manifest


__all__ = [
    "DATA_MANIFEST_KIND",
    "DATA_MANIFEST_SCHEMA_VERSION",
    "DATA_MANIFEST_V2_KIND",
    "DATA_MANIFEST_V2_SCHEMA_VERSION",
    "LEROBOT_META_PATHS",
    "TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE",
    "build_data_manifest",
    "build_robot_video_dataset_manifest",
    "canonical_data_manifest_sha256",
    "require_supported_data_manifest_header",
    "resolve_text_cache_index_descriptor_path",
    "selected_text_cache_prompts",
    "validate_data_manifest",
    "validate_robot_video_dataset_manifest",
]
