"""Collect paired LIBERO N=0/N=full demonstration utility labels.

This is deliberately a frozen-model data job, not a training entrypoint.  It
samples source indices before decoding any observations, records an immutable
manifest, and appends one durable JSONL record after each paired inference.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.gate.demo_utility import (
    collect_paired_utility,
    current_state_input_hashes,
    extract_current_state,
    parse_sample_identity,
    stable_sample_seed,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1
RECORD_AUGMENTATION_SCHEMA_VERSION = 1
SHA256_HEX_LENGTH = 64
SCIENTIFIC_SOURCE_FILES = (
    "experiments/libero/gate/collect_demo_utility.py",
    "experiments/libero/gate/demo_utility.py",
    "src/fastwam/datasets/lerobot/base_lerobot_dataset.py",
    "src/fastwam/datasets/lerobot/robot_video_dataset.py",
    "src/fastwam/models/wan22/fastwam_unified_shared.py",
    "src/fastwam/models/wan22/wan_video_vae.py",
    "src/fastwam/models/wan22/helpers/loader.py",
    "src/fastwam/models/wan22/helpers/io.py",
    "src/fastwam/models/wan22/helpers/state_dict_converters.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _scientific_data_config(resolved_data: Any) -> dict[str, Any]:
    """Return data semantics without host-specific artifact locations.

    Dataset and text-cache bytes are bound separately by content hashes in the
    manifest. Keeping their mount paths in this compatibility projection would
    make a scientifically identical resume fail after moving those bytes to a
    different host or mount point.
    """

    if not isinstance(resolved_data, Mapping):
        raise ValueError("Resolved data config must be a mapping")
    normalized = json.loads(_canonical_json(resolved_data))
    train = normalized.get("train")
    if not isinstance(train, dict):
        raise ValueError("Resolved data config must contain a train mapping")
    train.pop("dataset_dirs", None)
    train.pop("text_embedding_cache_dir", None)
    return normalized


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256, got {value!r}")
    return value


def _resolve_project_path(raw_path: Any, *, label: str) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError(f"{label} must be configured")
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _stable_file_provenance(raw_path: Any, *, label: str) -> dict[str, Any]:
    """Hash one artifact and reject concurrent replacement or mutation."""

    path = _resolve_project_path(raw_path, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    signature_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    before_signature = tuple(getattr(before, field) for field in signature_fields)
    after_signature = tuple(getattr(after, field) for field in signature_fields)
    if before_signature != after_signature:
        raise RuntimeError(f"{label} changed while it was being hashed: {path}")
    return {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def _assert_file_provenance_unchanged(
    provenance: Mapping[str, Any], *, label: str
) -> None:
    path = Path(str(provenance["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{label} disappeared after hashing: {path}")
    stat = path.stat()
    current = {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    expected = {key: int(provenance[key]) for key in current}
    if current != expected:
        raise RuntimeError(
            f"{label} changed after provenance capture: expected={expected}, current={current}"
        )


def _directory_tree_provenance(raw_path: Any, *, label: str) -> dict[str, Any]:
    """Hash all source files under a directory using relative paths and bytes."""

    root = _resolve_project_path(raw_path, label=label)
    if not root.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {root}")

    def listed_files() -> list[Path]:
        return sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and ".cache" not in path.relative_to(root).parts
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )

    files = listed_files()
    if not files:
        raise ValueError(f"{label} contains no source files: {root}")
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        artifact = _stable_file_provenance(path, label=f"{label} member")
        relative_path = path.relative_to(root).as_posix()
        total_size += int(artifact["size_bytes"])
        digest.update(
            _canonical_json(
                {
                    "relative_path": relative_path,
                    "size_bytes": artifact["size_bytes"],
                    "sha256": artifact["sha256"],
                }
            ).encode("utf-8")
        )
        digest.update(b"\n")
    after_relative_paths = [path.relative_to(root).as_posix() for path in listed_files()]
    before_relative_paths = [path.relative_to(root).as_posix() for path in files]
    if after_relative_paths != before_relative_paths:
        raise RuntimeError(f"{label} file set changed while it was being hashed: {root}")
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "excluded_relative_component": ".cache",
    }


def _assert_directory_tree_provenance_unchanged(
    provenance: Mapping[str, Any], *, label: str
) -> None:
    current = _directory_tree_provenance(provenance["path"], label=label)
    for key in ("sha256", "file_count", "total_size_bytes"):
        if current[key] != provenance[key]:
            raise RuntimeError(
                f"{label} changed during collection: {key} "
                f"expected={provenance[key]!r}, current={current[key]!r}"
            )


def _dataset_instantiation_path_overrides(
    dataset_source_artifacts: Sequence[Mapping[str, Any]],
    context_cache_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the exact absolute paths whose contents were fingerprinted."""

    dataset_dirs: list[str] = []
    for index, artifact in enumerate(dataset_source_artifacts):
        raw_path = artifact.get("path")
        if raw_path is None or not str(raw_path).strip():
            raise ValueError(f"dataset source artifact {index} is missing path")
        path = Path(str(raw_path))
        if not path.is_absolute():
            raise ValueError(f"dataset source artifact {index} path is not absolute: {path}")
        dataset_dirs.append(str(path))

    raw_cache_path = context_cache_artifact.get("path")
    if raw_cache_path is None or not str(raw_cache_path).strip():
        raise ValueError("text embedding cache artifact is missing path")
    cache_path = Path(str(raw_cache_path))
    if not cache_path.is_absolute():
        raise ValueError(f"text embedding cache artifact path is not absolute: {cache_path}")
    return {
        "dataset_dirs": dataset_dirs,
        "text_embedding_cache_dir": str(cache_path),
    }


def _resolve_vae_artifact(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the same VAE ModelConfig used by the model factory, then hash it."""

    from fastwam.models.wan22.helpers.loader import _resolve_configs

    _, _, vae_config, _ = _resolve_configs(
        model_id=str(cfg.model.model_id),
        tokenizer_model_id=str(cfg.model.tokenizer_model_id),
        redirect_common_files=bool(cfg.model.redirect_common_files),
    )
    vae_config.download_if_necessary()
    if not isinstance(vae_config.path, str):
        raise ValueError(
            "Collector requires the VAE loader to resolve exactly one file, "
            f"got {vae_config.path!r}"
        )
    return _stable_file_provenance(vae_config.path, label="VAE artifact")


def _scientific_source_provenance() -> dict[str, str]:
    """Hash label-generating source even when a new file is still untracked."""

    result: dict[str, str] = {}
    for relative_path in SCIENTIFIC_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Scientific source file is missing: {path}")
        result[relative_path] = _sha256_file(path)
    return result


def _resolve_existing_file(raw_path: Any, *, label: str) -> Path:
    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError(f"{label} must be provided")
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path)))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def resolve_dataset_stats_path(checkpoint_path: Path, explicit: Any = None) -> Path:
    """Resolve the exact pretrained normalization statistics used by the dataset."""

    if explicit is not None and str(explicit).strip():
        explicit_path = Path(
            os.path.expandvars(os.path.expanduser(str(explicit)))
        ).resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(
                "Explicit COLLECTOR.dataset_stats_path is not a file: "
                f"{explicit_path}"
            )
        return explicit_path

    candidates: list[Path] = []
    for parent in list(checkpoint_path.parents)[:5]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Set "
        f"COLLECTOR.dataset_stats_path explicitly. Tried: {attempted}"
    )


def _run_git(args: Sequence[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8", errors="replace")


def git_provenance() -> dict[str, Any]:
    """Capture the commit and dirty-worktree identity without mutating git state."""

    try:
        commit = str(_run_git(["rev-parse", "HEAD"])).strip()
        branch = str(_run_git(["rev-parse", "--abbrev-ref", "HEAD"])).strip()
        status = str(
            _run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
        )
        tracked_diff = _run_git(["diff", "--binary", "HEAD"], binary=True)
        assert isinstance(tracked_diff, bytes)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to capture git provenance in {PROJECT_ROOT}: {exc}") from exc

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
    }


def _normalize_ranges(raw_ranges: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "dataset_index",
        "dataset_id",
        "dataset_name",
        "start",
        "stop",
        "population",
    }
    ranges: list[dict[str, Any]] = []
    for raw in raw_ranges:
        missing = required - set(raw)
        if missing:
            raise ValueError(f"dataset_index_ranges entry is missing {sorted(missing)}: {raw}")
        item = {
            "dataset_index": int(raw["dataset_index"]),
            "dataset_id": str(raw["dataset_id"]),
            "dataset_name": str(raw["dataset_name"]),
            "start": int(raw["start"]),
            "stop": int(raw["stop"]),
            "population": int(raw["population"]),
        }
        if item["start"] < 0 or item["stop"] <= item["start"]:
            raise ValueError(f"Invalid dataset index range: {item}")
        if item["population"] != item["stop"] - item["start"]:
            raise ValueError(f"Range population mismatch: {item}")
        ranges.append(item)

    ranges.sort(key=lambda item: item["dataset_index"])
    if not ranges:
        raise ValueError("dataset_index_ranges() returned no strata")
    indices = [item["dataset_index"] for item in ranges]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate dataset_index values: {indices}")
    return ranges


def _derived_seed(base_seed: int, namespace: str, dataset_index: int | None = None) -> int:
    payload = f"{int(base_seed)}\0{namespace}\0{dataset_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _allocate_stratified_counts(
    ranges: Sequence[Mapping[str, Any]], num_samples: int
) -> list[int]:
    """Proportionally allocate samples while covering every stratum when possible."""

    populations = np.asarray([int(item["population"]) for item in ranges], dtype=np.int64)
    if np.any(populations <= 0):
        raise ValueError(f"Every stratum must be non-empty, got populations={populations.tolist()}")
    total = int(populations.sum())
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples > total:
        raise ValueError(f"Cannot sample {num_samples} states without replacement from {total}")

    allocations = np.zeros(len(ranges), dtype=np.int64)
    if num_samples >= len(ranges):
        allocations[:] = 1
    else:
        # With fewer samples than strata, choose the largest populations and use
        # dataset_index as the deterministic tie break.
        order = sorted(
            range(len(ranges)),
            key=lambda idx: (-int(populations[idx]), int(ranges[idx]["dataset_index"])),
        )
        allocations[order[:num_samples]] = 1
        return allocations.tolist()

    remaining = num_samples - int(allocations.sum())
    capacities = populations - allocations
    if remaining == 0:
        return allocations.tolist()
    capacity_total = int(capacities.sum())
    raw_extra = capacities.astype(np.float64) * (remaining / capacity_total)
    extra = np.floor(raw_extra).astype(np.int64)
    extra = np.minimum(extra, capacities)
    allocations += extra

    left = num_samples - int(allocations.sum())
    fractional = raw_extra - np.floor(raw_extra)
    order = sorted(
        range(len(ranges)),
        key=lambda idx: (-float(fractional[idx]), int(ranges[idx]["dataset_index"])),
    )
    while left > 0:
        progressed = False
        for idx in order:
            if allocations[idx] >= populations[idx]:
                continue
            allocations[idx] += 1
            left -= 1
            progressed = True
            if left == 0:
                break
        if not progressed:
            raise AssertionError("Stratified allocation exhausted capacity unexpectedly")

    if int(allocations.sum()) != num_samples or np.any(allocations > populations):
        raise AssertionError("Invalid stratified sample allocation")
    return allocations.tolist()


def build_stratified_sample_plan(
    raw_ranges: Iterable[Mapping[str, Any]], *, num_samples: int, seed: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Build a deterministic shuffled-without-replacement source-index plan."""

    ranges = _normalize_ranges(raw_ranges)
    allocations = _allocate_stratified_counts(ranges, int(num_samples))
    selected_by_stratum: list[list[int]] = []
    plan_strata: list[dict[str, Any]] = []

    for item, allocated in zip(ranges, allocations, strict=True):
        stratum_seed = _derived_seed(seed, "suite-stratum", item["dataset_index"])
        rng = np.random.default_rng(stratum_seed)
        population = int(item["population"])
        offsets = rng.permutation(population)[:allocated]
        selected = [int(item["start"] + int(offset)) for offset in offsets]
        if len(selected) != len(set(selected)):
            raise AssertionError("A stratum sampler produced duplicate source indices")
        selected_by_stratum.append(selected)
        plan_strata.append(
            {
                **item,
                "allocated": int(allocated),
                "seed": int(stratum_seed),
                "ordered_selected_source_indices_sha256": _sha256_json(selected),
            }
        )

    selected_indices = [index for group in selected_by_stratum for index in group]
    global_seed = _derived_seed(seed, "global-interleave")
    global_rng = np.random.default_rng(global_seed)
    global_rng.shuffle(selected_indices)
    if len(selected_indices) != int(num_samples):
        raise AssertionError("Sample plan length does not match num_samples")
    if len(selected_indices) != len(set(selected_indices)):
        raise AssertionError("Sample plan is not without replacement")
    return selected_indices, plan_strata


def _mixed_precision_dtype(value: Any) -> torch.dtype:
    key = str(value).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision={value!r}; expected no, fp16, or bf16")


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, Path):
            value = str(value)
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            value = str(value)
        result[str(key)] = value
    # Round-trip validation also catches NaN/Infinity.
    return json.loads(_canonical_json(result))


def _current_proprio(sample: Mapping[str, Any]) -> list[float]:
    proprio = sample.get("proprio")
    if not isinstance(proprio, torch.Tensor) or proprio.ndim != 2 or proprio.shape[0] < 1:
        raise ValueError(
            "Collector expects sample['proprio'] as [T, D] with at least one state, "
            f"got {type(proprio)} / {getattr(proprio, 'shape', None)}"
        )
    values = proprio[0].detach().to(device="cpu", dtype=torch.float32).tolist()
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Current normalized proprio contains non-finite values")
    return [float(value) for value in values]


def _write_jsonl(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_canonical_json(dict(value)) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def load_existing_record_index(
    records_path: Path,
    *,
    expected_manifest_fingerprint: str | None = None,
    expected_full_steps: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
    expected_git_sha: str | None = None,
    expected_base_seed: int | None = None,
) -> tuple[set[str], dict[int, str], int]:
    """Read existing records and reject corruption or incomplete completed rows."""

    sample_ids: set[str] = set()
    source_indices: dict[int, str] = {}
    count = 0
    if not records_path.exists():
        return sample_ids, source_indices, count

    with records_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {records_path}:{line_number}; refusing unsafe resume"
                ) from exc
            if not isinstance(record, dict) or not record.get("sample_id"):
                raise ValueError(f"Missing sample_id in {records_path}:{line_number}")
            if expected_full_steps is not None:
                try:
                    _validate_completed_record(record, full_steps=expected_full_steps)
                except (AssertionError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid completed record in {records_path}:{line_number}: {exc}"
                    ) from exc
            sample_id = str(record["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {records_path}")
            sample_ids.add(sample_id)
            metadata = record.get("source_metadata", {})
            if expected_manifest_fingerprint is not None:
                actual_fingerprint = record.get("manifest_compatibility_fingerprint")
                if actual_fingerprint != expected_manifest_fingerprint:
                    raise ValueError(
                        f"Record/manifest fingerprint mismatch in {records_path}:{line_number}: "
                        f"record={actual_fingerprint!r}, expected={expected_manifest_fingerprint!r}"
                    )
                if not isinstance(metadata, dict) or metadata.get("requested_sample_idx") is None:
                    raise ValueError(
                        f"Missing source_metadata.requested_sample_idx in "
                        f"{records_path}:{line_number}"
                    )
                requested = int(metadata["requested_sample_idx"])
                if metadata.get("source_sample_idx") is None:
                    raise ValueError(
                        f"Missing source_metadata.source_sample_idx in "
                        f"{records_path}:{line_number}"
                    )
                source = int(metadata["source_sample_idx"])
                if requested != source:
                    raise ValueError(
                        f"Completed record changed source identity in {records_path}:"
                        f"{line_number}: requested={requested}, source={source}"
                    )
                for field, expected_value in (
                    ("checkpoint_sha256", expected_checkpoint_sha256),
                    ("dataset_stats_sha256", expected_dataset_stats_sha256),
                    ("vae_sha256", expected_vae_sha256),
                    ("git_sha", expected_git_sha),
                ):
                    if expected_value is not None and record.get(field) != expected_value:
                        raise ValueError(
                            f"Record {field} mismatch in {records_path}:{line_number}: "
                            f"record={record.get(field)!r}, expected={expected_value!r}"
                        )
                expected_sample_id = (
                    f"{record['dataset_id']}/"
                    f"episode_{int(record['episode_index']):06d}/"
                    f"frame_{int(record['frame_index']):06d}"
                )
                if sample_id != expected_sample_id:
                    raise ValueError(
                        f"sample_id/source identity mismatch in {records_path}:{line_number}: "
                        f"{sample_id!r} != {expected_sample_id!r}"
                    )
                if expected_base_seed is not None:
                    identity = parse_sample_identity(record)
                    expected_seed = stable_sample_seed(expected_base_seed, identity)
                    if int(record["seed"]) != expected_seed:
                        raise ValueError(
                            f"Stable seed mismatch in {records_path}:{line_number}: "
                            f"record={record['seed']}, expected={expected_seed}"
                        )
            if isinstance(metadata, dict) and metadata.get("requested_sample_idx") is not None:
                source_idx = int(metadata["requested_sample_idx"])
                if source_idx in source_indices:
                    raise ValueError(
                        f"Duplicate requested_sample_idx={source_idx} in {records_path}"
                    )
                source_indices[source_idx] = sample_id
            count += 1
    return sample_ids, source_indices, count


def _validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    """Reject a manifest whose stored identity no longer matches its contents."""

    fingerprint = manifest.get("compatibility_fingerprint")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Manifest compatibility must be a mapping")
    expected_fingerprint = _sha256_json(compatibility)
    if fingerprint != expected_fingerprint:
        raise ValueError(
            "Immutable manifest compatibility fingerprint does not match its contents: "
            f"stored={fingerprint!r}, computed={expected_fingerprint!r}"
        )

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Manifest selection must be a mapping")
    selected = selection.get("ordered_selected_source_indices")
    if not isinstance(selected, list):
        raise ValueError("Manifest selection is missing the ordered source-index plan")
    selection_sha256 = _sha256_json(selected)
    if selection.get("ordered_selected_source_indices_sha256") != selection_sha256:
        raise ValueError("Immutable manifest selection digest does not match its plan")
    if int(selection.get("num_samples", -1)) != len(selected):
        raise ValueError("Immutable manifest selection count does not match its plan")
    if compatibility.get("selection_sha256") != selection_sha256:
        raise ValueError("Manifest compatibility is not bound to the stored selection plan")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Manifest artifacts must be a mapping")
    for artifact_name, compatibility_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("dataset_stats", "dataset_stats_sha256"),
        ("vae", "vae_sha256"),
    ):
        artifact = artifacts.get(artifact_name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Manifest artifact {artifact_name!r} must be a mapping")
        if artifact.get("sha256") != compatibility.get(compatibility_field):
            raise ValueError(
                f"Manifest artifact {artifact_name!r} is not bound to compatibility"
            )

    context_cache = artifacts.get("text_embedding_cache")
    if not isinstance(context_cache, Mapping) or context_cache.get("sha256") != compatibility.get(
        "context_cache_sha256"
    ):
        raise ValueError("Manifest text embedding cache is not bound to compatibility")

    dataset_sources = artifacts.get("dataset_sources")
    if not isinstance(dataset_sources, list):
        raise ValueError("Manifest dataset_sources must be a list")
    source_content = [
        {
            "dataset_name": str(item["dataset_name"]),
            "sha256": item["sha256"],
            "file_count": int(item["file_count"]),
            "total_size_bytes": int(item["total_size_bytes"]),
        }
        for item in dataset_sources
    ]
    if source_content != compatibility.get("dataset_source_content"):
        raise ValueError("Manifest dataset source artifacts are not bound to compatibility")
    if manifest.get("scientific_source_files") != compatibility.get(
        "scientific_source_files"
    ):
        raise ValueError("Manifest scientific source provenance is internally inconsistent")


def ensure_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create manifest once, or verify an existing manifest is exactly compatible."""

    expected = dict(payload)
    expected_fingerprint = str(expected["compatibility_fingerprint"])
    _validate_manifest_integrity(expected)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read existing immutable manifest: {path}") from exc
        if not isinstance(existing, Mapping):
            raise ValueError(f"Existing immutable manifest must be a mapping: {path}")
        _validate_manifest_integrity(existing)
        actual = existing.get("compatibility_fingerprint")
        if actual != expected_fingerprint:
            raise ValueError(
                "Existing manifest is incompatible with this run; use its original config "
                f"or a new output directory. existing={actual!r} expected={expected_fingerprint!r}"
            )
        return existing

    with path.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return expected


def _build_manifest(
    *,
    cfg: DictConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    stats_path: Path,
    stats_sha256: str,
    vae_artifact: Mapping[str, Any],
    dataset_source_artifacts: Sequence[Mapping[str, Any]],
    context_cache_artifact: Mapping[str, Any],
    git: Mapping[str, Any],
    ranges: Sequence[Mapping[str, Any]],
    task_tables: Mapping[int, Mapping[int, str]],
    plan_strata: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    collector = cfg.COLLECTOR
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config_sha256 = _sha256_json(resolved_config)
    normalized_task_tables = {
        str(int(dataset_index)): {
            str(int(task_index)): str(task)
            for task_index, task in sorted(table.items(), key=lambda item: int(item[0]))
        }
        for dataset_index, table in sorted(task_tables.items(), key=lambda item: int(item[0]))
    }
    source_files = _scientific_source_provenance()
    selection = {
        "algorithm": "proportional-suite-stratified-shuffle-without-replacement-v1",
        "base_seed": int(collector.seed),
        "global_interleave_seed": _derived_seed(int(collector.seed), "global-interleave"),
        "num_samples": len(selected_indices),
        "strata": list(plan_strata),
        "ordered_selected_source_indices": [int(index) for index in selected_indices],
        "ordered_selected_source_indices_sha256": _sha256_json(
            [int(index) for index in selected_indices]
        ),
    }
    compatibility = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_stats_sha256": stats_sha256,
        "vae_sha256": _require_sha256(vae_artifact.get("sha256"), field="vae.sha256"),
        "dataset_source_content": [
            {
                "dataset_name": str(item["dataset_name"]),
                "sha256": _require_sha256(
                    item.get("sha256"), field=f"dataset_source[{index}].sha256"
                ),
                "file_count": int(item["file_count"]),
                "total_size_bytes": int(item["total_size_bytes"]),
            }
            for index, item in enumerate(dataset_source_artifacts)
        ],
        "context_cache_sha256": _require_sha256(
            context_cache_artifact.get("sha256"), field="context_cache.sha256"
        ),
        "git_commit": git["commit"],
        "git_tracked_diff_sha256": git["tracked_diff_sha256"],
        "scientific_source_files": source_files,
        # Untracked files and operational resume/error policy remain in full
        # provenance below, but do not change scientific compatibility.
        "model_config_sha256": _sha256_json(resolved_config["model"]),
        "data_config_sha256": _sha256_json(
            _scientific_data_config(resolved_config["data"])
        ),
        "mixed_precision": str(cfg.get("mixed_precision", "bf16")),
        "execution_environment": {
            "device": str(collector.device),
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        },
        "dataset_index_ranges": list(ranges),
        "dataset_task_tables": normalized_task_tables,
        "selection_sha256": selection["ordered_selected_source_indices_sha256"],
        "collection_parameters": {
            "seed": int(collector.seed),
            "num_inference_steps": int(collector.num_inference_steps),
            "full_prefix_steps": int(collector.full_prefix_steps),
            "num_video_frames": int(collector.num_video_frames),
            "rand_device": str(collector.rand_device),
            "sigma_shift": (
                None if collector.get("sigma_shift") is None else float(collector.sigma_shift)
            ),
            "tiled": bool(collector.tiled),
            "force_custom_prefix": bool(collector.force_custom_prefix),
        },
    }
    compatibility_fingerprint = _sha256_json(compatibility)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "compatibility_fingerprint": compatibility_fingerprint,
        "compatibility": compatibility,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_path.stat().st_size,
                "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            },
            "dataset_stats": {
                "path": str(stats_path),
                "sha256": stats_sha256,
                "size_bytes": stats_path.stat().st_size,
                "mtime_ns": stats_path.stat().st_mtime_ns,
            },
            "vae": dict(vae_artifact),
            "dataset_sources": [dict(item) for item in dataset_source_artifacts],
            "text_embedding_cache": dict(context_cache_artifact),
        },
        "git": dict(git),
        "runtime": {
            "argv": sys.argv,
            "cwd": str(Path.cwd().resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model_environment": {
                key: os.environ.get(key)
                for key in (
                    "DIFFSYNTH_MODEL_BASE_PATH",
                    "DIFFSYNTH_SKIP_DOWNLOAD",
                    "DIFFSYNTH_DOWNLOAD_SOURCE",
                )
            },
        },
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "scientific_source_files": source_files,
        "dataset_index_ranges": list(ranges),
        "dataset_task_tables": normalized_task_tables,
        "selection": selection,
        "record_augmentation_schema_version": RECORD_AUGMENTATION_SCHEMA_VERSION,
    }


def _checkpoint_payload_provenance(
    model: torch.nn.Module, payload: Any
) -> dict[str, Any]:
    """Validate the state payload returned by load_checkpoint and keep only metadata."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "model.load_checkpoint() must return its mapping payload for provenance, "
            f"got {type(payload).__name__}"
        )
    top_level_keys = sorted(str(key) for key in payload)
    if "mot" not in payload:
        raise ValueError(
            "Demo Utility Collector requires a complete UniShare 'mot' checkpoint; "
            f"legacy/partial payload keys={top_level_keys} are not accepted"
        )
    state_key = "mot"
    target_module = getattr(model, "mot", None)
    checkpoint_state = payload[state_key]
    if not isinstance(checkpoint_state, Mapping) or not checkpoint_state:
        raise ValueError(f"Checkpoint {state_key!r} state must be a non-empty mapping")
    if target_module is None or not hasattr(target_module, "state_dict"):
        raise TypeError(f"Model has no state_dict-bearing target for checkpoint key {state_key!r}")

    target_state = target_module.state_dict()
    checkpoint_keys = {str(key) for key in checkpoint_state}
    target_keys = {str(key) for key in target_state}
    overlap = sorted(checkpoint_keys & target_keys)
    if not overlap:
        raise ValueError(
            f"Checkpoint {state_key!r} has no parameter names in common with the model"
        )
    shape_mismatches: list[dict[str, Any]] = []
    tensor_count = 0
    tensor_numel = 0
    for key, value in checkpoint_state.items():
        if torch.is_tensor(value):
            tensor_count += 1
            tensor_numel += int(value.numel())
        if key not in target_state or not hasattr(value, "shape"):
            continue
        checkpoint_shape = tuple(int(dim) for dim in value.shape)
        model_shape = tuple(int(dim) for dim in target_state[key].shape)
        if checkpoint_shape != model_shape:
            shape_mismatches.append(
                {"key": str(key), "checkpoint": list(checkpoint_shape), "model": list(model_shape)}
            )
    if shape_mismatches:
        raise ValueError(
            "Checkpoint/model shape mismatch after load: "
            + _canonical_json(shape_mismatches[:10])
        )

    step = payload.get("step")
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError(f"Checkpoint step must be scalar, got shape {tuple(step.shape)}")
        step = step.detach().cpu().item()
    elif isinstance(step, np.generic):
        step = step.item()
    if step is not None and not isinstance(step, (str, int, float, bool)):
        step = str(step)

    missing = sorted(target_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - target_keys)
    if missing or unexpected:
        raise ValueError(
            "Collector refuses a partial MoT checkpoint: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    proprio_module = getattr(model, "proprio_encoder", None)
    proprio_info: dict[str, Any] | None = None
    if proprio_module is not None:
        if "proprio_encoder" not in payload:
            raise ValueError(
                "Checkpoint is missing proprio_encoder weights required by this model"
            )
        proprio_state = payload["proprio_encoder"]
        if not isinstance(proprio_state, Mapping) or not proprio_state:
            raise ValueError("Checkpoint proprio_encoder state must be a non-empty mapping")
        target_proprio_state = proprio_module.state_dict()
        proprio_keys = {str(key) for key in proprio_state}
        target_proprio_keys = {str(key) for key in target_proprio_state}
        proprio_missing = sorted(target_proprio_keys - proprio_keys)
        proprio_unexpected = sorted(proprio_keys - target_proprio_keys)
        proprio_shape_mismatches = []
        for key in sorted(target_proprio_keys & proprio_keys):
            checkpoint_shape = tuple(int(dim) for dim in proprio_state[key].shape)
            model_shape = tuple(int(dim) for dim in target_proprio_state[key].shape)
            if checkpoint_shape != model_shape:
                proprio_shape_mismatches.append(
                    {"key": key, "checkpoint": checkpoint_shape, "model": model_shape}
                )
        if proprio_missing or proprio_unexpected or proprio_shape_mismatches:
            raise ValueError(
                "Checkpoint/proprio_encoder mismatch: "
                f"missing={proprio_missing}, unexpected={proprio_unexpected}, "
                f"shapes={proprio_shape_mismatches[:10]}"
            )
        proprio_info = {
            "state_key_count": len(proprio_keys),
            "state_keys_sha256": _sha256_json(sorted(proprio_keys)),
            "tensor_numel": sum(
                int(value.numel())
                for value in proprio_state.values()
                if torch.is_tensor(value)
            ),
        }
    elif "proprio_encoder" in payload:
        raise ValueError(
            "Checkpoint contains proprio_encoder but the instantiated model has none"
        )
    return {
        "top_level_keys": top_level_keys,
        "state_key": state_key,
        "step": step,
        "checkpoint_state_key_count": len(checkpoint_keys),
        "model_state_key_count": len(target_keys),
        "overlap_key_count": len(overlap),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_key_examples": missing[:20],
        "unexpected_key_examples": unexpected[:20],
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_tensor_numel": tensor_numel,
        "checkpoint_state_keys_sha256": _sha256_json(sorted(checkpoint_keys)),
        "proprio_encoder": proprio_info,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_device": str(getattr(model, "device", "unknown")),
        "model_torch_dtype": str(getattr(model, "torch_dtype", "unknown")),
        "video_scheduler_class": (
            f"{type(model.infer_video_scheduler).__module__}."
            f"{type(model.infer_video_scheduler).__qualname__}"
            if getattr(model, "infer_video_scheduler", None) is not None
            else None
        ),
        "action_scheduler_class": (
            f"{type(model.infer_action_scheduler).__module__}."
            f"{type(model.infer_action_scheduler).__qualname__}"
            if getattr(model, "infer_action_scheduler", None) is not None
            else None
        ),
    }


def _validate_endpoint_config(cfg: DictConfig) -> None:
    collector = cfg.COLLECTOR
    steps = int(collector.num_inference_steps)
    full = int(collector.full_prefix_steps)
    if steps <= 0:
        raise ValueError(f"COLLECTOR.num_inference_steps must be positive, got {steps}")
    if full != steps:
        raise ValueError(
            "Phase 2 only compares endpoint N=0 against N=num_inference_steps; "
            f"full_prefix_steps={full} must equal num_inference_steps={steps}"
        )
    if not bool(collector.force_custom_prefix):
        raise ValueError("COLLECTOR.force_custom_prefix must remain true for paired endpoints")

    expected_video_frames = 1 + (
        (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio)
    )
    if int(collector.num_video_frames) != expected_video_frames:
        raise ValueError(
            "COLLECTOR.num_video_frames must match the dataset/model horizon: "
            f"configured={collector.num_video_frames}, expected={expected_video_frames}"
        )


def _validate_record(record: Mapping[str, Any], *, full_steps: int) -> None:
    required = {
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "seed",
        "num_inference_steps",
        "n0",
        "nfull",
        "e0",
        "efull",
        "utility",
        "valid_length",
        "target_action_shape",
        "pred_n0_shape",
        "pred_nfull_shape",
        "input_hashes",
        "n0_route",
        "nfull_route",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Utility core record is missing fields: {sorted(missing)}")
    e0 = float(record["e0"])
    efull = float(record["efull"])
    utility = float(record["utility"])
    if not all(math.isfinite(value) for value in (e0, efull, utility)):
        raise ValueError("Utility record contains non-finite E0/Efull/U")
    if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"Utility mismatch: U={utility}, E0-Efull={e0 - efull}")
    if int(record["valid_length"]) <= 0:
        raise ValueError(f"valid_length must be positive, got {record['valid_length']}")
    if int(record["n0"]) != 0:
        raise ValueError(f"Expected n0=0, got {record['n0']}")
    if int(record["nfull"]) != int(full_steps):
        raise ValueError(f"Expected nfull={full_steps}, got {record['nfull']}")
    if int(record["num_inference_steps"]) != int(full_steps):
        raise ValueError(
            f"Expected num_inference_steps={full_steps}, got {record['num_inference_steps']}"
        )
    if int(record["episode_id"]) != int(record["episode_index"]):
        raise ValueError("episode_id must equal episode_index")
    if int(record["task_id"]) != int(record["task_index"]):
        raise ValueError("task_id must equal task_index")
    if record["task_id_source"] != "lerobot_task_index":
        raise ValueError("task_id_source must be 'lerobot_task_index'")

    target_shape = [int(value) for value in record["target_action_shape"]]
    if (
        target_shape != [int(value) for value in record["pred_n0_shape"]]
        or target_shape != [int(value) for value in record["pred_nfull_shape"]]
        or len(target_shape) != 2
        or int(record["valid_length"]) > target_shape[0]
    ):
        raise ValueError("target/N=0/N=full shapes and valid_length are inconsistent")

    for route_field, expected_prefix in (("n0_route", 0), ("nfull_route", full_steps)):
        route = record[route_field]
        if not isinstance(route, Mapping):
            raise ValueError(f"{route_field} must be a mapping")
        expected_route = {
            "inference_mode": "prefix",
            "video_prefix_steps": int(expected_prefix),
            "num_inference_steps": int(full_steps),
            "force_custom_prefix": True,
        }
        if dict(route) != expected_route:
            raise ValueError(
                f"{route_field} mismatch: got {dict(route)!r}, expected {expected_route!r}"
            )

    input_hashes = record["input_hashes"]
    expected_input_hash_keys = {
        "input_image",
        "proprio",
        "context",
        "context_mask",
        "valid_target_action",
        "action_is_pad",
        "combined",
    }
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != expected_input_hash_keys:
        raise ValueError(
            "input_hashes must contain exactly " f"{sorted(expected_input_hash_keys)}"
        )
    for name, value in input_hashes.items():
        _require_sha256(value, field=f"input_hashes.{name}")
    component_hashes = {
        name: input_hashes[name] for name in sorted(expected_input_hash_keys - {"combined"})
    }
    if input_hashes["combined"] != _sha256_json(component_hashes):
        raise ValueError("input_hashes.combined does not match component digests")

    source_metadata = record.get("source_metadata")
    if source_metadata is not None:
        if not isinstance(source_metadata, Mapping):
            raise ValueError("source_metadata must be a mapping")
        for record_field, metadata_field in (
            ("dataset_name", "dataset_name"),
            ("episode_index", "episode_index"),
            ("frame_index", "frame_index"),
            ("task_index", "task_index"),
            ("task", "task"),
        ):
            if str(record[record_field]) != str(source_metadata.get(metadata_field)):
                raise ValueError(
                    f"record/source metadata mismatch for {record_field}: "
                    f"{record[record_field]!r} != {source_metadata.get(metadata_field)!r}"
                )


def _validate_completed_record(record: Mapping[str, Any], *, full_steps: int) -> None:
    """Validate the durable record schema accepted as completed during resume."""

    _validate_record(record, full_steps=full_steps)
    required = {
        "schema_version",
        "collector_record_schema_version",
        "source_metadata",
        "current_proprio",
        "n0_latency_ms",
        "nfull_latency_ms",
        "total_latency_ms",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Completed utility record is missing fields: {sorted(missing)}")
    if int(record["schema_version"]) != 1:
        raise ValueError(f"Unsupported core schema_version={record['schema_version']!r}")
    if int(record["collector_record_schema_version"]) != RECORD_AUGMENTATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported collector_record_schema_version="
            f"{record['collector_record_schema_version']!r}"
        )
    if not isinstance(record["source_metadata"], Mapping):
        raise ValueError("Completed source_metadata must be a mapping")

    current_proprio = record["current_proprio"]
    if not isinstance(current_proprio, list) or not current_proprio:
        raise ValueError("current_proprio must be a non-empty JSON list")
    if not all(math.isfinite(float(value)) for value in current_proprio):
        raise ValueError("current_proprio contains non-finite values")

    if float(record["e0"]) < 0 or float(record["efull"]) < 0:
        raise ValueError("Action MSE values E0/Efull must be non-negative")
    for field in ("n0_latency_ms", "nfull_latency_ms", "total_latency_ms"):
        value = float(record[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative, got {value!r}")


def _assert_source_matches_plan(
    metadata: Mapping[str, Any], selected_index: int, ranges: Sequence[Mapping[str, Any]]
) -> None:
    requested = int(metadata.get("requested_sample_idx", -1))
    source = int(metadata.get("source_sample_idx", -1))
    if requested != selected_index or source != selected_index:
        raise AssertionError(
            "Strict dataset identity mismatch: "
            f"plan={selected_index}, requested={requested}, source={source}"
        )
    dataset_index = int(metadata["dataset_index"])
    matching = [item for item in ranges if int(item["dataset_index"]) == dataset_index]
    if len(matching) != 1:
        raise AssertionError(f"Unknown dataset_index={dataset_index} in sample metadata")
    item = matching[0]
    if not int(item["start"]) <= source < int(item["stop"]):
        raise AssertionError(f"Source index {source} is outside declared stratum {item}")


def _validate_existing_records_against_dataset(
    records_path: Path,
    *,
    dataset: Any,
    ranges: Sequence[Mapping[str, Any]],
    task_tables: Mapping[int, Mapping[int, str]],
) -> int:
    """Rebind every resumed row to its actual strict dataset state."""

    if not records_path.exists():
        return 0
    verified = 0
    with records_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_metadata = record["source_metadata"]
            selected_index = int(source_metadata["requested_sample_idx"])
            sample = dataset[selected_index]
            actual_raw = sample.get("metadata")
            if not isinstance(actual_raw, Mapping):
                raise ValueError(
                    f"Dataset sample {selected_index} is missing mapping-valued metadata"
                )
            actual_metadata = _json_safe_metadata(actual_raw)
            _assert_source_matches_plan(actual_metadata, selected_index, ranges)
            dataset_index = int(actual_metadata["dataset_index"])
            if dataset_index not in task_tables:
                raise ValueError(
                    f"No task table for resumed sample dataset_index={dataset_index}"
                )
            identity = parse_sample_identity(
                actual_metadata,
                task_by_index=task_tables[dataset_index],
            )
            expected_identity = identity.to_dict()
            for field in (
                "sample_id",
                "dataset_id",
                "dataset_name",
                "suite",
                "episode_index",
                "episode_id",
                "frame_index",
                "task_index",
                "task_id",
                "task_id_source",
                "task",
            ):
                if record.get(field) != expected_identity[field]:
                    raise ValueError(
                        "Resumed record does not match its real dataset identity at "
                        f"{records_path}:{line_number}: field={field}, "
                        f"record={record.get(field)!r}, dataset={expected_identity[field]!r}"
                    )

            actual_hashes = current_state_input_hashes(extract_current_state(sample))
            if record.get("input_hashes") != actual_hashes:
                raise ValueError(
                    "Resumed record input hashes do not match the real dataset state at "
                    f"{records_path}:{line_number}"
                )
            actual_proprio = _current_proprio(sample)
            if record.get("current_proprio") != actual_proprio:
                raise ValueError(
                    "Resumed record current_proprio does not match the real dataset state at "
                    f"{records_path}:{line_number}"
                )
            verified += 1
    return verified


def _prepare_output_files(output_dir: Path, *, resume: bool) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if not resume:
        existing = [path for path in (manifest_path, records_path, errors_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "resume=false requires a fresh output directory; found: "
                + ", ".join(str(path) for path in existing)
            )
    return manifest_path, records_path, errors_path


def collect(cfg: DictConfig) -> dict[str, int]:
    _validate_endpoint_config(cfg)
    collector = cfg.COLLECTOR
    checkpoint_path = _resolve_existing_file(cfg.get("ckpt"), label="ckpt")
    stats_path = resolve_dataset_stats_path(
        checkpoint_path, collector.get("dataset_stats_path")
    )
    output_dir = Path(
        os.path.expandvars(os.path.expanduser(str(collector.output_dir)))
    ).resolve()
    manifest_path, records_path, errors_path = _prepare_output_files(
        output_dir, resume=bool(collector.resume)
    )

    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another collector is already using {output_dir}") from exc

        LOGGER.info("Hashing checkpoint, stats, VAE, dataset sources, and text cache")
        checkpoint_artifact = _stable_file_provenance(
            checkpoint_path, label="UniShare checkpoint"
        )
        stats_artifact = _stable_file_provenance(stats_path, label="dataset stats")
        vae_artifact = _resolve_vae_artifact(cfg)
        dataset_source_artifacts = []
        for dataset_index, raw_dataset_dir in enumerate(cfg.data.train.dataset_dirs):
            source_artifact = _directory_tree_provenance(
                raw_dataset_dir,
                label=f"LIBERO source dataset {dataset_index}",
            )
            source_artifact["dataset_index"] = dataset_index
            source_artifact["dataset_name"] = Path(source_artifact["path"]).name
            dataset_source_artifacts.append(source_artifact)
        context_cache_artifact = _directory_tree_provenance(
            cfg.data.train.text_embedding_cache_dir,
            label="LIBERO text embedding cache",
        )
        checkpoint_sha256 = str(checkpoint_artifact["sha256"])
        stats_sha256 = str(stats_artifact["sha256"])
        git = git_provenance()
        misc.register_work_dir(output_dir)
        instantiation_paths = _dataset_instantiation_path_overrides(
            dataset_source_artifacts, context_cache_artifact
        )

        # Explicit overrides prevent accidental training split semantics, random
        # fallback, padding avoidance, or caller-CWD path resolution from
        # changing the requested identity.
        dataset = instantiate(
            cfg.data.train,
            **instantiation_paths,
            is_training_set=False,
            pretrained_norm_stats=str(stats_path),
            strict_getitem=True,
            return_metadata=True,
            skip_padding_as_possible=False,
        )
        if not hasattr(dataset, "dataset_index_ranges"):
            raise TypeError(
                "RobotVideoDataset must expose public dataset_index_ranges() for auditable sampling"
            )
        ranges = _normalize_ranges(dataset.dataset_index_ranges())
        source_names = [str(item["dataset_name"]) for item in dataset_source_artifacts]
        range_names = [str(item["dataset_name"]) for item in ranges]
        if source_names != range_names:
            raise ValueError(
                "Configured source dataset content fingerprints do not match MultiLeRobotDataset "
                f"order: sources={source_names}, ranges={range_names}"
            )
        population = sum(int(item["population"]) for item in ranges)
        if population != len(dataset):
            raise ValueError(
                f"dataset_index_ranges population={population} does not equal len(dataset)={len(dataset)}"
            )
        if not hasattr(dataset, "dataset_task_table"):
            raise TypeError(
                "RobotVideoDataset must expose public dataset_task_table(dataset_index) "
                "for task ID/string validation"
            )
        task_tables: dict[int, dict[int, str]] = {}
        for item in ranges:
            dataset_index = int(item["dataset_index"])
            raw_table = dataset.dataset_task_table(dataset_index)
            if not isinstance(raw_table, Mapping) or not raw_table:
                raise ValueError(
                    f"dataset_task_table({dataset_index}) must return a non-empty mapping"
                )
            task_tables[dataset_index] = {
                int(task_index): str(task) for task_index, task in raw_table.items()
            }
        selected_indices, plan_strata = build_stratified_sample_plan(
            ranges,
            num_samples=int(collector.num_samples),
            seed=int(collector.seed),
        )
        manifest_payload = _build_manifest(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            stats_path=stats_path,
            stats_sha256=stats_sha256,
            vae_artifact=vae_artifact,
            dataset_source_artifacts=dataset_source_artifacts,
            context_cache_artifact=context_cache_artifact,
            git=git,
            ranges=ranges,
            task_tables=task_tables,
            plan_strata=plan_strata,
            selected_indices=selected_indices,
        )
        manifest_existed = manifest_path.exists()
        if not manifest_existed:
            nonempty_orphan_outputs = [
                path
                for path in (records_path, errors_path)
                if path.exists() and path.stat().st_size > 0
            ]
            if nonempty_orphan_outputs:
                raise ValueError(
                    "Found non-empty Collector outputs without an immutable manifest: "
                    + ", ".join(str(path) for path in nonempty_orphan_outputs)
                )
            manifest = manifest_payload
        else:
            manifest = ensure_immutable_manifest(manifest_path, manifest_payload)
        manifest_fingerprint = str(manifest_payload["compatibility_fingerprint"])

        sample_ids, completed_source_indices, existing_count = load_existing_record_index(
            records_path,
            expected_manifest_fingerprint=manifest_fingerprint,
            expected_full_steps=int(collector.full_prefix_steps),
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_dataset_stats_sha256=stats_sha256,
            expected_vae_sha256=str(vae_artifact["sha256"]),
            expected_git_sha=str(git["commit"]),
            expected_base_seed=int(collector.seed),
        )
        selected_set = set(selected_indices)
        out_of_plan = set(completed_source_indices) - selected_set
        if out_of_plan:
            raise ValueError(
                "Existing records contain source indices outside the immutable plan: "
                f"{sorted(out_of_plan)[:10]}"
            )
        verified_existing = _validate_existing_records_against_dataset(
            records_path,
            dataset=dataset,
            ranges=ranges,
            task_tables=task_tables,
        )
        if verified_existing != existing_count:
            raise AssertionError(
                "Dataset-verified resume count differs from parsed completed rows: "
                f"verified={verified_existing}, parsed={existing_count}"
            )
        pending_indices = [
            index for index in selected_indices if index not in completed_source_indices
        ]
        LOGGER.info(
            "Collector plan: selected=%d existing=%d pending=%d output=%s",
            len(selected_indices),
            existing_count,
            len(pending_indices),
            output_dir,
        )
        if not pending_indices:
            if not manifest_existed:
                raise AssertionError("Complete records cannot exist without an immutable manifest")
            return {"selected": len(selected_indices), "existing": existing_count, "new": 0, "errors": 0}

        device = str(collector.device)
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("COLLECTOR.device is CUDA but torch.cuda.is_available() is false")
            if resolved_device.index is not None:
                torch.cuda.set_device(resolved_device)
        model = instantiate(
            cfg.model,
            model_dtype=_mixed_precision_dtype(cfg.get("mixed_precision", "bf16")),
            device=device,
        )
        model_paths = getattr(model, "model_paths", None)
        if not isinstance(model_paths, Mapping) or not model_paths.get("vae"):
            raise ValueError("Instantiated model did not report model_paths['vae']")
        actual_model_vae_path = _resolve_project_path(
            model_paths["vae"], label="model-reported VAE artifact"
        )
        expected_model_vae_path = Path(str(vae_artifact["path"]))
        if actual_model_vae_path != expected_model_vae_path:
            raise ValueError(
                "Preflight/model VAE path mismatch: "
                f"preflight={expected_model_vae_path}, model={actual_model_vae_path}"
            )
        _assert_file_provenance_unchanged(vae_artifact, label="VAE artifact")
        checkpoint_payload = model.load_checkpoint(str(checkpoint_path))
        _assert_file_provenance_unchanged(
            checkpoint_artifact, label="UniShare checkpoint"
        )
        checkpoint_load = _checkpoint_payload_provenance(model, checkpoint_payload)
        del checkpoint_payload
        gc.collect()
        model.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("Frozen Collector model still has trainable parameters")

        if manifest_existed:
            recorded_load = manifest.get("artifacts", {}).get("checkpoint", {}).get("load")
            if recorded_load != checkpoint_load:
                raise ValueError(
                    "Checkpoint load provenance differs from immutable manifest: "
                    f"recorded={recorded_load!r}, current={checkpoint_load!r}"
                )
        else:
            manifest_payload["artifacts"]["checkpoint"]["load"] = checkpoint_load
            manifest = ensure_immutable_manifest(manifest_path, manifest_payload)
        records_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)

        new_count = 0
        error_count = 0
        with records_path.open("a", encoding="utf-8") as records_stream, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_stream:
            for selected_index in tqdm(pending_indices, desc="paired demo utility"):
                try:
                    sample = dataset[selected_index]
                    metadata_raw = sample.get("metadata")
                    if not isinstance(metadata_raw, Mapping):
                        raise ValueError("Strict dataset sample is missing mapping-valued metadata")
                    metadata = _json_safe_metadata(metadata_raw)
                    _assert_source_matches_plan(metadata, selected_index, ranges)
                    dataset_index = int(metadata["dataset_index"])
                    if dataset_index not in task_tables:
                        raise ValueError(
                            f"No cached task table for sample dataset_index={dataset_index}"
                        )

                    with torch.inference_mode():
                        result = collect_paired_utility(
                            model,
                            sample,
                            metadata=metadata,
                            base_seed=int(collector.seed),
                            num_inference_steps=int(collector.num_inference_steps),
                            full_prefix_steps=int(collector.full_prefix_steps),
                            num_video_frames=int(collector.num_video_frames),
                            rand_device=str(collector.rand_device),
                            sigma_shift=(
                                None
                                if collector.get("sigma_shift") is None
                                else float(collector.sigma_shift)
                            ),
                            tiled=bool(collector.tiled),
                            force_custom_prefix=bool(collector.force_custom_prefix),
                            task_by_index=task_tables[dataset_index],
                        )
                    record = result.to_dict() if hasattr(result, "to_dict") else dict(result)
                    _validate_record(record, full_steps=int(collector.full_prefix_steps))
                    sample_id = str(record["sample_id"])
                    if sample_id in sample_ids:
                        raise ValueError(f"Duplicate sample_id generated by plan: {sample_id}")

                    record.update(
                        {
                            "collector_record_schema_version": RECORD_AUGMENTATION_SCHEMA_VERSION,
                            "source_metadata": metadata,
                            "current_proprio": _current_proprio(sample),
                            "manifest_compatibility_fingerprint": manifest_fingerprint,
                            "checkpoint_sha256": checkpoint_sha256,
                            "dataset_stats_sha256": stats_sha256,
                            "vae_sha256": vae_artifact["sha256"],
                            "git_sha": git["commit"],
                        }
                    )
                    _validate_completed_record(
                        record, full_steps=int(collector.full_prefix_steps)
                    )
                    _write_jsonl(records_stream, record)
                    sample_ids.add(sample_id)
                    completed_source_indices[selected_index] = sample_id
                    new_count += 1
                except Exception as exc:
                    error_count += 1
                    _write_jsonl(
                        errors_stream,
                        {
                            "timestamp_utc": _utc_now(),
                            "selected_source_index": int(selected_index),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "manifest_compatibility_fingerprint": manifest_fingerprint,
                        },
                    )
                    LOGGER.exception("Failed paired utility sample source_index=%d", selected_index)
                    if not bool(collector.continue_on_error):
                        raise
                    if error_count >= int(collector.max_errors):
                        raise RuntimeError(
                            f"Collector reached max_errors={collector.max_errors}"
                        ) from exc

        total_completed = len(completed_source_indices)
        if total_completed != len(selected_indices):
            raise RuntimeError(
                "Collection is incomplete; fix errors and resume with the same output directory. "
                f"completed={total_completed}/{len(selected_indices)}, errors_this_run={error_count}"
            )
        _assert_file_provenance_unchanged(
            checkpoint_artifact, label="UniShare checkpoint"
        )
        _assert_file_provenance_unchanged(stats_artifact, label="dataset stats")
        _assert_file_provenance_unchanged(vae_artifact, label="VAE artifact")
        for source_artifact in dataset_source_artifacts:
            _assert_directory_tree_provenance_unchanged(
                source_artifact,
                label=f"LIBERO source dataset {source_artifact['dataset_index']}",
            )
        _assert_directory_tree_provenance_unchanged(
            context_cache_artifact,
            label="LIBERO text embedding cache",
        )
        return {
            "selected": len(selected_indices),
            "existing": existing_count,
            "new": new_count,
            "errors": error_count,
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_demo_utility.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = collect(cfg)
    LOGGER.info("Demo utility collection complete: %s", summary)


if __name__ == "__main__":
    main()
