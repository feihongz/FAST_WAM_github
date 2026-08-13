"""Collect an independent four-seed validation grid for Utility Target V2.

This collector is deliberately separate from both the Phase-2.5 stability
collector and Target-V2 construction.  It evaluates the exact same 100 states
with base seeds 47--50; every cell is fresh paired N=0/N=full inference.  The
immutable output binds the source Phase-2.5 bytes, the Target-V2 bytes, the
ordered state plan, and all scientific artifacts needed to reproduce it.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.gate import collect_demo_utility as single
from experiments.libero.gate import collect_demo_utility_multiseed as phase25
from experiments.libero.gate import demo_utility_stability as stability
from experiments.libero.gate import demo_utility_target_v2 as target_v2
from experiments.libero.gate.demo_utility import (
    collect_paired_utility,
    current_state_input_hashes,
    extract_current_state,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1
VALIDATION_RECORD_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
AUDIT_KIND = "libero_demo_utility_target_v2_independent_validation"
COMPLETION_KIND = "libero_demo_utility_target_v2_independent_validation_completion"
V1_NUM_STATES = 100
VALIDATION_BASE_SEEDS = (47, 48, 49, 50)
GLOBAL_SEED_INDEX_OFFSET = 5
SCIENTIFIC_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            "experiments/libero/gate/collect_demo_utility_target_v2_validation.py",
            "experiments/libero/gate/demo_utility_target_v2.py",
            *phase25.SCIENTIFIC_SOURCE_FILES,
        )
    )
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _require_bundle_dir(raw_dir: Any, *, label: str) -> Path:
    if raw_dir is None or not str(raw_dir).strip():
        raise ValueError(f"COLLECTOR.{label}_dir must point to a completed bundle")
    return Path(os.path.expandvars(os.path.expanduser(str(raw_dir)))).resolve()


def _validate_validation_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if seeds != VALIDATION_BASE_SEEDS:
        raise ValueError(
            f"Independent validation base seeds must be exactly "
            f"{list(VALIDATION_BASE_SEEDS)}, got {list(seeds)}"
        )
    if set(seeds) & set(stability.DEFAULT_REPLICATE_BASE_SEEDS):
        raise ValueError("Independent validation seeds overlap Target-V2 source seeds")
    return seeds


def _validate_v1_config(cfg: DictConfig) -> tuple[int, ...]:
    collector = cfg.COLLECTOR
    single._validate_endpoint_config(cfg)
    if int(collector.num_states) != V1_NUM_STATES:
        raise ValueError(
            f"Target-V2 validation V1 requires num_states={V1_NUM_STATES}; "
            f"got {collector.num_states}"
        )
    seeds = _validate_validation_seeds(
        [int(value) for value in collector.validation_base_seeds]
    )
    if int(collector.global_seed_index_offset) != GLOBAL_SEED_INDEX_OFFSET:
        raise ValueError(
            f"global_seed_index_offset must equal {GLOBAL_SEED_INDEX_OFFSET}"
        )
    if int(collector.max_errors) <= 0:
        raise ValueError("COLLECTOR.max_errors must be positive")
    return seeds


def _resolve_phase25_bundle(
    raw_dir: Any,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = _require_bundle_dir(raw_dir, label="phase25")
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "Phase-2.5 directory must contain manifest.json and records.jsonl: "
            f"{root}"
        )
    manifest = _load_json(manifest_path, label="Phase-2.5 manifest")
    phase25._validate_manifest_integrity(manifest)
    return root, manifest_path, records_path, manifest


def _resolve_target_bundle_paths(raw_dir: Any) -> tuple[Path, Path, Path]:
    root = _require_bundle_dir(raw_dir, label="target_v2")
    manifest_path = root / target_v2.TARGET_MANIFEST_FILENAME
    targets_path = root / target_v2.TARGETS_FILENAME
    if not manifest_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError(
            "Target-V2 directory must contain manifest.json and targets.jsonl: "
            f"{root}"
        )
    return root, manifest_path, targets_path


def _scientific_source_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in SCIENTIFIC_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Scientific source file is missing: {path}")
        result[relative_path] = single._sha256_file(path)
    return result


def _phase25_compatibility_checks(
    *, cfg: DictConfig, manifest: Mapping[str, Any]
) -> None:
    compatibility = manifest["compatibility"]
    collector = cfg.COLLECTOR
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Resolved validation config must be a mapping")
    current = {
        "model_config_sha256": single._sha256_json(resolved["model"]),
        "data_config_sha256": single._sha256_json(
            single._scientific_data_config(resolved["data"])
        ),
        "mixed_precision": str(cfg.get("mixed_precision", "bf16")),
        "execution_environment": {
            "device": str(collector.device),
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        },
    }
    for field, current_value in current.items():
        if compatibility.get(field) != current_value:
            raise ValueError(
                "Validation scientific configuration differs from Phase 2.5: "
                f"field={field}, source={compatibility.get(field)!r}, "
                f"current={current_value!r}"
            )
    source_parameters = compatibility.get("collection_parameters", {})
    current_parameters = {
        "num_inference_steps": int(collector.num_inference_steps),
        "full_prefix_steps": int(collector.full_prefix_steps),
        "num_video_frames": int(collector.num_video_frames),
        "rand_device": str(collector.rand_device),
        "sigma_shift": (
            None if collector.get("sigma_shift") is None else float(collector.sigma_shift)
        ),
        "tiled": bool(collector.tiled),
        "force_custom_prefix": bool(collector.force_custom_prefix),
    }
    for field, current_value in current_parameters.items():
        if source_parameters.get(field) != current_value:
            raise ValueError(
                "Validation endpoint differs from Phase 2.5: "
                f"field={field}, source={source_parameters.get(field)!r}, "
                f"current={current_value!r}"
            )


def _selection_projection(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for fallback_order, record in enumerate(targets):
        input_hashes = record.get("input_hashes")
        if not isinstance(input_hashes, Mapping):
            raise ValueError("Target-V2 record has no input_hashes")
        projection.append(
            {
                "selection_order": int(record.get("selection_order", fallback_order)),
                "source_index": int(record["source_index"]),
                "sample_id": str(record["sample_id"]),
                "target_id": str(record["target_id"]),
                "target_sha256": str(record["target_sha256"]),
                "input_combined_sha256": str(input_hashes["combined"]),
            }
        )
    projection.sort(key=lambda item: item["selection_order"])
    if [item["selection_order"] for item in projection] != list(range(len(projection))):
        raise ValueError("Target-V2 selection order must be exactly 0..num_states-1")
    if len({item["source_index"] for item in projection}) != len(projection):
        raise ValueError("Target-V2 selection contains duplicate source indices")
    return projection


def _target_selection_source_sha(manifest: Mapping[str, Any]) -> str:
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Target-V2 manifest compatibility must be a mapping")
    value = compatibility.get("source_selection_plan_sha256")
    return single._require_sha256(value, field="source_selection_plan_sha256")


def _build_manifest(
    *,
    cfg: DictConfig,
    phase25_root: Path,
    phase25_manifest_path: Path,
    phase25_records_path: Path,
    phase25_manifest: Mapping[str, Any],
    target_root: Path,
    target_manifest_path: Path,
    targets_path: Path,
    target_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    base_seeds: Sequence[int],
    checkpoint_path: Path,
    stats_path: Path,
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    dataset_sources: Sequence[Mapping[str, Any]],
    vae: Mapping[str, Any],
    context_cache: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    collector = cfg.COLLECTOR
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved validation config must be a dictionary")
    ordered_targets = _selection_projection(targets)
    selection_sha256 = single._sha256_json(ordered_targets)
    phase25_manifest_sha256 = single._sha256_file(phase25_manifest_path)
    phase25_records_sha256 = single._sha256_file(phase25_records_path)
    target_manifest_sha256 = single._sha256_file(target_manifest_path)
    target_targets_sha256 = single._sha256_file(targets_path)
    phase25_selection_sha = str(phase25_manifest["compatibility"]["selection_plan_sha256"])
    target_selection_sha = _target_selection_source_sha(target_manifest)
    if target_selection_sha != phase25_selection_sha:
        raise ValueError("Target-V2 and Phase-2.5 selection-plan digests differ")
    global_seed_indices = [
        int(collector.global_seed_index_offset) + index
        for index in range(len(base_seeds))
    ]
    source_files = _scientific_source_provenance()
    compatibility = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "phase25_manifest_fingerprint": str(
            phase25_manifest["compatibility_fingerprint"]
        ),
        "phase25_manifest_sha256": phase25_manifest_sha256,
        "phase25_records_sha256": phase25_records_sha256,
        "phase25_selection_plan_sha256": phase25_selection_sha,
        "target_v2_manifest_fingerprint": str(
            target_manifest["compatibility_fingerprint"]
        ),
        "target_v2_manifest_sha256": target_manifest_sha256,
        "target_v2_targets_sha256": target_targets_sha256,
        "target_v2_selection_plan_sha256": target_selection_sha,
        "validation_selection_sha256": selection_sha256,
        "num_states": len(ordered_targets),
        "validation_base_seeds": [int(value) for value in base_seeds],
        "global_seed_indices": global_seed_indices,
        "expected_record_count": len(ordered_targets) * len(base_seeds),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "dataset_stats_sha256": str(stats["sha256"]),
        "vae_sha256": str(vae["sha256"]),
        "dataset_source_content": [
            {
                "dataset_name": str(item["dataset_name"]),
                "sha256": str(item["sha256"]),
                "file_count": int(item["file_count"]),
                "total_size_bytes": int(item["total_size_bytes"]),
            }
            for item in dataset_sources
        ],
        "context_cache_sha256": str(context_cache["sha256"]),
        "collection_git_commit": str(git["commit"]),
        "collection_git_tracked_diff_sha256": str(git["tracked_diff_sha256"]),
        "scientific_source_files": source_files,
        "model_config_sha256": single._sha256_json(resolved["model"]),
        "data_config_sha256": single._sha256_json(
            single._scientific_data_config(resolved["data"])
        ),
        "mixed_precision": str(cfg.get("mixed_precision", "bf16")),
        "execution_environment": {
            "device": str(collector.device),
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        },
        "collection_parameters": {
            "num_inference_steps": int(collector.num_inference_steps),
            "full_prefix_steps": int(collector.full_prefix_steps),
            "num_video_frames": int(collector.num_video_frames),
            "rand_device": str(collector.rand_device),
            "sigma_shift": (
                None
                if collector.get("sigma_shift") is None
                else float(collector.sigma_shift)
            ),
            "tiled": bool(collector.tiled),
            "force_custom_prefix": bool(collector.force_custom_prefix),
            "all_new_inference": True,
        },
    }
    fingerprint = single._sha256_json(compatibility)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "created_at_utc": single._utc_now(),
        "compatibility_fingerprint": fingerprint,
        "compatibility": compatibility,
        "phase25": {
            "directory": str(phase25_root),
            "manifest_path": str(phase25_manifest_path),
            "records_path": str(phase25_records_path),
            "manifest_fingerprint": str(phase25_manifest["compatibility_fingerprint"]),
            "manifest_sha256": phase25_manifest_sha256,
            "records_sha256": phase25_records_sha256,
            "selection_plan_sha256": phase25_selection_sha,
        },
        "target_v2": {
            "directory": str(target_root),
            "manifest_path": str(target_manifest_path),
            "targets_path": str(targets_path),
            "manifest_fingerprint": str(target_manifest["compatibility_fingerprint"]),
            "manifest_sha256": target_manifest_sha256,
            "targets_sha256": target_targets_sha256,
            "selection_plan_sha256": target_selection_sha,
        },
        "selection": {
            "algorithm": "exact-target-v2-source-order-v1",
            "num_states": len(ordered_targets),
            "ordered_targets": ordered_targets,
            "ordered_targets_sha256": selection_sha256,
        },
        "replicates": {
            "base_seeds": [int(value) for value in base_seeds],
            "global_seed_indices": global_seed_indices,
            "count": len(base_seeds),
            "expected_record_count": len(ordered_targets) * len(base_seeds),
            "all_new_inference": True,
        },
        "artifacts": {
            "checkpoint": {**dict(checkpoint), "path": str(checkpoint_path)},
            "dataset_stats": {**dict(stats), "path": str(stats_path)},
            "vae": dict(vae),
            "dataset_sources": [dict(item) for item in dataset_sources],
            "text_embedding_cache": dict(context_cache),
        },
        "git": dict(git),
        "runtime": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "resolved_config": resolved,
        "resolved_config_sha256": single._sha256_json(resolved),
        "scientific_source_files": source_files,
        "validation_record_schema_version": VALIDATION_RECORD_SCHEMA_VERSION,
    }


def _validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported validation manifest schema")
    if manifest.get("kind") != AUDIT_KIND:
        raise ValueError(f"Unexpected validation manifest kind={manifest.get('kind')!r}")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Validation manifest compatibility must be a mapping")
    if manifest.get("compatibility_fingerprint") != single._sha256_json(compatibility):
        raise ValueError("Validation manifest compatibility fingerprint is invalid")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("ordered_targets"), list
    ):
        raise ValueError("Validation manifest has no ordered target selection")
    ordered_targets = selection["ordered_targets"]
    selection_sha = single._sha256_json(ordered_targets)
    if selection.get("ordered_targets_sha256") != selection_sha:
        raise ValueError("Validation manifest selection digest is invalid")
    if compatibility.get("validation_selection_sha256") != selection_sha:
        raise ValueError("Validation compatibility is not bound to selection")
    if int(selection.get("num_states", -1)) != len(ordered_targets):
        raise ValueError("Validation manifest selection count is invalid")
    if int(compatibility.get("num_states", -1)) != len(ordered_targets):
        raise ValueError("Validation compatibility state count is invalid")

    phase25_binding = manifest.get("phase25")
    target_binding = manifest.get("target_v2")
    if not isinstance(phase25_binding, Mapping):
        raise ValueError("Validation manifest has no Phase-2.5 binding")
    if not isinstance(target_binding, Mapping):
        raise ValueError("Validation manifest has no Target-V2 binding")
    for field, compatibility_field in (
        ("manifest_fingerprint", "phase25_manifest_fingerprint"),
        ("manifest_sha256", "phase25_manifest_sha256"),
        ("records_sha256", "phase25_records_sha256"),
        ("selection_plan_sha256", "phase25_selection_plan_sha256"),
    ):
        if phase25_binding.get(field) != compatibility.get(compatibility_field):
            label = "Phase-2.5 " + field
            raise ValueError(f"{label} is not bound to compatibility")
    for field, compatibility_field in (
        ("manifest_fingerprint", "target_v2_manifest_fingerprint"),
        ("manifest_sha256", "target_v2_manifest_sha256"),
        ("targets_sha256", "target_v2_targets_sha256"),
        ("selection_plan_sha256", "target_v2_selection_plan_sha256"),
    ):
        if target_binding.get(field) != compatibility.get(compatibility_field):
            label = "Target-V2 " + field
            raise ValueError(f"{label} is not bound to compatibility")
    if phase25_binding.get("selection_plan_sha256") != target_binding.get(
        "selection_plan_sha256"
    ):
        raise ValueError("Phase-2.5 and Target-V2 selection bindings differ")

    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("Validation manifest has no replicate plan")
    seeds = [int(value) for value in replicates.get("base_seeds", [])]
    _validate_validation_seeds(seeds)
    if seeds != [int(value) for value in compatibility.get("validation_base_seeds", [])]:
        raise ValueError("Validation seed plan is not bound to compatibility")
    global_indices = [int(value) for value in replicates.get("global_seed_indices", [])]
    if global_indices != list(range(GLOBAL_SEED_INDEX_OFFSET, GLOBAL_SEED_INDEX_OFFSET + 4)):
        raise ValueError("Validation global seed indices must be exactly 5..8")
    if global_indices != [int(value) for value in compatibility.get("global_seed_indices", [])]:
        raise ValueError("Global seed indices are not bound to compatibility")
    if replicates.get("all_new_inference") is not True:
        raise ValueError("Every validation replicate must be new inference")
    expected_count = len(ordered_targets) * len(seeds)
    if int(replicates.get("expected_record_count", -1)) != expected_count:
        raise ValueError("Expected validation grid size is invalid")
    if int(compatibility.get("expected_record_count", -1)) != expected_count:
        raise ValueError("Compatibility expected record count is invalid")


def _ensure_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_manifest_integrity(payload)
    if path.exists():
        existing = _load_json(path, label="validation manifest")
        _validate_manifest_integrity(existing)
        if existing["compatibility_fingerprint"] != payload["compatibility_fingerprint"]:
            raise ValueError(
                "Existing validation manifest is incompatible with this run; "
                "use its original inputs/config or a new output directory"
            )
        return existing
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(payload)


def _prepare_output_files(output_dir: Path, *, resume: bool) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "manifest.json",
        output_dir / "records.jsonl",
        output_dir / "errors.jsonl",
    )
    if not resume:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "resume=false requires a fresh output directory; found: "
                + ", ".join(str(path) for path in existing)
            )
    return paths


def _jsonl_record_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Expected JSONL file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _ordered_validation_record_digest(path: Path) -> str:
    hashes: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed validation record at {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Validation record at {path}:{line_number} must be an object"
                )
            hashes.append(
                single._require_sha256(
                    row.get("validation_record_sha256"),
                    field=f"records[{line_number}].validation_record_sha256",
                )
            )
    return single._sha256_json(hashes)


def _completion_payload(
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the final byte-level seal after the complete grid is closed."""

    _validate_manifest_integrity(manifest)
    records_count = _jsonl_record_count(records_path)
    errors_count = _jsonl_record_count(errors_path)
    expected_count = int(manifest["compatibility"]["expected_record_count"])
    if records_count != expected_count:
        raise ValueError(
            f"Cannot seal incomplete validation records: {records_count}/{expected_count}"
        )
    if errors_count != 0:
        raise ValueError(
            f"Cannot seal validation output with {errors_count} recorded errors"
        )
    compatibility = manifest["compatibility"]
    payload = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "completed_at_utc": single._utc_now(),
        "validation_manifest_fingerprint": manifest["compatibility_fingerprint"],
        "validation_manifest_sha256": single._sha256_file(manifest_path),
        "records_sha256": single._sha256_file(records_path),
        "records_count": records_count,
        "ordered_validation_record_sha256_sha256": (
            _ordered_validation_record_digest(records_path)
        ),
        "errors_sha256": single._sha256_file(errors_path),
        "errors_count": errors_count,
        "expected_record_count": expected_count,
        "phase25_manifest_sha256": compatibility["phase25_manifest_sha256"],
        "phase25_records_sha256": compatibility["phase25_records_sha256"],
        "target_v2_manifest_sha256": compatibility["target_v2_manifest_sha256"],
        "target_v2_targets_sha256": compatibility["target_v2_targets_sha256"],
    }
    payload["completion_sha256"] = single._sha256_json(payload)
    return payload


def _validate_completion_seal(
    completion_path: Path,
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seal = _load_json(completion_path, label="validation completion seal")
    if int(seal.get("schema_version", -1)) != COMPLETION_SCHEMA_VERSION:
        raise ValueError("Unsupported validation completion schema")
    if seal.get("kind") != COMPLETION_KIND:
        raise ValueError("Validation completion kind is invalid")
    stored_digest = single._require_sha256(
        seal.get("completion_sha256"), field="completion_sha256"
    )
    digest_payload = {
        key: value for key, value in seal.items() if key != "completion_sha256"
    }
    if stored_digest != single._sha256_json(digest_payload):
        raise ValueError("Validation completion digest is invalid")
    current = _completion_payload(
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
    )
    for field, value in current.items():
        if field in {"completed_at_utc", "completion_sha256"}:
            continue
        if seal.get(field) != value:
            raise ValueError(f"Validation completion seal mismatch for {field}")
    return seal


def _ensure_completion_seal(
    completion_path: Path,
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if completion_path.exists():
        return _validate_completion_seal(
            completion_path,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
        )
    payload = _completion_payload(
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=completion_path.parent,
            prefix=".completion.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, completion_path)
        temporary_path = None
        directory_fd = os.open(completion_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return _validate_completion_seal(
        completion_path,
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
    )


def collect_validation_grid(
    *,
    dataset: Any,
    model: Any,
    target_records: Sequence[Mapping[str, Any]],
    base_seeds: Sequence[int],
    existing_keys: set[tuple[int, int]],
    infer_record: Callable[[Any, Any, Mapping[str, Any], int], Mapping[str, Any]],
    finalize_record: Callable[
        [Mapping[str, Any], Mapping[str, Any], int, int, Any], Mapping[str, Any]
    ],
    write_record: Callable[[Mapping[str, Any]], None],
    on_error: Callable[[int, int, BaseException], None] | None = None,
    continue_on_error: bool = False,
    max_errors: int = 1,
) -> dict[str, int]:
    """Run a state-outer/seed-inner grid with no source-measurement reuse."""

    seeds = _validate_validation_seeds(base_seeds)
    new_count = inferred_count = error_count = 0
    for target_record in target_records:
        source_index = int(target_record["source_index"])
        pending = [
            (validation_index, base_seed)
            for validation_index, base_seed in enumerate(seeds)
            if (source_index, validation_index) not in existing_keys
        ]
        if not pending:
            continue
        if model is None:
            raise RuntimeError("Validation inference is pending but model is not loaded")
        sample = dataset[source_index]
        for validation_index, base_seed in pending:
            try:
                utility_record = infer_record(model, sample, target_record, base_seed)
                completed = finalize_record(
                    utility_record,
                    target_record,
                    validation_index,
                    base_seed,
                    sample,
                )
                write_record(completed)
                existing_keys.add((source_index, validation_index))
                new_count += 1
                inferred_count += 1
            except Exception as exc:
                error_count += 1
                if on_error is not None:
                    on_error(source_index, validation_index, exc)
                if not continue_on_error:
                    raise
                if error_count >= int(max_errors):
                    raise RuntimeError(f"Collector reached max_errors={max_errors}") from exc
    return {
        "new": new_count,
        "inferred": inferred_count,
        "reused": 0,
        "errors": error_count,
    }


def _resolve_source_pilot_manifest(
    phase25_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Load the Pilot manifest needed for dataset/checkpoint structure checks."""

    binding = phase25_manifest.get("pilot")
    if not isinstance(binding, Mapping):
        raise ValueError("Phase-2.5 manifest has no Pilot binding")
    raw_path = binding.get("manifest_path")
    if raw_path is None:
        raise ValueError("Phase-2.5 manifest has no Pilot manifest path")
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path)))).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Phase-2.5's bound Pilot manifest is unavailable; restore the exact "
            f"source bundle at {path}"
        )
    expected_sha = single._require_sha256(
        binding.get("manifest_sha256"), field="phase25.pilot.manifest_sha256"
    )
    if single._sha256_file(path) != expected_sha:
        raise ValueError("Phase-2.5's bound Pilot manifest bytes have changed")
    manifest = _load_json(path, label="Phase-2.5 source Pilot manifest")
    single._validate_manifest_integrity(manifest)
    if manifest.get("compatibility_fingerprint") != binding.get("manifest_fingerprint"):
        raise ValueError("Phase-2.5 Pilot manifest fingerprint binding is invalid")
    return path, manifest


def _verify_target_source_binding(
    *,
    target_manifest: Mapping[str, Any],
    phase25_manifest: Mapping[str, Any],
    phase25_manifest_path: Path,
    phase25_records_path: Path,
) -> None:
    compatibility = target_manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Target-V2 manifest compatibility must be a mapping")
    expected = {
        "source_manifest_sha256": single._sha256_file(phase25_manifest_path),
        "source_records_sha256": single._sha256_file(phase25_records_path),
        "source_manifest_compatibility_fingerprint": str(
            phase25_manifest["compatibility_fingerprint"]
        ),
        "source_selection_plan_sha256": str(
            phase25_manifest["compatibility"]["selection_plan_sha256"]
        ),
    }
    for field, expected_value in expected.items():
        if compatibility.get(field) != expected_value:
            raise ValueError(
                f"Target-V2 {field} does not bind the supplied Phase-2.5 bundle"
            )


def _expected_validation_keys(
    targets: Sequence[Mapping[str, Any]], base_seeds: Sequence[int]
) -> set[tuple[int, int]]:
    return {
        (int(target["source_index"]), validation_index)
        for target in targets
        for validation_index in range(len(base_seeds))
    }


def _dataset_task_tables(
    dataset: Any,
    *,
    source_pilot_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[int, str]]]:
    ranges = single._normalize_ranges(dataset.dataset_index_ranges())
    expected_ranges = source_pilot_manifest["compatibility"]["dataset_index_ranges"]
    if phase25._scientific_dataset_ranges(ranges) != phase25._scientific_dataset_ranges(
        expected_ranges
    ):
        raise ValueError("Instantiated dataset ranges differ from Phase-2.5 source Pilot")
    task_tables: dict[int, dict[int, str]] = {}
    for item in ranges:
        dataset_index = int(item["dataset_index"])
        task_tables[dataset_index] = {
            int(key): str(value)
            for key, value in dataset.dataset_task_table(dataset_index).items()
        }
    normalized = {
        str(key): {str(inner_key): value for inner_key, value in table.items()}
        for key, table in task_tables.items()
    }
    if normalized != source_pilot_manifest["compatibility"]["dataset_task_tables"]:
        raise ValueError("Instantiated task tables differ from Phase-2.5 source Pilot")
    return ranges, task_tables


def collect(cfg: DictConfig) -> dict[str, int]:
    base_seeds = _validate_v1_config(cfg)
    collector = cfg.COLLECTOR
    (
        phase25_root,
        phase25_manifest_path,
        phase25_records_path,
        phase25_manifest,
    ) = _resolve_phase25_bundle(collector.phase25_dir)
    expected_phase25_manifest_sha = single._require_sha256(
        collector.expected_phase25_manifest_sha256,
        field="COLLECTOR.expected_phase25_manifest_sha256",
    )
    expected_phase25_records_sha = single._require_sha256(
        collector.expected_phase25_records_sha256,
        field="COLLECTOR.expected_phase25_records_sha256",
    )
    if single._sha256_file(phase25_manifest_path) != expected_phase25_manifest_sha:
        raise ValueError("Phase-2.5 manifest does not match preregistered SHA-256")
    if single._sha256_file(phase25_records_path) != expected_phase25_records_sha:
        raise ValueError("Phase-2.5 records do not match preregistered SHA-256")
    _phase25_compatibility_checks(cfg=cfg, manifest=phase25_manifest)

    target_root, target_manifest_path, targets_path = _resolve_target_bundle_paths(
        collector.target_v2_dir
    )
    expected_target_manifest_sha = single._require_sha256(
        collector.expected_target_v2_manifest_sha256,
        field="COLLECTOR.expected_target_v2_manifest_sha256",
    )
    expected_target_targets_sha = single._require_sha256(
        collector.expected_target_v2_targets_sha256,
        field="COLLECTOR.expected_target_v2_targets_sha256",
    )
    target_manifest, targets = target_v2.load_target_bundle(
        target_root,
        expected_manifest_sha256=expected_target_manifest_sha,
        expected_targets_sha256=expected_target_targets_sha,
        expected_num_states=V1_NUM_STATES,
    )
    _verify_target_source_binding(
        target_manifest=target_manifest,
        phase25_manifest=phase25_manifest,
        phase25_manifest_path=phase25_manifest_path,
        phase25_records_path=phase25_records_path,
    )
    validation_plan = target_v2.build_validation_plan(
        target_manifest, targets, base_seeds=base_seeds
    )
    expected_plan_count = V1_NUM_STATES * len(base_seeds)
    if len(validation_plan) != expected_plan_count:
        raise ValueError(
            f"Target-V2 validation plan has {len(validation_plan)} cells, "
            f"expected {expected_plan_count}"
        )

    output_dir = Path(
        os.path.expandvars(os.path.expanduser(str(collector.output_dir)))
    ).resolve()
    manifest_path, records_path, errors_path = _prepare_output_files(
        output_dir, resume=bool(collector.resume)
    )
    completion_path = output_dir / "completion.json"
    if not bool(collector.resume) and completion_path.exists():
        raise FileExistsError(
            f"resume=false requires a fresh output directory; found: {completion_path}"
        )
    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another collector is already using {output_dir}") from exc

        LOGGER.info("Hashing and rebinding Phase-2.5/Target-V2 scientific artifacts")
        source_pilot_manifest_path, source_pilot_manifest = (
            _resolve_source_pilot_manifest(phase25_manifest)
        )
        (
            checkpoint_path,
            stats_path,
            checkpoint_artifact,
            stats_artifact,
            dataset_source_artifacts,
            auxiliary_artifacts,
        ) = phase25._current_artifacts(cfg, pilot_manifest=source_pilot_manifest)
        vae_artifact = auxiliary_artifacts["vae"]
        context_cache_artifact = auxiliary_artifacts["text_embedding_cache"]

        source_compatibility = phase25_manifest["compatibility"]
        for label, current, expected in (
            ("checkpoint", checkpoint_artifact["sha256"], source_compatibility["checkpoint_sha256"]),
            ("dataset stats", stats_artifact["sha256"], source_compatibility["dataset_stats_sha256"]),
            ("VAE", vae_artifact["sha256"], source_compatibility["vae_sha256"]),
            ("text embedding cache", context_cache_artifact["sha256"], source_compatibility["context_cache_sha256"]),
        ):
            if current != expected:
                raise ValueError(f"{label} differs from Phase 2.5")
        source_dataset_content = [
            {
                "dataset_name": item["dataset_name"],
                "sha256": item["sha256"],
                "file_count": int(item["file_count"]),
                "total_size_bytes": int(item["total_size_bytes"]),
            }
            for item in dataset_source_artifacts
        ]
        if source_dataset_content != source_compatibility["dataset_source_content"]:
            raise ValueError("LIBERO dataset bytes/order differ from Phase 2.5")

        git = single.git_provenance()
        manifest_payload = _build_manifest(
            cfg=cfg,
            phase25_root=phase25_root,
            phase25_manifest_path=phase25_manifest_path,
            phase25_records_path=phase25_records_path,
            phase25_manifest=phase25_manifest,
            target_root=target_root,
            target_manifest_path=target_manifest_path,
            targets_path=targets_path,
            target_manifest=target_manifest,
            targets=targets,
            base_seeds=base_seeds,
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            checkpoint=checkpoint_artifact,
            stats=stats_artifact,
            dataset_sources=dataset_source_artifacts,
            vae=vae_artifact,
            context_cache=context_cache_artifact,
            git=git,
        )
        if not manifest_path.exists():
            orphans = [
                path
                for path in (records_path, errors_path)
                if path.exists() and path.stat().st_size > 0
            ]
            if orphans:
                raise ValueError(
                    "Found non-empty outputs without immutable manifest: "
                    + ", ".join(str(path) for path in orphans)
                )
        manifest = _ensure_immutable_manifest(manifest_path, manifest_payload)
        fingerprint = str(manifest["compatibility_fingerprint"])
        target_fingerprint = str(target_manifest["compatibility_fingerprint"])

        validation_kwargs = {
            "expected_base_seeds": base_seeds,
            "expected_validation_manifest_fingerprint": fingerprint,
            "expected_target_manifest_sha256": expected_target_manifest_sha,
            "expected_target_targets_sha256": expected_target_targets_sha,
            "expected_target_manifest_fingerprint": target_fingerprint,
            "expected_checkpoint_sha256": str(checkpoint_artifact["sha256"]),
            "expected_dataset_stats_sha256": str(stats_artifact["sha256"]),
            "expected_vae_sha256": str(vae_artifact["sha256"]),
        }
        existing = target_v2.load_validation_record_index(
            records_path, **validation_kwargs
        )
        expected_keys = _expected_validation_keys(targets, base_seeds)
        outside = set(existing) - expected_keys
        if outside:
            raise ValueError(
                "Existing validation records contain keys outside immutable plan: "
                f"{sorted(outside)[:10]}"
            )
        target_v2.validate_validation_grid(
            existing,
            targets,
            base_seeds=base_seeds,
            allow_incomplete=True,
        )
        pending_keys = expected_keys - set(existing)

        misc.register_work_dir(output_dir)
        instantiation_paths = single._dataset_instantiation_path_overrides(
            dataset_source_artifacts, context_cache_artifact
        )
        dataset = instantiate(
            cfg.data.train,
            **instantiation_paths,
            is_training_set=False,
            pretrained_norm_stats=str(stats_path),
            strict_getitem=True,
            return_metadata=True,
            skip_padding_as_possible=False,
        )
        ranges, task_tables = _dataset_task_tables(
            dataset, source_pilot_manifest=source_pilot_manifest
        )

        model = None
        if pending_keys:
            LOGGER.info("Independent validation cells are pending; loading model once")
            expected_load = source_pilot_manifest["artifacts"]["checkpoint"].get("load")
            if not isinstance(expected_load, Mapping):
                raise ValueError("Source Pilot manifest lacks checkpoint load provenance")
            model = phase25._load_frozen_model(
                cfg,
                checkpoint_path=checkpoint_path,
                checkpoint_artifact=checkpoint_artifact,
                vae_artifact=vae_artifact,
                expected_checkpoint_load=expected_load,
            )
        else:
            LOGGER.info("Validation grid is complete; skipping model load")

        records_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)
        existing_keys = set(existing)
        progress = tqdm(total=len(pending_keys), desc="Target-V2 independent validation")
        collection_git_sha = str(git["commit"])
        with records_path.open("a", encoding="utf-8") as records_stream, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_stream:

            def infer_record(
                current_model: Any,
                sample: Mapping[str, Any],
                target_record: Mapping[str, Any],
                base_seed: int,
            ) -> Mapping[str, Any]:
                metadata = single._json_safe_metadata(sample["metadata"])
                dataset_index = int(metadata["dataset_index"])
                with torch.inference_mode():
                    result = collect_paired_utility(
                        current_model,
                        sample,
                        metadata=metadata,
                        base_seed=int(base_seed),
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
                return result.to_dict() if hasattr(result, "to_dict") else dict(result)

            def finalize_record(
                utility_record: Mapping[str, Any],
                target_record: Mapping[str, Any],
                validation_index: int,
                base_seed: int,
                sample: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                source_index = int(target_record["source_index"])
                metadata_raw = sample.get("metadata")
                if not isinstance(metadata_raw, Mapping):
                    raise ValueError("Strict dataset sample is missing metadata")
                metadata = single._json_safe_metadata(metadata_raw)
                single._assert_source_matches_plan(metadata, source_index, ranges)
                if utility_record.get("sample_id") != target_record.get("sample_id"):
                    raise ValueError("Validation replicate changed sample identity")
                actual_hashes = current_state_input_hashes(extract_current_state(sample))
                if actual_hashes != target_record.get("input_hashes"):
                    raise ValueError("Dataset state differs from Target-V2 input hashes")
                actual_proprio = single._current_proprio(sample)
                target_proprio = target_record.get("current_proprio")
                if target_proprio is not None and actual_proprio != target_proprio:
                    raise ValueError("Dataset proprio differs from Target-V2 source state")
                completed = target_v2.augment_validation_record(
                    utility_record,
                    target_record=target_record,
                    validation_index=int(validation_index),
                    validation_base_seed=int(base_seed),
                    validation_manifest_compatibility_fingerprint=fingerprint,
                    target_manifest_sha256=expected_target_manifest_sha,
                    target_targets_sha256=expected_target_targets_sha,
                    target_manifest_compatibility_fingerprint=target_fingerprint,
                    collection_git_sha=collection_git_sha,
                )
                completed.update(
                    {
                        "validation_record_schema_version": VALIDATION_RECORD_SCHEMA_VERSION,
                        "source_metadata": metadata,
                        "current_proprio": actual_proprio,
                        "validation_manifest_compatibility_fingerprint": fingerprint,
                        "collection_git_sha": collection_git_sha,
                        "checkpoint_sha256": checkpoint_artifact["sha256"],
                        "dataset_stats_sha256": stats_artifact["sha256"],
                        "vae_sha256": vae_artifact["sha256"],
                        "git_sha": collection_git_sha,
                    }
                )
                completed["validation_record_sha256"] = (
                    target_v2.validation_record_sha256(completed)
                )
                if int(completed["validation_replicate_index"]) != int(validation_index):
                    raise ValueError("Validation replicate index changed during augmentation")
                if int(completed["global_seed_index"]) != GLOBAL_SEED_INDEX_OFFSET + int(
                    validation_index
                ):
                    raise ValueError("Validation global seed index is invalid")
                target_v2.validate_validation_record(completed, **validation_kwargs)
                return completed

            def write_record(record: Mapping[str, Any]) -> None:
                single._write_jsonl(records_stream, record)
                progress.update(1)

            def on_error(
                source_index: int, validation_index: int, exc: BaseException
            ) -> None:
                single._write_jsonl(
                    errors_stream,
                    {
                        "timestamp_utc": single._utc_now(),
                        "source_index": int(source_index),
                        "validation_replicate_index": int(validation_index),
                        "global_seed_index": GLOBAL_SEED_INDEX_OFFSET + int(validation_index),
                        "validation_base_seed": int(base_seeds[validation_index]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "validation_manifest_compatibility_fingerprint": fingerprint,
                    },
                )
                LOGGER.exception(
                    "Failed validation cell source_index=%d validation_index=%d",
                    source_index,
                    validation_index,
                )

            run_summary = collect_validation_grid(
                dataset=dataset,
                model=model,
                target_records=targets,
                base_seeds=base_seeds,
                existing_keys=existing_keys,
                infer_record=infer_record,
                finalize_record=finalize_record,
                write_record=write_record,
                on_error=on_error,
                continue_on_error=bool(collector.continue_on_error),
                max_errors=int(collector.max_errors),
            )
        progress.close()

        final_index = target_v2.load_validation_record_index(
            records_path, **validation_kwargs
        )
        if set(final_index) != expected_keys:
            missing = sorted(expected_keys - set(final_index))
            raise RuntimeError(
                "Independent validation is incomplete; fix errors and resume the same "
                f"output directory. completed={len(final_index)}/{len(expected_keys)}, "
                f"missing={missing[:10]}, errors_this_run={run_summary['errors']}"
            )
        grid_summary = target_v2.validate_validation_grid(
            final_index, targets, base_seeds=base_seeds
        )
        if int(grid_summary.get("completed_count", -1)) != expected_plan_count:
            raise RuntimeError("Validation core did not confirm the complete 400-cell grid")

        single._assert_file_provenance_unchanged(
            checkpoint_artifact, label="UniShare checkpoint"
        )
        single._assert_file_provenance_unchanged(stats_artifact, label="dataset stats")
        single._assert_file_provenance_unchanged(vae_artifact, label="VAE artifact")
        for artifact in dataset_source_artifacts:
            single._assert_directory_tree_provenance_unchanged(
                artifact, label=f"LIBERO source dataset {artifact['dataset_index']}"
            )
        single._assert_directory_tree_provenance_unchanged(
            context_cache_artifact, label="LIBERO text embedding cache"
        )
        for path, expected_sha, label in (
            (phase25_manifest_path, expected_phase25_manifest_sha, "Phase-2.5 manifest"),
            (phase25_records_path, expected_phase25_records_sha, "Phase-2.5 records"),
            (target_manifest_path, expected_target_manifest_sha, "Target-V2 manifest"),
            (targets_path, expected_target_targets_sha, "Target-V2 targets"),
            (
                source_pilot_manifest_path,
                str(phase25_manifest["pilot"]["manifest_sha256"]),
                "source Pilot manifest",
            ),
        ):
            if single._sha256_file(path) != expected_sha:
                raise RuntimeError(f"{label} changed during validation collection")
        completion = _ensure_completion_seal(
            completion_path,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
        )
        return {
            "selected_states": len(targets),
            "expected_records": len(expected_keys),
            "existing": len(existing),
            **run_summary,
            "completion_sha256": str(completion["completion_sha256"]),
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_demo_utility_target_v2_validation.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = collect(cfg)
    LOGGER.info("Target-V2 independent validation collection complete: %s", summary)


if __name__ == "__main__":
    main()
