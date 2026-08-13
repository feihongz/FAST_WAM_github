"""Collect a reproducible multi-seed LIBERO Demo-Utility stability grid.

Phase 2.5 does not train a Gate.  It selects 100 auditable states from the
immutable Pilot-500 run and evaluates the same paired N=0/N=full comparison at
five deterministic base seeds.  The Pilot base seed is copied byte-for-value
at the semantic record level; only the other four replicates run inference.

The implementation intentionally lives beside, rather than inside, the
single-seed collector.  That keeps the already-validated Pilot path unchanged.
"""

from __future__ import annotations

import fcntl
import gc
import json
import logging
import math
import os
import platform
import sys
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
from experiments.libero.gate import demo_utility_stability as stability
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
STABILITY_RECORD_SCHEMA_VERSION = 1
AUDIT_KIND = "libero_demo_utility_multiseed_stability"
V1_NUM_STATES = 100
V1_PARTIAL_TARGET = 16
SCIENTIFIC_SOURCE_FILES = (
    "experiments/libero/gate/collect_demo_utility_multiseed.py",
    "experiments/libero/gate/demo_utility_stability.py",
    "experiments/libero/gate/demo_utility.py",
    "src/fastwam/datasets/lerobot/base_lerobot_dataset.py",
    "src/fastwam/datasets/lerobot/robot_video_dataset.py",
    "src/fastwam/models/wan22/fastwam_unified_shared.py",
    "src/fastwam/models/wan22/wan_video_vae.py",
    "src/fastwam/models/wan22/helpers/loader.py",
    "src/fastwam/models/wan22/helpers/io.py",
    "src/fastwam/models/wan22/helpers/state_dict_converters.py",
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _resolve_pilot_bundle(raw_dir: Any) -> tuple[Path, Path, Path, dict[str, Any]]:
    if raw_dir is None or not str(raw_dir).strip():
        raise ValueError("COLLECTOR.pilot_dir must point to the completed Pilot-500 run")
    root = Path(os.path.expandvars(os.path.expanduser(str(raw_dir)))).resolve()
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "Pilot directory must contain manifest.json and records.jsonl: " f"{root}"
        )
    manifest = _load_json(manifest_path, label="Pilot manifest")
    single._validate_manifest_integrity(manifest)
    return root, manifest_path, records_path, manifest


def _selection_projection(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only stable, analysis-relevant state selection fields."""

    projection: list[dict[str, Any]] = []
    for fallback_order, record in enumerate(selected):
        source_metadata = record.get("source_metadata")
        if not isinstance(source_metadata, Mapping):
            raise ValueError("Selected Pilot record has no source_metadata")
        source_index = int(
            record.get("source_index", source_metadata.get("requested_sample_idx", -1))
        )
        if source_index < 0:
            raise ValueError("Selected Pilot record has no valid source index")
        projection.append(
            {
                "selection_order": int(record.get("selection_order", fallback_order)),
                "source_index": source_index,
                "sample_id": str(record["sample_id"]),
                "suite": str(record["suite"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "selection_bin": str(record["selection_bin"]),
                "pilot_utility": float(record.get("pilot_utility", record["utility"])),
                "valid_length": int(record["valid_length"]),
                "pilot_seed": int(record.get("pilot_seed", record["seed"])),
                "pilot_record_sha256": single._sha256_json(_pilot_record_payload(record)),
            }
        )
    projection.sort(key=lambda item: item["selection_order"])
    if [item["selection_order"] for item in projection] != list(range(len(projection))):
        raise ValueError("Selection order must be exactly 0..num_states-1")
    return projection


def _pilot_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove selector-only fields before hashing the immutable Pilot row."""

    selector_fields = {
        "selection_order",
        "selection_bin",
        "source_index",
        "pilot_seed",
        "pilot_e0",
        "pilot_efull",
        "pilot_utility",
        "pilot_valid_length",
        "pilot_input_combined_sha256",
        "pilot_manifest_compatibility_fingerprint",
    }
    return {key: value for key, value in record.items() if key not in selector_fields}


def _scientific_source_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in SCIENTIFIC_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Scientific source file is missing: {path}")
        result[relative_path] = single._sha256_file(path)
    return result


def _validate_v1_config(cfg: DictConfig) -> tuple[int, ...]:
    collector = cfg.COLLECTOR
    single._validate_endpoint_config(cfg)
    if int(collector.num_states) != V1_NUM_STATES:
        raise ValueError(
            f"Phase 2.5 V1 requires num_states={V1_NUM_STATES}; "
            f"got {collector.num_states}"
        )
    base_seeds = stability.validate_replicate_base_seeds(
        [int(value) for value in collector.replicate_base_seeds]
    )
    expected = tuple(int(value) for value in stability.DEFAULT_REPLICATE_BASE_SEEDS)
    if tuple(base_seeds) != expected:
        raise ValueError(f"Phase 2.5 V1 replicate seeds must be {expected}, got {base_seeds}")
    reuse_seed = int(collector.reuse_base_seed)
    if reuse_seed != int(collector.expected_pilot_base_seed) or reuse_seed not in base_seeds:
        raise ValueError(
            "reuse_base_seed must equal expected_pilot_base_seed and occur exactly once"
        )
    if list(base_seeds).count(reuse_seed) != 1:
        raise ValueError("reuse_base_seed must occur exactly once")
    if int(collector.max_errors) <= 0:
        raise ValueError("COLLECTOR.max_errors must be positive")
    return tuple(base_seeds)


def _pilot_compatibility_checks(
    *,
    cfg: DictConfig,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    records_path: Path,
) -> None:
    compatibility = manifest["compatibility"]
    collector = cfg.COLLECTOR
    if int(manifest.get("schema_version", -1)) != single.MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported Pilot manifest schema")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Resolved stability config must be a mapping")
    current_scientific_config = {
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
    for field, current_value in current_scientific_config.items():
        pilot_value = compatibility.get(field)
        if pilot_value != current_value:
            raise ValueError(
                "Stability scientific configuration differs from Pilot: "
                f"field={field}, pilot={pilot_value!r}, current={current_value!r}"
            )
    pilot_parameters = compatibility.get("collection_parameters", {})
    expected_parameters = {
        "seed": int(collector.expected_pilot_base_seed),
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
    }
    if dict(pilot_parameters) != expected_parameters:
        raise ValueError(
            "Pilot endpoint parameters differ from this stability audit: "
            f"pilot={pilot_parameters!r}, expected={expected_parameters!r}"
        )
    selection = manifest.get("selection", {})
    if int(selection.get("num_samples", -1)) != int(collector.expected_pilot_count):
        raise ValueError("Pilot manifest is not the expected completed Pilot sample plan")
    if single._sha256_file(manifest_path) == single._sha256_file(records_path):
        raise AssertionError("Pilot manifest and records unexpectedly have identical bytes")


def _current_artifacts(
    cfg: DictConfig,
    *,
    pilot_manifest: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    collector = cfg.COLLECTOR
    checkpoint_path = single._resolve_existing_file(cfg.get("ckpt"), label="ckpt")
    stats_path = single.resolve_dataset_stats_path(
        checkpoint_path, collector.get("dataset_stats_path")
    )
    checkpoint = single._stable_file_provenance(checkpoint_path, label="UniShare checkpoint")
    stats = single._stable_file_provenance(stats_path, label="dataset stats")
    vae = single._resolve_vae_artifact(cfg)
    dataset_sources: list[dict[str, Any]] = []
    for dataset_index, raw_path in enumerate(cfg.data.train.dataset_dirs):
        artifact = single._directory_tree_provenance(
            raw_path, label=f"LIBERO source dataset {dataset_index}"
        )
        artifact["dataset_index"] = dataset_index
        artifact["dataset_name"] = Path(artifact["path"]).name
        dataset_sources.append(artifact)
    context_cache = single._directory_tree_provenance(
        cfg.data.train.text_embedding_cache_dir,
        label="LIBERO text embedding cache",
    )

    expected = pilot_manifest["compatibility"]
    for label, current, expected_digest in (
        ("checkpoint", checkpoint, expected["checkpoint_sha256"]),
        ("dataset stats", stats, expected["dataset_stats_sha256"]),
        ("VAE", vae, expected["vae_sha256"]),
        ("text embedding cache", context_cache, expected["context_cache_sha256"]),
    ):
        if current["sha256"] != expected_digest:
            raise ValueError(
                f"{label} bytes differ from Pilot: "
                f"current={current['sha256']}, pilot={expected_digest}"
            )
    source_content = [
        {
            "dataset_name": item["dataset_name"],
            "sha256": item["sha256"],
            "file_count": item["file_count"],
            "total_size_bytes": item["total_size_bytes"],
        }
        for item in dataset_sources
    ]
    if source_content != expected["dataset_source_content"]:
        raise ValueError("Configured LIBERO dataset bytes/order differ from Pilot")
    return (
        checkpoint_path,
        stats_path,
        checkpoint,
        stats,
        dataset_sources,
        {"vae": vae, "text_embedding_cache": context_cache},
    )


def _build_manifest(
    *,
    cfg: DictConfig,
    pilot_root: Path,
    pilot_manifest_path: Path,
    pilot_records_path: Path,
    pilot_manifest: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
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
    assert isinstance(resolved, dict)
    selection_states = _selection_projection(selected)
    selection_sha256 = single._sha256_json(selection_states)
    source_files = _scientific_source_provenance()
    pilot_manifest_sha256 = single._sha256_file(pilot_manifest_path)
    pilot_records_sha256 = single._sha256_file(pilot_records_path)
    pilot_fingerprint = str(pilot_manifest["compatibility_fingerprint"])
    compatibility = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "pilot_manifest_fingerprint": pilot_fingerprint,
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "pilot_records_sha256": pilot_records_sha256,
        "selection_plan_sha256": selection_sha256,
        "num_states": len(selection_states),
        "replicate_base_seeds": [int(value) for value in base_seeds],
        "reuse_base_seed": int(collector.reuse_base_seed),
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_stats_sha256": stats["sha256"],
        "vae_sha256": vae["sha256"],
        "dataset_source_content": [
            {
                "dataset_name": item["dataset_name"],
                "sha256": item["sha256"],
                "file_count": int(item["file_count"]),
                "total_size_bytes": int(item["total_size_bytes"]),
            }
            for item in dataset_sources
        ],
        "context_cache_sha256": context_cache["sha256"],
        "pilot_git_commit": pilot_manifest["compatibility"]["git_commit"],
        "collection_git_commit": git["commit"],
        "collection_git_tracked_diff_sha256": git["tracked_diff_sha256"],
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
            "selection_seed": int(collector.selection_seed),
            "partial_target": V1_PARTIAL_TARGET,
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
        },
    }
    fingerprint = single._sha256_json(compatibility)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "created_at_utc": single._utc_now(),
        "compatibility_fingerprint": fingerprint,
        "compatibility": compatibility,
        "pilot": {
            "directory": str(pilot_root),
            "manifest_path": str(pilot_manifest_path),
            "records_path": str(pilot_records_path),
            "manifest_fingerprint": pilot_fingerprint,
            "manifest_sha256": pilot_manifest_sha256,
            "records_sha256": pilot_records_sha256,
        },
        "selection": {
            "algorithm": "phase2.5-five-bin-suite-task-episode-balanced-v1",
            "selection_seed": int(collector.selection_seed),
            "num_states": len(selection_states),
            "partial_target": V1_PARTIAL_TARGET,
            "ordered_states": selection_states,
            "ordered_states_sha256": selection_sha256,
        },
        "replicates": {
            "base_seeds": [int(value) for value in base_seeds],
            "count": len(base_seeds),
            "reuse_base_seed": int(collector.reuse_base_seed),
            "reuse_replicate_index": list(base_seeds).index(int(collector.reuse_base_seed)),
            "expected_record_count": len(selection_states) * len(base_seeds),
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
            "model_environment": {
                key: os.environ.get(key)
                for key in (
                    "DIFFSYNTH_MODEL_BASE_PATH",
                    "DIFFSYNTH_SKIP_DOWNLOAD",
                    "DIFFSYNTH_DOWNLOAD_SOURCE",
                )
            },
        },
        "resolved_config": resolved,
        "resolved_config_sha256": single._sha256_json(resolved),
        "scientific_source_files": source_files,
        "stability_record_schema_version": STABILITY_RECORD_SCHEMA_VERSION,
    }


def _validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("kind") != AUDIT_KIND:
        raise ValueError(f"Unexpected stability manifest kind={manifest.get('kind')!r}")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("Stability manifest compatibility must be a mapping")
    expected = single._sha256_json(compatibility)
    if manifest.get("compatibility_fingerprint") != expected:
        raise ValueError("Stability manifest compatibility fingerprint is invalid")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("ordered_states"), list
    ):
        raise ValueError("Stability manifest has no ordered state selection")
    states = selection["ordered_states"]
    state_sha = single._sha256_json(states)
    if selection.get("ordered_states_sha256") != state_sha:
        raise ValueError("Stability manifest selection digest is invalid")
    if compatibility.get("selection_plan_sha256") != state_sha:
        raise ValueError("Stability compatibility is not bound to selection")
    if int(selection.get("num_states", -1)) != len(states):
        raise ValueError("Stability manifest selection count is invalid")
    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("Stability manifest has no replicate plan")
    base_seeds = [int(value) for value in replicates.get("base_seeds", [])]
    if base_seeds != [int(value) for value in compatibility["replicate_base_seeds"]]:
        raise ValueError("Replicate plan is not bound to compatibility")
    if int(replicates.get("expected_record_count", -1)) != len(states) * len(base_seeds):
        raise ValueError("Expected stability grid size is invalid")
    pilot = manifest.get("pilot")
    if not isinstance(pilot, Mapping):
        raise ValueError("Stability manifest has no Pilot provenance")
    for field in ("manifest_fingerprint", "manifest_sha256", "records_sha256"):
        compatibility_field = (
            "pilot_manifest_fingerprint"
            if field == "manifest_fingerprint"
            else f"pilot_{field}"
        )
        if pilot.get(field) != compatibility.get(compatibility_field):
            raise ValueError(f"Pilot {field} is not bound to compatibility")


def _ensure_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_manifest_integrity(payload)
    if path.exists():
        existing = _load_json(path, label="stability manifest")
        _validate_manifest_integrity(existing)
        if existing["compatibility_fingerprint"] != payload["compatibility_fingerprint"]:
            raise ValueError(
                "Existing stability manifest is incompatible with this run; "
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


def _selection_source_index(record: Mapping[str, Any]) -> int:
    if record.get("source_index") is not None:
        return int(record["source_index"])
    metadata = record.get("source_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("requested_sample_idx") is None:
        raise ValueError("Selected record is missing source index")
    return int(metadata["requested_sample_idx"])


def collect_replicate_grid(
    *,
    dataset: Any,
    model: Any,
    selected_records: Sequence[Mapping[str, Any]],
    base_seeds: Sequence[int],
    reuse_base_seed: int,
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
    """Execute a state-outer/replicate-inner grid with injectable I/O.

    This small orchestration seam is deliberately independent of Hydra and GPU
    setup so tests can prove that each state is decoded once, Pilot replicates
    perform zero model calls, and resume skips completed composite keys.
    """

    seeds = tuple(int(value) for value in base_seeds)
    new_count = reused_count = inferred_count = error_count = 0
    for fallback_order, pilot_record in enumerate(selected_records):
        source_index = _selection_source_index(pilot_record)
        pending = [
            (replicate_index, base_seed)
            for replicate_index, base_seed in enumerate(seeds)
            if (source_index, replicate_index) not in existing_keys
        ]
        if not pending:
            continue
        sample = dataset[source_index]
        for replicate_index, base_seed in pending:
            try:
                if base_seed == int(reuse_base_seed):
                    utility_record = _pilot_record_payload(pilot_record)
                    reused = True
                else:
                    if model is None:
                        raise RuntimeError("New-inference replicate is pending but model is not loaded")
                    utility_record = infer_record(model, sample, pilot_record, base_seed)
                    reused = False
                completed = finalize_record(
                    utility_record,
                    pilot_record,
                    replicate_index,
                    base_seed,
                    sample,
                )
                write_record(completed)
                existing_keys.add((source_index, replicate_index))
                new_count += 1
                reused_count += int(reused)
                inferred_count += int(not reused)
            except Exception as exc:
                error_count += 1
                if on_error is not None:
                    on_error(source_index, replicate_index, exc)
                if not continue_on_error:
                    raise
                if error_count >= int(max_errors):
                    raise RuntimeError(f"Collector reached max_errors={max_errors}") from exc
    return {
        "new": new_count,
        "reused": reused_count,
        "inferred": inferred_count,
        "errors": error_count,
    }


def _load_frozen_model(
    cfg: DictConfig,
    *,
    checkpoint_path: Path,
    checkpoint_artifact: Mapping[str, Any],
    vae_artifact: Mapping[str, Any],
    expected_checkpoint_load: Mapping[str, Any],
) -> Any:
    device = str(cfg.COLLECTOR.device)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("COLLECTOR.device is CUDA but torch.cuda.is_available() is false")
        if resolved_device.index is not None:
            torch.cuda.set_device(resolved_device)
    model = instantiate(
        cfg.model,
        model_dtype=single._mixed_precision_dtype(cfg.get("mixed_precision", "bf16")),
        device=device,
    )
    model_paths = getattr(model, "model_paths", None)
    if not isinstance(model_paths, Mapping) or not model_paths.get("vae"):
        raise ValueError("Instantiated model did not report model_paths['vae']")
    actual_vae_path = single._resolve_project_path(
        model_paths["vae"], label="model-reported VAE artifact"
    )
    if actual_vae_path != Path(str(vae_artifact["path"])):
        raise ValueError("Preflight/model VAE path mismatch")
    single._assert_file_provenance_unchanged(vae_artifact, label="VAE artifact")
    payload = model.load_checkpoint(str(checkpoint_path))
    single._assert_file_provenance_unchanged(
        checkpoint_artifact, label="UniShare checkpoint"
    )
    checkpoint_load = single._checkpoint_payload_provenance(model, payload)
    del payload
    gc.collect()
    if dict(checkpoint_load) != dict(expected_checkpoint_load):
        raise ValueError(
            "Loaded checkpoint structure differs from Pilot manifest: "
            f"current={checkpoint_load!r}, pilot={expected_checkpoint_load!r}"
        )
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Frozen stability-audit model still has trainable parameters")
    return model


def collect(cfg: DictConfig) -> dict[str, int]:
    base_seeds = _validate_v1_config(cfg)
    collector = cfg.COLLECTOR
    pilot_root, pilot_manifest_path, pilot_records_path, pilot_manifest = (
        _resolve_pilot_bundle(collector.pilot_dir)
    )
    _pilot_compatibility_checks(
        cfg=cfg,
        manifest=pilot_manifest,
        manifest_path=pilot_manifest_path,
        records_path=pilot_records_path,
    )
    pilot_records = stability.load_pilot_records(
        pilot_records_path,
        expected_count=int(collector.expected_pilot_count),
        expected_base_seed=int(collector.expected_pilot_base_seed),
        expected_full_steps=int(collector.full_prefix_steps),
        manifest_path=pilot_manifest_path,
    )
    selected = stability.build_stability_selection(
        pilot_records,
        selection_seed=int(collector.selection_seed),
    )
    if len(selected) != V1_NUM_STATES:
        raise AssertionError(f"Selector returned {len(selected)} states, expected {V1_NUM_STATES}")

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

        LOGGER.info("Hashing and rebinding all Pilot scientific artifacts")
        (
            checkpoint_path,
            stats_path,
            checkpoint_artifact,
            stats_artifact,
            dataset_source_artifacts,
            auxiliary_artifacts,
        ) = _current_artifacts(cfg, pilot_manifest=pilot_manifest)
        vae_artifact = auxiliary_artifacts["vae"]
        context_cache_artifact = auxiliary_artifacts["text_embedding_cache"]
        git = single.git_provenance()
        manifest_payload = _build_manifest(
            cfg=cfg,
            pilot_root=pilot_root,
            pilot_manifest_path=pilot_manifest_path,
            pilot_records_path=pilot_records_path,
            pilot_manifest=pilot_manifest,
            selected=selected,
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

        existing = stability.load_stability_record_index(
            records_path,
            expected_stability_manifest_fingerprint=fingerprint,
            expected_full_steps=int(collector.full_prefix_steps),
            expected_pilot_manifest_fingerprint=str(
                pilot_manifest["compatibility_fingerprint"]
            ),
            expected_base_seeds=base_seeds,
        )
        expected_keys = {
            (_selection_source_index(record), replicate_index)
            for record in selected
            for replicate_index in range(len(base_seeds))
        }
        out_of_plan = set(existing) - expected_keys
        if out_of_plan:
            raise ValueError(
                "Existing records contain composite keys outside immutable plan: "
                f"{sorted(out_of_plan)[:10]}"
            )
        # A self-consistent row can still describe the wrong Pilot state.
        # Rebind every completed cell to the exact selected Pilot row before
        # treating it as resumable, even when the grid is still incomplete.
        stability.validate_complete_grid(
            existing, selected, base_seeds=base_seeds, allow_incomplete=True
        )
        pending_keys = expected_keys - set(existing)
        pending_inference = any(
            base_seeds[replicate_index] != int(collector.reuse_base_seed)
            for _, replicate_index in pending_keys
        )

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
        ranges = single._normalize_ranges(dataset.dataset_index_ranges())
        if ranges != pilot_manifest["compatibility"]["dataset_index_ranges"]:
            raise ValueError("Instantiated dataset ranges differ from Pilot")
        task_tables: dict[int, dict[int, str]] = {}
        for item in ranges:
            dataset_index = int(item["dataset_index"])
            table = dataset.dataset_task_table(dataset_index)
            task_tables[dataset_index] = {
                int(key): str(value) for key, value in table.items()
            }
        normalized_tables = {
            str(key): {str(inner_key): value for inner_key, value in table.items()}
            for key, table in task_tables.items()
        }
        if normalized_tables != pilot_manifest["compatibility"]["dataset_task_tables"]:
            raise ValueError("Instantiated dataset task tables differ from Pilot")

        model = None
        if pending_inference:
            LOGGER.info("New-inference cells are pending; loading frozen model once")
            expected_load = pilot_manifest["artifacts"]["checkpoint"].get("load")
            if not isinstance(expected_load, Mapping):
                raise ValueError("Pilot manifest lacks checkpoint load provenance")
            model = _load_frozen_model(
                cfg,
                checkpoint_path=checkpoint_path,
                checkpoint_artifact=checkpoint_artifact,
                vae_artifact=vae_artifact,
                expected_checkpoint_load=expected_load,
            )
        else:
            LOGGER.info("No new-inference cells are pending; skipping model load")

        records_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)
        existing_keys = set(existing)
        progress = tqdm(total=len(pending_keys), desc="multi-seed paired utility")
        collection_git_sha = str(git["commit"])

        with records_path.open("a", encoding="utf-8") as records_stream, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_stream:

            def infer_record(
                current_model: Any,
                sample: Mapping[str, Any],
                pilot_record: Mapping[str, Any],
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
                pilot_record: Mapping[str, Any],
                replicate_index: int,
                base_seed: int,
                sample: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                source_index = _selection_source_index(pilot_record)
                metadata_raw = sample.get("metadata")
                if not isinstance(metadata_raw, Mapping):
                    raise ValueError("Strict dataset sample is missing metadata")
                metadata = single._json_safe_metadata(metadata_raw)
                single._assert_source_matches_plan(metadata, source_index, ranges)
                if utility_record.get("sample_id") != pilot_record.get("sample_id"):
                    raise ValueError("Replicate changed selected sample identity")
                actual_hashes = current_state_input_hashes(extract_current_state(sample))
                if actual_hashes != pilot_record.get("input_hashes"):
                    raise ValueError("Selected dataset state differs from Pilot input hashes")
                actual_proprio = single._current_proprio(sample)
                if actual_proprio != pilot_record.get("current_proprio"):
                    raise ValueError("Selected dataset proprio differs from Pilot")
                completed = stability.augment_stability_record(
                    utility_record,
                    pilot_record=pilot_record,
                    selection_entry=pilot_record,
                    replicate_index=int(replicate_index),
                    replicate_base_seed=int(base_seed),
                    stability_manifest_compatibility_fingerprint=fingerprint,
                )
                completed.update(
                    {
                        "stability_record_schema_version": STABILITY_RECORD_SCHEMA_VERSION,
                        "source_metadata": metadata,
                        "current_proprio": actual_proprio,
                        "stability_manifest_compatibility_fingerprint": fingerprint,
                        "collection_git_sha": collection_git_sha,
                        "checkpoint_sha256": checkpoint_artifact["sha256"],
                        "dataset_stats_sha256": stats_artifact["sha256"],
                        "vae_sha256": vae_artifact["sha256"],
                    }
                )
                # The utility-generating git_sha is preserved for Pilot reuse;
                # new rows are explicitly attributed to the current collector.
                if int(base_seed) != int(collector.reuse_base_seed):
                    completed["git_sha"] = collection_git_sha
                stability.validate_stability_record(
                    completed,
                    expected_full_steps=int(collector.full_prefix_steps),
                    expected_stability_manifest_fingerprint=fingerprint,
                    expected_pilot_manifest_fingerprint=str(
                        pilot_manifest["compatibility_fingerprint"]
                    ),
                    expected_base_seeds=base_seeds,
                )
                return completed

            def write_record(record: Mapping[str, Any]) -> None:
                single._write_jsonl(records_stream, record)
                progress.update(1)

            def on_error(
                source_index: int, replicate_index: int, exc: BaseException
            ) -> None:
                single._write_jsonl(
                    errors_stream,
                    {
                        "timestamp_utc": single._utc_now(),
                        "selected_source_index": int(source_index),
                        "replicate_index": int(replicate_index),
                        "replicate_base_seed": int(base_seeds[replicate_index]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "stability_manifest_compatibility_fingerprint": fingerprint,
                    },
                )
                LOGGER.exception(
                    "Failed stability cell source_index=%d replicate_index=%d",
                    source_index,
                    replicate_index,
                )

            run_summary = collect_replicate_grid(
                dataset=dataset,
                model=model,
                selected_records=selected,
                base_seeds=base_seeds,
                reuse_base_seed=int(collector.reuse_base_seed),
                existing_keys=existing_keys,
                infer_record=infer_record,
                finalize_record=finalize_record,
                write_record=write_record,
                on_error=on_error,
                continue_on_error=bool(collector.continue_on_error),
                max_errors=int(collector.max_errors),
            )
        progress.close()

        final_index = stability.load_stability_record_index(
            records_path,
            expected_stability_manifest_fingerprint=fingerprint,
            expected_full_steps=int(collector.full_prefix_steps),
            expected_pilot_manifest_fingerprint=str(
                pilot_manifest["compatibility_fingerprint"]
            ),
            expected_base_seeds=base_seeds,
        )
        if set(final_index) != expected_keys:
            missing = sorted(expected_keys - set(final_index))
            raise RuntimeError(
                "Collection is incomplete; fix errors and resume the same output directory. "
                f"completed={len(final_index)}/{len(expected_keys)}, missing={missing[:10]}, "
                f"errors_this_run={run_summary['errors']}"
            )
        stability.validate_complete_grid(
            final_index,
            selected,
            base_seeds=base_seeds,
        )

        # Fail closed if any scientific input changed while collection ran.
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
        if single._sha256_file(pilot_manifest_path) != manifest["pilot"]["manifest_sha256"]:
            raise RuntimeError("Pilot manifest changed during stability collection")
        if single._sha256_file(pilot_records_path) != manifest["pilot"]["records_sha256"]:
            raise RuntimeError("Pilot records changed during stability collection")
        return {
            "selected_states": len(selected),
            "expected_records": len(expected_keys),
            "existing": len(existing),
            **run_summary,
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_demo_utility_multiseed.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = collect(cfg)
    LOGGER.info("Multi-seed Demo Utility stability collection complete: %s", summary)


if __name__ == "__main__":
    main()
