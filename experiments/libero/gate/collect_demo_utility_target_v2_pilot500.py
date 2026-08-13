"""Collect the exact Pilot-500 remainder and publish a combined Target V2.

This entrypoint is intentionally separate from the validated 100-state
stability collector.  It reuses that collector's model, dataset, and paired
inference seams without widening the original Phase-2.5 protocol.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from experiments.libero.gate import demo_utility_target_v2_pilot500 as expansion
from experiments.libero.gate.demo_utility import (
    collect_paired_utility,
    current_state_input_hashes,
    extract_current_state,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()
LOGGER = logging.getLogger(__name__)
EMPTY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
SCIENTIFIC_SOURCE_FILES = tuple(
    dict.fromkeys(
        (*phase25.SCIENTIFIC_SOURCE_FILES,
         "experiments/libero/gate/collect_demo_utility_target_v2_pilot500.py",
         "experiments/libero/gate/demo_utility_target_v2_pilot500.py",
         "experiments/libero/gate/demo_utility_target_v2.py",
         "configs/collect_libero_demo_utility_target_v2_pilot500.yaml")
    )
)
FORMAL_DATASET_NAMES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


def _require_bundle_dir(raw: Any, *, label: str) -> Path:
    if raw is None or not str(raw).strip():
        raise ValueError(f"{label} must be configured")
    path = Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _validate_config(cfg: DictConfig) -> tuple[int, ...]:
    single._validate_endpoint_config(cfg)
    collector = cfg.COLLECTOR
    if int(collector.num_states) != expansion.REMAINDER_STATE_COUNT:
        raise ValueError("Pilot-500 expansion num_states is frozen at 400")
    if int(collector.expected_pilot_count) != expansion.PILOT_STATE_COUNT:
        raise ValueError("Pilot-500 expansion requires the complete 500-state Pilot")
    if int(collector.expected_phase25_count) != expansion.EXISTING_TARGET_STATE_COUNT:
        raise ValueError("Pilot-500 expansion requires the exact 100-state Phase-2.5 panel")
    if int(collector.expected_pilot_base_seed) != 42:
        raise ValueError("Pilot-500 expansion requires the immutable Pilot seed 42")
    seeds = stability.validate_replicate_base_seeds(
        [int(value) for value in collector.replicate_base_seeds]
    )
    if tuple(seeds) != expansion.BASE_SEEDS:
        raise ValueError("Pilot-500 Target V2 seeds are frozen at 42--46")
    if int(collector.reuse_base_seed) != expansion.BASE_SEEDS[0]:
        raise ValueError("only seed 42 may be reused from Pilot")
    if not bool(collector.require_clean_tracked_diff):
        raise ValueError("formal Pilot-500 expansion requires a clean tracked diff")
    if bool(collector.continue_on_error):
        raise ValueError("formal Pilot-500 expansion must fail on the first error")
    if int(collector.max_errors) != 1:
        raise ValueError("formal Pilot-500 expansion max_errors is frozen at 1")
    required = (
        "pilot_dir", "phase25_dir", "existing_target_v2_dir", "output_dir",
        "remainder_target_v2_dir", "combined_target_v2_dir",
        "expected_pilot_manifest_sha256", "expected_pilot_records_sha256",
        "expected_phase25_manifest_sha256", "expected_phase25_records_sha256",
        "expected_phase25_selection_plan_sha256",
        "expected_existing_target_v2_manifest_sha256",
        "expected_existing_target_v2_targets_sha256",
    )
    for field in required:
        value = collector.get(field)
        if value is None or not str(value).strip():
            raise ValueError(f"COLLECTOR.{field} must be explicitly configured")
    raw_dataset_dirs = cfg.data.train.get("dataset_dirs")
    if raw_dataset_dirs is None or len(raw_dataset_dirs) != len(FORMAL_DATASET_NAMES):
        raise ValueError("data.train.dataset_dirs must explicitly contain four datasets")
    dataset_dirs = [Path(str(value)).expanduser() for value in raw_dataset_dirs]
    if any(not path.is_absolute() for path in dataset_dirs):
        raise ValueError("data.train.dataset_dirs must use absolute paths")
    if tuple(path.name for path in dataset_dirs) != FORMAL_DATASET_NAMES:
        raise ValueError(
            "data.train.dataset_dirs must preserve the frozen LIBERO suite order"
        )
    raw_cache = cfg.data.train.get("text_embedding_cache_dir")
    if raw_cache is None or not str(raw_cache).strip():
        raise ValueError("data.train.text_embedding_cache_dir must be explicitly configured")
    if not Path(str(raw_cache)).expanduser().is_absolute():
        raise ValueError("data.train.text_embedding_cache_dir must use an absolute path")
    return tuple(seeds)


def _load_bound_inputs(cfg: DictConfig):
    collector = cfg.COLLECTOR
    pilot_root, pilot_manifest_path, pilot_records_path, pilot_manifest = (
        phase25._resolve_pilot_bundle(collector.pilot_dir)
    )
    if single._sha256_file(pilot_manifest_path) != str(
        collector.expected_pilot_manifest_sha256
    ):
        raise ValueError("Pilot manifest differs from the preregistered SHA-256")
    if single._sha256_file(pilot_records_path) != str(
        collector.expected_pilot_records_sha256
    ):
        raise ValueError("Pilot records differ from the preregistered SHA-256")
    phase25._pilot_compatibility_checks(
        cfg=cfg,
        manifest=pilot_manifest,
        manifest_path=pilot_manifest_path,
        records_path=pilot_records_path,
    )
    pilot_records = stability.load_pilot_records(
        pilot_records_path,
        expected_count=expansion.PILOT_STATE_COUNT,
        expected_base_seed=42,
        expected_full_steps=int(collector.full_prefix_steps),
        manifest_path=pilot_manifest_path,
    )

    phase25_root = _require_bundle_dir(collector.phase25_dir, label="phase25_dir")
    phase25_manifest_path = phase25_root / "manifest.json"
    phase25_records_path = phase25_root / "records.jsonl"
    phase25_source = target_v2.load_verified_source_bundle(
        phase25_manifest_path,
        phase25_records_path,
        expected_manifest_sha256=str(collector.expected_phase25_manifest_sha256),
        expected_records_sha256=str(collector.expected_phase25_records_sha256),
        expected_selection_plan_sha256=str(
            collector.expected_phase25_selection_plan_sha256
        ),
        expected_num_states=expansion.EXISTING_TARGET_STATE_COUNT,
    )
    if phase25_source.manifest["pilot"]["manifest_sha256"] != single._sha256_file(
        pilot_manifest_path
    ):
        raise ValueError("Phase-2.5 source is not bound to the supplied Pilot manifest")
    if phase25_source.manifest["pilot"]["records_sha256"] != single._sha256_file(
        pilot_records_path
    ):
        raise ValueError("Phase-2.5 source is not bound to the supplied Pilot records")

    existing_root = _require_bundle_dir(
        collector.existing_target_v2_dir, label="existing_target_v2_dir"
    )
    existing_manifest, existing_targets = target_v2.load_target_bundle(
        existing_root,
        expected_manifest_sha256=str(
            collector.expected_existing_target_v2_manifest_sha256
        ),
        expected_targets_sha256=str(
            collector.expected_existing_target_v2_targets_sha256
        ),
        expected_num_states=expansion.EXISTING_TARGET_STATE_COUNT,
    )
    expected_source = existing_manifest["compatibility"]
    if expected_source["source_manifest_sha256"] != phase25_source.manifest_sha256:
        raise ValueError("existing Target V2 is not derived from supplied Phase-2.5 manifest")
    if expected_source["source_records_sha256"] != phase25_source.records_sha256:
        raise ValueError("existing Target V2 is not derived from supplied Phase-2.5 records")

    selected = expansion.build_remainder_selection(
        pilot_records,
        pilot_manifest,
        phase25_source.ordered_states,
    )
    return {
        "pilot_root": pilot_root,
        "pilot_manifest_path": pilot_manifest_path,
        "pilot_records_path": pilot_records_path,
        "pilot_manifest": pilot_manifest,
        "pilot_records": pilot_records,
        "phase25_source": phase25_source,
        "existing_root": existing_root,
        "existing_manifest": existing_manifest,
        "existing_targets": existing_targets,
        "selected": selected,
    }


def _scientific_source_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SCIENTIFIC_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source file is missing: {path}")
        result[relative] = single._sha256_file(path)
    return result


def _git_bytes(project_root: Path, args: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to inspect formal Git state: git {' '.join(args)}") from exc


def _assert_formal_git_state(
    *,
    project_root: Path = PROJECT_ROOT,
    scientific_source_files: Sequence[str] = SCIENTIFIC_SOURCE_FILES,
) -> dict[str, str]:
    """Require tracked-clean code and HEAD-identical scientific inputs.

    Unrelated untracked files do not change executed code and are allowed. A
    scientific source file, however, must be tracked and byte-identical to
    HEAD. The tracked-clean check rejects both staged and unstaged changes.
    """

    root = project_root.resolve()
    tracked_status = _git_bytes(
        root, ["status", "--porcelain=v1", "--untracked-files=no"]
    ).decode("utf-8", errors="replace")
    if tracked_status.strip():
        raise RuntimeError(
            "formal Pilot-500 collection requires a tracked-clean worktree; "
            f"tracked changes={tracked_status.splitlines()}"
        )
    result: dict[str, str] = {}
    for relative in scientific_source_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source file is missing: {path}")
        try:
            _git_bytes(root, ["ls-files", "--error-unmatch", "--", relative])
        except RuntimeError as exc:
            raise RuntimeError(
                f"scientific source file is not tracked by Git: {relative}"
            ) from exc
        head_bytes = _git_bytes(root, ["show", f"HEAD:{relative}"])
        working_bytes = path.read_bytes()
        if working_bytes != head_bytes:
            raise RuntimeError(
                f"scientific source file is not byte-identical to HEAD: {relative}"
            )
        result[relative] = single._sha256_bytes(working_bytes)
    return result


def _rehash_bound_components(
    *,
    manifest: Mapping[str, Any],
    dataset_source_artifacts: Sequence[Mapping[str, Any]],
    context_cache_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read every external scientific input immediately before sealing."""

    compatibility = manifest["compatibility"]
    pilot = manifest["pilot"]
    phase25_binding = manifest["excluded_phase25"]
    existing = manifest["existing_target_v2"]
    artifacts = manifest["artifacts"]

    def bound_file(path: Any, expected: Any, label: str) -> str:
        actual = single._stable_file_provenance(path, label=label)["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"{label} changed during Pilot-500 collection: "
                f"expected={expected}, current={actual}"
            )
        return str(actual)

    source_trees: list[dict[str, Any]] = []
    for artifact in dataset_source_artifacts:
        current = single._directory_tree_provenance(
            artifact["path"],
            label=f"LIBERO source dataset {artifact['dataset_index']}",
        )
        for field in ("sha256", "file_count", "total_size_bytes"):
            if current[field] != artifact[field]:
                raise RuntimeError(
                    f"LIBERO source dataset {artifact['dataset_index']} changed during "
                    f"Pilot-500 collection: {field}"
                )
        source_trees.append(
            {
                "dataset_index": int(artifact["dataset_index"]),
                "dataset_name": str(artifact["dataset_name"]),
                "sha256": str(current["sha256"]),
                "file_count": int(current["file_count"]),
                "total_size_bytes": int(current["total_size_bytes"]),
            }
        )
    current_context = single._directory_tree_provenance(
        context_cache_artifact["path"], label="LIBERO text embedding cache"
    )
    for field in ("sha256", "file_count", "total_size_bytes"):
        if current_context[field] != context_cache_artifact[field]:
            raise RuntimeError(
                f"LIBERO text embedding cache changed during Pilot-500 collection: {field}"
            )

    scientific_sources = _assert_formal_git_state()
    if scientific_sources != manifest["scientific_source_files"]:
        raise RuntimeError("scientific source files changed during Pilot-500 collection")
    return {
        "pilot": {
            "manifest_sha256": bound_file(
                pilot["manifest_path"], compatibility["pilot_manifest_sha256"],
                "Pilot manifest",
            ),
            "records_sha256": bound_file(
                pilot["records_path"], compatibility["pilot_records_sha256"],
                "Pilot records",
            ),
        },
        "phase25": {
            "manifest_sha256": bound_file(
                phase25_binding["manifest_path"],
                compatibility["phase25_manifest_sha256"],
                "Phase-2.5 manifest",
            ),
            "records_sha256": bound_file(
                phase25_binding["records_path"],
                compatibility["phase25_records_sha256"],
                "Phase-2.5 records",
            ),
        },
        "existing_target_v2": {
            "manifest_sha256": bound_file(
                existing["manifest_path"],
                compatibility["existing_target_v2_manifest_sha256"],
                "existing Target V2 manifest",
            ),
            "targets_sha256": bound_file(
                existing["targets_path"],
                compatibility["existing_target_v2_targets_sha256"],
                "existing Target V2 targets",
            ),
        },
        "artifacts": {
            "checkpoint_sha256": bound_file(
                artifacts["checkpoint"]["path"], compatibility["checkpoint_sha256"],
                "UniShare checkpoint",
            ),
            "dataset_stats_sha256": bound_file(
                artifacts["dataset_stats"]["path"],
                compatibility["dataset_stats_sha256"],
                "dataset stats",
            ),
            "vae_sha256": bound_file(
                artifacts["vae"]["path"], compatibility["vae_sha256"], "VAE"
            ),
            "dataset_sources": source_trees,
            "text_embedding_cache": {
                "sha256": str(current_context["sha256"]),
                "file_count": int(current_context["file_count"]),
                "total_size_bytes": int(current_context["total_size_bytes"]),
            },
        },
        "scientific_source_files": scientific_sources,
    }


def _build_manifest(
    *,
    cfg: DictConfig,
    inputs: Mapping[str, Any],
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
    selected = inputs["selected"]
    selection_states = phase25._selection_projection(selected)
    selection_sha = single._sha256_json(selection_states)
    source_files = _scientific_source_provenance()
    pilot_manifest = inputs["pilot_manifest"]
    phase25_source = inputs["phase25_source"]
    pilot_order_sha = expansion.sha256_json(
        expansion.pilot_ordered_source_indices(pilot_manifest)
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    compatibility = {
        "schema_version": 1,
        "kind": target_v2.SOURCE_BUNDLE_KIND,
        "purpose": expansion.EXPANSION_PURPOSE,
        "pilot_manifest_fingerprint": pilot_manifest["compatibility_fingerprint"],
        "pilot_manifest_sha256": single._sha256_file(inputs["pilot_manifest_path"]),
        "pilot_records_sha256": single._sha256_file(inputs["pilot_records_path"]),
        "phase25_manifest_sha256": phase25_source.manifest_sha256,
        "phase25_records_sha256": phase25_source.records_sha256,
        "phase25_selection_plan_sha256": phase25_source.selection_plan_sha256,
        "existing_target_v2_manifest_sha256": single._sha256_file(
            inputs["existing_root"] / "manifest.json"
        ),
        "existing_target_v2_targets_sha256": single._sha256_file(
            inputs["existing_root"] / "targets.jsonl"
        ),
        "pilot_ordered_source_indices_sha256": pilot_order_sha,
        "selection_plan_sha256": selection_sha,
        "num_states": len(selection_states),
        "replicate_base_seeds": list(base_seeds),
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
            "selection": "exact Pilot-500 order minus immutable Phase-2.5 100-state panel",
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
            "continue_on_error": False,
        },
    }
    fingerprint = single._sha256_json(compatibility)
    return {
        "schema_version": 1,
        "kind": target_v2.SOURCE_BUNDLE_KIND,
        "purpose": expansion.EXPANSION_PURPOSE,
        "created_at_utc": single._utc_now(),
        "compatibility_fingerprint": fingerprint,
        "compatibility": compatibility,
        "pilot": {
            "directory": str(inputs["pilot_root"]),
            "manifest_path": str(inputs["pilot_manifest_path"]),
            "records_path": str(inputs["pilot_records_path"]),
            "manifest_fingerprint": compatibility["pilot_manifest_fingerprint"],
            "manifest_sha256": compatibility["pilot_manifest_sha256"],
            "records_sha256": compatibility["pilot_records_sha256"],
        },
        "excluded_phase25": {
            "manifest_path": str(phase25_source.manifest_path),
            "records_path": str(phase25_source.records_path),
            "manifest_sha256": phase25_source.manifest_sha256,
            "records_sha256": phase25_source.records_sha256,
            "selection_plan_sha256": phase25_source.selection_plan_sha256,
            "state_count": phase25_source.num_states,
        },
        "existing_target_v2": {
            "directory": str(inputs["existing_root"]),
            "manifest_path": str(inputs["existing_root"] / "manifest.json"),
            "targets_path": str(inputs["existing_root"] / "targets.jsonl"),
            "manifest_sha256": compatibility[
                "existing_target_v2_manifest_sha256"
            ],
            "targets_sha256": compatibility[
                "existing_target_v2_targets_sha256"
            ],
            "state_count": expansion.EXISTING_TARGET_STATE_COUNT,
        },
        "selection": {
            "algorithm": "pilot500-order-minus-phase25-exact100-v1",
            "num_states": len(selection_states),
            "ordered_states": selection_states,
            "ordered_states_sha256": selection_sha,
            "pilot_ordered_source_indices_sha256": pilot_order_sha,
        },
        "replicates": {
            "base_seeds": list(base_seeds),
            "count": len(base_seeds),
            "reuse_base_seed": 42,
            "reuse_replicate_index": 0,
            "expected_record_count": expansion.EXPECTED_REMAINDER_RECORD_COUNT,
            "expected_reused_record_count": expansion.EXPECTED_REUSED_RECORD_COUNT,
            "expected_new_inference_count": expansion.EXPECTED_NEW_INFERENCE_COUNT,
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
                    "DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_SKIP_DOWNLOAD",
                    "DIFFSYNTH_DOWNLOAD_SOURCE",
                )
            },
        },
        "resolved_config": resolved,
        "resolved_config_sha256": single._sha256_json(resolved),
        "scientific_source_files": source_files,
        "stability_record_schema_version": stability.STABILITY_RECORD_SCHEMA_VERSION,
        "pilot500_expansion_record_schema_version": expansion.EXPANSION_RECORD_SCHEMA_VERSION,
    }


def _ensure_immutable_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    expansion.validate_expansion_manifest(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expansion.validate_expansion_manifest(existing)
        if existing["compatibility_fingerprint"] != payload["compatibility_fingerprint"]:
            raise ValueError("existing expansion manifest is incompatible with this run")
        return existing
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(payload)


def _prepare_output(output_dir: Path, *, resume: bool) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "manifest.json",
        output_dir / "records.jsonl",
        output_dir / "errors.jsonl",
        output_dir / "completion.json",
    )
    if not resume and any(path.exists() for path in paths):
        raise FileExistsError("resume=false requires a fresh expansion output directory")
    return paths


def _validate_resume_state(
    *, errors_path: Path, completion_path: Path, pending_keys: set[tuple[int, int]]
) -> None:
    """Fail closed for errored or contradictory resume directories."""

    if errors_path.exists() and errors_path.stat().st_size:
        raise ValueError(
            "formal expansion output already contains errors; use a fresh output directory"
        )
    if completion_path.exists() and pending_keys:
        raise ValueError("sealed expansion unexpectedly has pending cells")


def _publish_combined(
    cfg: DictConfig,
    *,
    inputs: Mapping[str, Any],
    manifest_path: Path,
    records_path: Path,
    completion_path: Path,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    source = target_v2.load_verified_source_bundle(
        manifest_path,
        records_path,
        expected_manifest_sha256=completion["manifest_sha256"],
        expected_records_sha256=completion["records_sha256"],
        expected_selection_plan_sha256=completion["selection_plan_sha256"],
        expected_num_states=expansion.REMAINDER_STATE_COUNT,
    )
    remainder_manifest, remainder_targets = target_v2.build_target_bundle(source)
    remainder_dir = Path(str(cfg.COLLECTOR.remainder_target_v2_dir)).expanduser().resolve()
    if remainder_dir.exists():
        loaded_manifest, loaded_targets = target_v2.load_target_bundle(
            remainder_dir,
            expected_num_states=expansion.REMAINDER_STATE_COUNT,
        )
        if loaded_manifest["compatibility_fingerprint"] != remainder_manifest[
            "compatibility_fingerprint"
        ] or target_v2.canonical_json(loaded_targets) != target_v2.canonical_json(
            remainder_targets
        ):
            raise ValueError("existing remainder Target V2 bundle differs from sealed source")
        remainder_manifest, remainder_targets = loaded_manifest, loaded_targets
    else:
        target_v2.write_target_bundle(remainder_dir, remainder_manifest, remainder_targets)
    remainder_manifest_path = remainder_dir / "manifest.json"
    remainder_targets_path = remainder_dir / "targets.jsonl"

    combined_manifest, combined_targets = expansion.build_combined_target_bundle(
        pilot_manifest=inputs["pilot_manifest"],
        pilot_records=inputs["pilot_records"],
        pilot_manifest_sha256=single._sha256_file(inputs["pilot_manifest_path"]),
        pilot_records_sha256=single._sha256_file(inputs["pilot_records_path"]),
        existing_manifest=inputs["existing_manifest"],
        existing_targets=inputs["existing_targets"],
        existing_manifest_sha256=single._sha256_file(inputs["existing_root"] / "manifest.json"),
        existing_targets_sha256=single._sha256_file(inputs["existing_root"] / "targets.jsonl"),
        remainder_manifest=remainder_manifest,
        remainder_targets=remainder_targets,
        remainder_manifest_sha256=single._sha256_file(remainder_manifest_path),
        remainder_targets_sha256=single._sha256_file(remainder_targets_path),
        expansion_completion_sha256=single._sha256_file(completion_path),
    )
    combined_dir = Path(str(cfg.COLLECTOR.combined_target_v2_dir)).expanduser().resolve()
    if combined_dir.exists():
        loaded_manifest, loaded_targets, loaded_completion = (
            expansion.load_combined_target_bundle(combined_dir)
        )
        if loaded_manifest["compatibility_fingerprint"] != combined_manifest[
            "compatibility_fingerprint"
        ] or target_v2.canonical_json(loaded_targets) != target_v2.canonical_json(
            combined_targets
        ):
            raise ValueError("existing combined Target V2 bundle differs from sealed inputs")
    else:
        expansion.write_combined_target_bundle(
            combined_dir, combined_manifest, combined_targets
        )
        _, _, loaded_completion = expansion.load_combined_target_bundle(combined_dir)
    return {
        "remainder_target_states": len(remainder_targets),
        "combined_target_states": len(combined_targets),
        "combined_high_confidence": combined_manifest["summary"]["high_confidence_count"],
        "combined_completion_sha256": loaded_completion["completion_sha256"],
    }


def collect(cfg: DictConfig) -> dict[str, Any]:
    base_seeds = _validate_config(cfg)
    collector = cfg.COLLECTOR
    inputs = _load_bound_inputs(cfg)
    selected = inputs["selected"]
    output_dir = Path(str(collector.output_dir)).expanduser().resolve()
    manifest_path, records_path, errors_path, completion_path = _prepare_output(
        output_dir, resume=bool(collector.resume)
    )
    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another collector is already using {output_dir}") from exc

        (
            checkpoint_path,
            stats_path,
            checkpoint_artifact,
            stats_artifact,
            dataset_source_artifacts,
            auxiliary_artifacts,
        ) = phase25._current_artifacts(cfg, pilot_manifest=inputs["pilot_manifest"])
        vae_artifact = auxiliary_artifacts["vae"]
        context_cache_artifact = auxiliary_artifacts["text_embedding_cache"]
        git = single.git_provenance()
        scientific_source_hashes = _assert_formal_git_state()
        if git["tracked_diff_sha256"] != EMPTY_SHA256:
            raise AssertionError("tracked-clean Git guard and provenance disagree")
        payload = _build_manifest(
            cfg=cfg,
            inputs=inputs,
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
        if payload["scientific_source_files"] != scientific_source_hashes:
            raise AssertionError("scientific source provenance changed during startup")
        if not manifest_path.exists():
            orphans = [
                path for path in (records_path, errors_path, completion_path)
                if path.exists() and path.stat().st_size
            ]
            if orphans:
                raise ValueError("non-empty expansion outputs exist without a manifest")
        manifest = _ensure_immutable_manifest(manifest_path, payload)
        fingerprint = manifest["compatibility_fingerprint"]
        compatibility = manifest["compatibility"]
        existing = expansion.load_expansion_record_index(
            records_path,
            expected_manifest_fingerprint=fingerprint,
            expected_pilot_manifest_fingerprint=compatibility[
                "pilot_manifest_fingerprint"
            ],
            expected_checkpoint_sha256=compatibility["checkpoint_sha256"],
            expected_dataset_stats_sha256=compatibility["dataset_stats_sha256"],
            expected_vae_sha256=compatibility["vae_sha256"],
        )
        expected_keys = {
            (phase25._selection_source_index(row), replicate_index)
            for row in selected
            for replicate_index in range(len(base_seeds))
        }
        if set(existing) - expected_keys:
            raise ValueError("existing expansion records contain keys outside plan")
        stability.validate_complete_grid(
            existing, selected, base_seeds=base_seeds, allow_incomplete=True
        )
        pending_keys = expected_keys - set(existing)
        _validate_resume_state(
            errors_path=errors_path, completion_path=completion_path,
            pending_keys=pending_keys,
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
        if phase25._scientific_dataset_ranges(ranges) != phase25._scientific_dataset_ranges(
            inputs["pilot_manifest"]["compatibility"]["dataset_index_ranges"]
        ):
            raise ValueError("instantiated dataset ranges differ from Pilot")
        task_tables = {
            int(item["dataset_index"]): {
                int(key): str(value)
                for key, value in dataset.dataset_task_table(
                    int(item["dataset_index"])
                ).items()
            }
            for item in ranges
        }
        normalized_tables = {
            str(key): {str(inner): value for inner, value in table.items()}
            for key, table in task_tables.items()
        }
        if normalized_tables != inputs["pilot_manifest"]["compatibility"][
            "dataset_task_tables"
        ]:
            raise ValueError("instantiated task tables differ from Pilot")

        pending_inference = any(
            base_seeds[index] != 42 for _, index in pending_keys
        )
        model = None
        if pending_inference:
            expected_load = inputs["pilot_manifest"]["artifacts"]["checkpoint"].get("load")
            if not isinstance(expected_load, Mapping):
                raise ValueError("Pilot manifest lacks checkpoint load provenance")
            model = phase25._load_frozen_model(
                cfg,
                checkpoint_path=checkpoint_path,
                checkpoint_artifact=checkpoint_artifact,
                vae_artifact=vae_artifact,
                expected_checkpoint_load=expected_load,
            )
        records_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)
        progress = tqdm(total=len(pending_keys), desc="Pilot-500 Target V2 remainder")
        existing_keys = set(existing)
        collection_git_sha = str(git["commit"])
        with records_path.open("a", encoding="utf-8") as records_stream, errors_path.open(
            "a", encoding="utf-8"
        ) as errors_stream:

            def infer_record(current_model, sample, pilot_record, base_seed):
                metadata = single._json_safe_metadata(sample["metadata"])
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
                        task_by_index=task_tables[int(metadata["dataset_index"])],
                    )
                return result.to_dict() if hasattr(result, "to_dict") else dict(result)

            def finalize_record(utility_record, pilot_record, replicate_index, base_seed, sample):
                source_index = phase25._selection_source_index(pilot_record)
                metadata = single._json_safe_metadata(sample["metadata"])
                single._assert_source_matches_plan(metadata, source_index, ranges)
                if utility_record.get("sample_id") != pilot_record.get("sample_id"):
                    raise ValueError("replicate changed Pilot sample identity")
                if current_state_input_hashes(extract_current_state(sample)) != pilot_record[
                    "input_hashes"
                ]:
                    raise ValueError("dataset state differs from Pilot input hashes")
                current_proprio = single._current_proprio(sample)
                if current_proprio != pilot_record["current_proprio"]:
                    raise ValueError("dataset proprio differs from Pilot")
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
                        "source_metadata": metadata,
                        "current_proprio": current_proprio,
                        "stability_manifest_compatibility_fingerprint": fingerprint,
                        "collection_git_sha": collection_git_sha,
                        "checkpoint_sha256": checkpoint_artifact["sha256"],
                        "dataset_stats_sha256": stats_artifact["sha256"],
                        "vae_sha256": vae_artifact["sha256"],
                    }
                )
                if int(base_seed) != 42:
                    completed["git_sha"] = collection_git_sha
                return expansion.seal_expansion_record(completed)

            def write_record(record):
                single._write_jsonl(records_stream, record)
                progress.update(1)

            def on_error(source_index, replicate_index, exc):
                single._write_jsonl(
                    errors_stream,
                    {
                        "timestamp_utc": single._utc_now(),
                        "source_index": int(source_index),
                        "replicate_index": int(replicate_index),
                        "replicate_base_seed": int(base_seeds[replicate_index]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "manifest_fingerprint": fingerprint,
                    },
                )

            run_summary = phase25.collect_replicate_grid(
                dataset=dataset,
                model=model,
                selected_records=selected,
                base_seeds=base_seeds,
                reuse_base_seed=42,
                existing_keys=existing_keys,
                infer_record=infer_record,
                finalize_record=finalize_record,
                write_record=write_record,
                on_error=on_error,
                continue_on_error=False,
                max_errors=1,
            )
        progress.close()

        final_index = expansion.load_expansion_record_index(
            records_path,
            expected_manifest_fingerprint=fingerprint,
            expected_pilot_manifest_fingerprint=compatibility[
                "pilot_manifest_fingerprint"
            ],
            expected_checkpoint_sha256=compatibility["checkpoint_sha256"],
            expected_dataset_stats_sha256=compatibility["dataset_stats_sha256"],
            expected_vae_sha256=compatibility["vae_sha256"],
        )
        if set(final_index) != expected_keys:
            missing = sorted(expected_keys - set(final_index))
            raise RuntimeError(f"expansion incomplete; missing={missing[:10]}")
        stability.validate_complete_grid(final_index, selected, base_seeds=base_seeds)
        component_hashes = _rehash_bound_components(
            manifest=manifest,
            dataset_source_artifacts=dataset_source_artifacts,
            context_cache_artifact=context_cache_artifact,
        )
        completion = expansion.ensure_completion_seal(
            completion_path,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
            component_hashes=component_hashes,
        )
        if _rehash_bound_components(
            manifest=manifest,
            dataset_source_artifacts=dataset_source_artifacts,
            context_cache_artifact=context_cache_artifact,
        ) != component_hashes:
            raise RuntimeError("scientific inputs changed while completion was sealed")
        publish = _publish_combined(
            cfg,
            inputs=inputs,
            manifest_path=manifest_path,
            records_path=records_path,
            completion_path=completion_path,
            completion=completion,
        )
        return {
            "selected_states": len(selected),
            "expected_records": len(expected_keys),
            "existing": len(existing),
            **run_summary,
            **publish,
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_demo_utility_target_v2_pilot500.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Pilot-500 Target V2 expansion summary: %s", collect(cfg))


if __name__ == "__main__":
    main()
