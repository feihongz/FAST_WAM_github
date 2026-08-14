"""Collect exact-V1 137-D Gate features for the sealed Pilot-500 remainder.

This is a deliberately thin, fail-closed specialization of
``collect_tiny_mlp_features``.  All scientific feature operations, row schemas,
tensor hashes, atomic progress files, and completion validation are reused
directly from the audited Target-100 collector.  This module only adds the
Phase-3b data boundary:

* the input Target-V2 bundle must contain exactly the 400 Pilot-500 remainder
  states;
* the sealed combined Target-500 bundle must bind those exact remainder bytes;
* the pre-registered follow-up document is an external SHA-256 trust anchor;
* no Validation4 path, hash, record, or loader exists in the configuration or
  runtime API.

An interrupted run publishes only private atomic ``.rows`` progress.  Every
resume rehydrates all 400 dataset states and compares their current-state input
hashes with Target V2 before a public four-file feature bundle can be sealed.
All upstream files and live dataset/cache artifacts are rehashed immediately
before either an incomplete return or final publication.
"""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import subprocess
import sys
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

from experiments.libero.gate import collect_tiny_mlp_features as base
from experiments.libero.gate import demo_utility_target_v2 as target_v2
from experiments.libero.gate.collect_demo_utility import (
    _dataset_instantiation_path_overrides,
    _directory_tree_provenance,
    _normalize_ranges,
    _resolve_existing_file,
    _resolve_vae_artifact,
    _scientific_data_config,
    _sha256_file,
    _stable_file_provenance,
    resolve_dataset_stats_path,
)
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()

LOGGER = logging.getLogger(__name__)

# Re-export the exact cache contract for consumers and tests. These aliases are
# intentionally objects from the Target-100 implementation, not forks.
BUNDLE_KIND = base.BUNDLE_KIND
COMPLETION_KIND = base.COMPLETION_KIND
BUNDLE_SCHEMA_VERSION = base.BUNDLE_SCHEMA_VERSION
FEATURE_RECORD_SCHEMA_VERSION = base.FEATURE_RECORD_SCHEMA_VERSION
COMPLETION_SCHEMA_VERSION = base.COMPLETION_SCHEMA_VERSION
MANIFEST_FILENAME = base.MANIFEST_FILENAME
INDEX_FILENAME = base.INDEX_FILENAME
FEATURES_FILENAME = base.FEATURES_FILENAME
COMPLETION_FILENAME = base.COMPLETION_FILENAME
PENDING_COMPLETION_FILENAME = ".completion.pending.json"
TENSOR_KEYS = base.TENSOR_KEYS
EXPECTED_DIMS = base.EXPECTED_DIMS

REMAINDER_STATE_COUNT = 400
COMBINED_STATE_COUNT = 500
TARGET_BASE_SEEDS = (42, 43, 44, 45, 46)
FORBIDDEN_VALIDATION_SEEDS = (47, 48, 49, 50)
COMBINED_KIND = "libero_demo_utility_target_v2_pilot500"
COMBINED_COMPLETION_KIND = f"{COMBINED_KIND}_completion"
FOLLOWUP_PROTOCOL_RELATIVE = "docs/GATE_OFFLINE_REMAINDER400_FOLLOWUP.md"
FOLLOWUP_STAGE = "libero_gate_remainder400_exact137_external_retest_v1"
FROZEN_EXTRACTOR_FINGERPRINT = (
    "975726ec657e117f2d0c0554e3aaf3a1e31eb343f5f97b9635f7fb4538987d7c"
)
FROZEN_PROJECTION_SHA256 = {
    "visual": "da3e212062de024fe7c07e670590fecd7bc456d3edb9c59890d1819f26cea50a",
    "instruction_mean": "13ddb4308889fdb2f9e2bd740abd3c82cccfb4bba26246f37a3fd3992bdfc280",
    "instruction_rms": "f25e240caa73a2286fea7836368248e4cc60878ebcb76635d90e066c596e76aa",
}
FROZEN_VISUAL_CONFIG = {
    "latent_channels": 48,
    "pooled_height": 2,
    "pooled_width": 4,
    "projection_dim": 64,
    "projection_seed": 20260815,
}
FROZEN_INSTRUCTION_CONFIG = {
    "context_dim": 4096,
    "projection_dim": 32,
    "mean_projection_seed": 20260816,
    "rms_projection_seed": 20260817,
}
FROZEN_PROPRIO_DIM = 8

SCIENTIFIC_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *base.SCIENTIFIC_SOURCE_FILES,
            "configs/collect_libero_gate_features_remainder400.yaml",
            FOLLOWUP_PROTOCOL_RELATIVE,
            "experiments/libero/gate/collect_tiny_mlp_features_remainder400.py",
        )
    )
)


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
        command = " ".join(args)
        raise RuntimeError(
            f"failed to inspect formal Git state: git {command}"
        ) from exc


def _assert_formal_git_state(
    *,
    project_root: Path = PROJECT_ROOT,
    scientific_source_files: Sequence[str] = SCIENTIFIC_SOURCE_FILES,
) -> dict[str, str]:
    """Require tracked-clean code and HEAD-identical scientific inputs.

    Unrelated untracked files are deliberately allowed. Both staged and
    unstaged tracked changes are rejected, and every scientific source must be
    tracked and byte-identical to its HEAD blob.
    """

    root = project_root.resolve()
    tracked_status = _git_bytes(
        root, ["status", "--porcelain=v1", "--untracked-files=no"]
    ).decode("utf-8", errors="replace")
    if tracked_status.strip():
        raise RuntimeError(
            "formal remainder-400 extraction requires a tracked-clean worktree; "
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
        working_bytes = path.read_bytes()
        if working_bytes != _git_bytes(root, ["show", f"HEAD:{relative}"]):
            raise RuntimeError(
                f"scientific source file is not byte-identical to HEAD: {relative}"
            )
        result[relative] = base.sha256_bytes(working_bytes)
    return result


def _git_head(project_root: Path = PROJECT_ROOT) -> str:
    value = _git_bytes(project_root.resolve(), ["rev-parse", "HEAD"]).decode(
        "ascii", errors="strict"
    ).strip()
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"formal Git HEAD is invalid: {value!r}")
    return value


def _formal_git_snapshot(
    *,
    project_root: Path = PROJECT_ROOT,
    scientific_source_files: Sequence[str] = SCIENTIFIC_SOURCE_FILES,
) -> dict[str, Any]:
    head_before = _git_head(project_root)
    sources = _assert_formal_git_state(
        project_root=project_root,
        scientific_source_files=scientific_source_files,
    )
    head_after = _git_head(project_root)
    if head_before != head_after:
        raise RuntimeError("formal Git HEAD changed during source verification")
    return {
        "head": head_before,
        "require_clean_tracked_diff": True,
        "scientific_source_files": sources,
    }


def _plain_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _validate_exact_numerical_contract(
    cfg: DictConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Reject every numerical override before any artifact or GPU access."""

    if str(cfg.get("mixed_precision")) != "bf16":
        raise ValueError("formal remainder-400 extraction requires mixed_precision=bf16")
    collector = cfg.FEATURE_COLLECTOR
    visual = _plain_mapping(collector.get("visual"), field="FEATURE_COLLECTOR.visual")
    instruction = _plain_mapping(
        collector.get("instruction"), field="FEATURE_COLLECTOR.instruction"
    )
    if visual != FROZEN_VISUAL_CONFIG:
        raise ValueError(
            "FEATURE_COLLECTOR.visual differs from the frozen exact-V1 contract"
        )
    if instruction != FROZEN_INSTRUCTION_CONFIG:
        raise ValueError(
            "FEATURE_COLLECTOR.instruction differs from the frozen exact-V1 contract"
        )
    proprio_dim = collector.get("proprio_dim")
    if isinstance(proprio_dim, bool) or int(proprio_dim) != FROZEN_PROPRIO_DIM:
        raise ValueError("FEATURE_COLLECTOR.proprio_dim is frozen at 8")
    if EXPECTED_DIMS != {
        "full": 137,
        "visual": 64,
        "instruction": 65,
        "proprio": 8,
    }:
        raise AssertionError("imported exact-V1 feature dimensions changed")

    projections = base.build_projection_matrices(
        latent_channels=visual["latent_channels"],
        pooled_height=visual["pooled_height"],
        pooled_width=visual["pooled_width"],
        visual_dim=visual["projection_dim"],
        visual_seed=visual["projection_seed"],
        context_dim=instruction["context_dim"],
        instruction_dim=instruction["projection_dim"],
        mean_seed=instruction["mean_projection_seed"],
        rms_seed=instruction["rms_projection_seed"],
    )
    actual_projection_hashes = {
        key: base.tensor_content_sha256(matrix) for key, matrix in projections.items()
    }
    if actual_projection_hashes != FROZEN_PROJECTION_SHA256:
        raise AssertionError(
            "exact-V1 projection matrix bytes differ from the sealed Target-100 contract"
        )
    extractor_config = {
        "visual": visual,
        "instruction": instruction,
        "proprio_dim": FROZEN_PROPRIO_DIM,
    }
    extractor = base._extractor_contract(extractor_config, projections)
    if extractor.get("extractor_fingerprint") != FROZEN_EXTRACTOR_FINGERPRINT:
        raise AssertionError(
            "exact-V1 extractor fingerprint differs from sealed Target-100"
        )
    return projections, extractor


def _formal_entry_preflight(
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    collector = cfg.FEATURE_COLLECTOR
    if collector.get("require_clean_tracked_diff") is not True:
        raise ValueError(
            "formal remainder-400 extraction requires require_clean_tracked_diff=true"
        )
    # This is intentionally the first non-config operation in collect().
    formal_git = _formal_git_snapshot()
    projections, extractor = _validate_exact_numerical_contract(cfg)
    return formal_git, projections, extractor


def _require_directory(value: Any, *, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} must be configured")
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{field} is missing or not a directory: {path}")
    return path


def _require_file_digest(path: Path, expected: Any, *, field: str) -> str:
    trust_anchor = base.require_sha256(expected, field=field)
    actual = _sha256_file(path)
    if actual != trust_anchor:
        raise ValueError(f"{field} differs from live file bytes: {path}")
    return actual


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed {label} row {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(row)
    return rows


def _canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{base.canonical_json(row)}\n" for row in rows).encode("utf-8")
    return base.sha256_bytes(payload)


def _component_projection(row: Mapping[str, Any], order: int, component: str) -> dict[str, Any]:
    return {
        "pilot500_selection_order": int(order),
        "source_index": int(row["source_index"]),
        "sample_id": str(row["sample_id"]),
        "target_id": str(row["target_id"]),
        "target_sha256": str(row["target_sha256"]),
        "component": component,
        "component_selection_order": int(row["selection_order"]),
        "source_pilot_record_sha256": str(row["source_pilot_record_sha256"]),
    }


def validate_combined_remainder_anchor(
    *,
    remainder_manifest: Mapping[str, Any],
    remainder_targets: Sequence[Mapping[str, Any]],
    remainder_manifest_sha256: str,
    remainder_records_sha256: str,
    combined_manifest: Mapping[str, Any],
    combined_targets: Sequence[Mapping[str, Any]],
    combined_manifest_sha256: str,
    combined_records_sha256: str,
    combined_completion: Mapping[str, Any],
    combined_completion_file_sha256: str,
) -> dict[str, Any]:
    """Validate that completed Target-500 seals these exact 400 target rows.

    The formal branch intentionally does not depend on the Pilot-500 producer
    module.  Revalidating the small consumer-side projection here keeps this PR
    stacked only on Target V2 while still rejecting component substitution,
    reordering, or completion tampering.
    """

    for field, value in (
        ("remainder_manifest_sha256", remainder_manifest_sha256),
        ("remainder_records_sha256", remainder_records_sha256),
        ("combined_manifest_sha256", combined_manifest_sha256),
        ("combined_records_sha256", combined_records_sha256),
        ("combined_completion_file_sha256", combined_completion_file_sha256),
    ):
        base.require_sha256(value, field=field)
    if len(remainder_targets) != REMAINDER_STATE_COUNT:
        raise ValueError("remainder Target V2 must contain exactly 400 states")
    if len(combined_targets) != COMBINED_STATE_COUNT:
        raise ValueError("combined Target V2 must contain exactly 500 states")

    target_v2.validate_target_manifest(remainder_manifest)
    target_v2._validate_targets_against_manifest(remainder_manifest, remainder_targets)
    if remainder_manifest.get("compatibility_fingerprint") != base.sha256_json(
        remainder_manifest.get("compatibility")
    ):
        raise ValueError("remainder compatibility fingerprint is invalid")
    if remainder_manifest.get("targets", {}).get(
        "canonical_records_sha256"
    ) != remainder_records_sha256:
        raise ValueError("remainder manifest is not bound to target record bytes")

    if combined_manifest.get("kind") != COMBINED_KIND or int(
        combined_manifest.get("schema_version", -1)
    ) != 1:
        raise ValueError("invalid combined Target-500 kind/schema")
    compatibility = combined_manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("combined Target-500 has no compatibility mapping")
    combined_fingerprint = base.require_sha256(
        combined_manifest.get("compatibility_fingerprint"),
        field="combined compatibility_fingerprint",
    )
    if combined_fingerprint != base.sha256_json(compatibility):
        raise ValueError("combined compatibility fingerprint is invalid")
    if list(compatibility.get("target_base_seeds", [])) != list(TARGET_BASE_SEEDS):
        raise ValueError("combined target seeds are not exactly 42--46")
    if int(compatibility.get("num_states", -1)) != COMBINED_STATE_COUNT:
        raise ValueError("combined compatibility state count is invalid")

    target_section = combined_manifest.get("targets")
    if not isinstance(target_section, Mapping):
        raise ValueError("combined Target-500 has no targets section")
    if int(target_section.get("count", -1)) != COMBINED_STATE_COUNT:
        raise ValueError("combined target count is invalid")
    if target_section.get("canonical_records_sha256") != combined_records_sha256:
        raise ValueError("combined manifest is not bound to target record bytes")
    if combined_records_sha256 != _canonical_jsonl_sha256(combined_targets):
        raise ValueError("combined target records are not canonical JSONL bytes")
    if compatibility.get("combined_targets_sha256") != combined_records_sha256:
        raise ValueError("combined compatibility is not bound to target records")

    completion_unhashed = {
        key: value
        for key, value in combined_completion.items()
        if key != "completion_sha256"
    }
    if combined_completion.get("kind") != COMBINED_COMPLETION_KIND or int(
        combined_completion.get("schema_version", -1)
    ) != 1:
        raise ValueError("invalid combined completion kind/schema")
    if base.require_sha256(
        combined_completion.get("completion_sha256"), field="combined completion_sha256"
    ) != base.sha256_json(completion_unhashed):
        raise ValueError("combined completion payload digest is invalid")
    expected_completion = {
        "manifest_sha256": combined_manifest_sha256,
        "targets_sha256": combined_records_sha256,
        "target_count": COMBINED_STATE_COUNT,
        "manifest_fingerprint": combined_fingerprint,
    }
    for field, expected in expected_completion.items():
        if combined_completion.get(field) != expected:
            raise ValueError(f"combined completion {field} is invalid")

    components = combined_manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("combined Target-500 has no component bindings")
    binding = components.get("remainder400")
    if not isinstance(binding, Mapping) or base.canonical_json(binding) != base.canonical_json(
        compatibility.get("remainder400")
    ):
        raise ValueError("combined remainder binding is missing or inconsistent")
    expected_binding = {
        "count": REMAINDER_STATE_COUNT,
        "manifest_sha256": remainder_manifest_sha256,
        "targets_sha256": remainder_records_sha256,
        "manifest_fingerprint": remainder_manifest["compatibility_fingerprint"],
        "source_manifest_sha256": remainder_manifest["source"]["manifest_sha256"],
        "source_records_sha256": remainder_manifest["source"]["records_sha256"],
    }
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            raise ValueError(f"combined remainder binding differs for {field}")

    policy = combined_manifest.get("policy")
    if not isinstance(policy, Mapping) or list(
        policy.get("independent_validation_seeds_excluded", [])
    ) != list(FORBIDDEN_VALIDATION_SEEDS):
        raise ValueError("combined bundle does not explicitly exclude seeds 47--50")

    selection = combined_manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("algorithm") != (
        "immutable-pilot500-order-existing100-plus-exact-remainder400-v1"
    ):
        raise ValueError("combined selection algorithm is invalid")
    states = selection.get("ordered_states")
    if not isinstance(states, list) or len(states) != COMBINED_STATE_COUNT:
        raise ValueError("combined ordered selection must contain 500 states")
    if selection.get("ordered_states_sha256") != base.sha256_json(states):
        raise ValueError("combined ordered selection digest is invalid")
    if compatibility.get("combined_selection_sha256") != base.sha256_json(states):
        raise ValueError("combined compatibility is not bound to selection")

    remainder_rows: list[tuple[int, Mapping[str, Any]]] = []
    component_counts = {"existing100": 0, "remainder400": 0}
    for order, (state, row) in enumerate(zip(states, combined_targets)):
        component = state.get("component")
        if component not in component_counts:
            raise ValueError("combined selection contains an unknown component")
        component_counts[str(component)] += 1
        if base.canonical_json(state) != base.canonical_json(
            _component_projection(row, order, str(component))
        ):
            raise ValueError("combined target row differs from ordered selection")
        if component == "remainder400":
            remainder_rows.append((int(state["component_selection_order"]), row))
    if component_counts != {"existing100": 100, "remainder400": 400}:
        raise ValueError("combined component counts are invalid")
    remainder_rows.sort(key=lambda item: item[0])
    if [order for order, _ in remainder_rows] != list(range(REMAINDER_STATE_COUNT)):
        raise ValueError("combined remainder component order is not exactly 0--399")
    for expected, (_, actual) in zip(remainder_targets, remainder_rows):
        if base.canonical_json(actual) != base.canonical_json(expected):
            raise ValueError("combined Target-500 does not contain the exact remainder row")

    return {
        "schema_version": 1,
        "kind": "libero_gate_remainder400_combined_anchor",
        "remainder_manifest_sha256": remainder_manifest_sha256,
        "remainder_records_sha256": remainder_records_sha256,
        "remainder_compatibility_fingerprint": remainder_manifest[
            "compatibility_fingerprint"
        ],
        "combined_manifest_sha256": combined_manifest_sha256,
        "combined_records_sha256": combined_records_sha256,
        "combined_completion_file_sha256": combined_completion_file_sha256,
        "combined_completion_sha256": combined_completion["completion_sha256"],
        "combined_compatibility_fingerprint": combined_fingerprint,
        "remainder_state_count": REMAINDER_STATE_COUNT,
        "combined_state_count": COMBINED_STATE_COUNT,
    }


def load_sealed_followup_targets(
    collector: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load the only label surface allowed to the feature collector."""

    remainder_dir = _require_directory(
        collector.get("remainder_target_v2_dir"), field="remainder_target_v2_dir"
    )
    remainder_manifest_path = remainder_dir / target_v2.TARGET_MANIFEST_FILENAME
    remainder_records_path = remainder_dir / target_v2.TARGETS_FILENAME
    remainder_manifest_sha = _require_file_digest(
        remainder_manifest_path,
        collector.get("expected_remainder_manifest_sha256"),
        field="expected_remainder_manifest_sha256",
    )
    remainder_records_sha = _require_file_digest(
        remainder_records_path,
        collector.get("expected_remainder_records_sha256"),
        field="expected_remainder_records_sha256",
    )
    remainder_manifest, remainder_targets = target_v2.load_target_bundle(
        remainder_dir,
        expected_manifest_sha256=remainder_manifest_sha,
        expected_targets_sha256=remainder_records_sha,
        expected_num_states=REMAINDER_STATE_COUNT,
    )

    combined_dir = _require_directory(
        collector.get("combined_target_v2_dir"), field="combined_target_v2_dir"
    )
    combined_manifest_path = combined_dir / "manifest.json"
    combined_records_path = combined_dir / "targets.jsonl"
    combined_completion_path = combined_dir / "completion.json"
    combined_manifest_sha = _require_file_digest(
        combined_manifest_path,
        collector.get("expected_combined_manifest_sha256"),
        field="expected_combined_manifest_sha256",
    )
    combined_records_sha = _require_file_digest(
        combined_records_path,
        collector.get("expected_combined_records_sha256"),
        field="expected_combined_records_sha256",
    )
    combined_completion_file_sha = _require_file_digest(
        combined_completion_path,
        collector.get("expected_combined_completion_file_sha256"),
        field="expected_combined_completion_file_sha256",
    )
    combined_manifest = base._load_json(
        combined_manifest_path, label="combined Target-500 manifest"
    )
    combined_targets = _load_jsonl(
        combined_records_path, label="combined Target-500 targets"
    )
    combined_completion = base._load_json(
        combined_completion_path, label="combined Target-500 completion"
    )
    anchor = validate_combined_remainder_anchor(
        remainder_manifest=remainder_manifest,
        remainder_targets=remainder_targets,
        remainder_manifest_sha256=remainder_manifest_sha,
        remainder_records_sha256=remainder_records_sha,
        combined_manifest=combined_manifest,
        combined_targets=combined_targets,
        combined_manifest_sha256=combined_manifest_sha,
        combined_records_sha256=combined_records_sha,
        combined_completion=combined_completion,
        combined_completion_file_sha256=combined_completion_file_sha,
    )
    return remainder_dir, remainder_manifest, remainder_targets, anchor


def _verify_followup_protocol(expected_sha256: Any) -> dict[str, str]:
    path = (PROJECT_ROOT / FOLLOWUP_PROTOCOL_RELATIVE).resolve()
    digest = _require_file_digest(
        path, expected_sha256, field="expected_followup_protocol_sha256"
    )
    return {"path": str(path), "sha256": digest}


def _assert_artifacts_match_followup(
    *,
    phase25: Mapping[str, Any],
    remainder_manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    vae: Mapping[str, Any],
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
    model_config_sha256: str,
    data_config_sha256: str,
) -> None:
    """Bind remainder labels and live artifacts without claiming its source is Phase2.5."""

    compatibility = phase25["compatibility"]
    source = remainder_manifest["source"]
    for label, artifact, field in (
        ("checkpoint", checkpoint, "checkpoint_sha256"),
        ("dataset stats", stats, "dataset_stats_sha256"),
        ("VAE", vae, "vae_sha256"),
    ):
        actual = str(artifact["sha256"])
        if actual != compatibility.get(field) or actual != source.get(field):
            raise ValueError(f"live {label} bytes differ from remainder / Phase-2.5")
    expected_sources = compatibility.get("dataset_source_content")
    actual_sources = [
        {
            "dataset_name": str(item["dataset_name"]),
            "sha256": str(item["sha256"]),
            "file_count": int(item["file_count"]),
            "total_size_bytes": int(item["total_size_bytes"]),
        }
        for item in dataset_sources
    ]
    if actual_sources != expected_sources:
        raise ValueError("live dataset source content differs from Phase-2.5")
    if context_cache["sha256"] != compatibility.get("context_cache_sha256"):
        raise ValueError("live text context cache differs from Phase-2.5")
    if model_config_sha256 != compatibility.get("model_config_sha256"):
        raise ValueError("resolved model config differs from Phase-2.5")
    if data_config_sha256 != compatibility.get("data_config_sha256"):
        raise ValueError("resolved scientific data config differs from Phase-2.5")


def augment_remainder400_manifest(
    manifest: Mapping[str, Any],
    *,
    pilot500_anchor: Mapping[str, Any],
    followup_protocol: Mapping[str, str],
    formal_git: Mapping[str, Any],
) -> dict[str, Any]:
    """Add Phase-3b anchors while retaining the exact V1 bundle schema."""

    result = copy.deepcopy(dict(manifest))
    compatibility = result.get("compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("base feature manifest has no mutable compatibility mapping")
    formal_sources = formal_git.get("scientific_source_files")
    if not isinstance(formal_sources, Mapping) or dict(formal_sources) != compatibility.get(
        "scientific_source_files"
    ):
        raise ValueError("formal Git source hashes differ from base feature manifest")
    git_head = str(formal_git.get("head", ""))
    if len(git_head) not in {40, 64}:
        raise ValueError("formal Git HEAD is missing or invalid")
    compatibility.update(
        {
            "followup_stage": FOLLOWUP_STAGE,
            "formal_git_head": git_head,
            "require_clean_tracked_diff": True,
            "pilot500_anchor": dict(pilot500_anchor),
            "followup_protocol_sha256": base.require_sha256(
                followup_protocol.get("sha256"), field="followup_protocol_sha256"
            ),
            "data_boundary": {
                "fit_labels": "remainder400_target5_only",
                "feature_rows": "remainder400_current_state_only",
                "original100_feature_rows_allowed": False,
                "independent_validation_records_allowed": False,
                "independent_validation_seeds_allowed": False,
            },
            "num_states": REMAINDER_STATE_COUNT,
        }
    )
    result["compatibility_fingerprint"] = base.sha256_json(compatibility)
    result["followup_protocol"] = dict(followup_protocol)
    result["pilot500_anchor"] = dict(pilot500_anchor)
    result["formal_git"] = copy.deepcopy(dict(formal_git))
    outputs = result.get("outputs", {}).get("features", {})
    if isinstance(outputs, dict):
        outputs["shapes"] = {
            key: [REMAINDER_STATE_COUNT, EXPECTED_DIMS[key]] for key in TENSOR_KEYS
        }
    base.validate_manifest(result)
    return result


def _content_snapshot(
    *,
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    vae: Mapping[str, Any],
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
    phase25_manifest_sha256: str,
    formal_git: Mapping[str, Any],
    pilot500_anchor: Mapping[str, Any],
    followup_protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_stats_sha256": stats["sha256"],
        "vae_sha256": vae["sha256"],
        "dataset_sources": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "file_count": item["file_count"],
                "total_size_bytes": item["total_size_bytes"],
            }
            for item in dataset_sources
        ],
        "context_cache": {
            "path": context_cache["path"],
            "sha256": context_cache["sha256"],
            "file_count": context_cache["file_count"],
            "total_size_bytes": context_cache["total_size_bytes"],
        },
        "phase25_manifest_sha256": phase25_manifest_sha256,
        "formal_git": copy.deepcopy(dict(formal_git)),
        "pilot500_anchor": dict(pilot500_anchor),
        "followup_protocol_sha256": followup_protocol_sha256,
    }


def _rehash_all_inputs(
    cfg: DictConfig,
    *,
    checkpoint_path: Path,
    stats_path: Path,
    phase_path: Path,
    expected_snapshot: Mapping[str, Any],
) -> None:
    """Fail if any upstream byte changed while GPU extraction was running."""

    collector = cfg.FEATURE_COLLECTOR
    _, _, _, anchor = load_sealed_followup_targets(collector)
    checkpoint = _stable_file_provenance(checkpoint_path, label="UniShare checkpoint")
    stats = _stable_file_provenance(stats_path, label="dataset stats")
    vae = _resolve_vae_artifact(cfg)
    dataset_sources: list[dict[str, Any]] = []
    for index, raw_path in enumerate(cfg.data.train.dataset_dirs):
        item = _directory_tree_provenance(raw_path, label=f"LIBERO source dataset {index}")
        item["dataset_index"] = index
        item["dataset_name"] = Path(str(item["path"])).name
        dataset_sources.append(item)
    context_cache = _directory_tree_provenance(
        cfg.data.train.text_embedding_cache_dir, label="LIBERO text embedding cache"
    )
    protocol = _verify_followup_protocol(
        collector.get("expected_followup_protocol_sha256")
    )
    current = _content_snapshot(
        checkpoint=checkpoint,
        stats=stats,
        vae=vae,
        dataset_sources=dataset_sources,
        context_cache=context_cache,
        phase25_manifest_sha256=_require_file_digest(
            phase_path,
            collector.get("expected_phase25_manifest_sha256"),
            field="expected_phase25_manifest_sha256",
        ),
        formal_git=_formal_git_snapshot(),
        pilot500_anchor=anchor,
        followup_protocol_sha256=protocol["sha256"],
    )
    if base.canonical_json(current) != base.canonical_json(expected_snapshot):
        raise ValueError("an upstream scientific input changed during feature extraction")


def _instantiate_dataset_and_contract(
    cfg: DictConfig,
    *,
    stats_path: Path,
    dataset_sources: Sequence[Mapping[str, Any]],
    context_cache: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[int, dict[int, str]]]:
    paths = _dataset_instantiation_path_overrides(dataset_sources, context_cache)
    dataset = instantiate(
        cfg.data.train,
        **paths,
        is_training_set=False,
        pretrained_norm_stats=str(stats_path),
        strict_getitem=True,
        return_metadata=True,
        skip_padding_as_possible=False,
    )
    if not hasattr(dataset, "dataset_index_ranges") or not hasattr(
        dataset, "dataset_task_table"
    ):
        raise TypeError("strict feature dataset lacks auditable range/task-table APIs")
    ranges = _normalize_ranges(dataset.dataset_index_ranges())
    if sum(int(item["population"]) for item in ranges) != len(dataset):
        raise ValueError("dataset ranges do not cover the instantiated dataset")
    if [item["dataset_name"] for item in ranges] != [
        item["dataset_name"] for item in dataset_sources
    ]:
        raise ValueError("dataset order differs from Phase-2.5 artifact order")
    task_tables: dict[int, dict[int, str]] = {}
    for item in ranges:
        index = int(item["dataset_index"])
        raw = dataset.dataset_task_table(index)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"dataset task table {index} is empty")
        task_tables[index] = {int(key): str(value) for key, value in raw.items()}
    return dataset, ranges, task_tables


def _load_frozen_vae(
    *, vae_path: str, cfg: DictConfig, device: torch.device
) -> tuple[torch.nn.Module, torch.dtype]:
    from fastwam.models.wan22.helpers.loader import _load_registered_model

    model = _load_registered_model(
        vae_path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=str(device),
    )
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("feature VAE is not frozen")
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("loaded feature VAE has no parameters") from exc
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported feature VAE dtype {dtype}")
    return model, dtype


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_recovery_file(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"recovery target is not a regular file: {path}")
    path.unlink()
    return True


def _recover_unsealed_output(
    *,
    output_dir: Path,
    progress_dir: Path,
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recover only crash-shaped outputs when no completion seal exists.

    A completion seal changes the policy from recovery to strict validation:
    no public or progress artifact is deleted once completion.json exists.
    """

    completion_path = output_dir / COMPLETION_FILENAME
    if completion_path.exists():
        base.validate_completion(output_dir)
        return {"sealed": True, "removed": []}

    removed: list[str] = []
    pending_completion_path = output_dir / PENDING_COMPLETION_FILENAME
    if _unlink_recovery_file(pending_completion_path):
        removed.append(pending_completion_path.name)
    expected_progress: set[str] = set()
    for target in targets:
        json_path, tensor_path = base._progress_paths(
            progress_dir, int(target["selection_order"])
        )
        expected_progress.update((json_path.name, tensor_path.name))
        if json_path.exists() != tensor_path.exists():
            for path in (json_path, tensor_path):
                if _unlink_recovery_file(path):
                    removed.append(str(path.relative_to(output_dir)))

    # Kill-safe atomic writers may leave only hidden temporary files. They are
    # never scientific rows and are safe to discard before rebuilding.
    for path in list(progress_dir.iterdir()):
        if path.name in expected_progress:
            continue
        if path.name.startswith(".") and (
            ".json." in path.name or ".safetensors." in path.name
        ):
            if _unlink_recovery_file(path):
                removed.append(str(path.relative_to(output_dir)))
            continue
        raise ValueError(f"unexpected progress artifact cannot be recovered: {path}")

    for filename in (INDEX_FILENAME, FEATURES_FILENAME):
        path = output_dir / filename
        if _unlink_recovery_file(path):
            removed.append(path.name)
    temporary_prefixes = tuple(
        f".{filename}."
        for filename in (
            MANIFEST_FILENAME,
            INDEX_FILENAME,
            FEATURES_FILENAME,
            COMPLETION_FILENAME,
            PENDING_COMPLETION_FILENAME,
        )
    )
    for path in list(output_dir.iterdir()):
        if path.is_file() and path.name.startswith(temporary_prefixes):
            if _unlink_recovery_file(path):
                removed.append(path.name)
    if removed:
        _fsync_directory(progress_dir)
        _fsync_directory(output_dir)
    return {"sealed": False, "removed": sorted(removed)}


def collect(cfg: DictConfig) -> dict[str, Any]:
    formal_git, projections, extractor = _formal_entry_preflight(cfg)
    collector = cfg.FEATURE_COLLECTOR
    expected_num_states = int(collector.expected_num_states)
    if expected_num_states != REMAINDER_STATE_COUNT:
        raise ValueError("Phase-3b feature cache is hard-locked to exact remainder 400")
    raw_max_new_rows = collector.get("max_new_rows")
    max_new_rows = None if raw_max_new_rows is None else int(raw_max_new_rows)
    if max_new_rows is not None and max_new_rows <= 0:
        raise ValueError("FEATURE_COLLECTOR.max_new_rows must be positive or null")

    checkpoint_path = _resolve_existing_file(cfg.get("ckpt"), label="ckpt")
    stats_path = resolve_dataset_stats_path(
        checkpoint_path, collector.get("dataset_stats_path")
    )
    remainder_dir, remainder_manifest, targets, pilot500_anchor = (
        load_sealed_followup_targets(collector)
    )
    phase_path = _resolve_existing_file(
        collector.get("phase25_manifest_path"), label="Phase-2.5 manifest"
    )
    phase25 = base._load_phase25_manifest(
        phase_path,
        expected_sha256=str(collector.get("expected_phase25_manifest_sha256")),
    )
    phase25["_manifest_path"] = str(phase_path)
    protocol = _verify_followup_protocol(
        collector.get("expected_followup_protocol_sha256")
    )

    output_dir = Path(
        os.path.expandvars(os.path.expanduser(str(collector.output_dir)))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another feature collector is using {output_dir}") from exc
        if not bool(collector.resume):
            existing = [path for path in output_dir.iterdir() if path.name != lock_path.name]
            if existing:
                raise FileExistsError("resume=false requires a fresh output directory")

        LOGGER.info("Hashing remainder-bound checkpoint/stats/VAE/dataset/context artifacts")
        checkpoint = _stable_file_provenance(checkpoint_path, label="UniShare checkpoint")
        stats = _stable_file_provenance(stats_path, label="dataset stats")
        vae = _resolve_vae_artifact(cfg)
        dataset_sources: list[dict[str, Any]] = []
        for index, raw_path in enumerate(cfg.data.train.dataset_dirs):
            item = _directory_tree_provenance(
                raw_path, label=f"LIBERO source dataset {index}"
            )
            item["dataset_index"] = index
            item["dataset_name"] = Path(str(item["path"])).name
            dataset_sources.append(item)
        context_cache = _directory_tree_provenance(
            cfg.data.train.text_embedding_cache_dir,
            label="LIBERO text embedding cache",
        )
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        data_config = _scientific_data_config(
            OmegaConf.to_container(cfg.data, resolve=True)
        )
        _assert_artifacts_match_followup(
            phase25=phase25,
            remainder_manifest=remainder_manifest,
            checkpoint=checkpoint,
            stats=stats,
            vae=vae,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
            model_config_sha256=base.sha256_json(model_config),
            data_config_sha256=base.sha256_json(data_config),
        )

        visual_cfg = collector.visual
        instruction_cfg = collector.instruction
        scientific_sources = formal_git["scientific_source_files"]
        manifest_payload = base._build_manifest(
            cfg=cfg,
            target_manifest=remainder_manifest,
            targets=targets,
            target_dir=remainder_dir,
            target_manifest_sha256=pilot500_anchor["remainder_manifest_sha256"],
            target_records_sha256=pilot500_anchor["remainder_records_sha256"],
            phase25=phase25,
            phase25_manifest_sha256=_sha256_file(phase_path),
            checkpoint=checkpoint,
            stats=stats,
            vae=vae,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
            extractor=extractor,
            scientific_sources=scientific_sources,
        )
        manifest_payload = augment_remainder400_manifest(
            manifest_payload,
            pilot500_anchor=pilot500_anchor,
            followup_protocol=protocol,
            formal_git=formal_git,
        )
        manifest_path = output_dir / MANIFEST_FILENAME
        manifest = base.ensure_immutable_manifest(manifest_path, manifest_payload)
        fingerprint = str(manifest["compatibility_fingerprint"])
        extractor_fingerprint = str(
            manifest["compatibility"]["extractor_fingerprint"]
        )
        dimensions = base._feature_dimensions_from_manifest(manifest)
        progress_dir = output_dir / ".rows"
        progress_dir.mkdir(exist_ok=True)
        recovery = _recover_unsealed_output(
            output_dir=output_dir,
            progress_dir=progress_dir,
            targets=targets,
        )
        if recovery["removed"]:
            LOGGER.warning("Recovered unsealed crash artifacts: %s", recovery["removed"])

        initial_snapshot = _content_snapshot(
            checkpoint=checkpoint,
            stats=stats,
            vae=vae,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
            phase25_manifest_sha256=_sha256_file(phase_path),
            formal_git=formal_git,
            pilot500_anchor=pilot500_anchor,
            followup_protocol_sha256=protocol["sha256"],
        )
        dataset, ranges, task_tables = _instantiate_dataset_and_contract(
            cfg,
            stats_path=stats_path,
            dataset_sources=dataset_sources,
            context_cache=context_cache,
        )
        pending = [
            target
            for target in targets
            if base._load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            is None
        ]
        LOGGER.info(
            "Remainder feature plan: total=%d existing=%d pending=%d",
            len(targets),
            len(targets) - len(pending),
            len(pending),
        )

        device = torch.device(str(collector.device))
        vae_model: torch.nn.Module | None = None
        vae_dtype: torch.dtype | None = None
        if pending and (max_new_rows is None or max_new_rows > 0):
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("FEATURE_COLLECTOR.device requests unavailable CUDA")
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)
            vae_model, vae_dtype = _load_frozen_vae(
                vae_path=str(vae["path"]), cfg=cfg, device=device
            )

        new_count = 0
        for target in tqdm(targets, desc="remainder400 current-state gate features"):
            # Always rehydrate and hash-check every state, including cached rows
            # and rows beyond an operational max_new_rows smoke budget.
            sample = dataset[int(target["source_index"])]
            state = base._validate_live_state(
                sample=sample,
                target=target,
                ranges=ranges,
                task_tables=task_tables,
            )
            existing = base._load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            if existing is not None:
                continue
            if max_new_rows is not None and new_count >= max_new_rows:
                continue
            if vae_model is None or vae_dtype is None:
                raise AssertionError("pending feature row has no frozen VAE model")

            def encode_current_image(image: torch.Tensor) -> torch.Tensor:
                value = base.prepare_vae_input(image, device=device, dtype=vae_dtype)
                with torch.inference_mode():
                    return vae_model.encode([value], device=device, tiled=False)

            with torch.inference_mode():
                features = base.extract_allowed_features(
                    input_image=state.input_image,
                    context=state.context,
                    proprio=state.proprio,
                    encode_current_image=encode_current_image,
                    projections=projections,
                    latent_channels=int(visual_cfg.latent_channels),
                    pooled_height=int(visual_cfg.pooled_height),
                    pooled_width=int(visual_cfg.pooled_width),
                    context_dim=int(instruction_cfg.context_dim),
                    proprio_dim=int(collector.proprio_dim),
                )
            record = base.build_feature_record(
                target, features, extractor_fingerprint=extractor_fingerprint
            )
            base.validate_feature_record(
                record,
                features,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            base._write_progress_row(
                progress_dir,
                record,
                features,
                extractor_fingerprint=extractor_fingerprint,
            )
            new_count += 1

        completed_progress = sum(
            base._load_progress_row(
                progress_dir,
                target,
                extractor_fingerprint=extractor_fingerprint,
                expected_dimensions=dimensions,
            )
            is not None
            for target in targets
        )
        _rehash_all_inputs(
            cfg,
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            phase_path=phase_path,
            expected_snapshot=initial_snapshot,
        )
        if completed_progress != REMAINDER_STATE_COUNT:
            return {
                "num_states": REMAINDER_STATE_COUNT,
                "existing": REMAINDER_STATE_COUNT - len(pending),
                "new": new_count,
                "progress_rows": completed_progress,
                "complete": False,
                "output_dir": str(output_dir),
            }

        rows, matrices = base._build_final_from_progress(
            progress_dir,
            targets,
            extractor_fingerprint=extractor_fingerprint,
            expected_dimensions=dimensions,
        )
        for key in TENSOR_KEYS:
            expected_shape = (REMAINDER_STATE_COUNT, EXPECTED_DIMS[key])
            if matrices[key].dtype != torch.float32 or tuple(matrices[key].shape) != expected_shape:
                raise AssertionError(
                    f"final {key} tensor must be float32 {expected_shape}, got "
                    f"{matrices[key].dtype} {tuple(matrices[key].shape)}"
                )
        index_path, features_path = base._publish_final_files(
            output_dir=output_dir,
            rows=rows,
            matrices=matrices,
            manifest_fingerprint=fingerprint,
        )
        completion_path = output_dir / COMPLETION_FILENAME
        pending_completion_path = output_dir / PENDING_COMPLETION_FILENAME
        if completion_path.exists():
            base.validate_completion(output_dir)
        else:
            completion = base.completion_payload(
                manifest_path=manifest_path,
                index_path=index_path,
                features_path=features_path,
                matrices=matrices,
                manifest_fingerprint=fingerprint,
                num_states=REMAINDER_STATE_COUNT,
            )
            base._atomic_write_bytes(
                pending_completion_path, base._serialize_json(completion)
            )
            if base._load_json(
                pending_completion_path, label="pending feature-cache completion"
            ) != completion:
                raise ValueError("pending completion bytes differ from computed payload")
            # The public seal must not exist until the final live-input check
            # succeeds. A kill anywhere before os.replace leaves only a private
            # pending file, which the next unsealed resume removes.
            _rehash_all_inputs(
                cfg,
                checkpoint_path=checkpoint_path,
                stats_path=stats_path,
                phase_path=phase_path,
                expected_snapshot=initial_snapshot,
            )
            if base._load_json(
                pending_completion_path, label="pending feature-cache completion"
            ) != completion:
                raise ValueError("pending completion changed before publication")
            os.replace(pending_completion_path, completion_path)
            _fsync_directory(output_dir)
            base.validate_completion(output_dir)
        return {
            "num_states": REMAINDER_STATE_COUNT,
            "existing": REMAINDER_STATE_COUNT - new_count,
            "new": new_count,
            "progress_rows": REMAINDER_STATE_COUNT,
            "complete": True,
            "output_dir": str(output_dir),
            "completion_sha256": base._load_json(
                completion_path, label="feature-cache completion"
            )["completion_sha256"],
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="collect_libero_gate_features_remainder400.yaml",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = collect(cfg)
    LOGGER.info("Gate remainder-400 current-state feature cache: %s", summary)


if __name__ == "__main__":
    main()
