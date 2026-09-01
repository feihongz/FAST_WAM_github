#!/usr/bin/env python3
"""Deterministically merge a complete Stage 2 Gate-label job.

The merge plan is recomputed from the immutable label contract. No directory
glob is accepted as input: every expected chunk must exist, and an unexpected
chunk-like JSON file makes the command fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    read_git_identity,
    sha256_file,
)
from fastwam.gating.artifacts import (
    build_label_artifact_context,
    load_complete_label_chunk_from_context,
    merge_label_chunks,
    validate_merged_label_artifact,
)
from fastwam.gating.contracts import require_sha256
from fastwam.gating.selection import (
    SelectionArtifacts,
    load_selection_artifacts,
    selected_rows_for_coverage,
)
from fastwam.gating.source_guard import capture_selected_source_snapshot
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()

_ROOT_KEYS = {
    "label_job_dir",
    "data_manifest",
    "episode_split",
    "label_contract",
    "output",
    "runtime",
}
_SUBSET_ROOT_KEYS = {"label_selection", "label_coverage"}
_LABEL_SELECTION_KEYS = {"directory", "expected_sha256"}
_LABEL_COVERAGE_KEYS = {"tier", "expected_sha256"}


def _resolved_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Stage 2 merge config must resolve to a mapping")
    keys = set(payload)
    has_selection = "label_selection" in keys
    has_coverage = "label_coverage" in keys
    if has_selection != has_coverage:
        raise ValueError(
            "label_selection and label_coverage must be provided together"
        )
    expected = _ROOT_KEYS | (_SUBSET_ROOT_KEYS if has_selection else set())
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise ValueError(
            "Stage 2 merge config fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return payload


def _exact_section(
    config: Mapping[str, Any],
    name: str,
    keys: set[str],
) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    if set(payload) != keys:
        missing = sorted(keys - set(payload))
        unexpected = sorted(set(payload) - keys)
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return payload


def _load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _repo_dir(runtime: Mapping[str, Any]) -> Path:
    configured = runtime.get("repo_dir")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _validated_git_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    identity = read_git_identity(_repo_dir(runtime))
    if bool(runtime["require_clean_git"]) and (
        identity.tracked_dirty or identity.untracked_source_files
    ):
        raise RuntimeError(
            "formal Stage 2 merge requires clean tracked files and no "
            "untracked source/config/test files"
        )
    return identity.as_dict()


def _basename(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."}:
        raise ValueError(f"{field} must be a non-empty local basename")
    if Path(value).name != value:
        raise ValueError(f"{field} must be a local basename")
    return value


def _expected_sha(spec: Mapping[str, Any], *, label: str) -> str:
    return require_sha256(
        spec.get("expected_sha256"),
        field=f"{label} expected_sha256",
    )


def _load_identity_inputs(
    resolved: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_spec = _exact_section(
        resolved, "data_manifest", {"path", "expected_sha256"}
    )
    split_spec = _exact_section(
        resolved,
        "episode_split",
        {"path", "expected_assignment_sha256"},
    )
    contract_spec = _exact_section(
        resolved, "label_contract", {"path", "expected_sha256"}
    )
    data_manifest = _load_json_mapping(
        manifest_spec["path"], label="Stage 2 data manifest"
    )
    episode_split = _load_json_mapping(
        split_spec["path"], label="Stage 2 episode split"
    )
    contract = _load_json_mapping(
        contract_spec["path"], label="Stage 2 label contract"
    )
    if data_manifest.get("manifest_sha256") != _expected_sha(
        manifest_spec, label="data manifest"
    ):
        raise ValueError("Stage 2 data manifest SHA256 mismatch")
    expected_assignment = require_sha256(
        split_spec["expected_assignment_sha256"],
        field="episode split expected_assignment_sha256",
    )
    if episode_split.get("assignment_sha256") != expected_assignment:
        raise ValueError("Stage 2 episode split assignment SHA256 mismatch")
    if contract.get("contract_sha256") != _expected_sha(
        contract_spec, label="label contract"
    ):
        raise ValueError("Stage 2 label contract SHA256 mismatch")
    return data_manifest, episode_split, contract


def _load_selection_binding(
    resolved: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any],
) -> tuple[SelectionArtifacts | None, str | None, Mapping[str, Any] | None]:
    """Load the optional committed selection and exact requested coverage."""

    if "label_selection" not in resolved:
        return None, None, None
    selection_spec = _exact_section(
        resolved, "label_selection", _LABEL_SELECTION_KEYS
    )
    coverage_spec = _exact_section(
        resolved, "label_coverage", _LABEL_COVERAGE_KEYS
    )
    directory = selection_spec["directory"]
    if not isinstance(directory, str) or not directory:
        raise ValueError("label_selection.directory must be a non-empty path")
    artifacts = load_selection_artifacts(
        Path(directory).expanduser().resolve(),
        data_manifest=data_manifest,
    )
    expected_selection = require_sha256(
        selection_spec["expected_sha256"],
        field="label selection expected_sha256",
    )
    actual_selection = require_sha256(
        artifacts.descriptor.get("selection_sha256"),
        field="label selection selection_sha256",
    )
    if actual_selection != expected_selection:
        raise ValueError("Stage 2 label selection SHA256 mismatch")
    tier = coverage_spec["tier"]
    if not isinstance(tier, str) or not tier:
        raise ValueError("label_coverage.tier must be a non-empty string")
    coverage = artifacts.coverages.get(tier)
    if coverage is None:
        raise ValueError(f"unknown Stage 2 label coverage tier: {tier}")
    expected_coverage = require_sha256(
        coverage_spec["expected_sha256"],
        field="label coverage expected_sha256",
    )
    actual_coverage = require_sha256(
        coverage.get("coverage_sha256"),
        field="label coverage coverage_sha256",
    )
    if actual_coverage != expected_coverage:
        raise ValueError("Stage 2 label coverage SHA256 mismatch")
    return artifacts, tier, coverage


def _check_planned_chunks(
    *,
    label_job_dir: Path,
    plans: tuple[Any, ...],
) -> tuple[Path, ...]:
    root = label_job_dir.resolve()
    planned_paths: list[Path] = []
    for plan in plans:
        path = Path(plan.path).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"planned label chunk escapes label_job_dir: {path}"
            ) from error
        planned_paths.append(path)
    if len(planned_paths) != len(set(planned_paths)):
        raise ValueError("label chunk plan contains duplicate paths")
    planned = set(planned_paths)
    discovered = {
        path.resolve()
        for path in root.glob("shard-*/chunk-*.json")
        if os.path.lexists(path)
    }
    missing = sorted(str(path) for path in planned - discovered)
    unexpected = sorted(str(path) for path in discovered - planned)
    if missing or unexpected:
        raise ValueError(
            "label chunk files differ from the immutable plan: "
            f"missing={missing}, unexpected={unexpected}"
        )
    non_files = sorted(str(path) for path in planned if not path.is_file())
    if non_files:
        raise ValueError(f"planned label chunks are not regular files: {non_files}")
    return tuple(planned_paths)


def _check_subset_planned_chunks(
    *,
    label_job_dir: Path,
    active_plans: tuple[Any, ...],
    known_plans: tuple[Any, ...],
    context: Any,
    selection_sha256: str,
) -> tuple[Path, ...]:
    """Require active chunks, permit valid inactive cohorts, reject unknowns."""

    root = label_job_dir.resolve()

    def indexed(
        plans: tuple[Any, ...],
        *,
        label: str,
    ) -> tuple[list[Path], dict[Path, Any]]:
        ordered: list[Path] = []
        by_path: dict[Path, Any] = {}
        for plan in plans:
            path = Path(plan.path).expanduser().resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"{label} label chunk escapes label_job_dir: {path}"
                ) from error
            if plan.cohort_index is None:
                raise ValueError(f"{label} subset plan has no cohort_index")
            if path in by_path:
                raise ValueError(f"{label} label chunk plan contains duplicate paths")
            ordered.append(path)
            by_path[path] = plan
        return ordered, by_path

    active_paths, active_by_path = indexed(active_plans, label="active")
    _, known_by_path = indexed(known_plans, label="known")
    if not set(active_by_path).issubset(known_by_path):
        raise ValueError("active label chunk plan is not part of the master selection")
    for path, active in active_by_path.items():
        known = known_by_path[path]
        if (
            active.cohort_index != known.cohort_index
            or active.shard_index != known.shard_index
            or active.chunk_index != known.chunk_index
            or active.planned_sample_ids != known.planned_sample_ids
        ):
            raise ValueError("active label chunk plan differs from master selection")

    discovered_entries = [
        path
        for path in root.rglob("chunk-*.json")
        if os.path.lexists(path)
    ]
    non_files = sorted(
        str(path)
        for path in discovered_entries
        if path.is_symlink() or not path.is_file()
    )
    if non_files:
        raise ValueError(f"planned label chunks are not regular files: {non_files}")
    discovered = {
        path.resolve() for path in discovered_entries
    }
    missing = sorted(str(path) for path in set(active_paths) - discovered)
    unexpected = sorted(str(path) for path in discovered - set(known_by_path))
    if missing or unexpected:
        raise ValueError(
            "label chunk files differ from the immutable coverage plan: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for path in sorted(discovered):
        plan = known_by_path[path]
        chunk = load_complete_label_chunk_from_context(
            path,
            context=context,
            planned_sample_ids=plan.planned_sample_ids,
            selection_sha256=selection_sha256,
            cohort_index=plan.cohort_index,
        )
        if (
            chunk["cohort_index"] != plan.cohort_index
            or chunk["shard_index"] != plan.shard_index
            or chunk["chunk_index"] != plan.chunk_index
        ):
            raise ValueError(
                "label chunk coordinates differ from the immutable plan: "
                f"path={path}"
            )
    return tuple(active_paths)


def _publish_file_no_clobber(source: Path, destination: Path) -> None:
    """Create ``destination`` atomically or require identical existing bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError(
                f"refusing to replace a non-regular merged artifact: {destination}"
            )
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(
                f"existing merged artifact differs from recomputed bytes: {destination}"
            )
    descriptor = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_merge_gate_labels(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact coverage, then publish rows followed by its manifest."""

    resolved = _resolved_config(config)
    runtime = _exact_section(
        resolved, "runtime", {"repo_dir", "require_clean_git"}
    )
    git_identity = _validated_git_identity(runtime)
    data_manifest, episode_split, contract = _load_identity_inputs(resolved)
    selection_artifacts, coverage_tier, coverage = _load_selection_binding(
        resolved,
        data_manifest=data_manifest,
    )
    if (
        selection_artifacts is not None
        and dict(selection_artifacts.episode_split) != episode_split
    ):
        raise ValueError("configured episode split differs from label selection")
    if contract.get("git_identity") != git_identity:
        raise RuntimeError(
            "merge Git identity differs from the label-generation contract"
        )
    source_snapshot = capture_selected_source_snapshot(data_manifest)
    source_snapshot.check_content()
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )

    chunk_size = contract["chunk_size"]
    label_job_dir = Path(str(resolved["label_job_dir"])).expanduser().resolve()

    # Local imports keep CLI parsing independent from the generation runtime.
    from fastwam.gating.label_job import (  # pylint: disable=import-outside-toplevel
        enumerate_label_samples,
        plan_label_chunks,
    )

    subset_merge_kwargs: dict[str, Any] = {}
    if selection_artifacts is None:
        samples = enumerate_label_samples(context)
        expected_sample_ids = sorted(sample.sample_id for sample in samples)
        if len(expected_sample_ids) != len(set(expected_sample_ids)):
            raise ValueError("label sample enumeration contains duplicate sample IDs")
        if len(expected_sample_ids) != int(data_manifest["num_frames"]):
            raise ValueError(
                "label sample enumeration does not cover the data manifest"
            )
        plans = plan_label_chunks(
            context=context,
            output_dir=label_job_dir,
            chunk_size=chunk_size,
            shard_indices=None,
        )
        planned_ids = sorted(
            sample_id
            for plan in plans
            for sample_id in plan.planned_sample_ids
        )
        if planned_ids != expected_sample_ids:
            raise ValueError(
                "label chunk plan does not exactly cover expected samples"
            )
        chunk_paths = _check_planned_chunks(
            label_job_dir=label_job_dir,
            plans=plans,
        )
        for plan, path in zip(plans, chunk_paths, strict=True):
            chunk = load_complete_label_chunk_from_context(
                path,
                context=context,
                planned_sample_ids=plan.planned_sample_ids,
            )
            if (
                chunk["shard_index"] != plan.shard_index
                or chunk["chunk_index"] != plan.chunk_index
            ):
                raise ValueError(
                    "label chunk coordinates differ from the immutable plan: "
                    f"path={path}"
                )
    else:
        if coverage_tier is None or coverage is None:
            raise RuntimeError(
                "subset selection and coverage binding was lost after validation"
            )
        selection_sha256 = require_sha256(
            selection_artifacts.descriptor["selection_sha256"],
            field="label selection selection_sha256",
        )
        coverage_sha256 = require_sha256(
            coverage["coverage_sha256"],
            field="label coverage coverage_sha256",
        )
        selected_rows = selected_rows_for_coverage(
            selection_artifacts,
            tier=coverage_tier,
        )
        expected_sample_ids = sorted(row["sample_id"] for row in selected_rows)
        if len(expected_sample_ids) != len(set(expected_sample_ids)):
            raise ValueError("label selection contains duplicate sample IDs")
        if len(expected_sample_ids) != coverage["sample_count"]:
            raise ValueError("label coverage sample_count differs from selection")
        if canonical_json_sha256(expected_sample_ids) != coverage[
            "sample_ids_sha256"
        ]:
            raise ValueError("label coverage sample IDs differ from selection")
        plans = plan_label_chunks(
            context=context,
            output_dir=label_job_dir,
            chunk_size=chunk_size,
            shard_indices=None,
            selection_artifacts=selection_artifacts,
            coverage_tier=coverage_tier,
        )
        planned_ids = sorted(
            sample_id
            for plan in plans
            for sample_id in plan.planned_sample_ids
        )
        if planned_ids != expected_sample_ids:
            raise ValueError(
                "label chunk plan does not exactly cover expected samples"
            )

        coverage_tiers = selection_artifacts.descriptor["coverage_tiers"]
        if not isinstance(coverage_tiers, list) or not coverage_tiers:
            raise ValueError("label selection has no coverage tiers")
        master_tier = coverage_tiers[-1]["tier"]
        master_coverage = selection_artifacts.coverages[master_tier]
        all_cohort_indices = [
            row["cohort_index"]
            for row in selection_artifacts.descriptor["cohorts"]
        ]
        if master_coverage["active_cohort_indices"] != all_cohort_indices:
            raise ValueError("last label coverage does not activate every cohort")
        known_plans = plan_label_chunks(
            context=context,
            output_dir=label_job_dir,
            chunk_size=chunk_size,
            shard_indices=None,
            selection_artifacts=selection_artifacts,
            coverage_tier=master_tier,
        )
        known_ids = sorted(
            sample_id
            for plan in known_plans
            for sample_id in plan.planned_sample_ids
        )
        if canonical_json_sha256(known_ids) != selection_artifacts.descriptor[
            "sample_ids_sha256"
        ]:
            raise ValueError("master chunk plan differs from label selection")
        chunk_paths = _check_subset_planned_chunks(
            label_job_dir=label_job_dir,
            active_plans=plans,
            known_plans=known_plans,
            context=context,
            selection_sha256=selection_sha256,
        )
        subset_merge_kwargs = {
            "selection_sha256": selection_sha256,
            "coverage_sha256": coverage_sha256,
            "active_cohort_indices": coverage["active_cohort_indices"],
        }

    output = _exact_section(
        resolved,
        "output",
        {
            "directory",
            "rows_file",
            "manifest_file",
            "expected_manifest_sha256",
        },
    )
    output_dir = Path(str(output["directory"])).expanduser().resolve()
    rows_path = output_dir / _basename(
        output["rows_file"], field="output.rows_file"
    )
    manifest_path = output_dir / _basename(
        output["manifest_file"], field="output.manifest_file"
    )
    expected_output_sha = output["expected_manifest_sha256"]
    if expected_output_sha not in (None, ""):
        expected_output_sha = require_sha256(
            expected_output_sha,
            field="output expected_manifest_sha256",
        )

    if os.path.lexists(manifest_path):
        if (
            manifest_path.is_symlink()
            or rows_path.is_symlink()
            or not rows_path.is_file()
            or not manifest_path.is_file()
        ):
            raise RuntimeError("existing merged manifest is incomplete or non-regular")
        existing = validate_merged_label_artifact(
            manifest_path,
            contract=contract,
            data_manifest=data_manifest,
            episode_split=episode_split,
            **subset_merge_kwargs,
        )
        if existing["expected_sample_ids_sha256"] != canonical_json_sha256(
            expected_sample_ids
        ):
            raise ValueError("existing merged artifact has different coverage")
        if expected_output_sha and existing["manifest_sha256"] != expected_output_sha:
            raise ValueError("existing merged label manifest SHA256 mismatch")
        return existing

    if os.path.lexists(rows_path) and (
        rows_path.is_symlink() or not rows_path.is_file()
    ):
        raise RuntimeError("existing orphan merged rows file is non-regular")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".stage2-merge-",
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        staged_rows = staging_dir / rows_path.name
        staged_manifest = staging_dir / manifest_path.name
        manifest = merge_label_chunks(
            chunk_paths,
            contract=contract,
            data_manifest=data_manifest,
            episode_split=episode_split,
            expected_sample_ids=expected_sample_ids,
            rows_output=staged_rows,
            manifest_output=staged_manifest,
            **subset_merge_kwargs,
        )
        staged_validated = validate_merged_label_artifact(
            staged_manifest,
            contract=contract,
            data_manifest=data_manifest,
            episode_split=episode_split,
            **subset_merge_kwargs,
        )
        if staged_validated != manifest:
            raise RuntimeError("staged merged label artifact changed during build")
        if (
            expected_output_sha
            and staged_validated["manifest_sha256"] != expected_output_sha
        ):
            raise ValueError("merged label manifest SHA256 mismatch")

        # Rows are data; the manifest is the commit marker. An interrupted
        # rows-only publication is safely completed on the next invocation
        # after byte-for-byte comparison with a freshly staged artifact.
        source_snapshot.check_stats()
        source_snapshot.check_content()
        _publish_file_no_clobber(staged_rows, rows_path)
        source_snapshot.check_stats()
        _publish_file_no_clobber(staged_manifest, manifest_path)

    validated = validate_merged_label_artifact(
        manifest_path,
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
        **subset_merge_kwargs,
    )
    if validated != manifest:
        raise RuntimeError("merged label artifact changed during publication")
    return validated


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="merge_gate_labels",
)
def main(config: DictConfig) -> None:
    manifest = run_merge_gate_labels(config)
    print(
        "Stage 2 Gate labels merged:\n"
        f"  rows: {manifest['row_count']}\n"
        f"  positives: {manifest['positive_count']}\n"
        f"  manifest_sha256: {manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
