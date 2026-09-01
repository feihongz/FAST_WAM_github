#!/usr/bin/env python3
"""Prepare immutable nested Stage 2 selection artifacts for LIBERO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastwam.alignment.checkpointing import sha256_file
from fastwam.gating.contracts import require_sha256
from fastwam.gating.selection import (
    build_libero_episode_strata,
    build_selection_artifacts,
    load_selection_artifacts,
    write_selection_artifacts,
)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise TypeError(f"{label} line {line_number} must be an object")
        rows.append(value)
    return rows


def _bound_metadata_path(root: dict[str, Any], relative_path: str) -> Path:
    records = [
        row
        for row in root.get("files", [])
        if isinstance(row, dict) and row.get("relative_path") == relative_path
    ]
    if len(records) != 1:
        raise ValueError(f"data manifest must bind exactly one {relative_path}")
    record = records[0]
    if record.get("role") != "metadata":
        raise ValueError(f"data manifest {relative_path} must have metadata role")
    path = Path(str(root["root"])).expanduser().resolve() / relative_path
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bound metadata must be a regular file: {path}")
    if path.stat().st_size != record.get("size_bytes"):
        raise ValueError(f"bound metadata size changed: {path}")
    if sha256_file(path) != require_sha256(
        record.get("sha256"), field=f"{relative_path} sha256"
    ):
        raise ValueError(f"bound metadata SHA256 changed: {path}")
    return path


def _libero_episode_task_indices(
    data_manifest: dict[str, Any],
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for dataset_index, root_value in enumerate(data_manifest["dataset_roots"]):
        if not isinstance(root_value, dict):
            raise TypeError("data manifest dataset root must be an object")
        root = root_value
        tasks_path = _bound_metadata_path(root, "meta/tasks.jsonl")
        episodes_path = _bound_metadata_path(root, "meta/episodes.jsonl")
        task_rows = _jsonl_objects(tasks_path, label="LIBERO tasks metadata")
        episode_rows = _jsonl_objects(
            episodes_path, label="LIBERO episodes metadata"
        )
        task_by_text: dict[str, int] = {}
        for row in task_rows:
            if set(row) != {"task_index", "task"}:
                raise ValueError("LIBERO task metadata fields differ")
            task_index = row["task_index"]
            task = row["task"]
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or task_index < 0
                or not isinstance(task, str)
                or not task
            ):
                raise ValueError("LIBERO task metadata is invalid")
            if task in task_by_text or task_index in task_by_text.values():
                raise ValueError("LIBERO task metadata contains duplicates")
            task_by_text[task] = task_index

        selected = set(root["selected_episodes"])
        boundary_lengths = {
            row["episode_index"]: row["length"]
            for row in root["episode_boundaries"]
        }
        observed: set[int] = set()
        for row in episode_rows:
            if set(row) != {"episode_index", "tasks", "length"}:
                raise ValueError("LIBERO episode metadata fields differ")
            episode_index = row["episode_index"]
            if episode_index not in selected:
                continue
            if episode_index in observed:
                raise ValueError("LIBERO episode metadata contains duplicates")
            observed.add(episode_index)
            tasks = row["tasks"]
            if not isinstance(tasks, list) or len(tasks) != 1 or tasks[0] not in task_by_text:
                raise ValueError("LIBERO episode must resolve to exactly one task")
            if row["length"] != boundary_lengths[episode_index]:
                raise ValueError("LIBERO episode length differs from data manifest")
            result.append(
                {
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "local_task_index": task_by_text[tasks[0]],
                }
            )
        if observed != selected:
            raise ValueError("LIBERO metadata does not cover selected episodes")
    return sorted(
        result, key=lambda row: (row["dataset_index"], row["episode_index"])
    )


def prepare_libero_selection(
    *,
    data_manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(data_manifest_path).expanduser().resolve()
    data_manifest = _json_object(manifest_path, label="Stage 2 data manifest")
    expected_manifest = require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    )
    if data_manifest.get("manifest_sha256") != expected_manifest:
        raise ValueError("Stage 2 data manifest SHA256 mismatch")
    task_indices = _libero_episode_task_indices(data_manifest)
    strata = build_libero_episode_strata(
        data_manifest, episode_task_indices=task_indices
    )
    expected = build_selection_artifacts(data_manifest, episode_strata=strata)
    root = Path(output_dir).expanduser().resolve()
    if (root / "label_selection.json").exists():
        actual = load_selection_artifacts(
            root, data_manifest=data_manifest, episode_strata=strata
        )
        if actual != expected:
            raise RuntimeError("existing selection artifacts differ from contract")
    else:
        write_selection_artifacts(
            root, expected, data_manifest=data_manifest, episode_strata=strata
        )
        actual = load_selection_artifacts(
            root, data_manifest=data_manifest, episode_strata=strata
        )
    return {
        "selection_dir": str(root),
        "selection_sha256": actual.descriptor["selection_sha256"],
        "episode_assignment_sha256": actual.episode_split["assignment_sha256"],
        "row_count": actual.descriptor["row_count"],
        "coverages": {
            name: dict(value) for name, value in actual.coverages.items()
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    summary = prepare_libero_selection(
        data_manifest_path=args.data_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
