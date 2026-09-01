from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fastwam.alignment.checkpointing import canonical_json_sha256
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.contracts import sample_id, validate_sample_identity
from fastwam.gating.selection import (
    SelectionArtifacts,
    build_libero_episode_strata,
    build_selection_artifacts,
    build_stratified_episode_split,
    load_selection_artifacts,
    selected_rows_for_coverage,
    validate_selection_artifacts,
    validate_stratified_episode_split,
    write_selection_artifacts,
)


def _manifest(
    root_episode_lengths: list[list[int]],
    *,
    episode_indices: list[list[int]] | None = None,
):
    roots = []
    total = 0
    if episode_indices is None:
        episode_indices = [list(range(len(lengths))) for lengths in root_episode_lengths]
    for dataset_index, (lengths, indices) in enumerate(
        zip(root_episode_lengths, episode_indices, strict=True)
    ):
        assert len(lengths) == len(indices)
        boundaries = []
        start = 0
        for episode_index, length in zip(indices, lengths, strict=True):
            boundaries.append(
                {
                    "episode_index": episode_index,
                    "from": start,
                    "to": start + length,
                    "length": length,
                }
            )
            start += length
        roots.append(
            {
                "dataset_index": dataset_index,
                "root": f"/data/libero-{dataset_index}",
                "selected_episodes": indices,
                "num_frames": start,
                "episode_boundaries": boundaries,
                "video_keys": [],
                "files": [],
            }
        )
        total += start
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": total,
        "dataset_roots": roots,
        "text_embedding_cache": {},
        "normalization_stats": {},
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    return manifest


def _task_rows(manifest, *, tasks_per_root=2):
    rows = []
    for root in manifest["dataset_roots"]:
        episodes = root["selected_episodes"]
        per_task = len(episodes) // tasks_per_root
        assert per_task * tasks_per_root == len(episodes)
        for position, episode_index in enumerate(episodes):
            rows.append(
                {
                    "dataset_index": root["dataset_index"],
                    "episode_index": episode_index,
                    "local_task_index": position // per_task,
                }
            )
    return rows


@pytest.fixture
def balanced_manifest():
    # Four LIBERO task strata, four episodes each. All official LIBERO
    # episodes are longer than 64; 80 keeps production-default count checks exact.
    return _manifest([[80] * 8, [80] * 8])


def _build_balanced(manifest):
    strata = build_libero_episode_strata(
        manifest, episode_task_indices=_task_rows(manifest)
    )
    artifacts = build_selection_artifacts(
        manifest,
        episode_strata=strata,
        validation_fraction=0.25,
        split_seed=17,
        selection_seed=23,
    )
    return strata, artifacts


def test_libero_strata_are_explicit_and_split_is_exact_and_balanced(
    balanced_manifest,
):
    task_rows = list(reversed(_task_rows(balanced_manifest)))
    strata = build_libero_episode_strata(
        balanced_manifest, episode_task_indices=task_rows
    )
    split = build_stratified_episode_split(
        balanced_manifest,
        episode_strata=list(reversed(strata)),
        validation_fraction=0.25,
        split_seed=17,
    )

    assert split["target_validation_episodes"] == 4
    assert sum(row["split"] == "validation" for row in split["assignments"]) == 4
    assert {row["validation_episodes"] for row in split["stratum_counts"]} == {1}
    assert {row["train_episodes"] for row in split["stratum_counts"]} == {3}
    assert validate_stratified_episode_split(
        deepcopy(split), balanced_manifest, episode_strata=strata
    ) == split

    # Input ordering cannot alter the signed assignment.
    assert split == build_stratified_episode_split(
        balanced_manifest,
        episode_strata=strata,
        validation_fraction=0.25,
        split_seed=17,
    )


def test_largest_remainder_keeps_exact_global_count():
    manifest = _manifest([[70] * 18])
    sizes = [3, 4, 5, 6]
    rows = []
    cursor = 0
    for task_index, size in enumerate(sizes):
        for episode_index in range(cursor, cursor + size):
            rows.append(
                {
                    "dataset_index": 0,
                    "episode_index": episode_index,
                    "stratum_id": f"task-{task_index}",
                }
            )
        cursor += size
    split = build_stratified_episode_split(
        manifest,
        episode_strata=rows,
        validation_fraction=0.25,
        split_seed=5,
    )

    counts = {
        row["stratum_id"]: row["validation_episodes"]
        for row in split["stratum_counts"]
    }
    assert split["target_validation_episodes"] == 5
    assert sum(counts.values()) == 5
    # Size three has the largest fractional remainder (.75).
    assert counts["task-0"] == 2
    assert all(counts[f"task-{index}"] == 1 for index in (1, 2, 3))


def test_default_nested_coverages_have_production_shape(balanced_manifest):
    strata, artifacts = _build_balanced(balanced_manifest)
    validated = validate_selection_artifacts(
        artifacts, data_manifest=balanced_manifest, episode_strata=strata
    )

    assert validated.descriptor["train_targets"] == [8, 16, 32, 64]
    assert validated.descriptor["validation_target"] == 32
    assert [row["cohort_id"] for row in validated.descriptor["cohorts"]] == [
        "train_rank_000_008",
        "train_rank_008_016",
        "train_rank_016_032",
        "train_rank_032_064",
        "validation_rank_000_032",
    ]

    expected = {
        "pilot": {"train": 12 * 8, "validation": 4 * 32},
        "medium": {"train": 12 * 16, "validation": 4 * 32},
        "formal": {"train": 12 * 32, "validation": 4 * 32},
        "cap": {"train": 12 * 64, "validation": 4 * 32},
    }
    sets = {}
    for tier, split_counts in expected.items():
        coverage = validated.coverages[tier]
        assert coverage["split_counts"] == split_counts
        assert coverage["sample_count"] == sum(split_counts.values())
        rows = selected_rows_for_coverage(validated, tier=tier)
        sets[tier] = {row["sample_id"] for row in rows}
    assert sets["pilot"] < sets["medium"] < sets["formal"] < sets["cap"]

    validation_sets = {
        tier: {
            row["sample_id"]
            for row in selected_rows_for_coverage(validated, tier=tier)
            if row["split"] == "validation"
        }
        for tier in expected
    }
    assert len({frozenset(value) for value in validation_sets.values()}) == 1


def test_rows_use_existing_semantic_sample_identity(balanced_manifest):
    _, artifacts = _build_balanced(balanced_manifest)
    row = artifacts.rows[37]
    identity = {
        key: row[key]
        for key in (
            "global_sample_index",
            "dataset_index",
            "episode_index",
            "frame_index",
            "dataset_frame_index",
        )
    }
    assert validate_sample_identity(balanced_manifest, identity) == identity
    assert sample_id(balanced_manifest, identity) == row["sample_id"]
    assert row["distance_to_episode_end"] == (
        row["episode_length"] - 1 - row["frame_index"]
    )


def test_dyadic_prefixes_span_trajectory(balanced_manifest):
    _, artifacts = _build_balanced(balanced_manifest)
    train_episode = next(
        row
        for row in artifacts.episode_split["assignments"]
        if row["split"] == "train"
    )
    episode_rows = sorted(
        (
            row
            for row in artifacts.rows
            if row["dataset_index"] == train_episode["dataset_index"]
            and row["episode_index"] == train_episode["episode_index"]
        ),
        key=lambda row: row["episode_selection_rank"],
    )
    assert [row["episode_selection_rank"] for row in episode_rows] == list(range(64))
    for prefix in (8, 16, 32, 64):
        # One logical bin in every prefix-sized temporal partition. XOR rotation
        # changes phase, but preserves this dyadic coverage invariant.
        assert {
            row["temporal_bin"] // (64 // prefix) for row in episode_rows[:prefix]
        } == set(range(prefix))


def test_short_episodes_saturate_without_duplicates():
    manifest = _manifest([[3, 5, 70, 71]])
    strata = [
        {
            "dataset_index": 0,
            "episode_index": episode_index,
            "stratum_id": "only-task",
        }
        for episode_index in range(4)
    ]
    artifacts = build_selection_artifacts(
        manifest,
        episode_strata=strata,
        validation_fraction=0.5,
        max_temporal_bins=64,
    )
    for episode_index, length in ((0, 3), (1, 5)):
        rows = [
            row for row in artifacts.rows if row["episode_index"] == episode_index
        ]
        # Validation caps at 32, train at 64; both exceed these short lengths.
        assert len(rows) == length
        assert {row["frame_index"] for row in rows} == set(range(length))
        assert len({row["sample_id"] for row in rows}) == length


@pytest.mark.parametrize("mode", ["missing", "duplicate", "invented"])
def test_explicit_libero_resolver_must_cover_manifest_exactly(
    balanced_manifest, mode
):
    rows = _task_rows(balanced_manifest)
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[-1]["episode_index"] = 999
    with pytest.raises(ValueError, match="cover manifest episodes exactly|duplicate"):
        build_libero_episode_strata(
            balanced_manifest, episode_task_indices=rows
        )


def test_tampering_is_rejected_even_when_inner_hash_is_resigned(balanced_manifest):
    strata, artifacts = _build_balanced(balanced_manifest)

    rows = [dict(row) for row in artifacts.rows]
    rows[0]["frame_index"] += 1
    tampered_rows = SelectionArtifacts(
        episode_split=artifacts.episode_split,
        descriptor=artifacts.descriptor,
        rows=tuple(rows),
        coverages=artifacts.coverages,
    )
    with pytest.raises(ValueError, match="rows SHA256"):
        validate_selection_artifacts(
            tampered_rows, data_manifest=balanced_manifest, episode_strata=strata
        )

    descriptor = deepcopy(dict(artifacts.descriptor))
    descriptor["selection_seed"] += 1
    tampered_descriptor = SelectionArtifacts(
        episode_split=artifacts.episode_split,
        descriptor=descriptor,
        rows=artifacts.rows,
        coverages=artifacts.coverages,
    )
    with pytest.raises(ValueError, match="descriptor SHA256"):
        validate_selection_artifacts(
            tampered_descriptor,
            data_manifest=balanced_manifest,
            episode_strata=strata,
        )

    coverages = deepcopy(dict(artifacts.coverages))
    coverages["formal"]["sample_count"] += 1
    unhashed = dict(coverages["formal"])
    unhashed.pop("coverage_sha256")
    coverages["formal"]["coverage_sha256"] = canonical_json_sha256(unhashed)
    resigned_coverage = SelectionArtifacts(
        episode_split=artifacts.episode_split,
        descriptor=artifacts.descriptor,
        rows=artifacts.rows,
        coverages=coverages,
    )
    with pytest.raises(ValueError, match="coverages differ"):
        validate_selection_artifacts(
            resigned_coverage,
            data_manifest=balanced_manifest,
            episode_strata=strata,
        )


def test_split_tampering_and_wrong_explicit_strata_are_rejected(balanced_manifest):
    strata, artifacts = _build_balanced(balanced_manifest)
    wrong = deepcopy(strata)
    wrong[0]["stratum_id"] = "wrong-task"
    with pytest.raises(ValueError):
        validate_stratified_episode_split(
            artifacts.episode_split,
            balanced_manifest,
            episode_strata=wrong,
        )

    split = deepcopy(dict(artifacts.episode_split))
    split["assignments"][0]["split"] = (
        "validation"
        if split["assignments"][0]["split"] == "train"
        else "train"
    )
    split["assignment_sha256"] = canonical_json_sha256(split["assignments"])
    with pytest.raises(ValueError, match="differs from its contract"):
        validate_stratified_episode_split(split, balanced_manifest)


def test_atomic_round_trip_and_noncanonical_jsonl_rejected(
    balanced_manifest, tmp_path
):
    strata, artifacts = _build_balanced(balanced_manifest)
    paths = write_selection_artifacts(
        tmp_path,
        artifacts,
        data_manifest=balanced_manifest,
        episode_strata=strata,
    )
    assert paths["descriptor"].name == "label_selection.json"
    loaded = load_selection_artifacts(
        tmp_path, data_manifest=balanced_manifest, episode_strata=strata
    )
    assert loaded == artifacts

    rows_path = paths["rows"]
    parsed = json.loads(rows_path.read_text().splitlines()[0])
    lines = rows_path.read_text().splitlines()
    lines[0] = json.dumps(parsed, sort_keys=False, indent=None)  # adds spaces
    rows_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="not canonically serialized"):
        load_selection_artifacts(
            tmp_path, data_manifest=balanced_manifest, episode_strata=strata
        )


def test_manifest_and_selection_configuration_fail_closed(balanced_manifest):
    bad_manifest = deepcopy(balanced_manifest)
    bad_manifest["num_frames"] += 1
    strata = build_libero_episode_strata(
        balanced_manifest, episode_task_indices=_task_rows(balanced_manifest)
    )
    with pytest.raises(ValueError, match="SHA256"):
        build_selection_artifacts(bad_manifest, episode_strata=strata)

    with pytest.raises(ValueError, match="last train target"):
        build_selection_artifacts(
            balanced_manifest,
            episode_strata=strata,
            train_targets=(8, 16, 32),
        )
    with pytest.raises(ValueError, match="power of two"):
        build_selection_artifacts(
            balanced_manifest,
            episode_strata=strata,
            max_temporal_bins=48,
            train_targets=(8, 16, 48),
        )
