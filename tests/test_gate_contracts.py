from copy import deepcopy

import pytest

import fastwam.gating.contracts as gate_contracts
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.contracts import (
    build_episode_split,
    build_episode_split_lookup,
    dataset_id,
    dataset_id_from_lookup,
    derive_pair_seeds,
    sample_id,
    sample_id_from_lookup,
    split_for_identity,
    validate_episode_split,
    validate_sample_identity,
    validate_sample_identity_with_lookup,
)


def _data_manifest():
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 12,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": "/data/a",
                "selected_episodes": [4, 8],
                "num_frames": 5,
                "episode_boundaries": [
                    {"episode_index": 4, "from": 0, "to": 3, "length": 3},
                    {"episode_index": 8, "from": 3, "to": 5, "length": 2},
                ],
                "video_keys": [],
                "files": [],
            },
            {
                "dataset_index": 1,
                "root": "/data/b",
                "selected_episodes": [1, 3, 7],
                "num_frames": 7,
                "episode_boundaries": [
                    {"episode_index": 1, "from": 0, "to": 2, "length": 2},
                    {"episode_index": 3, "from": 2, "to": 6, "length": 4},
                    {"episode_index": 7, "from": 6, "to": 7, "length": 1},
                ],
                "video_keys": [],
                "files": [],
            },
        ],
        "text_embedding_cache": {},
        "normalization_stats": {},
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    return manifest


def _identity():
    return {
        "global_sample_index": 8,
        "dataset_index": 1,
        "episode_index": 3,
        "frame_index": 1,
        "dataset_frame_index": 3,
    }


def test_sample_and_dataset_ids_bind_manifest_and_semantic_frame():
    manifest = _data_manifest()
    identity = _identity()

    assert validate_sample_identity(manifest, identity) == identity
    first = sample_id(manifest, identity)
    assert len(first) == 64
    assert len(dataset_id(manifest, 1)) == 64

    changed = dict(identity, dataset_frame_index=999)
    with pytest.raises(ValueError, match="dataset frame index"):
        sample_id(manifest, changed)
    changed = dict(
        identity,
        frame_index=2,
        dataset_frame_index=4,
        global_sample_index=9,
    )
    assert sample_id(manifest, changed) != first


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_index", 9, "out of range"),
        ("episode_index", 99, "absent"),
        ("frame_index", 4, "outside"),
        ("global_sample_index", 9, "global index"),
        ("dataset_frame_index", 9, "dataset frame index"),
    ],
)
def test_sample_identity_rejects_manifest_mismatch(field, value, message):
    identity = dict(_identity(), **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_sample_identity(_data_manifest(), identity)


def test_pair_seeds_are_stable_unique_and_shard_order_independent():
    stable_id = sample_id(_data_manifest(), _identity())
    seeds = derive_pair_seeds(
        sample_id_sha256=stable_id,
        base_seed=42,
        num_pairs=4,
    )

    assert seeds == derive_pair_seeds(
        sample_id_sha256=stable_id,
        base_seed=42,
        num_pairs=4,
    )
    assert len(seeds) == len(set(seeds)) == 4
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert seeds != derive_pair_seeds(
        sample_id_sha256=stable_id,
        base_seed=43,
        num_pairs=4,
    )


def test_episode_split_is_deterministic_disjoint_and_whole_episode():
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )

    assert validate_episode_split(split, manifest) == split
    assert split == build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )
    for counts in split["root_counts"]:
        assert counts["train_episodes"] >= 1
        assert counts["validation_episodes"] >= 1
    assert split_for_identity(split, manifest, _identity()) in {
        "train",
        "validation",
    }
    assignments = {
        (row["dataset_index"], row["episode_index"]): row["split"]
        for row in split["assignments"]
    }
    assert len(assignments) == 5


def test_prevalidated_split_lookup_avoids_per_sample_rebuild(monkeypatch):
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )
    lookup = build_episode_split_lookup(split, manifest)
    expected = split_for_identity(split, manifest, _identity())

    def fail_if_revalidated(*args, **kwargs):
        raise AssertionError("prevalidated lookup must not rebuild the split")

    monkeypatch.setattr(
        gate_contracts,
        "validate_episode_split",
        fail_if_revalidated,
    )
    assert split_for_identity(
        split,
        manifest,
        _identity(),
        lookup=lookup,
    ) == expected
    assert validate_sample_identity_with_lookup(_identity(), lookup) == _identity()
    assert sample_id_from_lookup(_identity(), lookup) == sample_id(
        manifest,
        _identity(),
    )
    assert dataset_id_from_lookup(1, lookup) == dataset_id(manifest, 1)


def test_prevalidated_identity_helpers_reject_wrong_frame_and_dataset():
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )
    lookup = build_episode_split_lookup(split, manifest)

    with pytest.raises(ValueError, match="global index"):
        sample_id_from_lookup(
            dict(_identity(), global_sample_index=9),
            lookup,
        )
    with pytest.raises(ValueError, match="dataset frame index"):
        sample_id_from_lookup(
            dict(_identity(), dataset_frame_index=4),
            lookup,
        )
    with pytest.raises(ValueError, match="out of range"):
        dataset_id_from_lookup(2, lookup)


def test_prevalidated_split_lookup_rejects_mismatched_artifacts():
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )
    lookup = build_episode_split_lookup(split, manifest)
    other_split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=8,
    )

    with pytest.raises(ValueError, match="assignment SHA256"):
        split_for_identity(
            other_split,
            manifest,
            _identity(),
            lookup=lookup,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("root_frames", "exactly cover"),
        ("total_frames", "do not sum"),
        ("boundary_gap", "contiguous"),
        ("episode_order", "follow selected_episodes"),
        ("boundary_count", "match selected_episodes"),
    ],
)
def test_self_hashed_structurally_invalid_manifest_is_rejected(
    mutation,
    message,
):
    manifest = _data_manifest()
    if mutation == "root_frames":
        manifest["dataset_roots"][0]["num_frames"] += 1
        manifest["num_frames"] += 1
    elif mutation == "total_frames":
        manifest["num_frames"] += 1
    elif mutation == "boundary_gap":
        manifest["dataset_roots"][0]["episode_boundaries"][1]["from"] += 1
    elif mutation == "episode_order":
        manifest["dataset_roots"][0]["selected_episodes"].reverse()
    elif mutation == "boundary_count":
        manifest["dataset_roots"][0]["episode_boundaries"].pop()
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)

    with pytest.raises((TypeError, ValueError), match=message):
        build_episode_split(
            manifest,
            validation_fraction=0.4,
            split_seed=7,
        )


def test_episode_split_and_manifest_tamper_fail_closed():
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.4,
        split_seed=7,
    )
    tampered = deepcopy(split)
    original = tampered["assignments"][0]["split"]
    tampered["assignments"][0]["split"] = (
        "train" if original == "validation" else "validation"
    )
    with pytest.raises(ValueError, match="differs"):
        validate_episode_split(tampered, manifest)

    tampered_manifest = deepcopy(manifest)
    tampered_manifest["dataset_roots"][0]["num_frames"] += 1
    with pytest.raises(ValueError, match="SHA256"):
        build_episode_split(
            tampered_manifest,
            validation_fraction=0.4,
            split_seed=7,
        )
