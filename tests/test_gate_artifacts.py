from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import fastwam.gating.artifacts as gate_artifacts
from fastwam.alignment.checkpointing import canonical_json_sha256
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.artifacts import (
    CHUNK_PLAN_ALGORITHM,
    COHORT_CHUNK_PLAN_ALGORITHM,
    ValidatedMergedLabelArtifact,
    build_label_chunk,
    build_label_contract,
    build_label_row,
    load_complete_label_chunk,
    load_validated_merged_label_artifact,
    merge_label_chunks,
    publish_json_atomic_no_clobber,
    shard_for_sample_id,
    validate_label_contract,
    validate_label_row,
    validate_merged_label_artifact,
    write_label_chunk_atomic,
)
from fastwam.gating.contracts import build_episode_split, sample_id


def _data_manifest():
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 7,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": "/data/a",
                "selected_episodes": [2, 5],
                "num_frames": 4,
                "episode_boundaries": [
                    {"episode_index": 2, "from": 0, "to": 2, "length": 2},
                    {"episode_index": 5, "from": 2, "to": 4, "length": 2},
                ],
                "video_keys": [],
                "files": [],
            },
            {
                "dataset_index": 1,
                "root": "/data/b",
                "selected_episodes": [1, 4],
                "num_frames": 3,
                "episode_boundaries": [
                    {"episode_index": 1, "from": 0, "to": 1, "length": 1},
                    {"episode_index": 4, "from": 1, "to": 3, "length": 2},
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


def _identities():
    rows = []
    global_index = 0
    for dataset_index, episodes in enumerate((((2, 2), (5, 2)), ((1, 1), (4, 2)))):
        dataset_frame_index = 0
        for episode_index, length in episodes:
            for frame_index in range(length):
                rows.append(
                    {
                        "global_sample_index": global_index,
                        "dataset_index": dataset_index,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "dataset_frame_index": dataset_frame_index,
                    }
                )
                global_index += 1
                dataset_frame_index += 1
    return rows


def _job(
    *,
    base_seed=42,
    num_shards=3,
    chunk_size=2,
    vae_sha256="f" * 64,
    label_runtime_config_sha256="1" * 64,
    chunk_plan_algorithm=CHUNK_PLAN_ALGORITHM,
):
    manifest = _data_manifest()
    split = build_episode_split(
        manifest,
        validation_fraction=0.5,
        split_seed=9,
    )
    contract = build_label_contract(
        data_manifest=manifest,
        episode_split=split,
        base_checkpoint_sha256="a" * 64,
        adapter_checkpoint_sha256="b" * 64,
        normalization_stats_sha256="c" * 64,
        data_config_sha256="d" * 64,
        vae_sha256=vae_sha256,
        label_runtime_config_sha256=label_runtime_config_sha256,
        git_identity={
            "commit": "e" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
        base_seed=base_seed,
        num_seed_pairs=3,
        relative_margin=0.05,
        num_shards=num_shards,
        chunk_size=chunk_size,
        chunk_plan_algorithm=chunk_plan_algorithm,
    )
    rows = []
    for index, identity in enumerate(_identities()):
        e0 = 1.0
        e10 = 0.8 if index % 2 == 0 else 1.0
        rows.append(
            build_label_row(
                contract=contract,
                data_manifest=manifest,
                episode_split=split,
                identity=identity,
                e0=e0,
                e10=e10,
                relative_gain=e0 - e10,
                label=e10 < 0.95 * e0,
                sample_weight=1.0,
                num_video_frames=5,
            )
        )
    rows.sort(key=lambda row: row["sample_id"])
    return manifest, split, contract, rows


def _chunks(tmp_path, *, contract, manifest, split, rows, prefix="job"):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["shard_index"], []).append(row)
    paths = []
    for shard_index, shard_rows in sorted(grouped.items()):
        shard_rows.sort(key=lambda row: row["sample_id"])
        path = tmp_path / f"{prefix}-shard{shard_index}-chunk0.json"
        write_label_chunk_atomic(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            shard_index=shard_index,
            chunk_index=0,
            planned_sample_ids=[row["sample_id"] for row in shard_rows],
            rows=shard_rows,
        )
        paths.append(path)
    return paths


def _cohort_chunks(
    tmp_path,
    *,
    contract,
    manifest,
    split,
    rows,
    selection_sha256,
    cohort_index,
):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["shard_index"], []).append(row)
    paths = []
    for shard_index, shard_rows in sorted(grouped.items()):
        shard_rows.sort(key=lambda row: row["sample_id"])
        path = (
            tmp_path
            / f"cohort-{cohort_index:05d}"
            / f"shard-{shard_index:05d}"
            / "chunk-00000000.json"
        )
        write_label_chunk_atomic(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            shard_index=shard_index,
            chunk_index=0,
            planned_sample_ids=[row["sample_id"] for row in shard_rows],
            rows=shard_rows,
            selection_sha256=selection_sha256,
            cohort_index=cohort_index,
        )
        paths.append(path)
    return paths


def _merged_artifact(tmp_path):
    manifest, split, contract, rows = _job()
    paths = _chunks(
        tmp_path,
        contract=contract,
        manifest=manifest,
        split=split,
        rows=rows,
    )
    manifest_path = tmp_path / "merged" / "manifest.json"
    rows_path = tmp_path / "merged" / "labels.jsonl"
    merged = merge_label_chunks(
        paths,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=[row["sample_id"] for row in rows],
        rows_output=rows_path,
        manifest_output=manifest_path,
    )
    return (
        manifest,
        split,
        contract,
        rows,
        manifest_path,
        rows_path,
        merged,
    )


def test_label_contract_is_self_hashed_and_binds_every_generation_identity():
    manifest, split, contract, _ = _job()

    assert validate_label_contract(
        contract,
        data_manifest=manifest,
        episode_split=split,
    ) == contract
    assert contract["num_inference_steps"] == 10
    assert contract["num_seed_pairs"] == 3
    assert contract["num_shards"] == 3
    assert contract["schema_version"] == 2
    assert contract["chunk_size"] == 2
    assert contract["vae_sha256"] == "f" * 64
    assert contract["label_runtime_config_sha256"] == "1" * 64
    assert contract["chunk_plan_algorithm"] == CHUNK_PLAN_ALGORITHM
    assert _job(chunk_plan_algorithm=CHUNK_PLAN_ALGORITHM)[2] == contract
    cohort_contract = _job(
        chunk_plan_algorithm=COHORT_CHUNK_PLAN_ALGORITHM
    )[2]
    assert validate_label_contract(cohort_contract) == cohort_contract
    assert (
        cohort_contract["chunk_plan_algorithm"]
        == COHORT_CHUNK_PLAN_ALGORITHM
    )
    assert cohort_contract["contract_sha256"] != contract["contract_sha256"]
    changed_plan_contract = _job(chunk_size=3)[2]
    assert changed_plan_contract["contract_sha256"] != contract["contract_sha256"]
    changed_vae_contract = _job(vae_sha256="2" * 64)[2]
    assert changed_vae_contract["contract_sha256"] != contract["contract_sha256"]
    changed_runtime_contract = _job(
        label_runtime_config_sha256="3" * 64,
    )[2]
    assert changed_runtime_contract["contract_sha256"] != contract["contract_sha256"]
    assert contract["data_manifest_sha256"] == manifest["manifest_sha256"]
    assert contract["episode_assignment_sha256"] == split["assignment_sha256"]

    tampered = deepcopy(contract)
    tampered["base_seed"] += 1
    with pytest.raises(ValueError, match="contract SHA256"):
        validate_label_contract(tampered)
    wrong_algorithm = deepcopy(contract)
    wrong_algorithm["chunk_plan_algorithm"] = "unknown"
    unhashed = dict(wrong_algorithm)
    unhashed.pop("contract_sha256")
    wrong_algorithm["contract_sha256"] = canonical_json_sha256(unhashed)
    with pytest.raises(ValueError, match="chunk plan algorithm"):
        validate_label_contract(wrong_algorithm)
    for field in ("vae_sha256", "label_runtime_config_sha256"):
        invalid_identity = deepcopy(contract)
        invalid_identity[field] = "not-a-sha256"
        unhashed = dict(invalid_identity)
        unhashed.pop("contract_sha256")
        invalid_identity["contract_sha256"] = canonical_json_sha256(unhashed)
        with pytest.raises(ValueError, match=field):
            validate_label_contract(invalid_identity)

    with pytest.raises(ValueError, match="exactly 10"):
        build_label_contract(
            data_manifest=manifest,
            episode_split=split,
            base_checkpoint_sha256="a" * 64,
            adapter_checkpoint_sha256="b" * 64,
            normalization_stats_sha256="c" * 64,
            data_config_sha256="d" * 64,
            vae_sha256="f" * 64,
            label_runtime_config_sha256="1" * 64,
            git_identity={
                "commit": "e" * 40,
                "tracked_dirty": False,
                "untracked_source_files": [],
            },
            base_seed=42,
            num_seed_pairs=3,
            relative_margin=0.05,
            num_inference_steps=9,
            num_shards=3,
            chunk_size=2,
        )


def test_label_row_recomputes_identity_seed_label_and_stable_shard():
    manifest, split, contract, rows = _job()
    row = rows[0]

    assert validate_label_row(
        row,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    ) == row
    assert row["shard_index"] == shard_for_sample_id(
        row["sample_id"], num_shards=contract["num_shards"]
    )
    assert len(set(row["seeds"])) == 3

    changed_seed = deepcopy(row)
    changed_seed["seeds"][0] += 1
    with pytest.raises(ValueError, match="seeds disagree"):
        validate_label_row(
            changed_seed,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )

    changed_label = dict(row, label=not row["label"])
    with pytest.raises(ValueError, match="label disagrees"):
        validate_label_row(
            changed_label,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )


def test_fixed_chunk_resume_requires_complete_self_hashed_plan(tmp_path):
    manifest, split, contract, rows = _job()
    shard = rows[0]["shard_index"]
    shard_rows = [row for row in rows if row["shard_index"] == shard]
    planned = [row["sample_id"] for row in shard_rows]
    path = tmp_path / "chunk.json"
    write_label_chunk_atomic(
        path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        shard_index=shard,
        chunk_index=7,
        planned_sample_ids=planned,
        rows=shard_rows,
    )

    loaded = load_complete_label_chunk(
        path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        planned_sample_ids=planned,
    )
    assert loaded["planned_row_count"] == loaded["row_count"] == len(planned)

    truncated = path.read_text(encoding="utf-8")[:-20]
    path.write_text(truncated, encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable or incomplete"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            planned_sample_ids=planned,
        )

    partial = build_label_chunk(
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        shard_index=shard,
        chunk_index=8,
        planned_sample_ids=planned[:-1],
        rows=shard_rows[:-1],
    )
    path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="external chunk plan"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            planned_sample_ids=planned,
        )


def test_cohort_chunk_v2_requires_exact_selection_and_cohort_binding(tmp_path):
    manifest, split, contract, rows = _job(
        chunk_plan_algorithm=COHORT_CHUNK_PLAN_ALGORITHM
    )
    selection_sha256 = "8" * 64
    shard = rows[0]["shard_index"]
    shard_rows = [row for row in rows if row["shard_index"] == shard]
    planned = [row["sample_id"] for row in shard_rows]
    path = tmp_path / "chunk-v2.json"
    write_label_chunk_atomic(
        path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        shard_index=shard,
        chunk_index=7,
        planned_sample_ids=planned,
        rows=shard_rows,
        selection_sha256=selection_sha256,
        cohort_index=3,
    )

    loaded = load_complete_label_chunk(
        path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        planned_sample_ids=planned,
        selection_sha256=selection_sha256,
        cohort_index=3,
    )
    assert loaded["schema_version"] == 2
    assert loaded["selection_sha256"] == selection_sha256
    assert loaded["cohort_index"] == 3

    with pytest.raises(ValueError, match="provided together"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            selection_sha256=selection_sha256,
        )
    with pytest.raises(ValueError, match="requires a selection/cohort binding"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )
    with pytest.raises(ValueError, match="selection SHA256 mismatch"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            selection_sha256="9" * 64,
            cohort_index=3,
        )

    tampered = deepcopy(loaded)
    tampered["cohort_index"] = 4
    unhashed = dict(tampered)
    unhashed.pop("chunk_sha256")
    tampered["chunk_sha256"] = canonical_json_sha256(unhashed)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort_index mismatch"):
        load_complete_label_chunk(
            path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            selection_sha256=selection_sha256,
            cohort_index=3,
        )


def test_merge_is_deterministic_and_validates_merged_artifact(tmp_path):
    manifest, split, contract, rows = _job()
    paths = _chunks(
        tmp_path,
        contract=contract,
        manifest=manifest,
        split=split,
        rows=rows,
    )
    expected = [row["sample_id"] for row in rows]

    first = merge_label_chunks(
        list(reversed(paths)),
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=expected,
        rows_output=tmp_path / "first" / "labels.jsonl",
        manifest_output=tmp_path / "first" / "manifest.json",
    )
    second = merge_label_chunks(
        paths,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=expected,
        rows_output=tmp_path / "second" / "labels.jsonl",
        manifest_output=tmp_path / "second" / "manifest.json",
    )

    assert first == second
    assert (tmp_path / "first" / "labels.jsonl").read_bytes() == (
        tmp_path / "second" / "labels.jsonl"
    ).read_bytes()
    assert first["row_count"] == len(rows)
    assert first["positive_count"] == sum(int(row["label"]) for row in rows)
    assert validate_merged_label_artifact(
        tmp_path / "first" / "manifest.json",
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    ) == first


def test_cohort_manifest_v2_reuses_chunks_across_coverage_expansion(tmp_path):
    manifest, split, contract, rows = _job(
        num_shards=1,
        chunk_plan_algorithm=COHORT_CHUNK_PLAN_ALGORITHM,
    )
    selection_sha256 = "8" * 64
    pilot_coverage_sha256 = "9" * 64
    expanded_coverage_sha256 = "a" * 64
    cohort_zero_rows = rows[:3]
    cohort_one_rows = rows[3:]
    cohort_zero_paths = _cohort_chunks(
        tmp_path / "chunks",
        contract=contract,
        manifest=manifest,
        split=split,
        rows=cohort_zero_rows,
        selection_sha256=selection_sha256,
        cohort_index=0,
    )
    cohort_one_paths = _cohort_chunks(
        tmp_path / "chunks",
        contract=contract,
        manifest=manifest,
        split=split,
        rows=cohort_one_rows,
        selection_sha256=selection_sha256,
        cohort_index=1,
    )

    pilot = merge_label_chunks(
        cohort_zero_paths,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=[row["sample_id"] for row in cohort_zero_rows],
        rows_output=tmp_path / "pilot" / "labels.jsonl",
        manifest_output=tmp_path / "pilot" / "manifest.json",
        selection_sha256=selection_sha256,
        coverage_sha256=pilot_coverage_sha256,
        active_cohort_indices=[0],
    )
    expanded = merge_label_chunks(
        cohort_zero_paths + cohort_one_paths,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=[row["sample_id"] for row in rows],
        rows_output=tmp_path / "expanded" / "labels.jsonl",
        manifest_output=tmp_path / "expanded" / "manifest.json",
        selection_sha256=selection_sha256,
        coverage_sha256=expanded_coverage_sha256,
        active_cohort_indices=[0, 1],
    )

    assert pilot["schema_version"] == expanded["schema_version"] == 2
    assert pilot["selection_sha256"] == expanded["selection_sha256"]
    assert pilot["coverage_sha256"] != expanded["coverage_sha256"]
    assert pilot["active_cohort_indices"] == [0]
    assert expanded["active_cohort_indices"] == [0, 1]
    assert pilot["chunks"] == [
        record for record in expanded["chunks"] if record["cohort_index"] == 0
    ]
    assert len(expanded["chunks"]) == 2
    assert {
        (record["cohort_index"], record["shard_index"], record["chunk_index"])
        for record in expanded["chunks"]
    } == {(0, 0, 0), (1, 0, 0)}
    assert validate_merged_label_artifact(
        tmp_path / "expanded" / "manifest.json",
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        selection_sha256=selection_sha256,
        coverage_sha256=expanded_coverage_sha256,
        active_cohort_indices=[0, 1],
    ) == expanded

    with pytest.raises(ValueError, match="label manifest keys differ"):
        validate_merged_label_artifact(
            tmp_path / "expanded" / "manifest.json",
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )
    with pytest.raises(ValueError, match="coverage SHA256 mismatch"):
        validate_merged_label_artifact(
            tmp_path / "expanded" / "manifest.json",
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            selection_sha256=selection_sha256,
            coverage_sha256="b" * 64,
            active_cohort_indices=[0, 1],
        )
    with pytest.raises(ValueError, match="duplicate label chunk cohort"):
        merge_label_chunks(
            cohort_zero_paths + cohort_one_paths + cohort_zero_paths,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            expected_sample_ids=[row["sample_id"] for row in rows],
            rows_output=tmp_path / "duplicate" / "labels.jsonl",
            manifest_output=tmp_path / "duplicate" / "manifest.json",
            selection_sha256=selection_sha256,
            coverage_sha256=expanded_coverage_sha256,
            active_cohort_indices=[0, 1],
        )


def test_validated_merged_loader_returns_recursive_immutable_snapshot(tmp_path):
    (
        manifest,
        split,
        contract,
        rows,
        manifest_path,
        rows_path,
        merged,
    ) = _merged_artifact(tmp_path)

    loaded = load_validated_merged_label_artifact(
        manifest_path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    )

    assert isinstance(loaded, ValidatedMergedLabelArtifact)
    assert loaded.manifest["manifest_sha256"] == merged["manifest_sha256"]
    assert tuple(row["sample_id"] for row in loaded.rows) == tuple(
        row["sample_id"] for row in rows
    )
    assert tuple(row["label"] for row in loaded.rows) == tuple(
        row["label"] for row in rows
    )
    with pytest.raises(FrozenInstanceError):
        loaded.rows = ()
    with pytest.raises(TypeError):
        loaded.manifest["row_count"] = 0
    with pytest.raises(TypeError):
        loaded.manifest["split_counts"]["train"] = 0
    with pytest.raises(TypeError):
        loaded.rows[0]["label"] = not loaded.rows[0]["label"]
    with pytest.raises(TypeError):
        loaded.rows[0]["seeds"][0] = 0

    rows_path.write_bytes(b'{"post_load":"replacement"}\n')
    assert len(loaded.rows) == len(rows)
    assert loaded.rows[0]["sample_id"] == rows[0]["sample_id"]


def test_validated_merged_loader_hashes_and_parses_one_jsonl_snapshot(
    tmp_path,
    monkeypatch,
):
    (
        manifest,
        split,
        contract,
        rows,
        manifest_path,
        rows_path,
        _,
    ) = _merged_artifact(tmp_path)
    original_read_bytes = Path.read_bytes
    rows_reads = 0

    def read_bytes_and_replace(path):
        nonlocal rows_reads
        snapshot = original_read_bytes(path)
        if path == rows_path:
            rows_reads += 1
            path.write_bytes(b'{"raced":"replacement"}\n')
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_bytes_and_replace)
    loaded = load_validated_merged_label_artifact(
        manifest_path,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
    )

    assert rows_reads == 1
    assert tuple(row["sample_id"] for row in loaded.rows) == tuple(
        row["sample_id"] for row in rows
    )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "extra"])
def test_merge_rejects_missing_duplicate_and_extra_coverage(tmp_path, failure):
    manifest, split, contract, rows = _job()
    paths = _chunks(
        tmp_path,
        contract=contract,
        manifest=manifest,
        split=split,
        rows=rows,
    )
    expected = [row["sample_id"] for row in rows]
    if failure == "missing":
        supplied_paths = paths[:-1]
        supplied_expected = expected
        message = "coverage"
    elif failure == "duplicate":
        supplied_paths = paths + [paths[0]]
        supplied_expected = expected
        message = "duplicate label chunk"
    else:
        supplied_paths = paths
        supplied_expected = expected[:-1]
        message = "coverage"

    with pytest.raises(ValueError, match=message):
        merge_label_chunks(
            supplied_paths,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            expected_sample_ids=supplied_expected,
            rows_output=tmp_path / failure / "labels.jsonl",
            manifest_output=tmp_path / failure / "manifest.json",
        )


def test_merge_rejects_mixed_and_corrupt_chunks(tmp_path):
    manifest, split, contract, rows = _job()
    paths = _chunks(
        tmp_path,
        contract=contract,
        manifest=manifest,
        split=split,
        rows=rows,
    )
    expected = [row["sample_id"] for row in rows]

    corrupt = json.loads(paths[0].read_text(encoding="utf-8"))
    corrupt["rows"][0]["sample_weight"] = 2.0
    paths[0].write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="chunk SHA256"):
        merge_label_chunks(
            paths,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            expected_sample_ids=expected,
            rows_output=tmp_path / "corrupt" / "labels.jsonl",
            manifest_output=tmp_path / "corrupt" / "manifest.json",
        )

    # A valid chunk from a different immutable contract is rejected before merge.
    _, _, other_contract, other_rows = _job(base_seed=43)
    other_paths = _chunks(
        tmp_path,
        contract=other_contract,
        manifest=manifest,
        split=split,
        rows=other_rows,
        prefix="other",
    )
    with pytest.raises(ValueError, match="mixed label contract"):
        merge_label_chunks(
            other_paths,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
            expected_sample_ids=expected,
            rows_output=tmp_path / "mixed" / "labels.jsonl",
            manifest_output=tmp_path / "mixed" / "manifest.json",
        )


def test_merged_manifest_and_jsonl_tamper_fail_closed(tmp_path):
    manifest, split, contract, rows = _job()
    paths = _chunks(
        tmp_path,
        contract=contract,
        manifest=manifest,
        split=split,
        rows=rows,
    )
    expected = [row["sample_id"] for row in rows]
    manifest_path = tmp_path / "merged" / "manifest.json"
    rows_path = tmp_path / "merged" / "labels.jsonl"
    merge_label_chunks(
        paths,
        contract=contract,
        data_manifest=manifest,
        episode_split=split,
        expected_sample_ids=expected,
        rows_output=rows_path,
        manifest_output=manifest_path,
    )

    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL SHA256"):
        validate_merged_label_artifact(
            manifest_path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )

    rows_path.write_text("", encoding="utf-8")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["rows_file_sha256"] = canonical_json_sha256("")
    with pytest.raises(ValueError, match="manifest SHA256|JSONL SHA256"):
        validate_merged_label_artifact(
            manifest_path,
            contract=contract,
            data_manifest=manifest,
            episode_split=split,
        )


def test_no_clobber_json_publish_preserves_existing_file_and_symlink(tmp_path):
    output = tmp_path / "nested" / "artifact.json"
    first_payload = {"kind": "first", "value": 1}

    assert publish_json_atomic_no_clobber(output, first_payload) is True
    first_bytes = output.read_bytes()
    assert json.loads(first_bytes) == first_payload
    assert publish_json_atomic_no_clobber(
        output, {"kind": "different", "value": 2}
    ) is False
    assert output.read_bytes() == first_bytes

    target = tmp_path / "symlink-target.json"
    target_bytes = b'{"owner":"external"}\n'
    target.write_bytes(target_bytes)
    symlink = tmp_path / "artifact-link.json"
    symlink.symlink_to(target)
    assert publish_json_atomic_no_clobber(symlink, {"replacement": True}) is False
    assert symlink.is_symlink()
    assert target.read_bytes() == target_bytes
    assert not list(tmp_path.rglob("*.tmp"))


def test_no_clobber_json_publish_cleans_temp_after_link_failure(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "nested" / "artifact.json"

    def deny_link(source, destination):
        del source, destination
        raise PermissionError("synthetic link failure")

    monkeypatch.setattr(gate_artifacts.os, "link", deny_link)
    with pytest.raises(PermissionError, match="synthetic link failure"):
        publish_json_atomic_no_clobber(output, {"complete": True})

    assert not output.exists()
    assert not list(tmp_path.rglob("*.tmp"))
