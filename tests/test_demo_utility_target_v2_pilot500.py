from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.libero.gate import demo_utility_stability as stability
from experiments.libero.gate import demo_utility_target_v2 as target_v2
from experiments.libero.gate import demo_utility_target_v2_pilot500 as expansion
from experiments.libero.gate.demo_utility import stable_sample_seed


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pilot(source_index: int) -> dict:
    episode = source_index
    dataset = "libero_goal_no_noops_lerobot"
    sample_id = f"{dataset}/episode_{episode:06d}/frame_000002"
    components = {
        "input_image": _sha(f"image-{source_index}"),
        "proprio": _sha(f"proprio-{source_index}"),
        "context": _sha(f"context-{source_index}"),
        "context_mask": _sha(f"mask-{source_index}"),
        "valid_target_action": _sha(f"action-{source_index}"),
        "action_is_pad": _sha(f"pad-{source_index}"),
    }
    utility = (source_index % 7 - 3) * 1e-3
    e0 = 0.2 + max(utility, 0.0)
    return {
        "schema_version": 1,
        "collector_record_schema_version": 1,
        "sample_id": sample_id,
        "dataset_id": dataset,
        "dataset_name": dataset,
        "suite": "libero_goal",
        "episode_index": episode,
        "episode_id": episode,
        "frame_index": 2,
        "task_index": source_index % 10,
        "task_id": source_index % 10,
        "task_id_source": "lerobot_task_index",
        "task": f"task {source_index % 10}",
        "seed": stable_sample_seed(42, dataset, episode, 2),
        "num_inference_steps": 10,
        "n0": 0,
        "nfull": 10,
        "e0": e0,
        "efull": e0 - utility,
        "utility": utility,
        "valid_length": 32,
        "target_action_shape": [32, 7],
        "pred_n0_shape": [32, 7],
        "pred_nfull_shape": [32, 7],
        "input_hashes": {**components, "combined": target_v2.sha256_json(components)},
        "n0_latency_ms": 1.0,
        "nfull_latency_ms": 2.0,
        "total_latency_ms": 3.0,
        "n0_route": {
            "inference_mode": "prefix",
            "video_prefix_steps": 0,
            "num_inference_steps": 10,
            "force_custom_prefix": True,
        },
        "nfull_route": {
            "inference_mode": "prefix",
            "video_prefix_steps": 10,
            "num_inference_steps": 10,
            "force_custom_prefix": True,
        },
        "current_proprio": [0.1, 0.2],
        "source_metadata": {
            "requested_sample_idx": source_index,
            "source_sample_idx": source_index,
            "dataset_name": dataset,
            "episode_index": episode,
            "frame_index": 2,
            "task_index": source_index % 10,
            "task": f"task {source_index % 10}",
        },
        "manifest_compatibility_fingerprint": "a" * 64,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "d" * 64,
        "vae_sha256": "e" * 64,
        "git_sha": "9" * 40,
    }


def _pilot_bundle():
    rows = [_pilot(index) for index in range(500)]
    order = list(range(500))
    manifest = {
        "compatibility_fingerprint": "a" * 64,
        "selection": {
            "num_samples": 500,
            "ordered_selected_source_indices": order,
            "ordered_selected_source_indices_sha256": target_v2.sha256_json(order),
        },
    }
    existing = [
        {
            "source_index": index,
            "sample_id": rows[index]["sample_id"],
            "pilot_record_sha256": stability.pilot_record_sha256(rows[index]),
        }
        for index in range(100)
    ]
    return manifest, rows, existing


def test_exact_remainder_is_disjoint_complete_and_in_pilot_order():
    manifest, rows, existing = _pilot_bundle()
    result = expansion.build_remainder_selection(rows, manifest, existing)
    assert len(result) == 400
    assert [row["source_index"] for row in result] == list(range(100, 500))
    assert [row["selection_order"] for row in result] == list(range(400))
    assert not ({row["source_index"] for row in result} & set(range(100)))
    assert {row["source_index"] for row in result} | set(range(100)) == set(range(500))


def test_remainder_fails_closed_on_existing_panel_tamper():
    manifest, rows, existing = _pilot_bundle()
    existing[0]["pilot_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Pilot row hash"):
        expansion.build_remainder_selection(rows, manifest, existing)


def _stability_row(
    pilot: dict,
    selection_order: int,
    replicate_index: int,
    manifest_fingerprint: str = "f" * 64,
) -> dict:
    base_seed = expansion.BASE_SEEDS[replicate_index]
    selected = {**pilot, "source_index": pilot["source_metadata"]["requested_sample_idx"],
                "selection_order": selection_order,
                "selection_bin": stability.utility_bin(pilot["utility"])}
    core = deepcopy(pilot)
    core["seed"] = stable_sample_seed(
        base_seed, pilot["dataset_id"], pilot["episode_index"], pilot["frame_index"]
    )
    row = stability.augment_stability_record(
        core,
        pilot_record=selected,
        selection_entry=selected,
        replicate_index=replicate_index,
        replicate_base_seed=base_seed,
        stability_manifest_compatibility_fingerprint=manifest_fingerprint,
        collection_git_sha="8" * 40,
    )
    return expansion.seal_expansion_record(row)


def test_expansion_row_hash_detects_self_consistent_field_mutation(tmp_path):
    row = _stability_row(_pilot(101), 0, 1)
    expansion.validate_expansion_record(row, expected_manifest_fingerprint="f" * 64)
    broken = deepcopy(row)
    broken["n0_latency_ms"] += 1.0
    with pytest.raises(ValueError, match="record SHA-256"):
        expansion.validate_expansion_record(broken)

    records = tmp_path / "records.jsonl"
    records.write_text(target_v2.canonical_json(row) + "\n", encoding="utf-8")
    index = expansion.load_expansion_record_index(
        records,
        expected_manifest_fingerprint="f" * 64,
        expected_pilot_manifest_fingerprint="a" * 64,
    )
    assert set(index) == {(101, 1)}


def test_seed_contract_excludes_independent_validation_seeds():
    assert expansion.BASE_SEEDS == (42, 43, 44, 45, 46)
    assert set(expansion.BASE_SEEDS).isdisjoint(target_v2.VALIDATION_BASE_SEEDS)
    assert expansion.EXPECTED_REMAINDER_RECORD_COUNT == 2000
    assert expansion.EXPECTED_REUSED_RECORD_COUNT == 400
    assert expansion.EXPECTED_NEW_INFERENCE_COUNT == 1600


def _component_target_bundle(pilots: list[dict], component: str):
    manifest_sha = _sha(f"{component}-source-manifest")
    records_sha = _sha(f"{component}-source-records")
    selection_sha = _sha(f"{component}-selection")
    input_sha = target_v2.sha256_json([
        {
            "selection_order": order,
            "source_index": pilot["source_metadata"]["requested_sample_idx"],
            "sample_id": pilot["sample_id"],
            "input_hashes": pilot["input_hashes"],
            "source_pilot_record_sha256": stability.pilot_record_sha256(pilot),
        }
        for order, pilot in enumerate(pilots)
    ])
    fingerprint = "f" * 64
    states = []
    record_index = {}
    for order, pilot in enumerate(pilots):
        source_index = pilot["source_metadata"]["requested_sample_idx"]
        states.append(
            {
                "selection_order": order,
                "source_index": source_index,
                "sample_id": pilot["sample_id"],
            }
        )
        for replicate_index in range(5):
            record_index[(source_index, replicate_index)] = _stability_row(
                pilot, order, replicate_index
            )
    source = target_v2.VerifiedSourceBundle(
        manifest_path=Path(f"/{component}/manifest.json"),
        records_path=Path(f"/{component}/records.jsonl"),
        manifest={
            "compatibility_fingerprint": fingerprint,
            "compatibility": {
                "checkpoint_sha256": "c" * 64,
                "dataset_stats_sha256": "d" * 64,
                "vae_sha256": "e" * 64,
                "pilot_manifest_fingerprint": "a" * 64,
            },
        },
        manifest_sha256=manifest_sha,
        records_sha256=records_sha,
        selection_plan_sha256=selection_sha,
        input_plan_sha256=input_sha,
        ordered_states=tuple(states),
        record_index=record_index,
    )
    return target_v2.build_target_bundle(source)


def test_combined_500_bundle_is_ordered_immutable_and_tamper_evident(tmp_path):
    pilot_manifest, pilots, _ = _pilot_bundle()
    existing_manifest, existing_targets = _component_target_bundle(
        pilots[:100], "existing100"
    )
    remainder_manifest, remainder_targets = _component_target_bundle(
        pilots[100:], "remainder400"
    )
    manifest, targets = expansion.build_combined_target_bundle(
        pilot_manifest=pilot_manifest,
        pilot_records=pilots,
        pilot_manifest_sha256=_sha("pilot-manifest"),
        pilot_records_sha256=_sha("pilot-records"),
        existing_manifest=existing_manifest,
        existing_targets=existing_targets,
        existing_manifest_sha256=_sha("existing-manifest"),
        existing_targets_sha256=_sha("existing-targets"),
        remainder_manifest=remainder_manifest,
        remainder_targets=remainder_targets,
        remainder_manifest_sha256=_sha("remainder-manifest"),
        remainder_targets_sha256=_sha("remainder-targets"),
        expansion_completion_sha256=_sha("expansion-completion"),
    )
    assert [row["source_index"] for row in targets] == list(range(500))
    assert [
        row["component"] for row in manifest["selection"]["ordered_states"]
    ] == ["existing100"] * 100 + ["remainder400"] * 400

    output = tmp_path / "combined"
    manifest_path, targets_path, completion_path = expansion.write_combined_target_bundle(
        output, manifest, targets
    )
    loaded_manifest, loaded_targets, completion = expansion.load_combined_target_bundle(
        output,
        expected_manifest_sha256=expansion.sha256_file(manifest_path),
        expected_targets_sha256=expansion.sha256_file(targets_path),
    )
    assert loaded_manifest == manifest
    assert loaded_targets == targets
    assert completion["target_count"] == 500
    with pytest.raises(FileExistsError, match="immutable"):
        expansion.write_combined_target_bundle(output, manifest, targets)

    broken = deepcopy(completion)
    broken["target_count"] = 499
    broken["completion_sha256"] = expansion.sha256_json(
        {key: value for key, value in broken.items() if key != "completion_sha256"}
    )
    completion_path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target count mismatch"):
        expansion.load_combined_target_bundle(output)


def _expansion_source_bundle(tmp_path):
    _, pilots, _ = _pilot_bundle()
    remainder = pilots[100:]
    states = [
        {
            "selection_order": order,
            "source_index": pilot["source_metadata"]["requested_sample_idx"],
            "sample_id": pilot["sample_id"],
            "suite": pilot["suite"],
            "task_index": pilot["task_index"],
            "episode_index": pilot["episode_index"],
            "frame_index": pilot["frame_index"],
            "selection_bin": stability.utility_bin(pilot["utility"]),
            "pilot_utility": pilot["utility"],
            "valid_length": pilot["valid_length"],
            "pilot_seed": pilot["seed"],
            "pilot_record_sha256": stability.pilot_record_sha256(pilot),
        }
        for order, pilot in enumerate(remainder)
    ]
    selection_sha = expansion.sha256_json(states)
    compatibility = {
        "schema_version": 1,
        "kind": target_v2.SOURCE_BUNDLE_KIND,
        "purpose": expansion.EXPANSION_PURPOSE,
        "pilot_manifest_fingerprint": "a" * 64,
        "pilot_manifest_sha256": "1" * 64,
        "pilot_records_sha256": "2" * 64,
        "phase25_manifest_sha256": "3" * 64,
        "phase25_records_sha256": "4" * 64,
        "phase25_selection_plan_sha256": "5" * 64,
        "existing_target_v2_manifest_sha256": "6" * 64,
        "existing_target_v2_targets_sha256": "7" * 64,
        "pilot_ordered_source_indices_sha256": expansion.sha256_json(list(range(500))),
        "selection_plan_sha256": selection_sha,
        "num_states": 400,
        "replicate_base_seeds": [42, 43, 44, 45, 46],
        "reuse_base_seed": 42,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "d" * 64,
        "vae_sha256": "e" * 64,
        "dataset_source_content": [{"dataset_name": "dataset", "sha256": "8" * 64,
                                    "file_count": 1, "total_size_bytes": 10}],
        "context_cache_sha256": "9" * 64,
        "collection_git_commit": "8" * 40,
    }
    fingerprint = expansion.sha256_json(compatibility)
    manifest = {
        "schema_version": 1,
        "kind": target_v2.SOURCE_BUNDLE_KIND,
        "purpose": expansion.EXPANSION_PURPOSE,
        "compatibility_fingerprint": fingerprint,
        "compatibility": compatibility,
        "pilot": {
            "manifest_fingerprint": "a" * 64,
            "manifest_sha256": "1" * 64,
            "records_sha256": "2" * 64,
        },
        "excluded_phase25": {
            "manifest_sha256": "3" * 64,
            "records_sha256": "4" * 64,
            "selection_plan_sha256": "5" * 64,
        },
        "existing_target_v2": {
            "manifest_sha256": "6" * 64,
            "targets_sha256": "7" * 64,
            "state_count": 100,
        },
        "scientific_source_files": {"collector.py": "a" * 64},
        "selection": {
            "num_states": 400,
            "ordered_states": states,
            "ordered_states_sha256": selection_sha,
            "pilot_ordered_source_indices_sha256": expansion.sha256_json(
                list(range(500))
            ),
        },
        "replicates": {
            "base_seeds": [42, 43, 44, 45, 46],
            "count": 5,
            "reuse_base_seed": 42,
            "reuse_replicate_index": 0,
            "expected_record_count": 2000,
            "expected_reused_record_count": 400,
            "expected_new_inference_count": 1600,
        },
    }
    records = [
        _stability_row(pilot, order, replicate_index, fingerprint)
        for order, pilot in enumerate(remainder)
        for replicate_index in range(5)
    ]
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    completion_path = tmp_path / "completion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records_path.write_text(
        "".join(target_v2.canonical_json(row) + "\n" for row in records),
        encoding="utf-8",
    )
    errors_path.write_text("", encoding="utf-8")
    return manifest, manifest_path, records_path, errors_path, completion_path


def _component_hashes(manifest):
    compatibility = manifest["compatibility"]
    return {
        "pilot": {"manifest_sha256": "1" * 64, "records_sha256": "2" * 64},
        "phase25": {"manifest_sha256": "3" * 64, "records_sha256": "4" * 64},
        "existing_target_v2": {
            "manifest_sha256": "6" * 64,
            "targets_sha256": "7" * 64,
        },
        "artifacts": {
            "checkpoint_sha256": "c" * 64,
            "dataset_stats_sha256": "d" * 64,
            "vae_sha256": "e" * 64,
            "dataset_sources": [
                {"dataset_index": 0, **compatibility["dataset_source_content"][0]}
            ],
            "text_embedding_cache": {
                "sha256": "9" * 64,
                "file_count": 1,
                "total_size_bytes": 10,
            },
        },
        "scientific_source_files": {"collector.py": "a" * 64},
    }


def test_formal_2000_row_completion_seal_is_exhaustive_and_tamper_evident(tmp_path):
    manifest, manifest_path, records_path, errors_path, completion_path = (
        _expansion_source_bundle(tmp_path)
    )
    seal = expansion.ensure_completion_seal(
        completion_path,
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
        component_hashes=_component_hashes(manifest),
    )
    assert seal["records_count"] == 2000
    assert seal["reused_record_count"] == 400
    assert seal["new_inference_record_count"] == 1600
    assert seal["errors_count"] == 0

    rows = records_path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(rows[-1])
    broken["utility"] += 1.0
    rows[-1] = target_v2.canonical_json(broken)
    records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid expansion row"):
        expansion.validate_completion_seal(
            completion_path,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
            component_hashes=_component_hashes(manifest),
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("pilot", "manifest_sha256"),
        ("phase25", "records_sha256"),
        ("existing_target_v2", "targets_sha256"),
        ("artifacts", "checkpoint_sha256"),
        ("artifacts", "vae_sha256"),
    ],
)
def test_completion_rejects_mutated_component_snapshot(tmp_path, section, field):
    manifest, manifest_path, records_path, errors_path, _ = _expansion_source_bundle(tmp_path)
    components = _component_hashes(manifest)
    components[section][field] = "0" * 64
    with pytest.raises(ValueError, match="component hash snapshot differs"):
        expansion.build_completion_payload(
            manifest_path=manifest_path, records_path=records_path,
            errors_path=errors_path, manifest=manifest, component_hashes=components,
        )
