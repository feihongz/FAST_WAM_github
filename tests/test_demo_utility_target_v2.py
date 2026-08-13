from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.libero.gate.demo_utility import stable_sample_seed
from experiments.libero.gate.demo_utility_stability import (
    augment_stability_record,
    pilot_record_sha256,
    utility_bin,
)
from experiments.libero.gate.demo_utility_target_v2 import (
    DEFAULT_DEADBAND_EPSILON,
    OFFICIAL_SOURCE_MANIFEST_SHA256,
    OFFICIAL_SOURCE_RECORDS_SHA256,
    OFFICIAL_SOURCE_SELECTION_PLAN_SHA256,
    SOURCE_BUNDLE_KIND,
    TARGET_BASE_SEEDS,
    TARGET_BUNDLE_KIND,
    VALIDATION_BASE_SEEDS,
    augment_validation_record,
    build_target_bundle,
    build_validation_plan,
    canonical_json,
    load_target_bundle,
    load_validation_record_index,
    load_verified_source_bundle,
    sha256_file,
    sha256_json,
    target_record_sha256,
    validate_target_manifest,
    validate_target_record,
    validate_validation_grid,
    validate_validation_record,
    validation_record_sha256,
    write_target_bundle,
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pilot_row(
    *,
    source_index: int,
    episode_index: int,
    utility: float,
    task_index: int = 0,
    suite: str = "libero_goal",
    pilot_fingerprint: str = "a" * 64,
) -> dict:
    dataset = f"{suite}_no_noops_lerobot"
    sample_id = f"{dataset}/episode_{episode_index:06d}/frame_000002"
    components = {
        "input_image": _sha_text(f"image-{source_index}"),
        "proprio": _sha_text(f"proprio-{source_index}"),
        "context": _sha_text(f"context-{source_index}"),
        "context_mask": _sha_text(f"mask-{source_index}"),
        "valid_target_action": _sha_text(f"action-{source_index}"),
        "action_is_pad": _sha_text(f"pad-{source_index}"),
    }
    hashes = {**components, "combined": sha256_json(components)}
    e0 = max(utility, 0.0) + 0.2
    efull = e0 - utility
    return {
        "schema_version": 1,
        "collector_record_schema_version": 1,
        "sample_id": sample_id,
        "dataset_id": dataset,
        "dataset_name": dataset,
        "suite": suite,
        "episode_index": episode_index,
        "episode_id": episode_index,
        "frame_index": 2,
        "task_index": task_index,
        "task_id": task_index,
        "task_id_source": "lerobot_task_index",
        "task": f"task {task_index}",
        "seed": stable_sample_seed(42, dataset, episode_index, 2),
        "num_inference_steps": 10,
        "n0": 0,
        "nfull": 10,
        "e0": e0,
        "efull": efull,
        "utility": utility,
        "valid_length": 32,
        "target_action_shape": [32, 7],
        "pred_n0_shape": [32, 7],
        "pred_nfull_shape": [32, 7],
        "input_hashes": hashes,
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
            "episode_index": episode_index,
            "frame_index": 2,
            "task_index": task_index,
            "task": f"task {task_index}",
        },
        "manifest_compatibility_fingerprint": pilot_fingerprint,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "d" * 64,
        "vae_sha256": "e" * 64,
        "git_sha": "9" * 40,
    }


def _write_source_bundle(
    root: Path,
    utility_sets: list[list[float]],
) -> tuple[Path, Path, str]:
    pilots = [
        _pilot_row(
            source_index=100 + index,
            episode_index=10 + index,
            task_index=index,
            utility=values[0],
        )
        for index, values in enumerate(utility_sets)
    ]
    states = []
    selections = []
    for order, pilot in enumerate(pilots):
        selection = {
            **pilot,
            "source_index": pilot["source_metadata"]["requested_sample_idx"],
            "selection_order": order,
            "selection_bin": utility_bin(pilot["utility"]),
        }
        selections.append(selection)
        states.append(
            {
                "selection_order": order,
                "source_index": selection["source_index"],
                "sample_id": pilot["sample_id"],
                "suite": pilot["suite"],
                "task_index": pilot["task_index"],
                "episode_index": pilot["episode_index"],
                "frame_index": pilot["frame_index"],
                "selection_bin": selection["selection_bin"],
                "pilot_utility": pilot["utility"],
                "valid_length": pilot["valid_length"],
                "pilot_seed": pilot["seed"],
                "pilot_record_sha256": pilot_record_sha256(pilot),
            }
        )
    selection_sha = sha256_json(states)
    compatibility = {
        "schema_version": 1,
        "kind": SOURCE_BUNDLE_KIND,
        "pilot_manifest_fingerprint": "a" * 64,
        "pilot_manifest_sha256": "1" * 64,
        "pilot_records_sha256": "2" * 64,
        "selection_plan_sha256": selection_sha,
        "num_states": len(states),
        "replicate_base_seeds": list(TARGET_BASE_SEEDS),
        "reuse_base_seed": 42,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "d" * 64,
        "vae_sha256": "e" * 64,
        "collection_git_commit": "f" * 40,
    }
    fingerprint = sha256_json(compatibility)
    manifest = {
        "schema_version": 1,
        "kind": SOURCE_BUNDLE_KIND,
        "compatibility_fingerprint": fingerprint,
        "compatibility": compatibility,
        "pilot": {
            "manifest_fingerprint": "a" * 64,
            "manifest_sha256": "1" * 64,
            "records_sha256": "2" * 64,
        },
        "selection": {
            "num_states": len(states),
            "ordered_states": states,
            "ordered_states_sha256": selection_sha,
        },
        "replicates": {
            "base_seeds": list(TARGET_BASE_SEEDS),
            "count": 5,
            "reuse_base_seed": 42,
            "reuse_replicate_index": 0,
            "expected_record_count": len(states) * 5,
        },
    }
    records = []
    for pilot, selection, values in zip(pilots, selections, utility_sets):
        for replicate_index, (base_seed, utility) in enumerate(
            zip(TARGET_BASE_SEEDS, values)
        ):
            core = deepcopy(pilot)
            core["seed"] = stable_sample_seed(
                base_seed,
                pilot["dataset_id"],
                pilot["episode_index"],
                pilot["frame_index"],
            )
            core["e0"] = max(utility, 0.0) + 0.2
            core["efull"] = core["e0"] - utility
            core["utility"] = utility
            records.append(
                augment_stability_record(
                    core,
                    pilot_record=pilot,
                    selection_entry=selection,
                    replicate_index=replicate_index,
                    replicate_base_seed=base_seed,
                    stability_manifest_compatibility_fingerprint=fingerprint,
                    collection_git_sha="f" * 40,
                )
            )
    root.mkdir()
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records_path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest_path, records_path, selection_sha


def _load_synthetic_source(tmp_path: Path, utility_sets: list[list[float]]):
    manifest_path, records_path, selection_sha = _write_source_bundle(
        tmp_path / "source", utility_sets
    )
    return load_verified_source_bundle(
        manifest_path,
        records_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_records_sha256=sha256_file(records_path),
        expected_selection_plan_sha256=selection_sha,
        expected_num_states=len(utility_sets),
    )


def test_source_loader_binds_file_hash_plan_and_every_grid_cell(tmp_path):
    source = _load_synthetic_source(tmp_path, [[0.01] * 5, [-0.01] * 5])
    assert source.num_states == 2
    assert len(source.record_index) == 10
    assert source.source_binding["base_seeds"] == list(TARGET_BASE_SEEDS)
    assert len(source.input_plan_sha256) == 64

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_verified_source_bundle(
            source.manifest_path,
            source.records_path,
            expected_manifest_sha256="0" * 64,
            expected_records_sha256=source.records_sha256,
            expected_selection_plan_sha256=source.selection_plan_sha256,
            expected_num_states=2,
        )
    with pytest.raises(ValueError, match="immutable plan"):
        load_verified_source_bundle(
            source.manifest_path,
            source.records_path,
            expected_manifest_sha256=source.manifest_sha256,
            expected_records_sha256=source.records_sha256,
            expected_selection_plan_sha256="0" * 64,
            expected_num_states=2,
        )


def test_source_loader_rejects_self_consistent_external_row_tamper(tmp_path):
    manifest_path, records_path, selection_sha = _write_source_bundle(
        tmp_path / "source", [[0.01] * 5]
    )
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[4]["current_proprio"] = [99.0]
    records_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cross-seed mismatch for current_proprio"):
        load_verified_source_bundle(
            manifest_path,
            records_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_records_sha256=sha256_file(records_path),
            expected_selection_plan_sha256=selection_sha,
            expected_num_states=1,
        )


def test_target_statistics_and_three_condition_confidence_rule(tmp_path):
    source = _load_synthetic_source(
        tmp_path,
        [
            [0.01, 0.01, 0.01, 0.01, 0.01],
            [-0.01, -0.01, -0.01, -0.01, -0.01],
            [0.001, 0.001, 0.001, 0.001, -0.0002],
        ],
    )
    manifest, targets = build_target_bundle(source)
    positive, negative, noisy = targets
    assert positive["utility_mean"] == pytest.approx(0.01)
    assert positive["utility_median"] == pytest.approx(0.01)
    assert positive["utility_sample_std"] == pytest.approx(0.0)
    assert positive["utility_sem"] == pytest.approx(0.0)
    assert positive["t95_ci_low"] == pytest.approx(0.01)
    assert positive["direction"] == "positive"
    assert positive["sign_agreement"] == 1.0
    assert positive["high_confidence"] is True
    assert positive["uncertain"] is False
    assert negative["direction"] == "negative"
    assert negative["high_confidence"] is True

    values = [0.001, 0.001, 0.001, 0.001, -0.0002]
    expected_std = statistics.stdev(values)
    expected_sem = expected_std / math.sqrt(5)
    assert noisy["utility_sample_std"] == pytest.approx(expected_std)
    assert noisy["utility_sem"] == pytest.approx(expected_sem)
    assert noisy["mean_outside_deadband"] is True
    assert noisy["direction_seed_count"] == 4
    assert noisy["enough_same_direction_seeds"] is True
    assert noisy["t95_ci_low"] <= DEFAULT_DEADBAND_EPSILON
    assert noisy["ci_clears_deadband"] is False
    assert noisy["high_confidence"] is False
    assert noisy["uncertain"] is True
    assert manifest["summary"] == {
        "num_states": 3,
        "high_confidence_count": 2,
        "uncertain_count": 1,
        "high_confidence_positive_count": 1,
        "high_confidence_negative_count": 1,
    }
    assert manifest["policy"]["candidate_training_weight"]["status"].startswith(
        "not_defined"
    )


def test_target_hash_manifest_and_atomic_bundle_are_tamper_evident(tmp_path):
    source = _load_synthetic_source(tmp_path, [[0.01] * 5, [-0.01] * 5])
    manifest, targets = build_target_bundle(source)
    validate_target_manifest(manifest)
    for target in targets:
        validate_target_record(target)
        assert target["target_sha256"] == target_record_sha256(target)

    output = tmp_path / "target-v2"
    manifest_path, targets_path = write_target_bundle(output, manifest, targets)
    loaded_manifest, loaded_targets = load_target_bundle(
        output,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_targets_sha256=sha256_file(targets_path),
        expected_num_states=2,
    )
    assert loaded_manifest["kind"] == TARGET_BUNDLE_KIND
    assert loaded_targets == targets
    with pytest.raises(FileExistsError, match="immutable"):
        write_target_bundle(output, manifest, targets)

    broken = deepcopy(loaded_targets)
    broken[0]["utility_mean"] += 1.0
    targets_path.write_text(
        "".join(canonical_json(row) + "\n" for row in broken), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="target_sha256"):
        load_target_bundle(
            output,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_targets_sha256=sha256_file(targets_path),
            expected_num_states=2,
        )


def _validation_core(target: dict, base_seed: int, utility: float = 0.002) -> dict:
    e0 = 0.2 + max(utility, 0.0)
    return {
        "sample_id": target["sample_id"],
        "dataset_id": target["dataset_id"],
        "dataset_name": target["dataset_name"],
        "suite": target["suite"],
        "episode_index": target["episode_index"],
        "episode_id": target["episode_id"],
        "frame_index": target["frame_index"],
        "task_index": target["task_index"],
        "task_id": target["task_id"],
        "task_id_source": target["task_id_source"],
        "task": target["task"],
        "seed": stable_sample_seed(
            base_seed,
            target["dataset_id"],
            target["episode_index"],
            target["frame_index"],
        ),
        "num_inference_steps": 10,
        "n0": 0,
        "nfull": 10,
        "e0": e0,
        "efull": e0 - utility,
        "utility": utility,
        "valid_length": target["valid_length"],
        "target_action_shape": target["target_action_shape"],
        "pred_n0_shape": target["target_action_shape"],
        "pred_nfull_shape": target["target_action_shape"],
        "input_hashes": target["input_hashes"],
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
        "git_sha": "b" * 40,
    }


def _validation_row(target: dict, index: int, target_manifest: dict) -> dict:
    return augment_validation_record(
        _validation_core(target, VALIDATION_BASE_SEEDS[index]),
        target_record=target,
        validation_index=index,
        validation_base_seed=VALIDATION_BASE_SEEDS[index],
        validation_manifest_compatibility_fingerprint="8" * 64,
        target_manifest_sha256="6" * 64,
        target_targets_sha256="7" * 64,
        target_manifest_compatibility_fingerprint=target_manifest[
            "compatibility_fingerprint"
        ],
        collection_git_sha="b" * 40,
    )


def test_validation_plan_record_and_strict_partial_resume_rebind(tmp_path):
    source = _load_synthetic_source(tmp_path, [[0.01] * 5])
    manifest, targets = build_target_bundle(source)
    target = targets[0]
    plan = build_validation_plan(manifest, targets)
    assert len(plan) == 4
    assert [row["validation_base_seed"] for row in plan] == list(
        VALIDATION_BASE_SEEDS
    )
    assert [row["validation_replicate_index"] for row in plan] == [0, 1, 2, 3]
    assert [row["global_seed_index"] for row in plan] == [5, 6, 7, 8]

    row = _validation_row(target, 0, manifest)
    validate_validation_record(
        row,
        expected_validation_manifest_fingerprint="8" * 64,
        expected_target_manifest_sha256="6" * 64,
        expected_target_targets_sha256="7" * 64,
        expected_target_manifest_fingerprint=manifest["compatibility_fingerprint"],
    )
    report = validate_validation_grid(
        {(target["source_index"], 0): row}, targets, allow_incomplete=True
    )
    assert report["completed_count"] == 1
    assert report["missing_count"] == 3

    tampered_latency = deepcopy(row)
    tampered_latency["n0_latency_ms"] += 1.0
    with pytest.raises(ValueError, match="validation_record_sha256"):
        validate_validation_record(tampered_latency)
    tampered_latency["validation_record_sha256"] = validation_record_sha256(
        tampered_latency
    )
    validate_validation_record(tampered_latency)

    broken = deepcopy(row)
    broken["target_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target rebind mismatch for target_sha256"):
        validate_validation_grid(
            {(target["source_index"], 0): broken},
            targets,
            allow_incomplete=True,
        )


def test_validation_complete_grid_and_resume_duplicate_rejection(tmp_path):
    source = _load_synthetic_source(tmp_path, [[0.01] * 5])
    manifest, targets = build_target_bundle(source)
    target = targets[0]
    rows = [_validation_row(target, index, manifest) for index in range(4)]
    index = {(target["source_index"], i): row for i, row in enumerate(rows)}
    report = validate_validation_grid(index, targets)
    assert report["is_complete"] is True
    assert report["completed_count"] == 4

    records_path = tmp_path / "validation.jsonl"
    records_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    loaded = load_validation_record_index(
        records_path,
        expected_validation_manifest_fingerprint="8" * 64,
        expected_target_manifest_sha256="6" * 64,
        expected_target_targets_sha256="7" * 64,
        expected_target_manifest_fingerprint=manifest["compatibility_fingerprint"],
    )
    assert set(loaded) == set(index)
    records_path.write_text(
        records_path.read_text(encoding="utf-8") + canonical_json(rows[0]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate validation resume key"):
        load_validation_record_index(records_path)


REAL_SOURCE = Path(
    "/root/feihong/FastWAM/utility_results/"
    "libero_demo_utility_multiseed100x5_730f27b_seed42_46"
)


@pytest.mark.skipif(not REAL_SOURCE.is_dir(), reason="formal Phase-2.5 source not mounted")
def test_formal_phase25_source_yields_preregistered_24_high_confidence_targets():
    source = load_verified_source_bundle(
        REAL_SOURCE / "manifest.json",
        REAL_SOURCE / "records.jsonl",
        expected_manifest_sha256=OFFICIAL_SOURCE_MANIFEST_SHA256,
        expected_records_sha256=OFFICIAL_SOURCE_RECORDS_SHA256,
        expected_selection_plan_sha256=OFFICIAL_SOURCE_SELECTION_PLAN_SHA256,
    )
    manifest, targets = build_target_bundle(source)
    assert len(targets) == 100
    assert manifest["summary"] == {
        "num_states": 100,
        "high_confidence_count": 24,
        "uncertain_count": 76,
        "high_confidence_positive_count": 12,
        "high_confidence_negative_count": 12,
    }
