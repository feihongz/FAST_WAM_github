from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy

import pytest

from experiments.libero.gate.demo_utility import stable_sample_seed
from experiments.libero.gate.demo_utility_stability import (
    DEFAULT_REPLICATE_BASE_SEEDS,
    DEFAULT_SUITE_BIN_QUOTAS,
    augment_stability_record,
    build_replicate_plan,
    build_stability_selection,
    load_pilot_records,
    load_stability_record_index,
    pilot_record_sha256,
    replicate_id,
    utility_bin,
    validate_complete_grid,
    validate_replicate_base_seeds,
    validate_stability_record,
)


def _sha_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pilot_row(
    *,
    source_index=7,
    suite="libero_goal",
    task_index=0,
    episode_index=1,
    frame_index=2,
    utility=0.1,
    valid_length=32,
    fingerprint="a" * 64,
):
    dataset = f"{suite}_no_noops_lerobot"
    sample_id = f"{dataset}/episode_{episode_index:06d}/frame_{frame_index:06d}"
    identity = {
        "dataset_name": dataset,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": task_index,
        "task": f"task {task_index}",
    }
    components = {
        "input_image": hashlib.sha256(f"image-{source_index}".encode()).hexdigest(),
        "proprio": hashlib.sha256(f"proprio-{source_index}".encode()).hexdigest(),
        "context": hashlib.sha256(f"context-{source_index}".encode()).hexdigest(),
        "context_mask": hashlib.sha256(f"mask-{source_index}".encode()).hexdigest(),
        "valid_target_action": hashlib.sha256(f"action-{source_index}".encode()).hexdigest(),
        "action_is_pad": hashlib.sha256(f"pad-{source_index}".encode()).hexdigest(),
    }
    hashes = {**components, "combined": _sha_json(components)}
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
        "frame_index": frame_index,
        "task_index": task_index,
        "task_id": task_index,
        "task_id_source": "lerobot_task_index",
        "task": f"task {task_index}",
        "seed": stable_sample_seed(42, dataset, episode_index, frame_index),
        "num_inference_steps": 10,
        "n0": 0,
        "nfull": 10,
        "e0": e0,
        "efull": efull,
        "utility": utility,
        "valid_length": valid_length,
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
            **identity,
        },
        "manifest_compatibility_fingerprint": fingerprint,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "d" * 64,
        "vae_sha256": "e" * 64,
        "git_sha": "f" * 40,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.001000001, "SP"),
        (0.001, "MP"),
        (0.000100001, "MP"),
        (0.0001, "NZ"),
        (-0.0001, "NZ"),
        (-0.000100001, "MN"),
        (-0.001, "MN"),
        (-0.001000001, "SN"),
    ],
)
def test_utility_bin_boundaries(value, expected):
    assert utility_bin(value) == expected


def test_replicate_seed_grid_and_ids_are_strict():
    assert validate_replicate_base_seeds([42, 43, 44, 45, 46]) == (
        42,
        43,
        44,
        45,
        46,
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_replicate_base_seeds([42, 43, 44, 46, 45])
    assert replicate_id("sample", 3, 45) == "sample/replicate_03_base_seed_45"


def _manifest(records):
    plan = [row["source_metadata"]["requested_sample_idx"] for row in records]
    selection_sha = _sha_json(plan)
    compatibility = {"selection_sha256": selection_sha}
    return {
        "compatibility": compatibility,
        "compatibility_fingerprint": _sha_json(compatibility),
        "selection": {
            "num_samples": len(plan),
            "ordered_selected_source_indices": plan,
            "ordered_selected_source_indices_sha256": selection_sha,
        },
    }


def test_strict_pilot_loader_reorders_to_manifest_and_rejects_tampering(tmp_path):
    rows = [_pilot_row(source_index=7), _pilot_row(source_index=8, episode_index=2)]
    manifest = _manifest(rows)
    for row in rows:
        row["manifest_compatibility_fingerprint"] = manifest["compatibility_fingerprint"]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_pilot_records(records_path, expected_count=2)
    assert [row["source_metadata"]["requested_sample_idx"] for row in loaded] == [7, 8]

    broken = deepcopy(rows)
    broken[1]["seed"] += 1
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in broken), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="seed mismatch"):
        load_pilot_records(records_path, expected_count=2)


def _synthetic_candidates():
    rows = []
    source = 0
    utility_by_bin = {"SP": 0.01, "SN": -0.01, "MP": 0.0005, "MN": -0.0005, "NZ": 0.0}
    for suite_index, (suite, quotas) in enumerate(DEFAULT_SUITE_BIN_QUOTAS.items()):
        for selection_bin, quota in quotas.items():
            for local in range(quota + 5):
                task_index = (source + local) % 10
                rows.append(
                    _pilot_row(
                        source_index=source,
                        suite=suite,
                        task_index=task_index,
                        episode_index=suite_index * 1000 + source,
                        utility=utility_by_bin[selection_bin],
                        valid_length=16 if source < 70 else 32,
                    )
                )
                source += 1
    return rows


def test_exact_selector_real_constraints_and_determinism():
    # Synthetic feasibility is awkward because all exact constraints interact;
    # the checked-in Pilot-500 is the authoritative integration fixture when present.
    path = (
        "/root/feihong/FastWAM/utility_results/"
        "libero_demo_utility_pilot500_508937a_seed42/records.jsonl"
    )
    from pathlib import Path

    if not Path(path).is_file():
        pytest.skip("Pilot-500 artifact is not available")
    records = load_pilot_records(path)
    selected = build_stability_selection(records)
    repeated = build_stability_selection(list(reversed(records)))

    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in repeated
    ]
    assert len(selected) == 100
    assert Counter(row["selection_bin"] for row in selected) == {
        "SP": 25,
        "SN": 25,
        "MP": 13,
        "MN": 12,
        "NZ": 25,
    }
    assert Counter(row["suite"] for row in selected) == {
        suite: 25 for suite in DEFAULT_SUITE_BIN_QUOTAS
    }
    assert sum(row["valid_length"] < 32 for row in selected) == 16
    task_counts = Counter((row["suite"], row["task_index"]) for row in selected)
    assert Counter(task_counts.values()) == {2: 20, 3: 20}
    assert len({(row["suite"], row["episode_index"]) for row in selected}) == 100
    assert all("source_metadata" in row for row in selected)

    plan = build_replicate_plan(selected)
    assert len(plan) == 500
    assert len({(row["source_index"], row["replicate_index"]) for row in plan}) == 500


def _selection_entry(pilot):
    return {
        **pilot,
        "source_index": pilot["source_metadata"]["requested_sample_idx"],
        "selection_order": 0,
        "selection_bin": utility_bin(pilot["utility"]),
    }


def test_augment_validate_and_resume_reject_corruption(tmp_path):
    pilot = _pilot_row()
    selection = _selection_entry(pilot)
    fingerprint = "b" * 64
    rep0 = augment_stability_record(
        pilot,
        pilot_record=pilot,
        selection_entry=selection,
        replicate_index=0,
        replicate_base_seed=42,
        stability_manifest_compatibility_fingerprint=fingerprint,
    )
    validate_stability_record(rep0, expected_stability_manifest_fingerprint=fingerprint)
    assert rep0["reused_from_pilot"] is True
    assert rep0["inference_origin"] == "pilot_reuse"
    assert rep0["source_pilot_record_sha256"] == pilot_record_sha256(pilot)

    new_core = deepcopy(pilot)
    new_core["seed"] = stable_sample_seed(
        43, pilot["dataset_id"], pilot["episode_index"], pilot["frame_index"]
    )
    new_core.update(e0=0.3, efull=0.2, utility=0.1)
    rep1 = augment_stability_record(
        new_core,
        pilot_record=pilot,
        selection_entry=selection,
        replicate_index=1,
        replicate_base_seed=43,
        stability_manifest_compatibility_fingerprint=fingerprint,
    )
    assert rep1["reused_from_pilot"] is False
    assert rep1["inference_origin"] == "new_inference"

    path = tmp_path / "stability.jsonl"
    path.write_text(json.dumps(rep0) + "\n" + json.dumps(rep1) + "\n", encoding="utf-8")
    index = load_stability_record_index(
        path, expected_stability_manifest_fingerprint=fingerprint
    )
    assert set(index) == {(7, 0), (7, 1)}
    report = validate_complete_grid(
        index,
        [selection],
        allow_incomplete=True,
    )
    assert report["completed_count"] == 2
    assert report["reused_from_pilot_count"] == 1
    assert report["new_inference_count"] == 1

    bad = deepcopy(rep1)
    bad["reused_from_pilot"] = True
    with pytest.raises(ValueError, match="reused_from_pilot"):
        validate_stability_record(bad)


def test_complete_grid_rejects_cross_replicate_state_mismatch():
    pilot = _pilot_row()
    selection = _selection_entry(pilot)
    fingerprint = "b" * 64
    index = {}
    for replicate_index, base_seed in enumerate(DEFAULT_REPLICATE_BASE_SEEDS):
        core = deepcopy(pilot)
        core["seed"] = stable_sample_seed(
            base_seed, pilot["dataset_id"], pilot["episode_index"], pilot["frame_index"]
        )
        record = augment_stability_record(
            core,
            pilot_record=pilot,
            selection_entry=selection,
            replicate_index=replicate_index,
            replicate_base_seed=base_seed,
            stability_manifest_compatibility_fingerprint=fingerprint,
        )
        index[(7, replicate_index)] = record
    index[(7, 4)]["current_proprio"] = [999.0]
    with pytest.raises(ValueError, match="selection rebind mismatch for current_proprio"):
        validate_complete_grid(index, [selection])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("sample_id", "libero_goal_no_noops_lerobot/episode_999999/frame_999999"),
        ("dataset_id", "libero_object_no_noops_lerobot"),
        ("selection_order", 9),
        ("selection_bin", "SN"),
        ("pilot_e0", 9.0),
        ("pilot_efull", 9.0),
        ("pilot_utility", -9.0),
        ("pilot_seed", 9),
        ("pilot_valid_length", 9),
        ("pilot_input_combined_sha256", "9" * 64),
        ("valid_length", 9),
        ("target_action_shape", [31, 7]),
        ("input_hashes", {"tampered": "state"}),
        ("current_proprio", [9.0]),
        ("source_metadata", {"requested_sample_idx": 7, "tampered": True}),
        ("source_pilot_record_sha256", "9" * 64),
        ("pilot_manifest_compatibility_fingerprint", "9" * 64),
        ("checkpoint_sha256", "9" * 64),
        ("dataset_stats_sha256", "9" * 64),
        ("vae_sha256", "9" * 64),
        ("git_sha", "9" * 40),
    ],
)
def test_partial_resume_strictly_rebinds_each_existing_row_to_pilot(field, replacement):
    pilot = _pilot_row()
    selection = _selection_entry(pilot)
    record = augment_stability_record(
        pilot,
        pilot_record=pilot,
        selection_entry=selection,
        replicate_index=0,
        replicate_base_seed=42,
        stability_manifest_compatibility_fingerprint="b" * 64,
    )
    record[field] = replacement

    # This is deliberately a 1/5 partial grid: rebinding must run before an
    # incomplete collection is accepted for resume.
    with pytest.raises(ValueError, match="selection rebind mismatch"):
        validate_complete_grid({(7, 0): record}, [selection], allow_incomplete=True)


def test_incomplete_default_still_rebinds_before_reporting_missing_cells():
    pilot = _pilot_row()
    selection = _selection_entry(pilot)
    record = augment_stability_record(
        pilot,
        pilot_record=pilot,
        selection_entry=selection,
        replicate_index=0,
        replicate_base_seed=42,
        stability_manifest_compatibility_fingerprint="b" * 64,
    )
    record["pilot_utility"] = -9.0
    with pytest.raises(ValueError, match="selection rebind mismatch for pilot_utility"):
        validate_complete_grid({(7, 0): record}, [selection])
