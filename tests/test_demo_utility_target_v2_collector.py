from __future__ import annotations

import copy
import json

import pytest

from experiments.libero.gate.collect_demo_utility import _sha256_json
from experiments.libero.gate.collect_demo_utility_target_v2_validation import (
    AUDIT_KIND,
    VALIDATION_BASE_SEEDS,
    _ensure_completion_seal,
    _validate_completion_seal,
    _validate_manifest_integrity,
    _validate_validation_seeds,
    collect_validation_grid,
)


class CountingDataset:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __getitem__(self, source_index: int) -> dict[str, int]:
        self.calls.append(source_index)
        return {"source_index": source_index}


def _target(source_index: int, order: int) -> dict:
    return {
        "target_id": f"target-v2/{order:03d}",
        "target_sha256": f"{source_index:064x}",
        "source_index": source_index,
        "selection_order": order,
        "sample_id": f"suite/episode_{source_index:06d}/frame_000000",
    }


def test_validation_seed_grid_is_frozen_and_disjoint_from_training_seeds():
    assert _validate_validation_seeds([47, 48, 49, 50]) == VALIDATION_BASE_SEEDS
    with pytest.raises(ValueError, match="exactly"):
        _validate_validation_seeds([47, 48, 49])
    with pytest.raises(ValueError, match="exactly"):
        _validate_validation_seeds([42, 48, 49, 50])


def test_grid_decodes_each_state_once_and_all_four_cells_are_new_inference():
    dataset = CountingDataset()
    targets = [_target(11, 0), _target(22, 1)]
    inference_calls: list[tuple[int, int]] = []
    written: list[dict] = []

    def infer(model, sample, target, base_seed):
        assert model == "frozen-model"
        inference_calls.append((sample["source_index"], base_seed))
        return {"seed": base_seed, "utility": base_seed / 100.0}

    def finalize(utility, target, validation_index, base_seed, sample):
        return {
            **utility,
            "source_index": sample["source_index"],
            "validation_replicate_index": validation_index,
            "validation_base_seed": base_seed,
            "global_seed_index": validation_index + 5,
            "inference_origin": "independent_validation",
            "reused_from_target": False,
        }

    summary = collect_validation_grid(
        dataset=dataset,
        model="frozen-model",
        target_records=targets,
        base_seeds=VALIDATION_BASE_SEEDS,
        existing_keys=set(),
        infer_record=infer,
        finalize_record=finalize,
        write_record=written.append,
    )

    assert dataset.calls == [11, 22]
    assert inference_calls == [
        (11, 47), (11, 48), (11, 49), (11, 50),
        (22, 47), (22, 48), (22, 49), (22, 50),
    ]
    assert [row["validation_base_seed"] for row in written] == [47, 48, 49, 50] * 2
    assert all(row["inference_origin"] == "independent_validation" for row in written)
    assert all(row["reused_from_target"] is False for row in written)
    assert summary == {"new": 8, "inferred": 8, "reused": 0, "errors": 0}


def test_grid_resume_uses_source_and_local_validation_index_composite_key():
    dataset = CountingDataset()
    targets = [_target(11, 0), _target(22, 1)]
    existing = {(11, 0), (11, 2)} | {(22, index) for index in range(4)}
    inference_calls: list[int] = []
    written: list[dict] = []

    def infer(model, sample, target, base_seed):
        inference_calls.append(base_seed)
        return {"seed": base_seed}

    summary = collect_validation_grid(
        dataset=dataset,
        model=object(),
        target_records=targets,
        base_seeds=VALIDATION_BASE_SEEDS,
        existing_keys=existing,
        infer_record=infer,
        finalize_record=lambda utility, target, index, seed, sample: {
            **utility,
            "source_index": sample["source_index"],
            "validation_replicate_index": index,
        },
        write_record=written.append,
    )

    assert dataset.calls == [11]
    assert inference_calls == [48, 50]
    assert [(row["source_index"], row["validation_replicate_index"]) for row in written] == [
        (11, 1),
        (11, 3),
    ]
    assert summary == {"new": 2, "inferred": 2, "reused": 0, "errors": 0}


def test_fully_resumed_grid_does_not_decode_data_or_require_a_model():
    dataset = CountingDataset()
    summary = collect_validation_grid(
        dataset=dataset,
        model=None,
        target_records=[_target(11, 0)],
        base_seeds=VALIDATION_BASE_SEEDS,
        existing_keys={(11, index) for index in range(4)},
        infer_record=lambda *args: pytest.fail("inference must not run"),
        finalize_record=lambda *args: pytest.fail("finalization must not run"),
        write_record=lambda *args: pytest.fail("writing must not run"),
    )
    assert dataset.calls == []
    assert summary == {"new": 0, "inferred": 0, "reused": 0, "errors": 0}


def _manifest() -> dict:
    ordered_targets = [
        {
            "selection_order": 0,
            "source_index": 11,
            "sample_id": "suite/episode_000011/frame_000000",
            "target_id": "target-v2/000",
            "target_sha256": "a" * 64,
        }
    ]
    selection_sha = _sha256_json(ordered_targets)
    compatibility = {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "phase25_manifest_fingerprint": "1" * 64,
        "phase25_manifest_sha256": "2" * 64,
        "phase25_records_sha256": "3" * 64,
        "phase25_selection_plan_sha256": "4" * 64,
        "target_v2_manifest_fingerprint": "5" * 64,
        "target_v2_manifest_sha256": "6" * 64,
        "target_v2_targets_sha256": "7" * 64,
        "target_v2_selection_plan_sha256": "4" * 64,
        "validation_selection_sha256": selection_sha,
        "num_states": 1,
        "validation_base_seeds": [47, 48, 49, 50],
        "global_seed_indices": [5, 6, 7, 8],
        "expected_record_count": 4,
    }
    return {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": _sha256_json(compatibility),
        "phase25": {
            "manifest_fingerprint": "1" * 64,
            "manifest_sha256": "2" * 64,
            "records_sha256": "3" * 64,
            "selection_plan_sha256": "4" * 64,
        },
        "target_v2": {
            "manifest_fingerprint": "5" * 64,
            "manifest_sha256": "6" * 64,
            "targets_sha256": "7" * 64,
            "selection_plan_sha256": "4" * 64,
        },
        "selection": {
            "num_states": 1,
            "ordered_targets": ordered_targets,
            "ordered_targets_sha256": selection_sha,
        },
        "replicates": {
            "base_seeds": [47, 48, 49, 50],
            "global_seed_indices": [5, 6, 7, 8],
            "count": 4,
            "expected_record_count": 4,
            "all_new_inference": True,
        },
    }


def test_validation_manifest_rejects_selection_source_and_target_tampering():
    manifest = _manifest()
    _validate_manifest_integrity(manifest)

    tampered_selection = copy.deepcopy(manifest)
    tampered_selection["selection"]["ordered_targets"][0]["source_index"] = 99
    with pytest.raises(ValueError, match="selection digest"):
        _validate_manifest_integrity(tampered_selection)

    tampered_phase25 = copy.deepcopy(manifest)
    tampered_phase25["phase25"]["records_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="Phase-2.5 records_sha256"):
        _validate_manifest_integrity(tampered_phase25)

    tampered_target = copy.deepcopy(manifest)
    tampered_target["target_v2"]["targets_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="Target-V2 targets_sha256"):
        _validate_manifest_integrity(tampered_target)


def test_validation_manifest_forbids_any_reuse_contract():
    manifest = _manifest()
    manifest["replicates"]["all_new_inference"] = False
    with pytest.raises(ValueError, match="new inference"):
        _validate_manifest_integrity(manifest)


def test_completion_seal_binds_final_records_and_zero_errors(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    completion_path = tmp_path / "completion.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    rows = [
        {"row": index, "validation_record_sha256": "a" * 64}
        for index in range(4)
    ]
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    errors_path.write_text("", encoding="utf-8")

    seal = _ensure_completion_seal(
        completion_path,
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
    )
    assert seal["records_count"] == 4
    assert seal["errors_count"] == 0
    assert completion_path.is_file()
    _validate_completion_seal(
        completion_path,
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
    )

    rows[-1]["row"] = 9
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="records_sha256"):
        _validate_completion_seal(
            completion_path,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
        )
